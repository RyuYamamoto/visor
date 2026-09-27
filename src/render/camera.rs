//! Pure camera math for the three view types (wgpu/egui-independent; unit tested).

use nalgebra::{Isometry3, Matrix4, Orthographic3, Perspective3, Point3, Vector3};

/// Vertical field of view [rad].
const FOVY: f32 = std::f32::consts::FRAC_PI_4;
const NEAR: f32 = 0.1;
const FAR: f32 = 1000.0;
/// Pitch limit (+/-89 deg). Combined with fixed up=+Z, keeps the camera off the gimbal singularity.
const PITCH_LIMIT: f32 = 89.0 * std::f32::consts::PI / 180.0;
const DISTANCE_MIN: f32 = 0.1;
const DISTANCE_MAX: f32 = 500.0;
/// Rotation sensitivity [rad/pt].
const ROTATE_SPEED: f32 = 0.008;
/// Scroll-zoom sensitivity [1/pt].
const ZOOM_SPEED: f32 = 0.002;

/// TopDownOrtho half-height (vertical half-extent) clamp [m]; matches the distance clamp's visible scale.
const HALF_HEIGHT_MIN: f32 = 0.1;
const HALF_HEIGHT_MAX: f32 = 500.0;
/// TopDownOrtho eye altitude above the look-at point [m]; large enough that ground geometry stays within near/far.
const TOPDOWN_ALTITUDE: f32 = 500.0;
/// FPS forward/back dolly sensitivity [m/pt].
const FPS_DOLLY_SPEED: f32 = 0.02;
/// FPS strafe (pan) sensitivity [m/pt].
const FPS_PAN_SPEED: f32 = 0.01;

/// Correction matrix mapping nalgebra Perspective3's OpenGL depth (-1..1) to wgpu's 0..1.
#[rustfmt::skip]
const OPENGL_TO_WGPU: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

/// Orbit-camera state around the target (world is ROS-style right-handed Z-up).
pub struct OrbitCamera {
    pub target: Point3<f32>,
    /// Azimuth around the world Z axis [rad].
    pub yaw: f32,
    /// Elevation from the XY plane [rad] (positive looks down).
    pub pitch: f32,
    /// Distance from the target [m].
    pub distance: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Point3::origin(),
            yaw: -45.0_f32.to_radians(),
            pitch: 35.0_f32.to_radians(),
            distance: 10.0,
        }
    }
}

impl OrbitCamera {
    pub fn eye(&self) -> Point3<f32> {
        self.target
            + self.distance
                * Vector3::new(
                    self.pitch.cos() * self.yaw.cos(),
                    self.pitch.cos() * self.yaw.sin(),
                    self.pitch.sin(),
                )
    }

    /// Left-drag rotation (delta in egui logical points; dragging down increases the look-down angle).
    pub fn rotate(&mut self, delta_x: f32, delta_y: f32) {
        self.yaw -= delta_x * ROTATE_SPEED;
        self.pitch = (self.pitch + delta_y * ROTATE_SPEED).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    /// Pan: move target so on-screen motion matches cursor motion on the plane at the target's depth.
    pub fn pan(&mut self, delta_x: f32, delta_y: f32, viewport_height_pt: f32) {
        if viewport_height_pt <= 0.0 {
            return;
        }
        let scale = 2.0 * self.distance * (FOVY * 0.5).tan() / viewport_height_pt;
        let forward = (self.target - self.eye()).normalize();
        let right = forward.cross(&Vector3::z()).normalize();
        let up_cam = right.cross(&forward);
        self.target += right * (-delta_x * scale) + up_cam * (delta_y * scale);
    }

    /// Wheel-scroll zoom (positive scroll moves closer).
    pub fn zoom_scroll(&mut self, scroll_y: f32) {
        self.set_distance(self.distance * (-scroll_y * ZOOM_SPEED).exp());
    }

    /// Multiplicative zoom, e.g. pinch (factor > 1 moves closer).
    pub fn zoom_factor(&mut self, factor: f32) {
        if factor > 0.0 {
            self.set_distance(self.distance / factor);
        }
    }

    fn set_distance(&mut self, distance: f32) {
        self.distance = distance.clamp(DISTANCE_MIN, DISTANCE_MAX);
    }

    /// Camera right/up unit vectors for billboarding point quads (derived the same way as the view matrix).
    pub fn basis(&self) -> (Vector3<f32>, Vector3<f32>) {
        // pitch is clamped to +/-89 deg, so forward is never parallel to Z.
        let forward = (self.target - self.eye()).normalize();
        let right = forward.cross(&Vector3::z()).normalize();
        let up = right.cross(&forward);
        (right, up)
    }

    /// View-projection (wgpu NDC depth 0..1); `follow` adds to both eye and target = translation-only follow.
    pub fn view_proj(&self, aspect: f32, follow: Vector3<f32>) -> Matrix4<f32> {
        let view = Isometry3::look_at_rh(&(self.eye() + follow), &(self.target + follow), &Vector3::z());
        let proj = Perspective3::new(aspect, FOVY, NEAR, FAR);
        OPENGL_TO_WGPU * proj.to_homogeneous() * view.to_homogeneous()
    }
}

/// The three selectable view types. Default = Orbit (the historical behavior).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViewType {
    #[default]
    Orbit,
    TopDownOrtho,
    Fps,
}

impl ViewType {
    pub const ALL: [ViewType; 3] = [ViewType::Orbit, ViewType::TopDownOrtho, ViewType::Fps];

    pub fn label(self) -> &'static str {
        match self {
            ViewType::Orbit => "Orbit",
            ViewType::TopDownOrtho => "TopDownOrtho",
            ViewType::Fps => "FPS",
        }
    }
}

/// Orthographic top-down camera: looks straight down world -Z, framing an XY region around `center`.
pub struct TopDownCamera {
    /// Look-at point on the ground plane (world XY) [m].
    pub center: [f32; 2],
    /// Screen rotation about the world Z axis [rad] (0 = world +Y points up on screen).
    pub rotation: f32,
    /// Half of the visible vertical extent [m] (the ortho zoom scale).
    pub half_height: f32,
}

impl Default for TopDownCamera {
    fn default() -> Self {
        Self {
            center: [0.0, 0.0],
            rotation: 0.0,
            half_height: 10.0,
        }
    }
}

impl TopDownCamera {
    /// Screen right/up unit vectors in world XY (used for both billboarding and panning).
    pub fn basis(&self) -> (Vector3<f32>, Vector3<f32>) {
        let right = Vector3::new(self.rotation.cos(), self.rotation.sin(), 0.0);
        let up = Vector3::new(-self.rotation.sin(), self.rotation.cos(), 0.0);
        (right, up)
    }

    /// Pan the look-at point on the ground plane so on-screen motion follows the cursor.
    pub fn pan(&mut self, delta_x: f32, delta_y: f32, viewport_height_pt: f32) {
        if viewport_height_pt <= 0.0 {
            return;
        }
        let scale = 2.0 * self.half_height / viewport_height_pt;
        let (right, up) = self.basis();
        let delta = right * (-delta_x * scale) + up * (delta_y * scale);
        self.center[0] += delta.x;
        self.center[1] += delta.y;
    }

    /// Rotate the view about the world Z axis (screen rotation).
    pub fn rotate_z(&mut self, delta_x: f32) {
        self.rotation += delta_x * ROTATE_SPEED;
    }

    /// Wheel-scroll zoom (positive scroll zooms in = smaller half-height).
    pub fn zoom_scroll(&mut self, scroll_y: f32) {
        self.set_half_height(self.half_height * (-scroll_y * ZOOM_SPEED).exp());
    }

    /// Multiplicative zoom, e.g. pinch (factor > 1 zooms in).
    pub fn zoom_factor(&mut self, factor: f32) {
        if factor > 0.0 {
            self.set_half_height(self.half_height / factor);
        }
    }

    fn set_half_height(&mut self, half_height: f32) {
        self.half_height = half_height.clamp(HALF_HEIGHT_MIN, HALF_HEIGHT_MAX);
    }

    /// View-projection (orthographic, wgpu NDC depth 0..1); `follow` translates the whole camera.
    pub fn view_proj(&self, aspect: f32, follow: Vector3<f32>) -> Matrix4<f32> {
        let target = Point3::new(self.center[0] + follow.x, self.center[1] + follow.y, follow.z);
        let eye = Point3::new(target.x, target.y, target.z + TOPDOWN_ALTITUDE);
        let (_, up) = self.basis();
        let view = Isometry3::look_at_rh(&eye, &target, &up);
        let half_w = self.half_height * aspect;
        let proj = Orthographic3::new(-half_w, half_w, -self.half_height, self.half_height, NEAR, FAR);
        OPENGL_TO_WGPU * proj.to_homogeneous() * view.to_homogeneous()
    }
}

/// First-person camera: an eye position with yaw/pitch look direction (perspective).
pub struct FpsCamera {
    /// Eye position in world space [m].
    pub eye: Point3<f32>,
    /// Azimuth around the world Z axis [rad].
    pub yaw: f32,
    /// Elevation from the XY plane [rad] (positive looks up).
    pub pitch: f32,
}

impl Default for FpsCamera {
    fn default() -> Self {
        Self {
            eye: Point3::new(-5.0, -5.0, 3.0),
            yaw: 45.0_f32.to_radians(),
            pitch: -20.0_f32.to_radians(),
        }
    }
}

impl FpsCamera {
    /// Unit forward (look) direction from yaw/pitch.
    fn forward(&self) -> Vector3<f32> {
        Vector3::new(
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
        )
    }

    /// Look around (yaw/pitch); pitch is clamped to +/-89 deg (dragging down looks down).
    pub fn look(&mut self, delta_x: f32, delta_y: f32) {
        self.yaw -= delta_x * ROTATE_SPEED;
        self.pitch = (self.pitch - delta_y * ROTATE_SPEED).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    /// Move forward/back along the look direction (positive scroll moves forward).
    pub fn dolly(&mut self, scroll_y: f32) {
        self.eye += self.forward() * (scroll_y * FPS_DOLLY_SPEED);
    }

    /// Strafe: move the eye along screen right/up (RViz FPS "push" convention: eye follows drag direction).
    pub fn pan(&mut self, delta_x: f32, delta_y: f32, _viewport_height_pt: f32) {
        let (right, up) = self.basis();
        self.eye += right * (delta_x * FPS_PAN_SPEED) + up * (-delta_y * FPS_PAN_SPEED);
    }

    /// Camera right/up unit vectors (up=+Z world convention; pitch clamp keeps forward off vertical).
    pub fn basis(&self) -> (Vector3<f32>, Vector3<f32>) {
        let forward = self.forward();
        let right = forward.cross(&Vector3::z()).normalize();
        let up = right.cross(&forward);
        (right, up)
    }

    /// View-projection (perspective, wgpu NDC depth 0..1); `follow` translates the whole camera.
    pub fn view_proj(&self, aspect: f32, follow: Vector3<f32>) -> Matrix4<f32> {
        let eye = self.eye + follow;
        let target = eye + self.forward();
        let view = Isometry3::look_at_rh(&eye, &target, &Vector3::z());
        let proj = Perspective3::new(aspect, FOVY, NEAR, FAR);
        OPENGL_TO_WGPU * proj.to_homogeneous() * view.to_homogeneous()
    }
}

/// Holds all three camera states plus the active `view_type`, so switching views preserves each view's pose.
#[derive(Default)]
pub struct ViewCameras {
    pub view_type: ViewType,
    pub orbit: OrbitCamera,
    pub topdown: TopDownCamera,
    pub fps: FpsCamera,
}

impl ViewCameras {
    /// View-projection of the active view (`follow` = Target Frame translation offset; zero = world-fixed).
    pub fn view_proj(&self, aspect: f32, follow: Vector3<f32>) -> Matrix4<f32> {
        match self.view_type {
            ViewType::Orbit => self.orbit.view_proj(aspect, follow),
            ViewType::TopDownOrtho => self.topdown.view_proj(aspect, follow),
            ViewType::Fps => self.fps.view_proj(aspect, follow),
        }
    }

    /// Screen right/up basis of the active view (for point-quad billboarding).
    pub fn basis(&self) -> (Vector3<f32>, Vector3<f32>) {
        match self.view_type {
            ViewType::Orbit => self.orbit.basis(),
            ViewType::TopDownOrtho => self.topdown.basis(),
            ViewType::Fps => self.fps.basis(),
        }
    }

    /// World point the active view is centered on (`follow` applied), used to measure the on-screen scale.
    pub fn focus_point(&self, follow: Vector3<f32>) -> Point3<f32> {
        let local = match self.view_type {
            ViewType::Orbit => self.orbit.target,
            ViewType::TopDownOrtho => {
                Point3::new(self.topdown.center[0], self.topdown.center[1], 0.0)
            }
            ViewType::Fps => self.fps.eye,
        };
        local + follow
    }

    /// Reset only the active view to its default pose (Zero button); other views and Target Frame are kept.
    pub fn reset_current(&mut self) {
        match self.view_type {
            ViewType::Orbit => self.orbit = OrbitCamera::default(),
            ViewType::TopDownOrtho => self.topdown = TopDownCamera::default(),
            ViewType::Fps => self.fps = FpsCamera::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Project a point to NDC (x, y, depth) via view_proj.
    fn project(camera: &OrbitCamera, point: Point3<f32>) -> (f32, f32, f32) {
        let clip = camera.view_proj(16.0 / 9.0, Vector3::zeros()) * point.to_homogeneous();
        (clip.x / clip.w, clip.y / clip.w, clip.z / clip.w)
    }

    /// Project a point via an arbitrary view-projection matrix.
    fn project_with(view_proj: Matrix4<f32>, point: Point3<f32>) -> (f32, f32, f32) {
        let clip = view_proj * point.to_homogeneous();
        (clip.x / clip.w, clip.y / clip.w, clip.z / clip.w)
    }

    #[test]
    fn initial_pose_projects_target_to_ndc_center() {
        let camera = OrbitCamera::default();
        let (x, y, depth) = project(&camera, camera.target);
        assert!(x.abs() < 1e-5, "x = {x}");
        assert!(y.abs() < 1e-5, "y = {y}");
        assert!(depth > 0.0 && depth < 1.0, "depth = {depth}");
    }

    #[test]
    fn eye_distance_matches_state_and_pitch_clamps() {
        let mut camera = OrbitCamera::default();
        assert!(((camera.eye() - camera.target).norm() - camera.distance).abs() < 1e-5);
        camera.rotate(0.0, 10_000.0);
        assert!((camera.pitch - PITCH_LIMIT).abs() < 1e-6);
        assert!(camera.view_proj(1.0, Vector3::zeros()).iter().all(|v| v.is_finite()));
        camera.rotate(0.0, -100_000.0);
        assert!((camera.pitch + PITCH_LIMIT).abs() < 1e-6);
        assert!(camera.view_proj(1.0, Vector3::zeros()).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn zoom_clamps_at_min_distance_without_flipping() {
        let mut camera = OrbitCamera::default();
        for _ in 0..1000 {
            camera.zoom_scroll(100.0);
        }
        assert_eq!(camera.distance, DISTANCE_MIN);
        camera.zoom_factor(1e9);
        assert_eq!(camera.distance, DISTANCE_MIN);
        for _ in 0..1000 {
            camera.zoom_scroll(-100.0);
        }
        assert_eq!(camera.distance, DISTANCE_MAX);
    }

    #[test]
    fn pan_by_half_viewport_height_moves_target_by_expected_scale() {
        let mut camera = OrbitCamera::default();
        let before = camera.target;
        let height = 600.0;
        camera.pan(0.0, height / 2.0, height);
        let moved = (camera.target - before).norm();
        let expected = camera.distance * (FOVY * 0.5).tan();
        assert!((moved - expected).abs() < 1e-4, "moved = {moved}, expected = {expected}");
    }

    #[test]
    fn basis_is_orthonormal_and_faces_screen_directions() {
        let camera = OrbitCamera::default();
        let (right, up) = camera.basis();
        let forward = (camera.target - camera.eye()).normalize();
        assert!((right.norm() - 1.0).abs() < 1e-5);
        assert!((up.norm() - 1.0).abs() < 1e-5);
        assert!(right.dot(&up).abs() < 1e-5);
        assert!(right.dot(&forward).abs() < 1e-5);
        assert!(up.dot(&forward).abs() < 1e-5);
        // With a look-down camera, up has a world +Z component (screen-up direction).
        assert!(up.z > 0.0);
        // right is world-horizontal (no Z component).
        assert!(right.z.abs() < 1e-5);
    }

    #[test]
    fn ndc_depth_preserves_camera_distance_order() {
        let camera = OrbitCamera::default();
        let eye = camera.eye();
        let toward = (camera.target - eye).normalize();
        let near_point = eye + toward * 2.0;
        let far_point = eye + toward * 20.0;
        let (_, _, depth_near) = project(&camera, near_point);
        let (_, _, depth_far) = project(&camera, far_point);
        assert!(depth_near > 0.0 && depth_far < 1.0);
        assert!(depth_near < depth_far, "near = {depth_near}, far = {depth_far}");
    }

    #[test]
    fn orbit_follow_offset_is_pure_translation() {
        let camera = OrbitCamera::default();
        let follow = Vector3::new(3.0, -2.0, 1.0);
        let point = Point3::new(0.5, 0.4, 0.2);
        let base = project(&camera, point);
        let with_follow = project_with(camera.view_proj(16.0 / 9.0, follow), point + follow);
        assert!((base.0 - with_follow.0).abs() < 1e-5);
        assert!((base.1 - with_follow.1).abs() < 1e-5);
        assert!((base.2 - with_follow.2).abs() < 1e-5);
    }

    #[test]
    fn topdown_center_projects_to_ndc_center() {
        let camera = TopDownCamera::default();
        let vp = camera.view_proj(16.0 / 9.0, Vector3::zeros());
        let (x, y, depth) = project_with(vp, Point3::new(camera.center[0], camera.center[1], 0.0));
        assert!(x.abs() < 1e-5, "x = {x}");
        assert!(y.abs() < 1e-5, "y = {y}");
        assert!(depth > 0.0 && depth < 1.0, "depth = {depth}");
    }

    #[test]
    fn topdown_is_orthographic_ignoring_depth_in_xy() {
        let camera = TopDownCamera::default();
        let vp = camera.view_proj(16.0 / 9.0, Vector3::zeros());
        let (x0, y0, _) = project_with(vp, Point3::new(2.0, 1.0, 0.0));
        let (x1, y1, _) = project_with(vp, Point3::new(2.0, 1.0, 3.0));
        assert!((x0 - x1).abs() < 1e-6, "x differs with depth: {x0} vs {x1}");
        assert!((y0 - y1).abs() < 1e-6, "y differs with depth: {y0} vs {y1}");
    }

    #[test]
    fn topdown_depth_orders_lower_z_farther() {
        let camera = TopDownCamera::default();
        let vp = camera.view_proj(16.0 / 9.0, Vector3::zeros());
        let (_, _, depth_high) = project_with(vp, Point3::new(0.0, 0.0, 2.0));
        let (_, _, depth_low) = project_with(vp, Point3::new(0.0, 0.0, -2.0));
        assert!(depth_high > 0.0 && depth_low < 1.0);
        assert!(depth_high < depth_low, "high z should be nearer: {depth_high} vs {depth_low}");
    }

    #[test]
    fn topdown_pan_moves_center_by_expected_scale() {
        let mut camera = TopDownCamera::default();
        let height = 600.0;
        camera.pan(0.0, height / 2.0, height);
        assert!((camera.center[1] - camera.half_height).abs() < 1e-4, "center = {:?}", camera.center);
        assert!(camera.center[0].abs() < 1e-6);
    }

    #[test]
    fn topdown_zoom_clamps_without_flipping() {
        let mut camera = TopDownCamera::default();
        for _ in 0..1000 {
            camera.zoom_scroll(100.0);
        }
        assert_eq!(camera.half_height, HALF_HEIGHT_MIN);
        for _ in 0..1000 {
            camera.zoom_scroll(-100.0);
        }
        assert_eq!(camera.half_height, HALF_HEIGHT_MAX);
    }

    #[test]
    fn topdown_basis_is_orthonormal_and_rotates() {
        let mut camera = TopDownCamera::default();
        let (right, up) = camera.basis();
        assert!((right.norm() - 1.0).abs() < 1e-6 && (up.norm() - 1.0).abs() < 1e-6);
        assert!(right.dot(&up).abs() < 1e-6);
        assert!((right - Vector3::x()).norm() < 1e-6 && (up - Vector3::y()).norm() < 1e-6);
        camera.rotate_z(1000.0);
        let (rotated_right, _) = camera.basis();
        assert!((rotated_right - Vector3::x()).norm() > 1e-3);
    }

    #[test]
    fn fps_forward_point_projects_to_center_and_pitch_clamps() {
        let mut camera = FpsCamera::default();
        let ahead = camera.eye + camera.forward() * 5.0;
        let (x, y, depth) = project_with(camera.view_proj(16.0 / 9.0, Vector3::zeros()), ahead);
        assert!(x.abs() < 1e-4 && y.abs() < 1e-4, "x = {x}, y = {y}");
        assert!(depth > 0.0 && depth < 1.0);
        camera.look(0.0, -1e6);
        assert!((camera.pitch - PITCH_LIMIT).abs() < 1e-6);
        camera.look(0.0, 1e6);
        assert!((camera.pitch + PITCH_LIMIT).abs() < 1e-6);
    }

    #[test]
    fn fps_dolly_moves_along_forward_and_pan_strafes() {
        let mut camera = FpsCamera::default();
        let forward = camera.forward();
        let before = camera.eye;
        camera.dolly(100.0);
        let moved = camera.eye - before;
        assert!((moved.normalize() - forward).norm() < 1e-5, "dolly not along forward");
        let (right, _) = camera.basis();
        let before = camera.eye;
        camera.pan(100.0, 0.0, 600.0);
        let strafe = camera.eye - before;
        assert!(strafe.dot(&forward).abs() < 1e-5, "strafe should not move along forward");
        assert!(strafe.dot(&right) > 0.0, "positive dx should strafe toward +right (RViz push)");
    }

    #[test]
    fn fps_basis_is_orthonormal() {
        let camera = FpsCamera::default();
        let (right, up) = camera.basis();
        let forward = camera.forward();
        assert!((right.norm() - 1.0).abs() < 1e-5 && (up.norm() - 1.0).abs() < 1e-5);
        assert!(right.dot(&up).abs() < 1e-5);
        assert!(right.dot(&forward).abs() < 1e-5);
        assert!(up.dot(&forward).abs() < 1e-5);
    }

    #[test]
    fn view_cameras_switch_preserves_state_and_reset_is_scoped() {
        let mut cameras = ViewCameras::default();
        assert_eq!(cameras.view_type, ViewType::Orbit);
        cameras.orbit.distance = 42.0;
        cameras.topdown.half_height = 33.0;
        cameras.view_type = ViewType::TopDownOrtho;
        cameras.reset_current();
        assert_eq!(cameras.topdown.half_height, TopDownCamera::default().half_height);
        assert_eq!(cameras.orbit.distance, 42.0, "reset must not touch other views");
        cameras.view_type = ViewType::Orbit;
        assert_eq!(cameras.orbit.distance, 42.0, "switching back preserves orbit state");
    }
}
