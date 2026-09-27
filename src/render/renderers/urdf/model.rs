//! URDF text -> LinkVisual intermediate representation (collision / inertial / joint are dropped; poses come from TF, so no FK here).

use egui::Color32;
use nalgebra::{Isometry3, Translation3, UnitQuaternion};

use crate::theme;

/// Drawing-relevant subset of a URDF `<robot>`.
#[derive(Debug, Clone, PartialEq)]
pub struct UrdfModel {
    pub robot_name: String,
    /// Number of `<link>` elements, including links without a visual (shown in the load result line).
    pub link_count: usize,
    /// link order -> per-link visual order. Deterministic, which is what makes GPU slot identity stable.
    pub visuals: Vec<LinkVisual>,
}

/// One `<visual>` of one link, flattened.
#[derive(Debug, Clone, PartialEq)]
pub struct LinkVisual {
    /// TF resolution key (this viewer treats a URDF link name as a TF frame name).
    pub link: String,
    /// Display name: `visual.name` if present, else `link#index`.
    pub name: String,
    /// Rigid transform from link frame to visual frame (xyz + fixed-axis rpy).
    pub origin: Isometry3<f64>,
    pub shape: Shape,
    /// Linear RGBA8 with straight (non-premultiplied) alpha, as mesh.wgsl expects.
    pub color: [u8; 4],
}

/// Geometry this renderer can draw, plus why it cannot draw the rest.
#[derive(Debug, Clone, PartialEq)]
pub enum Shape {
    Box {
        size: [f64; 3],
    },
    Cylinder {
        radius: f64,
        length: f64,
    },
    Sphere {
        radius: f64,
    },
    /// External mesh file; the URI is resolved and the file read on the loader worker, not here.
    Mesh {
        uri: String,
        scale: [f64; 3],
    },
    /// capsule / non-positive dimensions. `kind` is the list's geometry column, `reason` its state column.
    Unsupported {
        kind: &'static str,
        reason: String,
    },
}

impl Shape {
    /// Geometry kind shown in the settings_ui visual list.
    pub fn kind(&self) -> &'static str {
        match self {
            Shape::Box { .. } => "box",
            Shape::Cylinder { .. } => "cylinder",
            Shape::Sphere { .. } => "sphere",
            Shape::Mesh { .. } => "mesh",
            Shape::Unsupported { kind, .. } => kind,
        }
    }
}

/// Markers of an unexpanded xacro: urdf-rs drops every `xacro:*` element and cannot evaluate `${...}` / `$(...)`, so such a file would read as an empty or wrong robot unless it is expanded first.
const XACRO_MARKERS: [&str; 4] = ["<xacro:", "xmlns:xacro", "${", "$("];

/// Whether `text` needs xacro expansion before it is URDF (the caller runs `xacro::expand` when so). Expanded output may legitimately contain a literal `${` written as `$${`, so this is a routing decision, not a check `parse` repeats.
pub fn is_xacro(text: &str) -> bool {
    XACRO_MARKERS.iter().any(|m| text.contains(m))
}

/// Parse URDF text into the intermediate representation (the caller does the file I/O and any xacro expansion).
pub fn parse(text: &str) -> Result<UrdfModel, String> {
    // UrdfError's variants are private, so only its Display is usable.
    let robot = urdf_rs::read_from_string(text).map_err(|e| e.to_string())?;
    let mut visuals = Vec::new();
    for link in &robot.links {
        for (index, visual) in link.visual.iter().enumerate() {
            visuals.push(LinkVisual {
                link: link.name.clone(),
                name: visual
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("{}#{index}", link.name)),
                origin: pose_to_isometry(&visual.origin),
                shape: to_shape(&visual.geometry),
                color: resolve_color(visual.material.as_ref(), &robot.materials),
            });
        }
    }
    Ok(UrdfModel {
        robot_name: robot.name.clone(),
        link_count: robot.links.len(),
        visuals,
    })
}

/// URDF `<origin xyz rpy>` -> Isometry3 (fixed-axis roll(X)->pitch(Y)->yaw(Z), i.e. R = Rz*Ry*Rx).
fn pose_to_isometry(pose: &urdf_rs::Pose) -> Isometry3<f64> {
    let [x, y, z] = *pose.xyz;
    let [roll, pitch, yaw] = *pose.rpy;
    Isometry3::from_parts(
        Translation3::new(x, y, z),
        UnitQuaternion::from_euler_angles(roll, pitch, yaw),
    )
}

/// URDF geometry -> Shape (mesh / capsule are out of scope, and non-positive extents cannot be baked).
fn to_shape(geometry: &urdf_rs::Geometry) -> Shape {
    let non_positive = |kind: &'static str, dims: &[f64]| -> Option<Shape> {
        dims.iter()
            .any(|d| !d.is_finite() || *d <= 0.0)
            .then(|| Shape::Unsupported {
                kind,
                reason: "non-positive dimension".to_owned(),
            })
    };
    match geometry {
        urdf_rs::Geometry::Box { size } => {
            let size = [size[0], size[1], size[2]];
            non_positive("box", &size).unwrap_or(Shape::Box { size })
        }
        urdf_rs::Geometry::Cylinder { radius, length } => {
            non_positive("cylinder", &[*radius, *length]).unwrap_or(Shape::Cylinder {
                radius: *radius,
                length: *length,
            })
        }
        urdf_rs::Geometry::Sphere { radius } => {
            non_positive("sphere", &[*radius]).unwrap_or(Shape::Sphere { radius: *radius })
        }
        urdf_rs::Geometry::Mesh { filename, scale } => {
            let scale = scale.map(|s| *s).unwrap_or([1.0; 3]);
            // A zero or non-finite scale collapses the baked vertices, so there is nothing to draw.
            match scale.iter().all(|s| s.is_finite() && *s != 0.0) {
                true => Shape::Mesh {
                    uri: filename.clone(),
                    scale,
                },
                false => Shape::Unsupported {
                    kind: "mesh",
                    reason: format!("unusable mesh scale ({scale:?})"),
                },
            }
        }
        // Capsule is a urdf-rs extension, not part of the URDF spec (RViz does not draw it either).
        urdf_rs::Geometry::Capsule { .. } => Shape::Unsupported {
            kind: "capsule",
            reason: "capsule geometry not implemented".to_owned(),
        },
    }
}

/// Color resolution: the visual's own `<color rgba>`, else a top-level `<material name>` reference, else the theme default (`<texture>` ignored).
fn resolve_color(material: Option<&urdf_rs::Material>, top_level: &[urdf_rs::Material]) -> [u8; 4] {
    let rgba = material.and_then(|m| {
        m.color.as_ref().map(|c| *c.rgba).or_else(|| {
            top_level
                .iter()
                .find(|t| t.name == m.name)
                .and_then(|t| t.color.as_ref())
                .map(|c| *c.rgba)
        })
    });
    match rgba {
        Some([r, g, b, a]) => linear_rgba8(r, g, b, a),
        None => theme::to_linear_rgba8(theme::MESH_DEFAULT),
    }
}

/// Pack a URDF material rgba (0..1) as straight-alpha linear RGBA8 (from_rgba_unmultiplied would premultiply, darkening rgb twice with mesh.wgsl's `color.a * batch.alpha`).
fn linear_rgba8(r: f64, g: f64, b: f64, a: f64) -> [u8; 4] {
    let byte = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    let mut rgba = theme::to_linear_rgba8(Color32::from_rgb(byte(r), byte(g), byte(b)));
    rgba[3] = byte(a);
    rgba
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, Vector3};

    /// Wrap link bodies in a minimal `<robot>` so each test only shows the part it cares about.
    fn robot(body: &str) -> String {
        format!("<robot name=\"test_robot\">{body}</robot>")
    }

    fn single_visual(visual: &str) -> LinkVisual {
        let text = robot(&format!("<link name=\"base_link\">{visual}</link>"));
        let model = parse(&text).expect("parses");
        assert_eq!(model.robot_name, "test_robot");
        model.visuals.into_iter().next().expect("one visual")
    }

    #[test]
    fn visual_origin_composes_fixed_axis_rpy() {
        let visual = single_visual(
            r#"<visual>
                 <origin xyz="1 2 3" rpy="0.1 0.2 0.3"/>
                 <geometry><box size="1 1 1"/></geometry>
               </visual>"#,
        );
        assert_eq!(
            visual.origin.translation.vector,
            Vector3::new(1.0, 2.0, 3.0)
        );
        // URDF's fixed-axis roll->pitch->yaw is Rz(yaw) * Ry(pitch) * Rx(roll).
        let expected = Rotation3::from_axis_angle(&Vector3::z_axis(), 0.3)
            * Rotation3::from_axis_angle(&Vector3::y_axis(), 0.2)
            * Rotation3::from_axis_angle(&Vector3::x_axis(), 0.1);
        let actual = visual.origin.rotation.to_rotation_matrix();
        assert!(
            (actual.matrix() - expected.matrix()).abs().max() < 1e-12,
            "{actual:?} != {expected:?}"
        );
    }

    #[test]
    fn missing_origin_is_identity() {
        let visual =
            single_visual("<visual><geometry><sphere radius=\"0.5\"/></geometry></visual>");
        assert_eq!(visual.origin, Isometry3::identity());
        assert_eq!(visual.shape, Shape::Sphere { radius: 0.5 });
    }

    #[test]
    fn inline_material_color_wins() {
        let text = robot(
            r#"<material name="shared"><color rgba="0 0 1 1"/></material>
               <link name="base_link">
                 <visual>
                   <geometry><box size="1 1 1"/></geometry>
                   <material name="shared"><color rgba="1 0 0 1"/></material>
                 </visual>
               </link>"#,
        );
        let model = parse(&text).expect("parses");
        assert_eq!(model.visuals[0].color, linear_rgba8(1.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn named_material_reference_resolves_from_top_level() {
        let text = robot(
            r#"<material name="visor_cyan"><color rgba="0 0.9 1 1"/></material>
               <link name="base_link">
                 <visual>
                   <geometry><box size="1 1 1"/></geometry>
                   <material name="visor_cyan"/>
                 </visual>
               </link>"#,
        );
        let model = parse(&text).expect("parses");
        assert_eq!(model.visuals[0].color, linear_rgba8(0.0, 0.9, 1.0, 1.0));
    }

    #[test]
    fn unknown_material_name_and_texture_only_fall_back_to_default() {
        let default = theme::to_linear_rgba8(theme::MESH_DEFAULT);
        let unknown = single_visual(
            r#"<visual>
                 <geometry><box size="1 1 1"/></geometry>
                 <material name="nowhere"/>
               </visual>"#,
        );
        assert_eq!(unknown.color, default);
        let textured = single_visual(
            r#"<visual>
                 <geometry><box size="1 1 1"/></geometry>
                 <material name="tex"><texture filename="wood.png"/></material>
               </visual>"#,
        );
        assert_eq!(textured.color, default);
        let no_material =
            single_visual("<visual><geometry><box size=\"1 1 1\"/></geometry></visual>");
        assert_eq!(no_material.color, default);
    }

    #[test]
    fn material_alpha_is_kept_as_straight_alpha() {
        let opaque = single_visual(
            r#"<visual>
                 <geometry><box size="1 1 1"/></geometry>
                 <material name="red"><color rgba="1 0 0 1"/></material>
               </visual>"#,
        );
        let translucent = single_visual(
            r#"<visual>
                 <geometry><box size="1 1 1"/></geometry>
                 <material name="red"><color rgba="1 0 0 0.5"/></material>
               </visual>"#,
        );
        // rgb stays identical (no premultiplication) and only alpha differs.
        assert_eq!(opaque.color[..3], translucent.color[..3]);
        assert_eq!(opaque.color[3], 255);
        assert_eq!(translucent.color[3], 128);
    }

    #[test]
    fn multiple_visuals_keep_link_then_visual_order() {
        let text = robot(
            r#"<link name="a">
                 <visual><geometry><box size="1 1 1"/></geometry></visual>
                 <visual name="a_named"><geometry><sphere radius="1"/></geometry></visual>
               </link>
               <link name="b">
                 <visual><geometry><cylinder radius="1" length="2"/></geometry></visual>
                 <visual><geometry><box size="2 2 2"/></geometry></visual>
               </link>"#,
        );
        let model = parse(&text).expect("parses");
        assert_eq!(model.link_count, 2);
        let ids: Vec<(&str, &str, &str)> = model
            .visuals
            .iter()
            .map(|v| (v.link.as_str(), v.name.as_str(), v.shape.kind()))
            .collect();
        assert_eq!(
            ids,
            vec![
                ("a", "a#0", "box"),
                ("a", "a_named", "sphere"),
                ("b", "b#0", "cylinder"),
                ("b", "b#1", "box"),
            ]
        );
    }

    #[test]
    fn collision_inertial_and_joints_are_ignored() {
        let text = robot(
            r#"<link name="base_link">
                 <inertial>
                   <origin xyz="9 9 9"/>
                   <mass value="1"/>
                   <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
                 </inertial>
                 <visual><geometry><box size="1 1 1"/></geometry></visual>
                 <collision>
                   <origin xyz="5 5 5"/>
                   <geometry><box size="3 3 3"/></geometry>
                 </collision>
               </link>
               <link name="collision_only">
                 <collision><geometry><box size="1 1 1"/></geometry></collision>
               </link>
               <joint name="j" type="revolute">
                 <origin xyz="7 7 7" rpy="1 1 1"/>
                 <parent link="base_link"/>
                 <child link="collision_only"/>
                 <axis xyz="0 0 1"/>
                 <limit lower="-1" upper="1" effort="0" velocity="1"/>
               </joint>"#,
        );
        let model = parse(&text).expect("parses");
        assert_eq!(model.link_count, 2);
        // A collision-only link emits nothing, and no joint/collision/inertial origin leaks in (no FK).
        assert_eq!(model.visuals.len(), 1);
        assert_eq!(model.visuals[0].link, "base_link");
        assert_eq!(model.visuals[0].origin, Isometry3::identity());
    }

    #[test]
    fn vendor_extensions_are_dropped() {
        // urdf-rs's preprocessing keeps only link/joint/material under <robot>, so <gazebo> never reaches us.
        let text = robot(
            r#"<gazebo reference="base_link"><material>Gazebo/Blue</material></gazebo>
               <link name="base_link">
                 <visual><geometry><box size="1 1 1"/></geometry></visual>
               </link>"#,
        );
        let model = parse(&text).expect("parses");
        assert_eq!(model.visuals.len(), 1);
    }

    #[test]
    fn mesh_keeps_its_uri_and_scale_while_capsule_stays_unsupported() {
        let plain = single_visual(
            "<visual><geometry><mesh filename=\"package://foo/bar.stl\"/></geometry></visual>",
        );
        assert_eq!(
            plain.shape,
            Shape::Mesh {
                uri: "package://foo/bar.stl".to_owned(),
                scale: [1.0; 3]
            }
        );
        assert_eq!(plain.shape.kind(), "mesh");
        let scaled = single_visual(
            "<visual><geometry><mesh filename=\"/abs/path.dae\" scale=\"2 0.5 -1\"/></geometry></visual>",
        );
        assert_eq!(
            scaled.shape,
            Shape::Mesh {
                uri: "/abs/path.dae".to_owned(),
                scale: [2.0, 0.5, -1.0]
            }
        );
        let capsule = single_visual(
            "<visual><geometry><capsule radius=\"1\" length=\"2\"/></geometry></visual>",
        );
        let Shape::Unsupported { kind, reason } = &capsule.shape else {
            panic!("expected Unsupported, got {:?}", capsule.shape);
        };
        assert_eq!(*kind, "capsule");
        assert!(reason.contains("not implemented"), "reason={reason}");
    }

    #[test]
    fn a_zero_or_non_finite_mesh_scale_is_unsupported() {
        for scale in ["0 1 1", "1 nan 1", "1 1 inf"] {
            let visual = single_visual(&format!(
                "<visual><geometry><mesh filename=\"a.stl\" scale=\"{scale}\"/></geometry></visual>"
            ));
            assert!(
                matches!(
                    &visual.shape,
                    Shape::Unsupported { kind, reason } if *kind == "mesh" && reason.contains("scale")
                ),
                "{scale} -> {:?}",
                visual.shape
            );
        }
    }

    #[test]
    fn non_positive_dimensions_become_unsupported() {
        for geometry in [
            "<box size=\"1 0 1\"/>",
            "<box size=\"-1 1 1\"/>",
            "<cylinder radius=\"0\" length=\"1\"/>",
            "<cylinder radius=\"1\" length=\"0\"/>",
            "<sphere radius=\"0\"/>",
        ] {
            let visual =
                single_visual(&format!("<visual><geometry>{geometry}</geometry></visual>"));
            assert!(
                matches!(
                    &visual.shape,
                    Shape::Unsupported { reason, .. } if reason == "non-positive dimension"
                ),
                "{geometry} -> {:?}",
                visual.shape
            );
        }
    }

    #[test]
    fn invalid_xml_and_malformed_attributes_return_err() {
        for text in [
            "<robot name=\"x\"><link name=\"a\"></robot>",
            "",
            "<robot name=\"x\"><link name=\"a\"><visual><geometry><box size=\"1 1\"/></geometry></visual></link></robot>",
            "<robot name=\"x\"><link name=\"a\"><visual><geometry><box size=\"1 1 1\"/></geometry><material/></visual></link></robot>",
        ] {
            assert!(parse(text).is_err(), "expected Err for {text:?}");
        }
        // An empty robot is valid and simply has no visuals.
        let empty = parse("<robot name=\"empty\"/>").expect("empty robot parses");
        assert!(empty.visuals.is_empty());
        // serde-xml-rs ignores the root element's name, so arbitrary XML parses as an empty robot (surfaced later as "no visuals").
        let other = parse("<not_a_robot/>").expect("root name is not checked");
        assert!(other.visuals.is_empty());
    }

    #[test]
    fn xacro_is_detected_for_routing_and_not_rejected_by_parse() {
        // Links defined inside a xacro macro are dropped by urdf-rs, so unexpanded text reads as an empty robot: that is why the caller must expand first.
        let macro_file = r#"<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
             <xacro:macro name="sr">
               <link name="base_link">
                 <visual><geometry><box size="1 1 1"/></geometry></visual>
               </link>
             </xacro:macro>
           </robot>"#;
        assert!(is_xacro(macro_file));
        assert!(
            parse(macro_file)
                .expect("urdf-rs tolerates it")
                .visuals
                .is_empty()
        );
        // Property / argument substitutions mark a file as xacro even outside macros.
        for text in [
            r#"<robot name="r"><link name="a"><visual><origin xyz="0 0 ${h}"/><geometry><box size="1 1 1"/></geometry></visual></link></robot>"#,
            r#"<robot name="r"><xacro:include filename="$(find pkg)/urdf/x.xacro"/></robot>"#,
        ] {
            assert!(is_xacro(text), "{text}");
        }
        let plain = robot(
            r#"<link name="a"><visual><geometry><box size="1 1 1"/></geometry></visual></link>"#,
        );
        assert!(!is_xacro(&plain));
        assert!(parse(&plain).is_ok());
        // Expanded output may carry a literal `${...}` (written `$${...}` in the xacro); parse must not mistake it for unexpanded xacro.
        let escaped = robot(
            r#"<link name="lit${x}"><visual><geometry><box size="1 1 1"/></geometry></visual></link>"#,
        );
        assert_eq!(
            parse(&escaped)
                .expect("literal braces are plain text")
                .visuals[0]
                .link,
            "lit${x}"
        );
    }

    #[test]
    fn bundled_sample_urdf_parses() {
        let text = std::fs::read_to_string(super::super::sample_urdf_path("sample_robot.urdf"))
            .expect("bundled sample readable");
        let model = parse(&text).expect("bundled sample parses");
        assert_eq!(model.robot_name, "visor_sample");
        let links: Vec<&str> = model.visuals.iter().map(|v| v.link.as_str()).collect();
        assert_eq!(
            links,
            vec!["base_link", "laser_link", "arm_link", "hand_link"]
        );
        // Every visual in the primitive-only sample is drawable.
        assert!(
            model
                .visuals
                .iter()
                .all(|v| !matches!(v.shape, Shape::Unsupported { .. }))
        );
    }
}
