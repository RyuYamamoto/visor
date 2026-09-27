//! mesh-loader Scene -> 28B posed-mesh vertices. Pure and GPU-free, so the URDF scale bake, the up-axis correction and the normal rules are all unit-testable.

use mesh_loader::Scene;

use super::collada;
use crate::render::{MeshBatch, MeshBatchBuilder};
use crate::theme;

/// COLLADA `<up_axis>`. mesh-loader ignores the element, so this renderer applies the rotation itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UpAxis {
    /// ROS convention, and what every other format this renderer reads assumes.
    #[default]
    Z,
    Y,
    X,
}

impl UpAxis {
    /// Rotate a model-space vector so the file's up axis becomes +Z.
    fn apply(self, v: [f32; 3]) -> [f32; 3] {
        match self {
            UpAxis::Z => v,
            UpAxis::Y => [v[0], -v[2], v[1]],
            UpAxis::X => [-v[2], v[1], v[0]],
        }
    }
}

/// How far into a file to look for `<up_axis>` (COLLADA puts `<asset>` first, right after the root element).
const UP_AXIS_SCAN_BYTES: usize = 64 * 1024;

/// Read `<up_axis>` out of raw COLLADA bytes; anything else (STL/OBJ, or a DAE without the element) is Z-up.
pub fn detect_up_axis(bytes: &[u8]) -> UpAxis {
    const TAG: &[u8] = b"<up_axis>";
    let head = &bytes[..bytes.len().min(UP_AXIS_SCAN_BYTES)];
    let Some(start) = head.windows(TAG.len()).position(|w| w == TAG) else {
        return UpAxis::Z;
    };
    let rest = &head[start + TAG.len()..];
    let end = rest.iter().position(|b| *b == b'<').unwrap_or(rest.len());
    match String::from_utf8_lossy(&rest[..end]).trim() {
        "Y_UP" => UpAxis::Y,
        "X_UP" => UpAxis::X,
        _ => UpAxis::Z,
    }
}

/// How to interpret one mesh file: what the URDF contributes, plus the facts read out of the bytes themselves.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshParams {
    /// URDF `<mesh scale>`, baked into the vertices because SceneBatch.model is rigid-only.
    pub scale: [f64; 3],
    /// Linear RGBA8 used when the file carries no color of its own (URDF `<material>` or the theme default).
    pub fallback_rgba: [u8; 4],
    pub up_axis: UpAxis,
    /// COLLADA per-material face groups in `Scene::meshes` order; empty for other formats and for documents collada.rs declined.
    pub material_groups: Vec<Vec<collada::Group>>,
}

impl Default for MeshParams {
    fn default() -> Self {
        Self {
            scale: [1.0; 3],
            fallback_rgba: theme::to_linear_rgba8(theme::MESH_DEFAULT),
            up_axis: UpAxis::Z,
            material_groups: Vec::new(),
        }
    }
}

/// Bake result, including the counts the settings_ui list reports.
#[derive(Debug, Clone)]
pub struct Built {
    pub batch: MeshBatch,
    pub triangles: usize,
    /// Triangles dropped as degenerate (zero area with no usable vertex normal) or out of index range.
    pub skipped: usize,
}

/// Scene -> one batch covering every mesh in the file. Err when nothing drawable is left, so a zero-vertex batch never reaches the GPU.
pub fn build(scene: &Scene, params: &MeshParams) -> Result<Built, String> {
    let scale = params.scale;
    if scale.iter().any(|s| !s.is_finite() || *s == 0.0) {
        return Err(format!("unusable mesh scale {scale:?}"));
    }
    let scale = scale.map(|s| s as f32);
    let inverse_scale = scale.map(|s| 1.0 / s);
    // A negative determinant mirrors the geometry, which reverses the winding; the inverse-transpose already flips the normals.
    let mirrored = scale[0] * scale[1] * scale[2] < 0.0;
    let total: usize = scene.meshes.iter().map(|m| m.faces.len()).sum();
    if total == 0 {
        return Err("file contains no triangles".to_owned());
    }
    let mut builder = MeshBatchBuilder::with_capacity(total * 3);
    let mut triangles = 0;
    let mut skipped = 0;
    for (index, mesh) in scene.meshes.iter().enumerate() {
        let material = scene.materials.get(index);
        let opacity = material
            .and_then(|m| m.opacity)
            .filter(|o| o.is_finite() && *o > 0.0)
            .unwrap_or(1.0);
        let mesh_base = match material.and_then(|m| m.color.diffuse) {
            Some(diffuse) => linear_rgba8(diffuse, opacity),
            None => with_opacity(params.fallback_rgba, opacity),
        };
        // A COLOR input that is opaque white everywhere says nothing (Blender writes one for any unpainted colour attribute) and would otherwise override every material in the file.
        let colors: &[[f32; 4]] = match mesh.colors[0].iter().all(|c| *c == [1.0, 1.0, 1.0, 1.0]) {
            true => &[],
            false => &mesh.colors[0],
        };
        // Faces arrive in group order, so one cursor is enough to know which <triangles>/<polylist> material a face takes; a table whose total disagrees with the mesh is ignored rather than trusted.
        let groups = params
            .material_groups
            .get(index)
            .filter(|groups| groups.iter().map(|g| g.faces).sum::<usize>() == mesh.faces.len())
            .map_or(&[][..], Vec::as_slice);
        let mut group = 0;
        let mut group_end = groups.first().map_or(usize::MAX, |g| g.faces);
        for (face_index, face) in mesh.faces.iter().enumerate() {
            while face_index >= group_end && group + 1 < groups.len() {
                group += 1;
                group_end += groups[group].faces;
            }
            let base = groups
                .get(group)
                .and_then(|g| g.diffuse)
                .map_or(mesh_base, |diffuse| linear_rgba8(diffuse, opacity));
            let order = if mirrored { [0, 2, 1] } else { [0, 1, 2] };
            let corners: [usize; 3] = order.map(|c| face[c] as usize);
            if corners.iter().any(|c| *c >= mesh.vertices.len()) {
                skipped += 1;
                continue;
            }
            let positions = corners.map(|c| {
                let v = params.up_axis.apply(mesh.vertices[c]);
                [v[0] * scale[0], v[1] * scale[1], v[2] * scale[2]]
            });
            let normals = corners.map(|c| {
                mesh.normals.get(c).and_then(|n| {
                    let n = params.up_axis.apply(*n);
                    normalize([
                        n[0] * inverse_scale[0],
                        n[1] * inverse_scale[1],
                        n[2] * inverse_scale[2],
                    ])
                })
            });
            let fallback_normal = match normals.iter().all(Option::is_some) {
                true => Some([0.0; 3]),
                false => normalize(cross(
                    sub(positions[1], positions[0]),
                    sub(positions[2], positions[0]),
                )),
            };
            let Some(fallback_normal) = fallback_normal else {
                skipped += 1;
                continue;
            };
            for (i, corner) in corners.into_iter().enumerate() {
                let rgba = match colors.get(corner) {
                    Some(color) => linear_rgba8(*color, opacity),
                    None => base,
                };
                builder.push_vertex(positions[i], normals[i].unwrap_or(fallback_normal), rgba);
            }
            triangles += 1;
        }
    }
    if triangles == 0 {
        return Err(format!("all {total} triangle(s) are degenerate"));
    }
    Ok(Built {
        batch: builder.build(),
        triangles,
        skipped,
    })
}

/// Float RGBA (sRGB, straight alpha) -> the linear RGBA8 the mesh pipeline expects, matching model::linear_rgba8.
fn linear_rgba8(rgba: [f32; 4], opacity: f32) -> [u8; 4] {
    let lut = theme::srgb_to_linear_u8();
    let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    [
        lut[byte(rgba[0]) as usize],
        lut[byte(rgba[1]) as usize],
        lut[byte(rgba[2]) as usize],
        byte(rgba[3] * opacity),
    ]
}

/// Scale an already-linear color's alpha by the embedded material opacity.
fn with_opacity(rgba: [u8; 4], opacity: f32) -> [u8; 4] {
    let mut out = rgba;
    out[3] = (rgba[3] as f32 * opacity.clamp(0.0, 1.0)).round() as u8;
    out
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Unit vector, or None when the input is zero-length or not finite (a missing normal, or a degenerate triangle).
fn normalize(v: [f32; 3]) -> Option<[f32; 3]> {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    (length.is_finite() && length > f32::EPSILON).then(|| v.map(|c| c / length))
}

/// Absolute path of a mesh fixture (tests load through mesh-loader, so the real parse path is covered).
#[cfg(test)]
pub fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/mesh")
        .join(name)
}

/// Binary STL bytes (80B header + u32 count + 50B per triangle) built here rather than checked in, so the layout stays readable.
#[cfg(test)]
pub fn binary_stl(triangles: &[([f32; 3], [[f32; 3]; 3])]) -> Vec<u8> {
    let mut bytes = vec![0u8; 80];
    bytes.extend_from_slice(&(triangles.len() as u32).to_le_bytes());
    for (normal, vertices) in triangles {
        for v in normal.iter().chain(vertices.iter().flatten()) {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes.extend_from_slice(&0u16.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::MESH_STRIDE;

    /// Decode the packed 28B layout back into (position, normal, rgba).
    fn vertices(batch: &MeshBatch) -> Vec<([f32; 3], [f32; 3], [u8; 4])> {
        (0..batch.count as usize)
            .map(|i| {
                let at = i * MESH_STRIDE;
                let f32_at = |offset: usize| {
                    f32::from_le_bytes(
                        batch.bytes[at + offset..at + offset + 4]
                            .try_into()
                            .expect("4 bytes"),
                    )
                };
                let position = std::array::from_fn(|c| f32_at(c * 4));
                let normal = std::array::from_fn(|c| f32_at(12 + c * 4));
                let rgba = std::array::from_fn(|c| batch.bytes[at + 24 + c]);
                (position, normal, rgba)
            })
            .collect()
    }

    fn load(name: &str) -> Scene {
        let path = fixture_path(name);
        mesh_loader::Loader::default()
            .load(&path)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    fn from_slice(bytes: &[u8], name: &str) -> Scene {
        mesh_loader::Loader::default()
            .load_from_slice(bytes, name)
            .expect("parses")
    }

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    /// Load a fixture together with the material grouping the loader passes alongside it.
    fn load_grouped(name: &str) -> (Scene, MeshParams) {
        let bytes = std::fs::read(fixture_path(name)).expect("read fixture");
        let params = MeshParams {
            material_groups: collada::material_groups(&bytes).unwrap_or_default(),
            ..Default::default()
        };
        (from_slice(&bytes, name), params)
    }

    #[test]
    fn each_material_group_of_one_geometry_keeps_its_own_color() {
        let (scene, params) = load_grouped("triangle_two_materials.dae");
        // mesh-loader merges the two groups into one mesh and reports only the first material.
        assert_eq!(scene.meshes.len(), 1);
        assert_eq!(scene.materials.len(), 1);
        let built = build(&scene, &params).expect("builds");
        assert_eq!((built.triangles, built.skipped), (3, 0));
        let verts = vertices(&built.batch);
        let red = super::linear_rgba8([1.0, 0.0, 0.0, 1.0], 1.0);
        let blue = super::linear_rgba8([0.0, 0.0, 1.0, 1.0], 1.0);
        assert!(verts[..3].iter().all(|v| v.2 == red), "{:?}", verts[0].2);
        assert!(verts[3..].iter().all(|v| v.2 == blue), "{:?}", verts[3].2);
    }

    #[test]
    fn a_grouping_that_does_not_match_the_mesh_is_ignored() {
        let (scene, mut params) = load_grouped("triangle_two_materials.dae");
        params.material_groups = vec![vec![collada::Group {
            faces: 99,
            diffuse: Some([0.0, 1.0, 0.0, 1.0]),
        }]];
        // A table whose face total disagrees would colour the wrong triangles, so the mesh material is used instead.
        let built = build(&scene, &params).expect("builds");
        let red = super::linear_rgba8([1.0, 0.0, 0.0, 1.0], 1.0);
        assert!(vertices(&built.batch).iter().all(|v| v.2 == red));
    }

    #[test]
    fn an_all_white_color_layer_does_not_override_the_material() {
        let dae = std::fs::read_to_string(fixture_path("triangle_vertex_color.dae"))
            .expect("read fixture")
            .replace("1 0 0 0 0 1 1 1 1", "1 1 1 1 1 1 1 1 1");
        let (scene, params) = (
            from_slice(dae.as_bytes(), "white.dae"),
            MeshParams {
                material_groups: collada::material_groups(dae.as_bytes()).unwrap_or_default(),
                ..Default::default()
            },
        );
        // Blender writes such a layer for any unpainted color attribute; trusting it hides every material in the file.
        let built = build(&scene, &params).expect("builds");
        let green = super::linear_rgba8([0.0, 1.0, 0.0, 1.0], 0.5);
        assert!(vertices(&built.batch).iter().all(|v| v.2 == green));
    }

    #[test]
    fn ascii_stl_keeps_positions_and_file_normals() {
        let built = build(&load("triangle_ascii.stl"), &MeshParams::default()).expect("builds");
        assert_eq!((built.triangles, built.skipped), (2, 0));
        let verts = vertices(&built.batch);
        assert_eq!(verts.len(), 6);
        assert!(close(verts[0].0, [0.0, 0.0, 0.0]));
        assert!(close(verts[1].0, [1.0, 0.0, 0.0]));
        assert!(close(verts[2].0, [0.0, 1.0, 0.0]));
        assert!(verts[..3].iter().all(|v| close(v.1, [0.0, 0.0, 1.0])));
        assert!(verts[3..].iter().all(|v| close(v.1, [0.0, 1.0, 0.0])));
        // STL carries no material, so every vertex takes the URDF/theme fallback color.
        let fallback = MeshParams::default().fallback_rgba;
        assert!(verts.iter().all(|v| v.2 == fallback));
    }

    #[test]
    fn binary_stl_falls_back_to_face_normals_when_the_file_stores_zeros() {
        let bytes = binary_stl(&[
            (
                [0.0, 0.0, 1.0],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ),
            (
                [0.0, 0.0, 0.0],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ),
        ]);
        let built = build(&from_slice(&bytes, "zero_normal.stl"), &MeshParams::default())
            .expect("builds");
        assert_eq!((built.triangles, built.skipped), (2, 0));
        let verts = vertices(&built.batch);
        // The stored normal is used as-is, and the zeroed one is replaced by the triangle's own face normal.
        assert!(verts.iter().all(|v| close(v.1, [0.0, 0.0, 1.0])));
    }

    #[test]
    fn z_up_dae_positions_pass_through_with_the_embedded_material_color() {
        let built = build(&load("triangle_z_up.dae"), &MeshParams::default()).expect("builds");
        assert_eq!(built.triangles, 1);
        let verts = vertices(&built.batch);
        assert!(close(verts[0].0, [0.0, 0.0, 0.0]));
        assert!(close(verts[1].0, [1.0, 0.0, 0.0]));
        assert!(close(verts[2].0, [0.0, 2.0, 0.0]));
        // The DAE's own diffuse (pure green) wins over the fallback.
        let green = super::linear_rgba8([0.0, 1.0, 0.0, 1.0], 1.0);
        assert!(verts.iter().all(|v| v.2 == green));
        assert_ne!(green, MeshParams::default().fallback_rgba);
    }

    #[test]
    fn y_up_dae_is_rotated_onto_z_up() {
        let bytes = std::fs::read(fixture_path("triangle_y_up.dae")).expect("readable");
        assert_eq!(detect_up_axis(&bytes), UpAxis::Y);
        let params = MeshParams {
            up_axis: detect_up_axis(&bytes),
            ..Default::default()
        };
        let built = build(&load("triangle_y_up.dae"), &params).expect("builds");
        let verts = vertices(&built.batch);
        // (x, y, z) -> (x, -z, y): the file's +Y vertex becomes +Z, and the +Z normal becomes -Y.
        assert!(close(verts[1].0, [1.0, 0.0, 0.0]));
        assert!(close(verts[2].0, [0.0, 0.0, 2.0]));
        assert!(verts.iter().all(|v| close(v.1, [0.0, -1.0, 0.0])));
        // X_UP maps the file's +X onto +Z the same way.
        assert_eq!(UpAxis::X.apply([1.0, 0.0, 0.0]), [0.0, 0.0, 1.0]);
        assert_eq!(detect_up_axis(b"no asset element here"), UpAxis::Z);
    }

    #[test]
    fn unit_scale_and_node_transforms_are_not_applied_twice() {
        // mesh-loader already folds <unit meter="0.01"> into the vertices; this renderer must not scale again.
        let cm = build(&load("triangle_unit_cm.dae"), &MeshParams::default()).expect("builds");
        let verts = vertices(&cm.batch);
        assert!(close(verts[1].0, [0.01, 0.0, 0.0]));
        assert!(close(verts[2].0, [0.0, 0.02, 0.0]));
        // Node <matrix> translation likewise arrives already applied.
        let offset = build(&load("triangle_node_offset.dae"), &MeshParams::default()).expect("builds");
        let verts = vertices(&offset.batch);
        assert!(close(verts[0].0, [0.5, -0.25, 0.0]));
        assert!(close(verts[1].0, [1.5, -0.25, 0.0]));
        assert!(close(verts[2].0, [0.5, 1.75, 0.0]));
    }

    #[test]
    fn scale_is_baked_into_positions_and_inverse_transposed_into_normals() {
        let bytes = binary_stl(&[(
            [0.0, 1.0, 0.0],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        )]);
        let scene = from_slice(&bytes, "scaled.stl");
        let uniform = build(
            &scene,
            &MeshParams {
                scale: [2.0; 3],
                ..Default::default()
            },
        )
        .expect("builds");
        let verts = vertices(&uniform.batch);
        assert!(close(verts[1].0, [2.0, 0.0, 0.0]));
        assert!(verts.iter().all(|v| close(v.1, [0.0, 1.0, 0.0])));
        // Non-uniform: the normal goes through the inverse scale and is renormalized, so it stays a unit vector.
        let squashed = build(
            &from_slice(
                &binary_stl(&[(
                    [0.0, 0.6, 0.8],
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
                )]),
                "squashed.stl",
            ),
            &MeshParams {
                scale: [1.0, 2.0, 0.5],
                ..Default::default()
            },
        )
        .expect("builds");
        let verts = vertices(&squashed.batch);
        assert!(close(verts[2].0, [0.0, 2.0, 0.5]));
        let expected = super::normalize([0.0, 0.6 / 2.0, 0.8 / 0.5]).expect("unit");
        assert!(verts.iter().all(|v| close(v.1, expected)));
        assert!(verts.iter().all(|v| {
            let n = v.1;
            ((n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt() - 1.0).abs() < 1e-5
        }));
    }

    #[test]
    fn negative_scale_reverses_the_winding_and_mirrors_the_normal() {
        let bytes = binary_stl(&[(
            [0.0, 0.0, 1.0],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )]);
        let scene = from_slice(&bytes, "mirror.stl");
        let mirrored = build(
            &scene,
            &MeshParams {
                scale: [1.0, 1.0, -1.0],
                ..Default::default()
            },
        )
        .expect("builds");
        let verts = vertices(&mirrored.batch);
        // Corners come out in 0, 2, 1 order so the mirrored triangle still winds CCW around its normal.
        assert!(close(verts[0].0, [0.0, 0.0, 0.0]));
        assert!(close(verts[1].0, [0.0, 1.0, 0.0]));
        assert!(close(verts[2].0, [1.0, 0.0, 0.0]));
        // Inverse-transpose already points the normal outward again on the mirrored side.
        assert!(verts.iter().all(|v| close(v.1, [0.0, 0.0, -1.0])));
        let geometric = cross(sub(verts[1].0, verts[0].0), sub(verts[2].0, verts[0].0));
        assert!(geometric[2] < 0.0, "winding disagrees with the normal");
    }

    #[test]
    fn degenerate_triangles_are_dropped_and_counted() {
        let bytes = binary_stl(&[
            (
                [0.0, 0.0, 0.0],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            ),
            (
                [0.0, 0.0, 1.0],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ),
        ]);
        let built = build(&from_slice(&bytes, "degenerate.stl"), &MeshParams::default())
            .expect("builds");
        assert_eq!((built.triangles, built.skipped), (1, 1));
        assert_eq!(built.batch.count, 3);
        // A file where nothing survives yields no batch at all.
        let all_bad = binary_stl(&[(
            [0.0, 0.0, 0.0],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        )]);
        let error = build(&from_slice(&all_bad, "flat.stl"), &MeshParams::default()).unwrap_err();
        assert!(error.contains("degenerate"), "{error}");
        let empty = build(&Scene::default(), &MeshParams::default()).unwrap_err();
        assert!(empty.contains("no triangles"), "{empty}");
    }

    #[test]
    fn vertex_colors_win_over_the_material_and_opacity_reaches_the_alpha() {
        let built =
            build(&load("triangle_vertex_color.dae"), &MeshParams::default()).expect("builds");
        let verts = vertices(&built.batch);
        let red = super::linear_rgba8([1.0, 0.0, 0.0, 1.0], 0.5);
        let blue = super::linear_rgba8([0.0, 0.0, 1.0, 1.0], 0.5);
        let white = super::linear_rgba8([1.0, 1.0, 1.0, 1.0], 0.5);
        assert_eq!([verts[0].2, verts[1].2, verts[2].2], [red, blue, white]);
        // The effect's transparency 0.5 lands in the alpha (the slider multiplies on top of it).
        assert!(verts.iter().all(|v| v.2[3] == 128));
    }

    #[test]
    fn unusable_scale_is_rejected() {
        let scene = load("triangle_z_up.dae");
        for scale in [[0.0, 1.0, 1.0], [1.0, f64::NAN, 1.0], [1.0, 1.0, f64::INFINITY]] {
            let params = MeshParams {
                scale,
                ..Default::default()
            };
            assert!(build(&scene, &params).is_err(), "{scale:?}");
        }
    }
}
