//! URDF primitive meshes (box / cylinder / sphere) baked at the visual origin: dimensions go into the vertices (model matrix is rigid-only), flat faces get face normals and curved ones analytic normals.

use std::f32::consts::{PI, TAU};

use crate::render::{MeshBatch, MeshBatchBuilder};

use super::model::Shape;

/// Circumference segments of a cylinder (a robot's main silhouette, so finer than render's CYLINDER_SEGMENTS = 12).
const CYLINDER_SEGMENTS: usize = 24;
/// Meridian (longitude) divisions of a sphere.
const SPHERE_MERIDIANS: usize = 24;
/// Latitude bands of a sphere (both polar caps included).
const SPHERE_PARALLELS: usize = 12;

/// +Z face normal (cylinder top cap).
const UP: [f32; 3] = [0.0, 0.0, 1.0];
/// -Z face normal (cylinder bottom cap).
const DOWN: [f32; 3] = [0.0, 0.0, -1.0];

/// Shape -> baked mesh. None when there is nothing drawable, so a zero-vertex batch never reaches the GPU.
pub fn bake(shape: &Shape, rgba: [u8; 4]) -> Option<MeshBatch> {
    match shape {
        Shape::Box { size } if positive(size) => Some(box_mesh(*size, rgba)),
        Shape::Cylinder { radius, length } if positive(&[*radius, *length]) => {
            Some(cylinder_mesh(*radius, *length, rgba))
        }
        Shape::Sphere { radius } if positive(&[*radius]) => Some(sphere_mesh(*radius, rgba)),
        _ => None,
    }
}

/// All extents are finite and > 0 (model.rs already rejects these, but bake stays safe on its own).
fn positive(dims: &[f64]) -> bool {
    dims.iter().all(|d| d.is_finite() && *d > 0.0)
}

/// Vertex count of one box mesh (TriangleList). Used for capacity reservation and tests.
pub fn box_vertex_count() -> usize {
    6 * 2 * 3
}

/// Vertex count of one cylinder mesh (side 2 + caps 2 triangles per segment).
pub fn cylinder_vertex_count() -> usize {
    CYLINDER_SEGMENTS * 4 * 3
}

/// Vertex count of one sphere mesh (2 triangles per band per meridian, minus one at each polar cap).
pub fn sphere_vertex_count() -> usize {
    SPHERE_MERIDIANS * (SPHERE_PARALLELS * 2 - 2) * 3
}

/// Origin-centered axis-aligned box (URDF `<box size="x y z">`), one flat axis normal per face.
pub fn box_mesh(size: [f64; 3], rgba: [u8; 4]) -> MeshBatch {
    let half: [f32; 3] = std::array::from_fn(|i| size[i] as f32 * 0.5);
    // (normal axis, sign, u axis, v axis) with u x v = normal, so the quad below winds CCW seen from outside.
    const FACES: [(usize, f32, usize, usize); 6] = [
        (0, 1.0, 1, 2),
        (0, -1.0, 2, 1),
        (1, 1.0, 2, 0),
        (1, -1.0, 0, 2),
        (2, 1.0, 0, 1),
        (2, -1.0, 1, 0),
    ];
    let mut builder = MeshBatchBuilder::with_capacity(box_vertex_count());
    for (axis, sign, u, v) in FACES {
        let mut normal = [0.0; 3];
        normal[axis] = sign;
        let corner = |su: f32, sv: f32| {
            let mut p = [0.0; 3];
            p[axis] = sign * half[axis];
            p[u] = su * half[u];
            p[v] = sv * half[v];
            p
        };
        let quad = [
            corner(-1.0, -1.0),
            corner(1.0, -1.0),
            corner(1.0, 1.0),
            corner(-1.0, 1.0),
        ];
        for i in [0, 1, 2, 0, 2, 3] {
            builder.push_vertex(quad[i], normal, rgba);
        }
    }
    builder.build()
}

/// Origin-centered cylinder along Z (URDF `<cylinder radius length>`): smooth radial side normals + flat caps.
pub fn cylinder_mesh(radius: f64, length: f64, rgba: [u8; 4]) -> MeshBatch {
    let (r, half) = (radius as f32, length as f32 * 0.5);
    let radial = |i: usize| {
        let angle = TAU * (i % CYLINDER_SEGMENTS) as f32 / CYLINDER_SEGMENTS as f32;
        [angle.cos(), angle.sin(), 0.0]
    };
    let rim = |normal: [f32; 3], z: f32| [normal[0] * r, normal[1] * r, z];
    let mut builder = MeshBatchBuilder::with_capacity(cylinder_vertex_count());
    for i in 0..CYLINDER_SEGMENTS {
        let (ni, nj) = (radial(i), radial(i + 1));
        let (bottom_i, bottom_j) = (rim(ni, -half), rim(nj, -half));
        let (top_i, top_j) = (rim(ni, half), rim(nj, half));
        // Side quad: per-vertex radial normals (smooth), CCW seen from outside.
        for (p, n) in [
            (bottom_i, ni),
            (bottom_j, nj),
            (top_j, nj),
            (bottom_i, ni),
            (top_j, nj),
            (top_i, ni),
        ] {
            builder.push_vertex(p, n, rgba);
        }
        // Cap fans around each center, wound so their normals point away from the body.
        for p in [[0.0, 0.0, half], top_i, top_j] {
            builder.push_vertex(p, UP, rgba);
        }
        for p in [[0.0, 0.0, -half], bottom_j, bottom_i] {
            builder.push_vertex(p, DOWN, rgba);
        }
    }
    builder.build()
}

/// Origin-centered sphere (URDF `<sphere radius>`) as a lat/long grid; normals are the normalized positions.
pub fn sphere_mesh(radius: f64, rgba: [u8; 4]) -> MeshBatch {
    let r = radius as f32;
    // Ring k sits at polar angle phi = PI*k/P (k = 0 is the +Z pole, k = P the -Z pole).
    let point = |k: usize, m: usize| -> ([f32; 3], [f32; 3]) {
        let phi = PI * k as f32 / SPHERE_PARALLELS as f32;
        let theta = TAU * (m % SPHERE_MERIDIANS) as f32 / SPHERE_MERIDIANS as f32;
        let normal = [phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos()];
        (normal.map(|c| c * r), normal)
    };
    let mut builder = MeshBatchBuilder::with_capacity(sphere_vertex_count());
    for band in 0..SPHERE_PARALLELS {
        for m in 0..SPHERE_MERIDIANS {
            let (upper_a, upper_b) = (point(band, m), point(band, m + 1));
            let (lower_a, lower_b) = (point(band + 1, m), point(band + 1, m + 1));
            // Both triangles wind CCW seen from outside; at each pole one of them collapses, so skip it.
            if band + 1 != SPHERE_PARALLELS {
                for (p, n) in [upper_a, lower_a, lower_b] {
                    builder.push_vertex(p, n, rgba);
                }
            }
            if band != 0 {
                for (p, n) in [upper_a, lower_b, upper_b] {
                    builder.push_vertex(p, n, rgba);
                }
            }
        }
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::MESH_STRIDE;

    /// Decode the packed 28B layout back into (position, normal, rgba) so the geometry can be asserted.
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

    /// Triangles as [(position, normal); 3] groups.
    fn triangles(batch: &MeshBatch) -> Vec<[([f32; 3], [f32; 3]); 3]> {
        vertices(batch)
            .chunks(3)
            .map(|t| std::array::from_fn(|i| (t[i].0, t[i].1)))
            .collect()
    }

    fn norm(v: [f32; 3]) -> f32 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    /// Geometric normal of a triangle (cross of its edges), to verify the winding is outward.
    fn face_normal(tri: &[([f32; 3], [f32; 3]); 3]) -> [f32; 3] {
        let e1: [f32; 3] = std::array::from_fn(|c| tri[1].0[c] - tri[0].0[c]);
        let e2: [f32; 3] = std::array::from_fn(|c| tri[2].0[c] - tri[0].0[c]);
        [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ]
    }

    #[test]
    fn box_mesh_has_expected_extent_and_axis_normals() {
        let size = [0.4, 0.3, 0.1];
        let batch = box_mesh(size, [1, 2, 3, 4]);
        assert_eq!(batch.count as usize, box_vertex_count());
        assert_eq!(box_vertex_count(), 36);
        let verts = vertices(&batch);
        // Every vertex is a corner of the origin-centered box.
        for (position, _, _) in &verts {
            for c in 0..3 {
                assert!(
                    (position[c].abs() - size[c] as f32 * 0.5).abs() < 1e-6,
                    "position {position:?} is not on the box surface"
                );
            }
        }
        // 6 axis-aligned face normals, 6 vertices each.
        let mut counts = std::collections::BTreeMap::new();
        for (_, normal, _) in &verts {
            *counts.entry(format!("{normal:?}")).or_insert(0) += 1;
        }
        assert_eq!(counts.len(), 6);
        assert!(counts.values().all(|c| *c == 6), "{counts:?}");
        for (_, normal, _) in &verts {
            assert!((norm(*normal) - 1.0).abs() < 1e-6);
            // Exactly one component is +/-1 and the others are zero.
            assert_eq!(normal.iter().filter(|c| c.abs() == 1.0).count(), 1);
        }
        // Winding is outward: the geometric normal agrees with the stored one.
        for tri in triangles(&batch) {
            assert!(dot(face_normal(&tri), tri[0].1) > 0.0);
        }
    }

    #[test]
    fn cylinder_mesh_spans_length_on_z_and_side_normals_are_radial() {
        let (radius, length) = (0.04, 0.06);
        let batch = cylinder_mesh(radius, length, [9, 9, 9, 255]);
        assert_eq!(batch.count as usize, cylinder_vertex_count());
        assert_eq!(cylinder_vertex_count(), 288);
        let half = length as f32 * 0.5;
        let mut side = 0;
        let mut caps = 0;
        for (position, normal, _) in vertices(&batch) {
            assert!(position[2].abs() <= half + 1e-6);
            let xy = (position[0] * position[0] + position[1] * position[1]).sqrt();
            assert!(xy <= radius as f32 + 1e-6);
            if normal[2] == 0.0 {
                // Side: normal is horizontal, unit length, and the vertex sits on the rim.
                side += 1;
                assert!((norm(normal) - 1.0).abs() < 1e-6);
                assert!((xy - radius as f32).abs() < 1e-6);
                // Radial: the normal points along the vertex's own xy direction.
                assert!((normal[0] * radius as f32 - position[0]).abs() < 1e-6);
                assert!((normal[1] * radius as f32 - position[1]).abs() < 1e-6);
            } else {
                // Caps: axis-aligned normal matching the cap's z.
                caps += 1;
                assert_eq!(normal, if position[2] > 0.0 { UP } else { DOWN });
                assert!((position[2].abs() - half).abs() < 1e-6);
            }
        }
        assert_eq!(side, CYLINDER_SEGMENTS * 6);
        assert_eq!(caps, CYLINDER_SEGMENTS * 6);
        // The full length is spanned (extreme z on both ends is present).
        let zs: Vec<f32> = vertices(&batch).iter().map(|v| v.0[2]).collect();
        assert!(zs.iter().any(|z| (*z - half).abs() < 1e-6));
        assert!(zs.iter().any(|z| (*z + half).abs() < 1e-6));
        for tri in triangles(&batch) {
            assert!(dot(face_normal(&tri), tri[0].1) > 0.0);
        }
    }

    #[test]
    fn sphere_mesh_vertices_lie_on_radius_with_outward_normals() {
        let radius = 0.05;
        let batch = sphere_mesh(radius, [7, 7, 7, 255]);
        assert_eq!(batch.count as usize, sphere_vertex_count());
        assert_eq!(sphere_vertex_count(), 1584);
        for (position, normal, _) in vertices(&batch) {
            assert!((norm(position) - radius as f32).abs() < 1e-6);
            assert!((norm(normal) - 1.0).abs() < 1e-6);
            // Normal is the normalized position (smooth shading) and points outward.
            for c in 0..3 {
                assert!((normal[c] * radius as f32 - position[c]).abs() < 1e-6);
            }
            assert!(dot(normal, position) > 0.0);
        }
        // No degenerate triangles survive at the poles.
        for tri in triangles(&batch) {
            assert!(norm(face_normal(&tri)) > 0.0, "degenerate triangle");
            assert!(dot(face_normal(&tri), tri[0].1) > 0.0);
        }
    }

    #[test]
    fn bake_returns_none_for_unsupported_and_non_positive_shapes() {
        assert!(
            bake(
                &Shape::Unsupported {
                    kind: "mesh",
                    reason: "not implemented".to_owned()
                },
                [0; 4]
            )
            .is_none()
        );
        assert!(bake(&Shape::Sphere { radius: 0.0 }, [0; 4]).is_none());
        assert!(bake(&Shape::Sphere { radius: -1.0 }, [0; 4]).is_none());
        assert!(
            bake(
                &Shape::Box {
                    size: [1.0, 0.0, 1.0]
                },
                [0; 4]
            )
            .is_none()
        );
        assert!(
            bake(
                &Shape::Cylinder {
                    radius: 0.1,
                    length: f64::NAN
                },
                [0; 4]
            )
            .is_none()
        );
        // Positive shapes all bake.
        for shape in [
            Shape::Box { size: [1.0; 3] },
            Shape::Cylinder {
                radius: 0.5,
                length: 2.0,
            },
            Shape::Sphere { radius: 0.5 },
        ] {
            assert!(bake(&shape, [0; 4]).is_some());
        }
    }

    #[test]
    fn bake_writes_the_given_color() {
        let rgba = [12, 34, 56, 200];
        for shape in [
            Shape::Box { size: [1.0; 3] },
            Shape::Cylinder {
                radius: 0.5,
                length: 2.0,
            },
            Shape::Sphere { radius: 0.5 },
        ] {
            let batch = bake(&shape, rgba).expect("drawable");
            assert!(vertices(&batch).iter().all(|v| v.2 == rgba));
        }
    }
}
