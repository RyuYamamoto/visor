//! Per-material face grouping of a COLLADA `<geometry>`: mesh-loader 0.1.13 concatenates every primitive group of one geometry into a single mesh and keeps only the first group's material (`// TODO: multiple materials from geometry.mesh.primitives?` in its collada/instance.rs), so a body with five bound materials arrives flat; assimp (what RViz uses) splits the same file per group, which is why RViz showed colours and we did not. We re-read only the grouping from the same bytes and hand geometry.rs a colour per face range, leaving the batch layout and the GPU side untouched.

use std::collections::HashMap;

/// One primitive group of a `<geometry>`, in the document order mesh-loader concatenates them in.
#[derive(Debug, Clone, PartialEq)]
pub struct Group {
    /// Faces this group contributes; mirrors mesh-loader's `vertex_indices_size`.
    pub faces: usize,
    /// Diffuse of the bound material (sRGB, straight alpha); None when the binding does not resolve or the material is textured.
    pub diffuse: Option<[f32; 4]>,
}

/// Group tables per `<geometry>` in document order (the order mesh-loader fills `Scene::meshes` in). None means "do not group this file" — not COLLADA, or a primitive kind whose face count we would have to guess — and leaves the caller on mesh-loader's one material per mesh.
pub fn material_groups(bytes: &[u8]) -> Option<Vec<Vec<Group>>> {
    let text = std::str::from_utf8(bytes).ok()?;
    let doc = roxmltree::Document::parse(text).ok()?;
    let root = doc.root_element();
    if root.tag_name().name() != "COLLADA" {
        return None;
    }
    // `effect` / `material` / `geometry` each live in one library element, so scanning descendants stays correct and keeps document order.
    let mut effects: HashMap<&str, [f32; 4]> = HashMap::new();
    for effect in root.descendants().filter(|n| n.has_tag_name("effect")) {
        let Some(id) = effect.attribute("id") else {
            continue;
        };
        // A <diffuse> holding a <texture> rather than a <color> leaves the group uncoloured, as before.
        let diffuse = effect
            .descendants()
            .find(|n| n.has_tag_name("diffuse"))
            .and_then(|d| d.children().find(|c| c.has_tag_name("color")))
            .and_then(|c| parse_rgba(c.text().unwrap_or_default()));
        if let Some(diffuse) = diffuse {
            effects.insert(id, diffuse);
        }
    }
    let mut materials: HashMap<&str, &str> = HashMap::new();
    for material in root.descendants().filter(|n| n.has_tag_name("material")) {
        let Some(id) = material.attribute("id") else {
            continue;
        };
        let effect = material
            .children()
            .find(|c| c.has_tag_name("instance_effect"))
            .and_then(|c| c.attribute("url"));
        if let Some(effect) = effect {
            materials.insert(id, effect.trim_start_matches('#'));
        }
    }
    // mesh-loader flattens the per-instance_geometry bindings into one symbol table; staying in step with it matters more than one geometry instanced twice with different bindings.
    let mut bound: HashMap<&str, [f32; 4]> = HashMap::new();
    for binding in root
        .descendants()
        .filter(|n| n.has_tag_name("instance_material"))
    {
        let (Some(symbol), Some(target)) =
            (binding.attribute("symbol"), binding.attribute("target"))
        else {
            continue;
        };
        let color = materials
            .get(target.trim_start_matches('#'))
            .and_then(|effect| effects.get(effect));
        if let Some(color) = color {
            bound.insert(symbol, *color);
        }
    }

    let mut per_geometry = Vec::new();
    for geometry in root.descendants().filter(|n| n.has_tag_name("geometry")) {
        let Some(mesh) = geometry.children().find(|c| c.has_tag_name("mesh")) else {
            per_geometry.push(Vec::new());
            continue;
        };
        let mut groups = Vec::new();
        for primitive in mesh.children().filter(roxmltree::Node::is_element) {
            let faces = match primitive.tag_name().name() {
                "triangles" => primitive.attribute("count")?.parse().ok()?,
                "polylist" => polylist_faces(primitive)?,
                // Not primitives, and lines contribute no triangles (mesh-loader drops them too).
                "source" | "vertices" | "extra" | "lines" | "linestrips" => continue,
                // A wrong tristrips/trifans/polygons face count would colour every later group wrong, so decline the file instead of guessing.
                _ => return None,
            };
            groups.push(Group {
                faces,
                diffuse: primitive
                    .attribute("material")
                    .and_then(|symbol| bound.get(symbol).copied()),
            });
        }
        per_geometry.push(groups);
    }
    Some(per_geometry)
}

/// Triangles a `<polylist>` contributes after fan triangulation, counted the way mesh-loader counts it.
fn polylist_faces(primitive: roxmltree::Node<'_, '_>) -> Option<usize> {
    let text = primitive
        .children()
        .find(|c| c.has_tag_name("vcount"))?
        .text()
        .unwrap_or_default();
    let mut faces = 0usize;
    for field in text.split_ascii_whitespace() {
        faces += field.parse::<usize>().ok()?.saturating_sub(2);
    }
    Some(faces)
}

/// COLLADA `<color>` text -> RGBA, accepting three components as fully opaque.
fn parse_rgba(text: &str) -> Option<[f32; 4]> {
    let mut rgba = [0.0, 0.0, 0.0, 1.0];
    let mut count = 0;
    for field in text.split_ascii_whitespace() {
        if count == 4 {
            return None;
        }
        rgba[count] = field.parse::<f32>().ok().filter(|v| v.is_finite())?;
        count += 1;
    }
    (count >= 3).then_some(rgba)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> Vec<u8> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/mesh")
            .join(name);
        std::fs::read(path).expect("read fixture")
    }

    #[test]
    fn a_two_material_geometry_reports_both_groups_in_order() {
        let groups = material_groups(&fixture("triangle_two_materials.dae")).expect("groups");
        assert_eq!(groups.len(), 1, "one <geometry>");
        assert_eq!(
            groups[0],
            vec![
                Group {
                    faces: 1,
                    diffuse: Some([1.0, 0.0, 0.0, 1.0])
                },
                Group {
                    faces: 2,
                    diffuse: Some([0.0, 0.0, 1.0, 1.0])
                },
            ]
        );
    }

    #[test]
    fn a_single_material_geometry_still_reports_one_group() {
        let groups = material_groups(&fixture("triangle_vertex_color.dae")).expect("groups");
        assert_eq!(
            groups,
            vec![vec![Group {
                faces: 1,
                diffuse: Some([0.0, 1.0, 0.0, 1.0])
            }]]
        );
    }

    #[test]
    fn non_collada_bytes_are_left_alone() {
        assert_eq!(material_groups(b"solid x\nendsolid x\n"), None);
        assert_eq!(material_groups(&[0u8, 159, 146, 150]), None);
        assert_eq!(material_groups(b"<?xml version=\"1.0\"?><robot/>"), None);
    }

    #[test]
    fn an_unsupported_primitive_kind_declines_the_whole_file() {
        let text = String::from_utf8(fixture("triangle_two_materials.dae"))
            .expect("utf-8")
            .replace("polylist", "trifans");
        assert_eq!(material_groups(text.as_bytes()), None);
    }

    #[test]
    fn polylist_faces_are_fan_triangulated_like_mesh_loader() {
        let doc = roxmltree::Document::parse(
            "<polylist count=\"3\"><vcount>3 4 5</vcount><p/></polylist>",
        )
        .expect("parses");
        assert_eq!(polylist_faces(doc.root_element()), Some(1 + 2 + 3));
        // Degenerate counts contribute nothing rather than underflowing.
        let doc =
            roxmltree::Document::parse("<polylist><vcount>2 1 0</vcount></polylist>").expect("ok");
        assert_eq!(polylist_faces(doc.root_element()), Some(0));
    }

    #[test]
    fn color_text_needs_three_or_four_finite_components() {
        assert_eq!(parse_rgba("1 0 0 1"), Some([1.0, 0.0, 0.0, 1.0]));
        assert_eq!(
            parse_rgba(" 0.5  0.25 0.125 "),
            Some([0.5, 0.25, 0.125, 1.0])
        );
        assert_eq!(parse_rgba("1 0"), None);
        assert_eq!(parse_rgba("1 0 0 1 0"), None);
        assert_eq!(parse_rgba("1 0 nan"), None);
    }
}
