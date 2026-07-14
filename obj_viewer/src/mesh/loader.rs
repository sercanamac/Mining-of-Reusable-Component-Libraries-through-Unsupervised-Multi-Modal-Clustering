use anyhow::Result;
use glam::Vec3;

#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
#[cfg(not(target_arch = "wasm32"))]
use std::io::{BufRead, BufReader};
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

/// Raw mesh data loaded from OBJ file
pub struct RawMesh {
    pub positions: Vec<Vec3>,
    pub indices: Vec<u32>,
}

/// Parse OBJ from string content (works on all platforms including WASM)
pub fn parse_obj_str(content: &str) -> Result<RawMesh> {
    let mut positions = Vec::new();
    let mut indices = Vec::new();

    for line in content.lines() {
        let line = line.trim();

        // Skip comments and empty lines
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "v" => {
                // Vertex: v x y z [w]
                if parts.len() >= 4 {
                    let x: f32 = parts[1].parse()?;
                    let y: f32 = parts[2].parse()?;
                    let z: f32 = parts[3].parse()?;
                    positions.push(Vec3::new(x, y, z));
                }
            }
            "f" => {
                // Face: f v1 v2 v3 ... or f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 ...
                let face_indices: Vec<u32> = parts[1..]
                    .iter()
                    .filter_map(|s| {
                        // Handle "v/vt/vn" or "v//vn" or "v" formats
                        s.split('/').next()?.parse::<u32>().ok()
                    })
                    .map(|i| i.saturating_sub(1)) // Convert to 0-indexed
                    .collect();

                // Triangulate polygon using fan triangulation
                if face_indices.len() >= 3 {
                    for i in 1..face_indices.len() - 1 {
                        indices.push(face_indices[0]);
                        indices.push(face_indices[i]);
                        indices.push(face_indices[i + 1]);
                    }
                }
            }
            _ => {
                // Ignore other directives (vn, vt, mtllib, usemtl, etc.)
            }
        }
    }

    if positions.is_empty() {
        anyhow::bail!("OBJ file contains no vertices");
    }

    if indices.is_empty() {
        anyhow::bail!("OBJ file contains no faces");
    }

    log::info!(
        "Parsed OBJ: {} vertices, {} triangles",
        positions.len(),
        indices.len() / 3
    );

    Ok(RawMesh { positions, indices })
}

/// Load OBJ from filesystem (native only)
#[cfg(not(target_arch = "wasm32"))]
pub fn load_obj(path: &Path) -> Result<RawMesh> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let content: String = reader
        .lines()
        .collect::<std::io::Result<Vec<_>>>()?
        .join("\n");
    parse_obj_str(&content)
}
