use crate::renderer::Vertex;
use glam::Vec3;

use super::loader::RawMesh;

/// Processed mesh with computed normals and bounding information
#[derive(Clone)]
pub struct ProcessedMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub center: Vec3,
    pub radius: f32,
}

impl ProcessedMesh {
    /// Process a raw mesh: compute normals and bounding information
    pub fn from_raw(raw: &RawMesh) -> Self {
        // Accumulate face normals to vertices (area-weighted smooth normals)
        let mut normals = vec![Vec3::ZERO; raw.positions.len()];

        for chunk in raw.indices.chunks(3) {
            if chunk.len() < 3 {
                continue;
            }

            let i0 = chunk[0] as usize;
            let i1 = chunk[1] as usize;
            let i2 = chunk[2] as usize;

            // Bounds check
            if i0 >= raw.positions.len() || i1 >= raw.positions.len() || i2 >= raw.positions.len() {
                continue;
            }

            let v0 = raw.positions[i0];
            let v1 = raw.positions[i1];
            let v2 = raw.positions[i2];

            let edge1 = v1 - v0;
            let edge2 = v2 - v0;

            // Cross product gives area-weighted normal
            let face_normal = edge1.cross(edge2);

            normals[i0] += face_normal;
            normals[i1] += face_normal;
            normals[i2] += face_normal;
        }

        // Compute bounding box while building vertices
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);

        let vertices: Vec<Vertex> = raw
            .positions
            .iter()
            .zip(normals.iter())
            .map(|(&pos, &norm)| {
                min = min.min(pos);
                max = max.max(pos);

                Vertex::new(pos.to_array(), norm.normalize_or_zero().to_array())
            })
            .collect();

        let center = (min + max) * 0.5;
        let radius = (max - min).length() * 0.5;

        log::info!(
            "Mesh bounds: center={:?}, radius={:.2}",
            center,
            radius
        );

        Self {
            vertices,
            indices: raw.indices.clone(),
            center,
            radius,
        }
    }
}
