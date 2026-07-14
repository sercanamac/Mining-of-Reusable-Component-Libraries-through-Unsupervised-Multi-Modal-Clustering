use bytemuck::{Pod, Zeroable};
use glam::Mat4;

use super::OrbitCamera;

/// Camera uniform data sent to GPU
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct CameraUniform {
    /// Combined view-projection matrix
    view_proj: [[f32; 4]; 4],
    /// Camera position in world space (w unused, for alignment)
    view_position: [f32; 4],
    /// Light position in world space (w unused)
    light_position: [f32; 4],
    /// Material diffuse color (RGB + alpha)
    material_color: [f32; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            view_position: [0.0, 0.0, 5.0, 1.0],
            light_position: [10.0, 10.0, 10.0, 1.0],
            material_color: [0.7, 0.75, 0.8, 1.0],
        }
    }

    /// Set the material diffuse color (RGB, 0.0-1.0) with full opacity.
    pub fn set_material_color(&mut self, r: f32, g: f32, b: f32) {
        self.material_color = [r, g, b, 1.0];
    }

    /// Set the material color with custom alpha (0.0-1.0) for transparency.
    pub fn set_material_color_alpha(&mut self, r: f32, g: f32, b: f32, a: f32) {
        self.material_color = [r, g, b, a];
    }

    /// Tell the shader to use per-vertex colors instead of the uniform color.
    pub fn use_vertex_colors(&mut self) {
        self.material_color[3] = 0.0; // a < 0.5 → shader reads vertex color
    }

    /// Update uniform data from camera state
    pub fn update(&mut self, camera: &OrbitCamera, aspect: f32) {
        self.view_proj = camera.view_projection_matrix(aspect).to_cols_array_2d();

        let pos = camera.position();
        self.view_position = [pos.x, pos.y, pos.z, 1.0];

        // Light follows camera (headlight style)
        self.light_position = [pos.x, pos.y, pos.z, 1.0];
    }
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self::new()
    }
}
