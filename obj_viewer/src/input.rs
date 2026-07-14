use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta};
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::camera::OrbitCamera;

/// Actions that can be triggered by keyboard input
pub enum KeyAction {
    Quit,
    ResetCamera,
}

/// Tracks input state and handles input events
pub struct InputState {
    pub left_mouse_pressed: bool,
    pub right_mouse_pressed: bool,
    pub last_mouse_pos: Option<(f64, f64)>,
    pub rotate_sensitivity: f32,
    pub zoom_sensitivity: f32,
}

impl InputState {
    pub fn new() -> Self {
        Self {
            left_mouse_pressed: false,
            right_mouse_pressed: false,
            last_mouse_pos: None,
            rotate_sensitivity: 0.005,
            zoom_sensitivity: 0.1,
        }
    }

    /// Handle mouse button press/release
    pub fn handle_mouse_button(&mut self, button: MouseButton, state: ElementState) {
        let pressed = state == ElementState::Pressed;
        match button {
            MouseButton::Left => self.left_mouse_pressed = pressed,
            MouseButton::Right => self.right_mouse_pressed = pressed,
            _ => {}
        }
    }

    /// Handle cursor movement, updating camera if dragging
    pub fn handle_cursor_moved(&mut self, position: (f64, f64), camera: &mut OrbitCamera) {
        if let Some(last) = self.last_mouse_pos {
            let delta_x = (position.0 - last.0) as f32;
            let delta_y = (position.1 - last.1) as f32;

            if self.left_mouse_pressed {
                camera.rotate(delta_x, delta_y, self.rotate_sensitivity);
            }
        }
        self.last_mouse_pos = Some(position);
    }

    /// Handle mouse scroll for zooming
    pub fn handle_scroll(&mut self, delta: MouseScrollDelta, camera: &mut OrbitCamera) {
        let scroll = match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 100.0,
        };
        camera.zoom(scroll, self.zoom_sensitivity);
    }

    /// Handle keyboard input, returning action if applicable
    pub fn handle_keyboard(&mut self, event: &KeyEvent) -> Option<KeyAction> {
        if event.state != ElementState::Pressed {
            return None;
        }

        match event.physical_key {
            PhysicalKey::Code(KeyCode::Escape) | PhysicalKey::Code(KeyCode::KeyQ) => {
                Some(KeyAction::Quit)
            }
            PhysicalKey::Code(KeyCode::KeyR) => Some(KeyAction::ResetCamera),
            _ => None,
        }
    }
}

impl Default for InputState {
    fn default() -> Self {
        Self::new()
    }
}
