use glam::{EulerRot, Mat4, Quat, Vec3};

/// Orbit camera that rotates around a target point
pub struct OrbitCamera {
    /// Point to orbit around (typically mesh center)
    pub target: Vec3,
    /// Distance from target
    pub distance: f32,
    /// Horizontal rotation in radians
    pub yaw: f32,
    /// Vertical rotation in radians (clamped)
    pub pitch: f32,
    /// Field of view in radians
    pub fov_y: f32,
    /// Near clipping plane
    pub near: f32,
    /// Far clipping plane
    pub far: f32,

    // Constraints
    pub min_distance: f32,
    pub max_distance: f32,
    pub min_pitch: f32,
    pub max_pitch: f32,
}

impl OrbitCamera {
    pub fn new(target: Vec3, distance: f32) -> Self {
        Self {
            target,
            distance,
            yaw: 0.0,
            pitch: 0.3, // Slight downward angle
            fov_y: 45.0_f32.to_radians(),
            near: 0.1,
            far: 1000.0,
            min_distance: 0.5,
            max_distance: 500.0,
            min_pitch: -std::f32::consts::FRAC_PI_2 + 0.1,
            max_pitch: std::f32::consts::FRAC_PI_2 - 0.1,
        }
    }

    /// Fit camera to view a mesh with given center and bounding radius
    pub fn fit_to_mesh(&mut self, center: Vec3, radius: f32) {
        self.target = center;

        // Calculate distance to fit mesh in view
        let half_fov = self.fov_y / 2.0;
        self.distance = radius / half_fov.tan() * 1.0; // tight fit, maximize object in frame

        // Adjust constraints based on mesh size
        self.min_distance = radius * 0.1;
        self.max_distance = radius * 10.0;
        self.near = (radius * 0.01).max(0.001);
        self.far = radius * 20.0;

        log::info!(
            "Camera fit: distance={:.2}, near={:.4}, far={:.2}",
            self.distance,
            self.near,
            self.far
        );
    }

    /// Rotate camera based on mouse delta
    pub fn rotate(&mut self, delta_x: f32, delta_y: f32, sensitivity: f32) {
        self.yaw -= delta_x * sensitivity;
        self.pitch -= delta_y * sensitivity;
        self.pitch = self.pitch.clamp(self.min_pitch, self.max_pitch);
    }

    /// Zoom camera (change distance)
    pub fn zoom(&mut self, delta: f32, sensitivity: f32) {
        self.distance *= 1.0 - delta * sensitivity;
        self.distance = self.distance.clamp(self.min_distance, self.max_distance);
    }

    /// Reset camera to default orientation
    pub fn reset(&mut self, center: Vec3, radius: f32) {
        self.yaw = 0.0;
        self.pitch = 0.3;
        self.fit_to_mesh(center, radius);
    }

    /// Get camera position in world space
    pub fn position(&self) -> Vec3 {
        let rotation = Quat::from_euler(EulerRot::YXZ, self.yaw, self.pitch, 0.0);
        let offset = rotation * Vec3::new(0.0, 0.0, self.distance);
        self.target + offset
    }

    /// Get view matrix (world to camera space)
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position(), self.target, Vec3::Y)
    }

    /// Get projection matrix
    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, aspect, self.near, self.far)
    }

    /// Get combined view-projection matrix
    pub fn view_projection_matrix(&self, aspect: f32) -> Mat4 {
        self.projection_matrix(aspect) * self.view_matrix()
    }
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self::new(Vec3::ZERO, 5.0)
    }
}
