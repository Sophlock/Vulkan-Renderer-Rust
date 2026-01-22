use glam::Mat4;
use crate::application::assets::asset_traits::CameraInterface;
use super::transform::Transform;

pub struct Camera {
    pub transform: Transform,
    pub fov: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn new(fov: f32, aspect: f32, near: f32, far: f32) -> Self {
        Self { transform: Transform::default(), fov, aspect, near, far }
    }
}

impl CameraInterface for Camera {
    fn view_projection(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov, self.aspect, self.near, self.far) * self.transform.matrix()
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            transform: Transform::default(),
            fov: 80.0,
            aspect: 16.0 / 9.0,
            near: 0.001,
            far: 100000.0,
        }
    }
}