use glam::Mat4;

use super::transform::Transform;
use crate::application::assets::asset_traits::CameraInterface;

pub struct Camera {
    pub transform: Transform,
    pub fov: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
    pub speed: f32,
    pub rot_speed: f32,
}

impl Camera {
    pub fn new(fov: f32, aspect: f32, near: f32, far: f32) -> Self {
        Self {
            transform: Transform::default(),
            fov,
            aspect,
            near,
            far,
            ..Self::default()
        }
    }
}

impl CameraInterface for Camera {
    fn view_projection(&self) -> Mat4 {
        let mut persp =
            Mat4::perspective_rh(self.fov.to_radians(), self.aspect, self.near, self.far);
        persp.w_axis.w *= -1.0;
        persp * self.transform.matrix().inverse()
    }

    fn transform(&self) -> Transform {
        self.transform
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
            speed: 15f32,
            rot_speed: 3f32,
        }
    }
}
