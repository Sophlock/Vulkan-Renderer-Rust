use glam::{Mat4, Vec3};

use crate::application::{
    assets::asset_traits::{CameraInterface, RHICameraInterface},
    renderer::Renderer,
};

pub struct VKCamera {
    view_projection: Mat4,
    location: glam::Vec3,
}

impl RHICameraInterface for VKCamera {
    type RHI = Renderer;

    fn create<T: CameraInterface>(source: &T, _: &Self::RHI) -> Self {
        Self {
            view_projection: source.view_projection(),
            location: source.transform().location
        }
    }

    fn view_projection(&self) -> Mat4 {
        self.view_projection
    }

    fn location(&self) -> Vec3 {
        self.location
    }
}
