use glam::Mat4;
use crate::application::assets::asset_traits::{CameraInterface, RHICameraInterface};
use crate::application::renderer::Renderer;

pub struct VKCamera {
    view_projection: Mat4
}

impl RHICameraInterface for VKCamera {
    type RHI = Renderer;

    fn create<T: CameraInterface>(source: &T, rhi: &Self::RHI) -> Self {
        Self {
            view_projection: source.view_projection(),
        }
    }
}