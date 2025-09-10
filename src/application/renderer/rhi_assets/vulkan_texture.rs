use std::sync::Arc;
use vulkano::image::view::ImageView;
use crate::application::assets::asset_traits::{RHITextureInterface, TextureInterface};
use crate::application::renderer::Renderer;

pub struct VKTexture {
    image: Arc<ImageView>
}

impl RHITextureInterface for VKTexture {
    type RHI = Renderer;

    fn create<T: TextureInterface>(source: &T, rhi: &Self::RHI) -> Self {
        todo!()
    }
}