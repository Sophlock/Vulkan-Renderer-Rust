use crate::application::assets::asset_traits::{ModelInterface, RHIModelInterface};
use crate::application::renderer::Renderer;

pub struct VKModel {
    
}

impl RHIModelInterface for VKModel {
    type RHI = Renderer;

    fn create<T: ModelInterface>(source: &T, rhi: &Self::RHI) -> Self {
        todo!()
    }
}