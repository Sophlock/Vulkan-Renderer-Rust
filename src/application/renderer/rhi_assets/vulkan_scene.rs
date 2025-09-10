use crate::application::assets::asset_traits::{ModelInterface, RHISceneInterface, SceneInterface};
use crate::application::renderer::Renderer;
use crate::application::renderer::rhi_assets::vulkan_model::VKModel;

pub struct VKScene {
    
}

impl RHISceneInterface for VKScene {
    type RHI = Renderer;

    fn create<T: SceneInterface>(source: &T, rhi: &Self::RHI) -> Self {
        let _: Vec<_> = source.models().iter().map(|model| model.rhi::<VKModel>(rhi)).collect();
        todo!()
    }
}