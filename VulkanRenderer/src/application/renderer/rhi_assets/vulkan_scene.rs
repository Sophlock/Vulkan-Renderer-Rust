use crate::application::{
    assets::asset_traits::{CameraInterface, ModelInterface, RHISceneInterface, SceneInterface},
    renderer::{
        Renderer,
        rhi_assets::{vulkan_camera::VKCamera, vulkan_model::VKModel},
    },
};
use crate::application::assets::asset_traits::{RHIInterface, RHIModelInterface};
use crate::application::renderer::rhi_assets::RHIResourceManager;

pub struct VKScene {
    models: Vec<VKModel>,
    camera: VKCamera,
}

impl RHISceneInterface for VKScene {
    type RHI = Renderer;

    fn create<T: SceneInterface>(source: &T, rhi: &Self::RHI, resource_manager: &mut RHIResourceManager) -> Self {
        let models = source
            .models()
            .iter()
            .map(|model| VKModel::create(model, rhi, resource_manager))
            .collect();
        Self {
            models,
            camera: source.camera().rhi(rhi),
        }
    }

    fn models(&self) -> &[VKModel] {
        self.models.as_slice()
    }

    fn camera(&self) -> &VKCamera {
        &self.camera
    }
}
