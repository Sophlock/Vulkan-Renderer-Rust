use crate::application::{
    assets::asset_traits::{CameraInterface, RHIModelInterface, RHISceneInterface, SceneInterface},
    rhi::{
        rhi_assets::{vulkan_camera::VKCamera, vulkan_model::VKModel, RHIResourceManager},
        VKRHI,
    },
};

pub struct VKScene {
    models: Vec<VKModel>,
    camera: VKCamera,
}

impl RHISceneInterface for VKScene {
    type RHI = VKRHI;

    fn create<T: SceneInterface>(
        source: &T,
        rhi: &Self::RHI,
        resource_manager: &mut RHIResourceManager,
    ) -> Self {
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
