use crate::application::{
    assets::asset_traits::{CameraInterface, RHISceneInterface, SceneInterface},
    rhi::{
        VKRHI,
        rhi_assets::{
            RHIHandle, RHIResourceManager, vulkan_camera::VKCamera, vulkan_model::VKModel,
        },
    },
};

pub struct VKScene {
    models: Vec<RHIHandle<VKModel>>,
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
            .map(|model| resource_manager.create_model(model.clone()))
            .collect();
        Self {
            models,
            camera: source.camera().rhi(rhi),
        }
    }

    fn models(&self) -> &[RHIHandle<VKModel>] {
        self.models.as_slice()
    }

    fn camera(&self) -> &VKCamera {
        &self.camera
    }

    fn set_camera(&mut self, camera: VKCamera) {
        self.camera = camera;
    }
}
