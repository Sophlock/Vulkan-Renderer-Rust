use crate::application::{
    assets::asset_traits::{CameraInterface, ModelInterface, RHISceneInterface, SceneInterface},
    renderer::{
        Renderer,
        rhi_assets::{vulkan_camera::VKCamera, vulkan_model::VKModel},
    },
};

pub struct VKScene {
    models: Vec<VKModel>,
    camera: VKCamera,
}

impl RHISceneInterface for VKScene {
    type RHI = Renderer;

    fn create<T: SceneInterface>(source: &T, rhi: &Self::RHI) -> Self {
        let models = source
            .models()
            .iter()
            .map(|model| model.rhi::<VKModel>(rhi))
            .collect();
        Self {
            models,
            camera: source.camera().rhi(rhi),
        }
    }

    fn models(&self) -> &[VKModel] {
        self.models.as_slice()
    }
}
