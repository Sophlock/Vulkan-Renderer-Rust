use crate::application::assets::asset_traits::{ModelInterface, RHIInterface, RHIModelInterface, RHIResource};
use crate::application::renderer::rhi_assets::vulkan_material::VKMaterial;
use crate::application::renderer::rhi_assets::vulkan_mesh::VKMesh;
use crate::application::renderer::rhi_assets::RHIHandle;
use crate::application::renderer::Renderer;
use glam::Mat4;
use AssetSystem::resource_management::Resource;

pub struct VKModel {
    transform: Mat4,
    mesh: RHIHandle<VKMesh>,
    material: RHIHandle<VKMaterial>,
    uuid: usize
}

impl RHIResource for VKModel {
    fn uuid_mut(&mut self) -> &mut usize {
        &mut self.uuid
    }
}

impl Resource for VKModel {
    fn set_uuid(&mut self, uuid: usize) {
        self.uuid = uuid;
    }
}

impl RHIModelInterface for VKModel {
    type RHI = Renderer;

    fn create<T: ModelInterface>(source: &T, rhi: &Self::RHI) -> Self {
        let mesh = source.mesh();
        let material = source.material();
        Self {
            transform: source.transform().matrix(),
            mesh: rhi.resource_manager_mut().create_mesh(mesh),
            material: rhi.resource_manager_mut().create_material(material),
            uuid: 0
        }
    }
}