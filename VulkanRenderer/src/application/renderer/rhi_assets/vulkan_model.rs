use asset_system::resource_management::Resource;
use glam::Mat4;

use crate::application::{
    assets::asset_traits::{ModelInterface, RHIModelInterface, RHIResource},
    renderer::{
        rhi_assets::{
            vulkan_material_instance::VKMaterialInstance, vulkan_mesh::VKMesh, RHIHandle,
            RHIResourceManager,
        },
        Renderer,
    },
};

pub struct VKModel {
    transform: Mat4,
    mesh: RHIHandle<VKMesh>,
    material: RHIHandle<VKMaterialInstance>,
    uuid: usize,
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

    fn create<T: ModelInterface>(
        source: &T,
        _: &Self::RHI,
        resource_manager: &mut RHIResourceManager,
    ) -> Self {
        let mesh = source.mesh();
        let material = source.material();
        Self {
            transform: source.transform().matrix(),
            mesh: resource_manager.create_mesh(mesh),
            material: resource_manager.create_material_instance(material),
            uuid: 0,
        }
    }

    fn mesh(&self) -> RHIHandle<VKMesh> {
        self.mesh.clone()
    }

    fn material(&self) -> RHIHandle<VKMaterialInstance> {
        self.material.clone()
    }

    fn transform(&self) -> Mat4 {
        self.transform
    }
}
