use asset_system::resource_management::Resource;
use glam::Mat4;

use crate::application::{
    assets::asset_traits::{ModelInterface, RHIInterface, RHIModelInterface, RHIResource},
    renderer::{
        Renderer,
        rhi_assets::{
            RHIHandle, vulkan_material_instance::VKMaterialInstance, vulkan_mesh::VKMesh,
        },
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

    fn create<T: ModelInterface>(source: &T, rhi: &Self::RHI) -> Self {
        let mesh = source.mesh();
        let material = source.material();
        let mut resource_manager = rhi.resource_manager_mut();
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
}
