use glam::Mat4;
use crate::application::assets::asset_traits::{ModelInterface, RHIInterface, RHIModelInterface};
use crate::application::renderer::Renderer;
use crate::application::renderer::rhi_assets::RHIHandle;
use crate::application::renderer::rhi_assets::vulkan_material::VKMaterial;
use crate::application::renderer::rhi_assets::vulkan_mesh::VKMesh;

pub struct VKModel {
    transform: Mat4,
    mesh: RHIHandle<VKMesh>,
    material: RHIHandle<VKMaterial>
}

impl RHIModelInterface for VKModel {
    type RHI = Renderer;

    fn create<'a, T: ModelInterface>(source: &'a T, rhi: &mut Self::RHI) -> Self {
        let mesh = source.mesh();
        let material = source.material();
        todo!();
        Self {
            transform: source.transform().matrix(),
            mesh: rhi.resource_manager_mut().create_mesh(mesh),
            material: rhi.resource_manager_mut().create_material(material.clone())
        }
    }
}