use crate::application::assets::asset_traits::{
    MeshInterface, RHIMeshInterface, RHIResource, RHITextureInterface, TextureInterface,
};
use crate::application::renderer::Renderer;
use crate::application::renderer::rhi_assets::vulkan_mesh::VKMesh;
use crate::application::renderer::rhi_assets::vulkan_texture::VKTexture;
use crate::application::resource_management::ResourceManager;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

pub mod vulkan_camera;
pub mod vulkan_material;
pub mod vulkan_mesh;
pub mod vulkan_model;
pub mod vulkan_scene;
pub mod vulkan_texture;

struct RHIResourceManager {
    resources: ResourceManager,
    asset_to_rhi: HashMap<usize, usize>,
}

struct RHIHandle<T: RHIResource + 'static> {
    uuid: usize,
    phantom: PhantomData<T>,
}

impl RHIResourceManager {
    fn new() -> Self {
        Self {
            resources: ResourceManager::new(),
            asset_to_rhi: HashMap::new(),
        }
    }

    fn create_texture<T: TextureInterface>(
        &mut self,
        source: &T,
        rhi: &Renderer,
    ) -> RHIHandle<VKTexture> {
        let asset_id = source.asset_metadata().uuid();
        let tex = VKTexture::create(source, rhi);
        let id = self.resources.add(tex);
        self.asset_to_rhi.insert(asset_id, id);
        RHIHandle::<VKTexture>::new(id)
    }

    fn create_mesh<T: MeshInterface>(&mut self, source: &T, rhi: &Renderer) -> RHIHandle<VKMesh> {
        let asset_id = source.asset_metadata().uuid();
        let mesh = VKMesh::create(source, rhi);
        let id = self.resources.add(mesh);
        self.asset_to_rhi.insert(asset_id, id);
        RHIHandle::<VKMesh>::new(id)
    }
}

impl<T: RHIResource + 'static> RHIHandle<T> {
    fn new(uuid: usize) -> Self {
        Self {
            uuid,
            phantom: PhantomData,
        }
    }

    fn get<'a>(&self, manager: &'a RHIResourceManager) -> Option<&'a T> {
        manager.resources.get::<T>(self.uuid)
    }
}
