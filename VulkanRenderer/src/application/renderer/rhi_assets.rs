use crate::application::{
    assets::asset_traits::{
        MaterialInterface, MeshInterface, RHIMaterialInterface, RHIMeshInterface, RHIResource,
        RHITextureInterface, TextureInterface,
    },
    renderer::{
        rhi_assets::{vulkan_material::VKMaterial, vulkan_mesh::VKMesh, vulkan_texture::VKTexture},
        Renderer,
    },
};
use std::{
    cell::RefCell,
    collections::HashMap,
    marker::PhantomData,
    ops::Deref,
    rc::{Rc, Weak},
    sync::Arc
};
use AssetSystem::assets::AssetHandle;
use AssetSystem::resource_management::ResourceManager;

pub mod vulkan_camera;
pub mod vulkan_material;
pub mod vulkan_mesh;
pub mod vulkan_model;
pub mod vulkan_scene;
pub mod vulkan_texture;

pub struct RHIResourceManager {
    resources: ResourceManager,
    asset_to_rhi: HashMap<usize, usize>,
    asset_manager: Arc<ResourceManager>,
    rhi: Option<Weak<RefCell<Renderer>>>,
}

pub struct RHIHandle<T: RHIResource + 'static> {
    uuid: usize,
    phantom: PhantomData<T>,
}

impl RHIResourceManager {
    pub fn new(asset_manager: Arc<ResourceManager>) -> Self {
        Self {
            resources: ResourceManager::new(),
            asset_to_rhi: HashMap::new(),
            asset_manager,
            rhi: None,
        }
    }

    pub fn register_rhi(&mut self, rhi: &Rc<RefCell<Renderer>>) {
        self.rhi = Some(Rc::downgrade(rhi));
    }

    pub fn create_texture<T: TextureInterface + 'static>(
        &mut self,
        source: AssetHandle<T>,
    ) -> RHIHandle<VKTexture> {
        let source_data = source.get(self.asset_manager()).unwrap();
        let asset_id = source_data.asset_metadata().uuid();
        let tex = VKTexture::create(source_data, self.rhi().as_ref().borrow().deref());
        let id = self.resources.add(tex);
        self.asset_to_rhi.insert(asset_id, id);
        RHIHandle::<VKTexture>::new(id)
    }

    pub fn create_mesh<T: MeshInterface + 'static>(
        &mut self,
        source: AssetHandle<T>,
    ) -> RHIHandle<VKMesh> {
        let source_data = source.get(self.asset_manager()).unwrap();
        let asset_id = source_data.asset_metadata().uuid();
        let mesh = VKMesh::create(source_data, self.rhi().as_ref().borrow().deref());
        let id = self.resources.add(mesh);
        self.asset_to_rhi.insert(asset_id, id);
        RHIHandle::<VKMesh>::new(id)
    }

    pub fn create_material<T: MaterialInterface + 'static>(
        &mut self,
        source: AssetHandle<T>,
    ) -> RHIHandle<VKMaterial> {
        let source_data = source.get(self.asset_manager()).unwrap();
        let asset_id = source_data.asset_metadata().uuid();
        let mesh = VKMaterial::create(source_data, self.rhi().as_ref().borrow().deref());
        let id = self.resources.add(mesh);
        self.asset_to_rhi.insert(asset_id, id);
        RHIHandle::<VKMaterial>::new(id)
    }

    fn rhi(&self) -> Rc<RefCell<Renderer>> {
        self.rhi.as_ref().unwrap().upgrade().unwrap()
    }

    fn asset_manager(&self) -> &ResourceManager {
        self.asset_manager.as_ref()
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
