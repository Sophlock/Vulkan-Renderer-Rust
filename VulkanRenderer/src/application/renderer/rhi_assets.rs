use crate::application::assets::asset_traits::{MaterialInstanceInterface, RHIMaterialInstanceInterface};
use crate::application::assets::asset_traits::RHIModelInterface;
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
use crate::application::assets::asset_traits::ModelInterface;
use crate::application::renderer::rhi_assets::vulkan_material_instance::VKMaterialInstance;
use crate::application::renderer::rhi_assets::vulkan_model::VKModel;

pub mod vulkan_camera;
pub mod vulkan_material;
pub mod vulkan_mesh;
pub mod vulkan_model;
pub mod vulkan_scene;
pub mod vulkan_texture;
pub mod vulkan_material_instance;

pub struct RHIResourceManager {
    resources: ResourceManager,
    asset_to_rhi: HashMap<usize, usize>,
    asset_manager: Arc<ResourceManager>,
    rhi: Option<Weak<Renderer>>,
}

pub struct RHIHandle<T: RHIResource + 'static> {
    uuid: usize,
    _phantom: PhantomData<T>,
}

macro_rules! implement_rhi_resource {
    ($create_func:ident, $rhi_type:ident, $asset_interface:ident) => {
        pub fn $create_func<T: $asset_interface + 'static>(
            &mut self,
            source: AssetHandle<T>,
        ) -> RHIHandle<$rhi_type> {
            let source_data = source.get(self.asset_manager()).unwrap();
            let asset_id = source_data.asset_metadata().uuid();
            let tex = $rhi_type::create(source_data, self.rhi().as_ref());
            let id = self.resources.add(tex);
            self.asset_to_rhi.insert(asset_id, id);
            RHIHandle::<$rhi_type>::new(id)
    }
    };
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

    pub fn register_rhi(&mut self, rhi: &Rc<Renderer>) {
        self.rhi = Some(Rc::downgrade(rhi));
    }

    implement_rhi_resource!(create_texture, VKTexture, TextureInterface);
    implement_rhi_resource!(create_mesh, VKMesh, MeshInterface);
    implement_rhi_resource!(create_material, VKMaterial, MaterialInterface);
    implement_rhi_resource!(create_material_instance, VKMaterialInstance, MaterialInstanceInterface);
    implement_rhi_resource!(create_model, VKModel, ModelInterface);

    fn rhi(&self) -> Rc<Renderer> {
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
            _phantom: PhantomData,
        }
    }

    pub fn get<'a>(&self, manager: &'a RHIResourceManager) -> Option<&'a T> {
        manager.resources.get::<T>(self.uuid)
    }
}

impl<T: RHIResource> Clone for RHIHandle<T> {
    fn clone(&self) -> Self {
        Self {
            uuid: self.uuid,
            _phantom: PhantomData
        }
    }
}
