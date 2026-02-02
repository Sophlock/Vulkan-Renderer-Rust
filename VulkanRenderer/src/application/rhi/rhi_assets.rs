use std::{
    cell::{Ref, RefCell},
    collections::HashMap,
    marker::PhantomData,
    ops::Deref,
    rc::{Rc, Weak},
    sync::Arc,
};

use asset_system::{assets::AssetHandle, resource_management::ResourceManager};

use crate::application::{
    assets::asset_traits::{
        MaterialInstanceInterface, MaterialInterface, MeshInterface, ModelInterface,
        RHIMaterialInstanceInterface, RHIMaterialInterface, RHIMeshInterface, RHIModelInterface,
        RHIResource, RHITextureInterface, TextureInterface,
    },
    rhi::{
        VKRHI,
        rhi_assets::{
            vulkan_material::VKMaterial, vulkan_material_instance::VKMaterialInstance,
            vulkan_mesh::VKMesh, vulkan_model::VKModel, vulkan_texture::VKTexture,
        },
    },
};

pub mod vulkan_camera;
pub mod vulkan_material;
pub mod vulkan_material_instance;
pub mod vulkan_mesh;
pub mod vulkan_model;
pub mod vulkan_scene;
pub mod vulkan_texture;

pub struct RHIResourceManager {
    resources: ResourceManager,
    asset_to_rhi: HashMap<usize, usize>,
    asset_manager: Arc<RefCell<ResourceManager>>,
    rhi: Option<Weak<VKRHI>>,
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
            let asset_manager_arc = self.asset_manager.clone();
            let asset_manager = asset_manager_arc.borrow();
            let source_data = source.get(asset_manager.deref()).unwrap();
            let asset_id = source_data.asset_metadata().uuid();
            if let Some(id) = self.asset_to_rhi.get(&asset_id) {
                RHIHandle::<$rhi_type>::new(id.clone())
            }
            else {
                let new_rhi = $rhi_type::create(source_data, self.rhi().as_ref(), self);
                let id = self.resources.add(new_rhi);
                self.asset_to_rhi.insert(asset_id, id);
                RHIHandle::<$rhi_type>::new(id)
            }
        }
    };
}

impl RHIResourceManager {
    pub fn new(asset_manager: Arc<RefCell<ResourceManager>>) -> Self {
        Self {
            resources: ResourceManager::new(),
            asset_to_rhi: HashMap::new(),
            asset_manager,
            rhi: None,
        }
    }

    pub fn register_rhi(&mut self, rhi: &Rc<VKRHI>) {
        self.rhi = Some(Rc::downgrade(rhi));
    }

    implement_rhi_resource!(create_texture, VKTexture, TextureInterface);
    implement_rhi_resource!(create_mesh, VKMesh, MeshInterface);
    implement_rhi_resource!(create_material, VKMaterial, MaterialInterface);
    implement_rhi_resource!(
        create_material_instance,
        VKMaterialInstance,
        MaterialInstanceInterface
    );
    implement_rhi_resource!(create_model, VKModel, ModelInterface);

    fn rhi(&self) -> Rc<VKRHI> {
        self.rhi.as_ref().unwrap().upgrade().unwrap()
    }

    fn asset_manager(&self) -> Ref<ResourceManager> {
        self.asset_manager.borrow()
    }
    
    pub fn resource_iterator<T: RHIResource + 'static>(&self) -> Option<impl Iterator<Item=&T>> {
        self.resources.get_iter()
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
    
    pub fn id(&self) -> usize {
        self.uuid
    }
}

impl<T: RHIResource> Clone for RHIHandle<T> {
    fn clone(&self) -> Self {
        Self {
            uuid: self.uuid,
            _phantom: PhantomData,
        }
    }
}
