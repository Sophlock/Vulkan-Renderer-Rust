use std::{
    any::TypeId,
    cell::{Ref, RefCell},
    collections::HashMap,
    marker::PhantomData,
    rc::{Rc, Weak},
    sync::Arc,
};

use asset_system::{assets::AssetHandle, resource_management::ResourceManager};
use vulkano::{
    DeviceSize,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryAllocator},
};

use crate::application::{
    assets::{
        AssetManager::AssetManager,
        asset_traits::{
            MaterialInstanceInterface, MaterialInterface, MeshInterface, ModelInterface,
            RHIMaterialInstanceInterface, RHIMaterialInterface, RHIMeshInterface,
            RHIModelInterface, RHIResource, RHITextureInterface, TextureInterface,
        },
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
    asset_manager: Arc<RefCell<AssetManager>>,
    rhi: Option<Weak<VKRHI>>,
    shared_buffers: HashMap<TypeId, SharedBuffer>,
}

struct SharedBuffer {
    buffer: Subbuffer<[u8]>,
    next_index: DeviceSize,
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
            let source_data = source.get(asset_manager.resource_manager()).unwrap();
            let asset_id = source_data.asset_metadata().uuid();
            if let Some(id) = self.asset_to_rhi.get(&asset_id) {
                RHIHandle::<$rhi_type>::new(id.clone())
            } else {
                let new_rhi = $rhi_type::create(source_data, self.rhi().as_ref(), self);
                let id = self.resources.add(new_rhi);
                self.asset_to_rhi.insert(asset_id, id);
                RHIHandle::<$rhi_type>::new(id)
            }
        }
    };
}

impl RHIResourceManager {
    pub fn new(asset_manager: Arc<RefCell<AssetManager>>) -> Self {
        Self {
            resources: ResourceManager::new(),
            asset_to_rhi: HashMap::new(),
            asset_manager,
            rhi: None,
            shared_buffers: HashMap::new(),
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

    fn asset_manager(&self) -> Ref<AssetManager> {
        self.asset_manager.borrow()
    }

    pub fn resource_iterator<T: RHIResource + 'static>(&self) -> Option<impl Iterator<Item = &T>> {
        self.resources.get_iter()
    }

    pub fn index(&self, uuid: usize) -> Option<usize> {
        self.resources.index(uuid)
    }

    pub fn request_from_shared_buffer<T: BufferContents>(
        &mut self,
        num: usize,
    ) -> Option<Subbuffer<[T]>> {
        let buffer = self.shared_buffers.get_mut(&TypeId::of::<T>())?;
        let bytes_required = (num * size_of::<T>()) as DeviceSize;
        let new_index = buffer.next_index + bytes_required;
        if buffer.buffer.size() < new_index {
            None
        } else {
            let result = buffer
                .buffer
                .clone()
                .slice(buffer.next_index..new_index)
                .reinterpret::<[T]>();
            buffer.next_index = new_index;
            Some(result)
        }
    }

    pub fn allocate_shared_buffer<T: BufferContents>(&mut self, num: usize, usage: BufferUsage) {
        self.shared_buffers.insert(
            TypeId::of::<T>(),
            SharedBuffer::new(
                (size_of::<T>() * num) as DeviceSize,
                self.rhi
                    .as_ref()
                    .unwrap()
                    .upgrade()
                    .unwrap()
                    .buffer_allocator
                    .clone(),
                usage,
            ),
        );
    }

    pub fn shared_buffer<T: BufferContents>(&self) -> Option<&Subbuffer<[T]>> {
        Some(
            self.shared_buffers
                .get(&TypeId::of::<T>())?
                .buffer
                .reinterpret_ref::<[T]>(),
        )
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

impl SharedBuffer {
    fn new(size: DeviceSize, allocator: Arc<dyn MemoryAllocator>, usage: BufferUsage) -> Self {
        let buffer = Buffer::new_slice(
            allocator,
            BufferCreateInfo {
                usage,
                ..BufferCreateInfo::default()
            },
            AllocationCreateInfo::default(),
            size,
        )
        .unwrap();
        Self {
            buffer,
            next_index: 0,
        }
    }
}
