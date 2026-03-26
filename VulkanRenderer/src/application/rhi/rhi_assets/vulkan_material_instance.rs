use std::{cell::RefCell, ops::Deref, sync::Arc};

use asset_system::resource_management::Resource;
use vulkano::{
    descriptor_set::{DescriptorSet, allocator::DescriptorSetAllocator},
    memory::allocator::MemoryAllocator,
};

use crate::application::{
    assets::asset_traits::{MaterialInstanceInterface, RHIMaterialInstanceInterface, RHIResource},
    rhi::{
        VKRHI,
        rhi_assets::{RHIHandle, RHIResourceManager, vulkan_material::VKMaterial},
        shader_cursor::ShaderCursor,
        shader_object::ShaderObjectQueue,
    },
};

pub struct VKMaterialInstance {
    //shader_object: Arc<ShaderObject>,
    material: RHIHandle<VKMaterial>,
    uuid: usize,
}

impl VKMaterialInstance {
    fn new(
        material: RHIHandle<VKMaterial>,
        descriptor_allocator: &Arc<dyn DescriptorSetAllocator>,
        buffer_allocator: &Arc<dyn MemoryAllocator>,
        in_flight_frames: usize,
        resource_manager: &RHIResourceManager,
        update_queue: Arc<RefCell<ShaderObjectQueue>>,
    ) -> Self {
        /*let shader_object = ShaderObject::new(
            material
                .get(resource_manager)
                .unwrap()
                .shader_object_layout()
                .clone(),
            descriptor_allocator,
            buffer_allocator,
            in_flight_frames as u32,
            update_queue
        );*/
        Self {
            //shader_object,
            material,
            uuid: 0,
        }
    }

    pub fn material(&self) -> RHIHandle<VKMaterial> {
        self.material.clone()
    }

    pub fn shader_cursor(&self) -> ShaderCursor {
        //ShaderCursor::new(self.shader_object.clone())
        unimplemented!()
    }

    pub fn descriptor_sets(&self) -> &[Arc<DescriptorSet>] {
        //self.shader_object.descriptor_sets()
        unimplemented!()
    }
}

impl RHIResource for VKMaterialInstance {
    fn uuid_mut(&mut self) -> &mut usize {
        &mut self.uuid
    }
}

impl Resource for VKMaterialInstance {
    fn set_uuid(&mut self, uuid: usize) {
        self.uuid = uuid;
    }
}

impl RHIMaterialInstanceInterface for VKMaterialInstance {
    type RHI = VKRHI;

    fn create<T: MaterialInstanceInterface>(
        source: &T,
        rhi: &Self::RHI,
        resource_manager: &mut RHIResourceManager,
    ) -> Self {
        VKMaterialInstance::new(
            resource_manager.create_material(source.material()),
            &rhi.descriptor_allocator,
            &rhi.buffer_allocator,
            rhi.frames_in_flight,
            resource_manager.deref(),
            rhi.shader_object_update_queue().clone(),
        )
    }
}
