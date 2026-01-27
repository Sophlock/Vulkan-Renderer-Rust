use std::ops::{Deref, DerefMut};
use crate::application::renderer::rhi_assets::vulkan_material::VKMaterial;
use crate::application::renderer::rhi_assets::{RHIHandle, RHIResourceManager};
use crate::application::renderer::shader_object::ShaderObject;
use std::sync::Arc;
use vulkano::descriptor_set::allocator::DescriptorSetAllocator;
use vulkano::descriptor_set::DescriptorSet;
use vulkano::memory::allocator::MemoryAllocator;
use AssetSystem::resource_management::Resource;
use crate::application::assets::asset_traits::{MaterialInstanceInterface, RHIInterface, RHIMaterialInstanceInterface, RHIResource};
use crate::application::renderer::Renderer;
use crate::application::renderer::shader_cursor::ShaderCursor;

pub struct VKMaterialInstance {
    shader_object: ShaderObject,
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
    ) -> Self {
        let shader_object = ShaderObject::new(
            material
                .get(resource_manager)
                .unwrap()
                .shader_object_layout()
                .clone(),
            descriptor_allocator,
            buffer_allocator,
            in_flight_frames as u32,
        );
        Self {
            shader_object,
            material,
            uuid: 0,
        }
    }

    pub fn material(&self) -> RHIHandle<VKMaterial> {
        self.material.clone()
    }

    pub fn shader_cursor(&mut self) -> ShaderCursor {
        ShaderCursor::new(&mut self.shader_object)
    }
    
    pub fn descriptor_sets(&self) -> &[Arc<DescriptorSet>] {
        self.shader_object.descriptor_sets()
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
    type RHI = Renderer;

    fn create<T: MaterialInstanceInterface>(source: &T, rhi: &Self::RHI) -> Self {
        let mut resource_manager = rhi.resource_manager_mut();
        VKMaterialInstance::new(
            resource_manager.create_material(source.material()),
            &rhi.descriptor_allocator,
            &rhi.buffer_allocator,
            rhi.frames_in_flight,
            resource_manager.deref(),
        )
    }
}
