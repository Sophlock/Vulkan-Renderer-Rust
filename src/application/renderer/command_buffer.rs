use std::sync::Arc;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
};
use vulkano::{
    command_buffer::allocator::{
        StandardCommandBufferAllocator,
        StandardCommandBufferAllocatorCreateInfo,
    },
    device::Device,
};

pub struct CommandBufferInterface {
    allocator: Arc<StandardCommandBufferAllocator>,
}

impl CommandBufferInterface {
    pub fn new(device: Arc<Device>, image_count: usize) -> Self {
        let allocator = StandardCommandBufferAllocator::new(
            device,
            StandardCommandBufferAllocatorCreateInfo {
                primary_buffer_count: image_count,
                secondary_buffer_count: 0,
                ..StandardCommandBufferAllocatorCreateInfo::default()
            },
        )
        .into();
        Self { allocator }
    }

    pub fn primary_command_buffer(
        &self,
        queue_family_index: u32,
    ) -> AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> {
        AutoCommandBufferBuilder::primary(
            self.allocator.clone(),
            queue_family_index,
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap()
    }
}
