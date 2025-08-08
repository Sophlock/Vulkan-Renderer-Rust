use std::sync::Arc;
use vulkano::{
    command_buffer::{
        CommandBufferLevel,
        allocator::{
            CommandBufferAlloc, CommandBufferAllocator, StandardCommandBufferAllocator,
            StandardCommandBufferAllocatorCreateInfo,
        }
    },
    device::Device
};

pub struct CommandBufferInterface {
    allocator: StandardCommandBufferAllocator,
    primary_command_buffer_allocation: CommandBufferAlloc,
}

impl CommandBufferInterface {
    pub fn new(device: Arc<Device>, image_count: usize, queue_family_index: u32) -> Self {
        let allocator = StandardCommandBufferAllocator::new(
            device,
            StandardCommandBufferAllocatorCreateInfo {
                primary_buffer_count: image_count,
                secondary_buffer_count: 0,
                ..StandardCommandBufferAllocatorCreateInfo::default()
            },
        );
        let primary_command_buffer_allocation = allocator
            .allocate(queue_family_index, CommandBufferLevel::Primary)
            .unwrap();

        Self {
            allocator,
            primary_command_buffer_allocation,
        }
    }
}
