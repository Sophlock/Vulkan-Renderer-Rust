use std::sync::Arc;

use vulkano::{
    DeviceSize, Validated, VulkanError,
    buffer::{
        AllocateBufferError, Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer,
    },
    command_buffer::{CopyBufferInfoTyped, PrimaryCommandBufferAbstract},
    device::Queue,
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
    sync::{GpuFuture, Sharing},
};

use super::command_buffer::CommandBufferInterface;

pub fn buffer_from_slice<T: BufferContents + Copy>(
    allocator: Arc<dyn MemoryAllocator>,
    command_buffer_interface: &CommandBufferInterface,
    queue: Arc<Queue>,
    data: &[T],
    usage: BufferUsage,
    memory_type_filter: MemoryTypeFilter,
) -> Result<Subbuffer<[T]>, Validated<AllocateBufferError>> {
    let create_info = BufferCreateInfo {
        sharing: Sharing::Exclusive,
        usage: usage | BufferUsage::TRANSFER_DST,
        ..BufferCreateInfo::default()
    };
    let alloc_info = AllocationCreateInfo {
        memory_type_filter,
        ..AllocationCreateInfo::default()
    };
    let staging_buffer = Buffer::from_iter(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..create_info.clone()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..alloc_info.clone()
        },
        data.iter().copied(),
    )?;
    let buffer =
        Buffer::new_slice::<T>(allocator, create_info, alloc_info, data.len() as DeviceSize)?;
    copy_buffer_to_buffer(
        staging_buffer,
        buffer.clone(),
        command_buffer_interface,
        queue,
    )
    .unwrap();
    Ok(buffer)
}

pub fn copy_buffer_to_buffer<T: BufferContents>(
    src_buffer: Subbuffer<[T]>,
    dst_buffer: Subbuffer<[T]>,
    command_buffer_interface: &CommandBufferInterface,
    queue: Arc<Queue>,
) -> Result<(), Validated<VulkanError>> {
    let mut cb = command_buffer_interface.primary_command_buffer(queue.queue_family_index());
    cb.copy_buffer(CopyBufferInfoTyped::buffers(src_buffer, dst_buffer))?;
    cb.build()?
        .execute(queue)
        .unwrap()
        .then_signal_fence_and_flush()?
        .wait(None)
}

pub fn copy_slice_to_buffer_staged<T: BufferContents + Copy>(
    src_slice: &[T],
    dst_buffer: Subbuffer<[T]>,
    allocator: Arc<dyn MemoryAllocator>,
    command_buffer_interface: &CommandBufferInterface,
    queue: Arc<Queue>,
) -> Result<(), Validated<VulkanError>> {
    let staging_buffer = Buffer::from_iter(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..BufferCreateInfo::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..AllocationCreateInfo::default()
        },
        src_slice.iter().copied(),
    )
    .unwrap();
    copy_buffer_to_buffer(staging_buffer, dst_buffer, command_buffer_interface, queue)
}
