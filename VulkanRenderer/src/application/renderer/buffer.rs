use super::command_buffer::CommandBufferInterface;
use std::sync::Arc;
use vulkano::{
    command_buffer::{CopyBufferInfoTyped, PrimaryCommandBufferAbstract},
    device::Queue,
    sync::GpuFuture,
    buffer::{
        Buffer,
        BufferCreateInfo,
        Subbuffer,
        AllocateBufferError,
        BufferContents,
        BufferUsage
    },
    device::Device,
    memory::{
        allocator::{
            MemoryTypeFilter,
            AllocationCreateInfo,
            StandardMemoryAllocator
        }
    },
    sync::Sharing,
    DeviceSize,
    Validated,
    VulkanError
};

pub fn buffer_from_slice<T: BufferContents + Copy>(
    device: Arc<Device>,
    command_buffer_interface: &CommandBufferInterface,
    queue: Arc<Queue>,
    data: &[T],
    usage: BufferUsage,
    memory_type_filter: MemoryTypeFilter,
) -> Result<Subbuffer<[T]>, Validated<AllocateBufferError>> {
    let create_info = BufferCreateInfo {
        sharing: Sharing::Exclusive,
        size: 0,
        usage: usage | BufferUsage::TRANSFER_DST,
        ..BufferCreateInfo::default()
    };
    let alloc = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let alloc_info = AllocationCreateInfo {
        memory_type_filter,
        ..AllocationCreateInfo::default()
    };
    let staging_buffer = Buffer::from_iter(
        alloc.clone(),
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
    let buffer = Buffer::new_slice::<T>(alloc, create_info, alloc_info, data.len() as DeviceSize)?;
    copy_buffer_to_buffer(staging_buffer, buffer.clone(), command_buffer_interface, queue).unwrap();
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
