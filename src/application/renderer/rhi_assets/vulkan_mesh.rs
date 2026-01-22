use crate::application::{
    assets::asset_traits::{Vertex, Index, MeshInterface},
    renderer::{buffer::buffer_from_slice, command_buffer::CommandBufferInterface},
};
use std::sync::Arc;
use vulkano::device::Queue;
use vulkano::{
    buffer::{BufferUsage, Subbuffer},
    device::Device,
    memory::allocator::MemoryTypeFilter,
};
use crate::application::assets::asset_traits::{RHIInterface, RHIMeshInterface, RHIResource};
use crate::application::renderer::Renderer;
use crate::application::resource_management::Resource;

pub struct VKMesh {
    vertex_buffer: Subbuffer<[Vertex]>,
    index_buffer: Subbuffer<[Index]>,
    uuid: usize,
}

impl VKMesh {
    fn new<Mesh: MeshInterface>(
        mesh: &Mesh,
        device: &Arc<Device>,
        command_buffer_interface: &CommandBufferInterface,
        queue: &Arc<Queue>,
    ) -> Self {
        let vertex_buffer = buffer_from_slice(
            device.clone(),
            command_buffer_interface,
            queue.clone(),
            mesh.vertices(),
            BufferUsage::VERTEX_BUFFER,
            MemoryTypeFilter::PREFER_DEVICE,
        )
        .unwrap();
        let index_buffer = buffer_from_slice(
            device.clone(),
            command_buffer_interface,
            queue.clone(),
            mesh.indices(),
            BufferUsage::INDEX_BUFFER,
            MemoryTypeFilter::PREFER_DEVICE,
        )
        .unwrap();

        Self {
            vertex_buffer,
            index_buffer,
            uuid: 0,
        }
    }
}

impl RHIResource for VKMesh {
    fn uuid_mut(&mut self) -> &mut usize {
        &mut self.uuid
    }
}

impl RHIMeshInterface for VKMesh {
    type RHI = Renderer;

    fn create<T: MeshInterface>(source: &T, rhi: &Self::RHI) -> Self {
        Self::new(source, &rhi.device, &rhi.command_buffer_interface, &rhi.queues.graphics_queue)
    }
}
