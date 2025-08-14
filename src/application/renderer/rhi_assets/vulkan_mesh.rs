use crate::application::{
    assets::mesh::{Index, MeshInterface, Vertex},
    renderer::{buffer::buffer_from_slice, command_buffer::CommandBufferInterface},
};
use std::sync::Arc;
use vulkano::device::Queue;
use vulkano::{
    buffer::{BufferUsage, Subbuffer},
    device::Device,
    memory::allocator::MemoryTypeFilter,
};

struct VKMesh {
    vertex_buffer: Subbuffer<[Vertex]>,
    index_buffer: Subbuffer<[Index]>,
}

impl VKMesh {
    pub fn new<Mesh: MeshInterface>(
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
        }
    }
}
