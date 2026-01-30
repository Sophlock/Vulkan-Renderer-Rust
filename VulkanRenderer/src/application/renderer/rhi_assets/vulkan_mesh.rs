use std::sync::Arc;

use asset_system::resource_management::Resource;
use vulkano::{
    buffer::{BufferUsage, Subbuffer},
    device::Queue,
    memory::allocator::{MemoryAllocator, MemoryTypeFilter},
};

use crate::application::{
    assets::asset_traits::{Index, MeshInterface, RHIMeshInterface, RHIResource, Vertex},
    renderer::{
        Renderer, buffer::buffer_from_slice, command_buffer::CommandBufferInterface,
        rhi_assets::RHIResourceManager,
    },
};

pub struct VKMesh {
    vertex_buffer: Subbuffer<[Vertex]>,
    index_buffer: Subbuffer<[Index]>,
    uuid: usize,
}

impl VKMesh {
    fn new<Mesh: MeshInterface>(
        mesh: &Mesh,
        allocator: &Arc<dyn MemoryAllocator>,
        command_buffer_interface: &CommandBufferInterface,
        queue: &Arc<Queue>,
    ) -> Self {
        let vertex_buffer = buffer_from_slice(
            allocator.clone(),
            command_buffer_interface,
            queue.clone(),
            mesh.vertices(),
            BufferUsage::VERTEX_BUFFER,
            MemoryTypeFilter::PREFER_DEVICE,
        )
        .unwrap();
        let index_buffer = buffer_from_slice(
            allocator.clone(),
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

    pub fn vertex(&self) -> &Subbuffer<[Vertex]> {
        &self.vertex_buffer
    }

    pub fn index(&self) -> &Subbuffer<[Index]> {
        &self.index_buffer
    }
}

impl Resource for VKMesh {
    fn set_uuid(&mut self, uuid: usize) {
        self.uuid = uuid;
    }
}

impl RHIResource for VKMesh {
    fn uuid_mut(&mut self) -> &mut usize {
        &mut self.uuid
    }
}

impl RHIMeshInterface for VKMesh {
    type RHI = Renderer;

    fn create<T: MeshInterface>(
        source: &T,
        rhi: &Self::RHI,
        _: &mut RHIResourceManager,
    ) -> Self {
        Self::new(
            source,
            &rhi.buffer_allocator,
            &rhi.command_buffer_interface,
            &rhi.queues.graphics_queue,
        )
    }
}
