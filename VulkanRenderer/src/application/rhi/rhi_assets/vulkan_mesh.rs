use std::sync::Arc;

use asset_system::resource_management::{Resource, ResourceManager};
use vulkano::{
    buffer::{BufferUsage, Subbuffer},
    device::Queue,
    memory::allocator::{MemoryAllocator, MemoryTypeFilter},
};

use crate::application::rhi::buffer::copy_slice_to_buffer_staged;
use crate::application::{
    assets::asset_traits::{Index, MeshInterface, RHIMeshInterface, RHIResource, Vertex},
    rhi::{
        VKRHI, buffer::buffer_from_slice, command_buffer::CommandBufferInterface,
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

    fn new_preallocated<Mesh: MeshInterface>(
        mesh: &Mesh,
        resource_manager: &mut RHIResourceManager,
        allocator: &Arc<dyn MemoryAllocator>,
        command_buffer_interface: &CommandBufferInterface,
        queue: &Arc<Queue>,
    ) -> Self {
        let vertex_buffer = resource_manager
            .request_from_shared_buffer(mesh.vertices().len())
            .unwrap();
        copy_slice_to_buffer_staged(
            mesh.vertices(),
            vertex_buffer.clone(),
            allocator.clone(),
            command_buffer_interface,
            queue.clone(),
        )
        .unwrap();
        let index_buffer = resource_manager
            .request_from_shared_buffer(mesh.indices().len())
            .unwrap();
        copy_slice_to_buffer_staged(
            mesh.indices(),
            index_buffer.clone(),
            allocator.clone(),
            command_buffer_interface,
            queue.clone(),
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

    pub fn index_offset(&self) -> usize {
        self.index_buffer.offset() as usize / size_of::<Index>()
    }
    pub fn index_size(&self) -> usize {
        self.index_buffer.len() as usize
    }
    pub fn vertex_offset(&self) -> usize {
        self.vertex_buffer.offset() as usize / size_of::<Vertex>()
    }
    pub fn vertex_size(&self) -> usize {
        self.vertex_buffer.len() as usize
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
    type RHI = VKRHI;

    fn create<T: MeshInterface>(
        source: &T,
        rhi: &Self::RHI,
        resource_manager: &mut RHIResourceManager,
    ) -> Self {
        /*Self::new(
            source,
            &rhi.buffer_allocator,
            &rhi.command_buffer_interface,
            &rhi.queues.graphics_queue,
        )*/
        Self::new_preallocated(
            source,
            resource_manager,
            &rhi.buffer_allocator,
            &rhi.command_buffer_interface,
            &rhi.queues.graphics_queue,
        )
    }
}
