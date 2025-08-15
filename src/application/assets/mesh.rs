use glam::{Vec2, Vec3};
use vulkano::buffer::BufferContents;
use vulkano::pipeline::graphics::vertex_input;
use super::{AssetMetadata, Asset};

#[derive(BufferContents, Copy, Clone, vertex_input::Vertex)]
#[repr(C)]
pub struct Vertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: Vec3,
    #[format(R32G32B32_SFLOAT)]
    pub normal: Vec3,
    #[format(R32G32B32_SFLOAT)]
    pub tangent: Vec3,
    #[format(R32G32_SFLOAT)]
    pub texture_coordinates: Vec2,
}

#[derive(BufferContents, Copy, Clone)]
#[repr(C)]
pub struct Index {
    pub index: u32,
}

pub struct Mesh {
    asset_metadata: AssetMetadata,

    vertices: Vec<Vertex>,
    indices: Vec<Index>,
}

impl Asset for Mesh {
    fn asset_metadata(&self) -> &AssetMetadata {
        &self.asset_metadata
    }
}

pub trait MeshInterface {
    fn vertices(&self) -> &[Vertex];
    fn indices(&self) -> &[Index];
}

impl MeshInterface for Mesh {
    fn vertices(&self) -> &[Vertex] {
        self.vertices.as_ref()
    }

    fn indices(&self) -> &[Index] {
        self.indices.as_ref()
    }
}