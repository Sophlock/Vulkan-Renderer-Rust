use glam::{Vec2, Vec3};
use vulkano::buffer::BufferContents;
use super::{AssetMetadata, Asset};

#[derive(BufferContents, Copy, Clone)]
#[repr(C)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub tangent: Vec3,
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