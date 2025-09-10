use crate::application::assets::asset_traits::{Index, MeshInterface, Vertex};
use super::{Asset, AssetMetadata};

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

impl MeshInterface for Mesh {
    fn vertices(&self) -> &[Vertex] {
        self.vertices.as_ref()
    }

    fn indices(&self) -> &[Index] {
        self.indices.as_ref()
    }
}