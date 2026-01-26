use crate::application::assets::asset_traits::{Index, MeshInterface, Vertex};
use std::path::Path;
use AssetSystem::assets::AssetMetadata;
use AssetSystem::Asset;

#[derive(Asset)]
pub struct Mesh {
    asset_metadata: AssetMetadata,

    vertices: Vec<Vertex>,
    indices: Vec<Index>,
}

impl Mesh {
    pub fn new(path: impl AsRef<Path>, asset_metadata: AssetMetadata) -> Self {
        let (doc, buffers, _) = gltf::import(path).unwrap();
        let mesh = doc.meshes().next().unwrap();
        let prim = mesh.primitives().next().unwrap();
        let read = prim.reader(|buffer| Some(&buffers[buffer.index()]));

        let pos = read.read_positions().unwrap();
        let normal = read.read_normals().unwrap();
        let tangent = read.read_tangents().unwrap();
        let uv = read.read_tex_coords(0).unwrap();

        let vertices = pos.zip(normal).zip(tangent).zip(uv.into_f32())
            .map(|(((p, n), t), tex)| {
                Vertex{
                    position: p.into(),
                    normal: n.into(),
                    tangent: [t[0], t[1], t[2]].into(),
                    texture_coordinates: tex.into(),
                }
            }).collect();

        let indices = read.read_indices().unwrap().into_u32().map(|index| {Index{index}}).collect();

        Self {
            vertices,
            indices,
            asset_metadata
        }
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