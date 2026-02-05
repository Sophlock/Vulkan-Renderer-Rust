use std::path::Path;

use asset_system::{Asset, assets::AssetMetadata};

use crate::application::assets::asset_traits::{Index, MeshInterface, Vertex};

#[derive(Asset)]
pub struct Mesh {
    asset_metadata: AssetMetadata,

    vertices: Vec<Vertex>,
    indices: Vec<Index>,
}

impl Mesh {
    pub fn new(name: String, path: impl AsRef<Path>) -> Self {
        let (doc, buffers, _) = gltf::import(path).unwrap();
        let mesh = doc.meshes().next().unwrap();
        let prim = mesh.primitives().next().unwrap();
        let read = prim.reader(|buffer| Some(&buffers[buffer.index()]));

        let pos = read.read_positions().unwrap();
        let normal = read.read_normals().unwrap();
        let tangent = read.read_tangents().unwrap();
        let uv = read.read_tex_coords(0).unwrap();

        let vertices = pos
            .zip(normal)
            .zip(tangent)
            .zip(uv.into_f32())
            .map(|(((p, n), t), tex)| Vertex {
                position: p,
                normal: n,
                tangent: *t.first_chunk().unwrap(),
                texture_coordinates: tex,
            })
            .collect();

        let indices = read
            .read_indices()
            .unwrap()
            .into_u32()
            .map(|index| Index { index })
            .collect();

        Self {
            vertices,
            indices,
            asset_metadata: AssetMetadata::new(name),
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
