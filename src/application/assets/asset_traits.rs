use glam::{Vec2, Vec3};
use vulkano::buffer::BufferContents;
use vulkano::pipeline::graphics::vertex_input;

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

pub trait RHIInterface {
    type MeshType: RHIMeshInterface;
    type TextureType: RHITextureInterface;
}

pub trait MeshInterface : Sized {
    fn vertices(&self) -> &[Vertex];
    fn indices(&self) -> &[Index];
    fn rhi<RHIType: RHIMeshInterface>(&self, rhi: &RHIType::RHI) -> RHIType {
        RHIType::create(self, rhi)
    }
}

pub trait RHIMeshInterface {
    type RHI: RHIInterface;
    fn create<T: MeshInterface>(source: &T, rhi: &Self::RHI) -> Self;
}

pub trait TextureInterface : Sized {
    fn pixels(&self) -> &[u32];

    fn rhi<RHIType: RHITextureInterface>(&self, rhi: &RHIType::RHI) -> RHIType {
        RHIType::create(self, rhi)
    }
}

pub trait RHITextureInterface {
    type RHI: RHIInterface;
    fn create<T: TextureInterface>(source: &T, rhi: &Self::RHI) -> Self;
}

pub trait ModelInterface : Sized {
    fn rhi<RHIType: RHIModelInterface>(&self, rhi: &RHIType::RHI) -> RHIType {
        RHIType::create(self, rhi)
    }
}

pub trait RHIModelInterface {
    type RHI: RHIInterface;
    fn create<T: ModelInterface>(source: &T, rhi: &Self::RHI) -> Self;
}

pub trait SceneInterface : Sized {
    type ModelType: ModelInterface;
    fn models(&self) -> &Vec<Self::ModelType>;
    fn rhi<RHIType: RHISceneInterface>(&self, rhi: &RHIType::RHI) -> RHIType {
        RHIType::create(self, rhi)
    }
}

pub trait RHISceneInterface {
    type RHI: RHIInterface;
    fn create<T: SceneInterface>(source: &T, rhi: &Self::RHI) -> Self;
}