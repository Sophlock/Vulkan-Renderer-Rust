use glam::{Mat4, Vec2, Vec3};
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
    type CameraType: RHICameraInterface;
    type ModelType: RHIModelInterface;
    type SceneType: RHISceneInterface;
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

pub trait CameraInterface : Sized {
    fn view_projection(&self) -> Mat4;
    fn rhi<RHIType: RHICameraInterface>(&self, rhi: &RHIType::RHI) -> RHIType {
        RHIType::create(self, rhi)
    }
}

pub trait RHICameraInterface {
    type RHI: RHIInterface;
    fn create<T: CameraInterface>(source: &T, rhi: &Self::RHI) -> Self;
}

pub trait SceneInterface : Sized {
    type ModelType: ModelInterface;
    type CameraType: CameraInterface;
    fn models(&self) -> &Vec<Self::ModelType>;
    fn camera(&self) -> &Self::CameraType;
    fn rhi<RHIType: RHISceneInterface>(&self, rhi: &RHIType::RHI) -> RHIType {
        RHIType::create(self, rhi)
    }
}

pub trait RHISceneInterface {
    type RHI: RHIInterface;
    fn create<T: SceneInterface>(source: &T, rhi: &Self::RHI) -> Self;
}

pub trait MaterialInterface : Sized {
    fn rhi<RHIType: RHIMaterialInterface>(&self, rhi: &RHIType::RHI) -> RHIType {
        RHIType::create(self, rhi)
    }
}

pub trait RHIMaterialInterface {
    type RHI: RHIInterface;
    fn create<T: MaterialInterface>(source: &T, rhi: &Self::RHI) -> Self;
}

pub trait MaterialInstanceInterface : Sized {
    fn rhi<RHIType: RHIMaterialInstanceInterface>(&self, rhi: &RHIType::RHI) -> RHIType {
        RHIType::create(self, rhi)
    }
}

pub trait RHIMaterialInstanceInterface {
    type RHI: RHIInterface;
    fn create<T: MaterialInstanceInterface>(source: &T, rhi: &Self::RHI) -> Self;
}