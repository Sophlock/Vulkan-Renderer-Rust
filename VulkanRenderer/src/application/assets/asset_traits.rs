use std::cell::{Ref, RefMut};

use asset_system::{
    assets::{Asset, AssetHandle},
    resource_management::{Resource, ResourceManager},
};
use glam::{Mat4, Vec2, Vec3};
use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input};

use crate::application::{
    renderer::rhi_assets::{RHIHandle, RHIResourceManager},
    scene::transform::Transform,
};

#[derive(BufferContents, Copy, Clone, vertex_input::Vertex)]
#[repr(C)]
pub struct Vertex {
    #[name("input.position")]
    #[format(R32G32B32_SFLOAT)]
    pub position: Vec3,
    #[name("input.normal")]
    #[format(R32G32B32_SFLOAT)]
    pub normal: Vec3,
    #[name("input.tangent")]
    #[format(R32G32B32_SFLOAT)]
    pub tangent: Vec3,
    #[name("input.textureCoordinate")]
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
    type MaterialType: RHIMaterialInterface;
    type MaterialInstanceType: RHIMaterialInstanceInterface;
    type CameraType: RHICameraInterface;
    type ModelType: RHIModelInterface;
    type SceneType: RHISceneInterface;

    fn resource_manager(&self) -> Ref<RHIResourceManager>;
    fn resource_manager_mut(&self) -> RefMut<RHIResourceManager>;
}

pub trait RHIResource: Resource {
    fn uuid_mut(&mut self) -> &mut usize;

    fn set_uuid(&mut self, uuid: usize) {
        *self.uuid_mut() = uuid;
    }
}

pub trait MeshInterface: Asset {
    fn vertices(&self) -> &[Vertex];
    fn indices(&self) -> &[Index];
    /*fn rhi<RHIType: RHIMeshInterface>(&self, rhi: &RHIType::RHI) -> RHIType {
        RHIType::create(self, rhi)
    }*/
}

pub trait RHIMeshInterface: RHIResource {
    type RHI: RHIInterface;
    fn create<T: MeshInterface>(
        source: &T,
        rhi: &Self::RHI,
        resource_manager: &mut RHIResourceManager,
    ) -> Self;
}

pub trait TextureInterface: Asset {
    fn pixels(&self) -> &[u8];

    fn size(&self) -> [u32; 3];

    /*fn rhi<RHIType: RHITextureInterface>(&self, rhi: &RHIType::RHI) -> RHIType {
        RHIType::create(self, rhi)
    }*/
}

pub trait RHITextureInterface: RHIResource {
    type RHI: RHIInterface;
    fn create<T: TextureInterface>(
        source: &T,
        rhi: &Self::RHI,
        resource_manager: &mut RHIResourceManager,
    ) -> Self;
}

pub trait ModelInterface: Asset {
    /*fn rhi<RHIType: RHIModelInterface>(&self, rhi: &RHIType::RHI) -> RHIType {
        RHIType::create(self, rhi)
    }*/

    type MeshType: MeshInterface + 'static;
    type MaterialType: MaterialInstanceInterface + 'static;

    fn transform(&self) -> Transform;
    fn mesh(&self) -> AssetHandle<Self::MeshType>;
    fn material(&self) -> AssetHandle<Self::MaterialType>;
}

pub trait RHIModelInterface: RHIResource {
    type RHI: RHIInterface;
    fn create<T: ModelInterface>(
        source: &T,
        rhi: &Self::RHI,
        resource_manager: &mut RHIResourceManager,
    ) -> Self;

    fn mesh(&self) -> RHIHandle<<<Self as RHIModelInterface>::RHI as RHIInterface>::MeshType>;
    fn material(
        &self,
    ) -> RHIHandle<<<Self as RHIModelInterface>::RHI as RHIInterface>::MaterialInstanceType>;

    fn transform(&self) -> Mat4;
}

pub trait CameraInterface: Sized {
    fn view_projection(&self) -> Mat4;
    fn transform(&self) -> Transform;
    fn rhi<RHIType: RHICameraInterface>(&self, rhi: &RHIType::RHI) -> RHIType {
        RHIType::create(self, rhi)
    }
}

pub trait RHICameraInterface {
    type RHI: RHIInterface;
    fn create<T: CameraInterface>(source: &T, rhi: &Self::RHI) -> Self;

    fn view_projection(&self) -> Mat4;
    fn location(&self) -> Vec3;
}

pub trait SceneInterface: Sized {
    type ModelType: ModelInterface;
    type CameraType: CameraInterface;
    fn models(&self) -> &Vec<Self::ModelType>;
    fn camera(&self) -> &Self::CameraType;
    /*fn rhi<RHIType: RHISceneInterface>(&self, rhi: &RHIType::RHI) -> RHIType {
        RHIType::create(self, rhi)
    }*/
}

pub trait RHISceneInterface {
    type RHI: RHIInterface;
    fn create<T: SceneInterface>(
        source: &T,
        rhi: &Self::RHI,
        resource_manager: &mut RHIResourceManager,
    ) -> Self;
    fn models(&self) -> &[<<Self as RHISceneInterface>::RHI as RHIInterface>::ModelType];
    fn camera(&self) -> &<<Self as RHISceneInterface>::RHI as RHIInterface>::CameraType;
}

pub trait MaterialInterface: Asset {
    fn module(&self) -> &str;
    fn material(&self) -> &str;
    /*fn rhi<RHIType: RHIMaterialInterface>(&self, rhi: &RHIType::RHI) -> RHIType {
        RHIType::create(self, rhi)
    }*/
}

pub trait RHIMaterialInterface: RHIResource {
    type RHI: RHIInterface;
    fn create<T: MaterialInterface>(
        source: &T,
        rhi: &Self::RHI,
        resource_manager: &mut RHIResourceManager,
    ) -> Self;
}

pub trait MaterialInstanceInterface: Asset {
    type MaterialType: MaterialInterface + 'static;

    /*fn rhi<RHIType: RHIMaterialInstanceInterface>(&self, rhi: &RHIType::RHI) -> RHIType {
        RHIType::create(self, rhi)
    }*/

    fn material(&self) -> AssetHandle<Self::MaterialType>;
}

pub trait RHIMaterialInstanceInterface: RHIResource {
    type RHI: RHIInterface;
    fn create<T: MaterialInstanceInterface>(
        source: &T,
        rhi: &Self::RHI,
        resource_manager: &mut RHIResourceManager,
    ) -> Self;
}
