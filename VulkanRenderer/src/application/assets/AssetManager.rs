use std::{cell::RefCell, marker::PhantomData, path::Path, sync::Arc};

use asset_system::{assets::AssetHandle, resource_management::ResourceManager};

use crate::application::{
    assets::{material::Material, material_instance::MaterialInstance, mesh::Mesh},
    scene::{model::Model, transform::Transform},
};

pub struct AssetManager {
    resource_manager: ResourceManager,
}

impl AssetManager {
    pub fn new() -> Arc<RefCell<AssetManager>> {
        Arc::new(RefCell::new(Self {
            resource_manager: ResourceManager::new(),
        }))
    }

    pub fn add_mesh(&mut self, name: &str, path: impl AsRef<Path>) -> AssetHandle<Mesh> {
        AssetHandle::<Mesh> {
            uuid: self.resource_manager.add(Mesh::new(name.into(), path)),
            _phantom: PhantomData,
        }
    }

    pub fn add_material(
        &mut self,
        name: &str,
        module: &str,
        material_type: &str,
    ) -> AssetHandle<Material> {
        AssetHandle::<Material> {
            uuid: self.resource_manager.add(Material::new(
                name.into(),
                module.into(),
                material_type.into(),
            )),
            _phantom: PhantomData,
        }
    }

    pub fn add_material_instance(
        &mut self,
        name: &str,
        material: AssetHandle<Material>,
    ) -> AssetHandle<MaterialInstance> {
        AssetHandle::<MaterialInstance> {
            uuid: self
                .resource_manager
                .add(MaterialInstance::new(name.into(), material)),
            _phantom: PhantomData,
        }
    }

    pub fn add_model(
        &mut self,
        name: &str,
        transform: Transform,
        mesh: AssetHandle<Mesh>,
        material: AssetHandle<MaterialInstance>,
    ) -> AssetHandle<Model> {
        AssetHandle::<Model> {
            uuid: self
                .resource_manager
                .add(Model::new(name.into(), transform, mesh, material)),
            _phantom: PhantomData,
        }
    }

    pub fn resource_manager(&self) -> &ResourceManager {
        &self.resource_manager
    }
}
