use asset_system::{
    assets::{AssetHandle, AssetMetadata},
    Asset,
};
use egui_winit_vulkano::egui::Ui;

use super::transform::Transform;
use crate::application::assets::{
    asset_traits::ModelInterface, material_instance::MaterialInstance, mesh::Mesh,
};

#[derive(Asset)]
pub struct Model {
    pub transform: Transform,
    pub mesh: AssetHandle<Mesh>,
    pub material: AssetHandle<MaterialInstance>,
    asset_metadata: AssetMetadata,
}

impl Model {
    pub fn draw_gui(&mut self, ui: &mut Ui) {}
}

impl Model {
    pub fn new(
        name: String,
        transform: Transform,
        mesh: AssetHandle<Mesh>,
        material: AssetHandle<MaterialInstance>,
    ) -> Self {
        Self {
            transform,
            mesh,
            material,
            asset_metadata: AssetMetadata::new(name),
        }
    }
}

impl ModelInterface for Model {
    type MeshType = Mesh;
    type MaterialType = MaterialInstance;

    fn transform(&self) -> Transform {
        self.transform.clone()
    }

    fn mesh(&self) -> AssetHandle<Mesh> {
        self.mesh.clone()
    }

    fn material(&self) -> AssetHandle<MaterialInstance> {
        self.material.clone()
    }
}
