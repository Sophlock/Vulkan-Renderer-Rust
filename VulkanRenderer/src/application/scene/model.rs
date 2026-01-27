use super::transform::Transform;
use crate::application::assets::asset_traits::ModelInterface;
use crate::application::assets::{material::Material, mesh::Mesh};
use egui_winit_vulkano::egui::Ui;
use AssetSystem::Asset;
use AssetSystem::assets::{AssetHandle, AssetMetadata};
use crate::application::assets::material_instance::MaterialInstance;

#[derive(Asset)]
pub struct Model {
    pub transform: Transform,
    pub mesh: AssetHandle<Mesh>,
    pub material: AssetHandle<MaterialInstance>,
    asset_metadata: AssetMetadata
}

impl Model {
    pub fn draw_gui(&mut self, ui: &mut Ui) {}
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
