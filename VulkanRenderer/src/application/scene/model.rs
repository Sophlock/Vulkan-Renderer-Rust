use super::transform::Transform;
use crate::application::assets::asset_traits::{MaterialInterface, MeshInterface, ModelInterface};
use crate::application::assets::material::Material;
use crate::application::assets::{AssetHandle, mesh::Mesh};
use egui_winit_vulkano::egui::Ui;

pub struct Model {
    pub transform: Transform,
    pub mesh: AssetHandle<Mesh>,
    pub material: AssetHandle<Material>,
}

impl Model {
    pub fn draw_gui(&mut self, ui: &mut Ui) {}
}

impl ModelInterface for Model {
    type MeshType = Mesh;
    type MaterialType = Material;

    fn transform(&self) -> Transform {
        self.transform.clone()
    }

    fn mesh(&self) -> AssetHandle<Mesh> {
        self.mesh.clone()
    }

    fn material(&self) -> AssetHandle<Material> {
        self.material.clone()
    }
}
