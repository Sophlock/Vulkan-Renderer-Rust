use super::transform::Transform;
use crate::application::assets::{mesh::Mesh, AssetHandle};
use egui_winit_vulkano::egui::Ui;
use crate::application::assets::asset_traits::{MaterialInterface, MeshInterface, ModelInterface};
use crate::application::assets::material::Material;

pub struct Model {
    pub transform: Transform,
    pub mesh: AssetHandle<Mesh>,
    pub material: AssetHandle<Material>
}

impl Model {
    pub fn draw_gui(&mut self, ui: &mut Ui) {

    }
}

impl ModelInterface for Model {
    fn transform(&self) -> Transform {
        self.transform.clone()
    }

    fn mesh(&self) -> AssetHandle<impl MeshInterface> {
        self.mesh.clone()
    }

    fn material(&self) -> AssetHandle<impl MaterialInterface> {
        self.material.clone()
    }
}