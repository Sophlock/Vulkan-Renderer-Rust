use super::transform::Transform;
use crate::application::assets::{mesh::Mesh, AssetHandle};
use egui_winit_vulkano::egui::Ui;

pub struct Model {
    pub transform: Transform,
    pub mesh: AssetHandle<Mesh>
}

impl Model {
    pub fn draw_gui(&mut self, ui: &mut Ui) {

    }
}