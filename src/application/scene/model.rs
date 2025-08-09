use egui_winit_vulkano::egui::Ui;
use super::transform::Transform;

pub struct Model {
    pub transform: Transform,
}

impl Model {
    pub fn draw_gui(&mut self, ui: &mut Ui) {

    }
}