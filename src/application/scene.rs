use egui_winit_vulkano::egui::Ui;
use model::Model;

mod model;
mod transform;

pub struct Scene {
    pub models: Vec<Model>,
}

impl Scene {
    pub fn new() -> Self {
        Self { models: vec![] }
    }

    pub fn draw_gui(&mut self, gui: &mut Ui) {
        self.models.iter_mut().for_each(|model| {
            model.draw_gui(gui);
        })
    }
}
