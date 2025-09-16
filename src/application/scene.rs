use egui_winit_vulkano::egui::Ui;
use model::Model;
use crate::application::assets::asset_traits::SceneInterface;
use crate::application::scene::camera::Camera;

mod model;
mod transform;
mod camera;

pub struct Scene {
    pub models: Vec<Model>,
    pub camera: Camera
}

impl Scene {
    pub fn new() -> Self {
        Self { models: vec![], camera: Camera::default() }
    }

    pub fn draw_gui(&mut self, gui: &mut Ui) {
        self.models.iter_mut().for_each(|model| {
            model.draw_gui(gui);
        })
    }
}

impl SceneInterface for Scene {
    type ModelType = Model;
    type CameraType = Camera;

    fn models(&self) -> &Vec<Self::ModelType> {
        &self.models
    }

    fn camera(&self) -> &Self::CameraType {
        &self.camera
    }
}
