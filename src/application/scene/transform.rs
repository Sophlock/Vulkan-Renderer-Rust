use egui_winit_vulkano::egui;
use egui_winit_vulkano::egui::Ui;
use glam::{Mat4, Quat, Vec3};

pub struct Transform {
    pub location: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            location: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.location)
    }
}

pub fn draw_transform_to_gui(ui: &mut Ui, transform: &mut Transform) {
    draw_vec3_to_gui(ui, &mut transform.location);
    draw_rotation_to_gui(ui, &mut transform.rotation);
    draw_vec3_to_gui(ui, &mut transform.scale);
}

pub fn draw_vec3_to_gui(ui: &mut Ui, vec: &mut Vec3) {
    ui.horizontal(|ui| {
        ui.add(egui::DragValue::new(&mut vec.x));
        ui.add(egui::DragValue::new(&mut vec.y));
        ui.add(egui::DragValue::new(&mut vec.z));
    });
}

pub fn draw_rotation_to_gui(ui: &mut Ui, rot: &mut Quat) {
    ui.horizontal(|ui| {
        let (mut yaw, mut pitch, mut roll) = rot.to_euler(glam::EulerRot::XYZ);
       ui.drag_angle_tau(&mut yaw);
        ui.drag_angle(&mut pitch);
        ui.drag_angle_tau(&mut roll);
        rot.clone_from(&Quat::from_euler(glam::EulerRot::XYZ, yaw, pitch, roll));
    });
}
