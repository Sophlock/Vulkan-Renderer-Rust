mod assets;
mod input;
mod renderer;
mod rhi;
mod scene;

use std::{
    cell::RefCell,
    ops::DerefMut,
    rc::Rc,
    sync::Arc,
    time::{Duration, SystemTime},
};

use egui_winit_vulkano::{
    egui,
    egui::{Color32, Frame},
};
use gilrs::Gilrs;
use glam::{EulerRot, Quat, Vec3};
use rhi::VKRHI;
use vulkano::buffer::BufferUsage;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::ActiveEventLoop,
    window::WindowId,
};
use winit_input_map::{InputCode, InputMap, input_map};

use crate::{
    AppEvent,
    application::{
        assets::{
            AssetManager::AssetManager,
            asset_traits::{Index, RHIInterface, RHISceneInterface, RendererInterface, Vertex},
            material::Material,
        },
        input::InputAction,
        renderer::VKRenderer,
        rhi::rhi_assets::vulkan_scene::VKScene,
        scene::{Scene, transform::Transform},
    },
};

pub struct Application {
    renderer: Option<Rc<VKRenderer>>,
    asset_manager: Arc<RefCell<AssetManager>>,
    rhi_scene_proxy: Option<VKScene>,
    scene: Scene,
    input: InputMap<InputAction>,
    time_measurement: TimeMeasureSystem,
    gilrs: Gilrs,
}

impl Application {
    pub fn new() -> Self {
        let asset_manager = AssetManager::new();
        let scene = Self::scene(asset_manager.borrow_mut().deref_mut());
        Self {
            renderer: None,
            rhi_scene_proxy: None,
            asset_manager,
            scene,
            input: Self::build_input_map(),
            time_measurement: TimeMeasureSystem::new(),
            gilrs: Gilrs::new().unwrap(),
        }
    }

    fn scene(asset_manager: &mut AssetManager) -> Scene {
        let num_materials = 1;
        let num_instances = 200000;

        let mut scene = Scene::new();
        let mesh = asset_manager.add_mesh("TestMesh", "resources/assets/meshes/sphere.glb");

        use rand::prelude::*;

        // Get an RNG:
        let mut rng = rand::rng();
        let bounds = -200f32..200f32;

        for i in 0..num_materials {
            let material = asset_manager.add_material(
                format!("TestMat_{}", i).as_str(),
                "Materials/basicMaterials",
                "SingleColorUnlitMaterial",
            );
            let material_instance = asset_manager
                .add_material_instance(format!("TestMatInst_{}", i).as_str(), material);

            for j in 0..num_instances {
                let transform = Transform {
                    location: Vec3::new(
                        rng.random_range(bounds.clone()),
                        rng.random_range(bounds.clone()),
                        rng.random_range(bounds.clone()),
                    ),
                    ..Transform::default()
                };
                scene.models.push(asset_manager.add_model(
                    format!("TestModel_{}_{}", i, j).as_str(),
                    transform,
                    mesh.clone(),
                    material_instance.clone(),
                ));
            }
        }
        scene.camera.transform.location = Vec3::new(0., 0., 2.);
        scene
    }

    fn build_input_map() -> InputMap<InputAction> {
        {
            use InputAction::*;
            use winit_input_map::base_input_codes::*;
            input_map!(
                (LeftClick, MouseButton::Left),
                (RightClick, MouseButton::Right),
                (ScrollClick, MouseButton::Middle),
                (ScrollUp, MouseScrollUp),
                (ScrollDown, MouseScrollDown),
                (MouseRight, MouseMoveRight),
                (MouseLeft, MouseMoveLeft),
                (MouseUp, MouseMoveUp),
                (MouseDown, MouseMoveDown),
                (Forward, KeyW),
                (Back, KeyS),
                (Left, KeyA),
                (Right, KeyD),
                (Down, KeyQ),
                (Up, KeyE),
                (BothClick, [MouseButton::Left, MouseButton::Right])
            )
        }
    }

    fn update_scene_proxy(&mut self, rhi: &VKRHI) {
        // TODO: This should be more lazy
        self.rhi_scene_proxy = Some(VKScene::create(
            &self.scene,
            rhi,
            rhi.resource_manager_mut().deref_mut(),
        ));
    }

    fn update_aspect_ratio(&mut self, x: u32, y: u32) {
        self.scene.camera.aspect = x as f32 / y as f32;
    }

    fn tick(&mut self, delta_time: f32) {
        use InputAction::*;

        let mouse_move = self.input.dir(MouseLeft, MouseRight, MouseUp, MouseDown);
        let cam_move = self.input.dir(Forward, Back, Right, Left);
        let scroll = self.input.axis(ScrollUp, ScrollDown);

        let mut cam_euler = self.scene.camera.transform.rotation.to_euler(EulerRot::YXZ);
        cam_euler.2 = 0.;

        let left = self.input.pressing(LeftClick);
        let right = self.input.pressing(RightClick);
        let both = self.input.pressing(BothClick);

        let cam = &mut self.scene.camera;

        if both {
            let right = cam.transform.right();
            let up = cam.transform.up();
            cam.transform.location -=
                (right * mouse_move.0 + up * mouse_move.1) * cam.speed * delta_time;
        } else if left {
            let forward = cam.transform.forward();
            let forward_proj = forward.with_y(0.).normalize();

            cam.transform.location += forward_proj * mouse_move.1 * cam.speed * delta_time;
            cam_euler.0 += mouse_move.0 * cam.rot_speed * delta_time;
        } else if right {
            cam_euler.0 += mouse_move.0 * cam.rot_speed * delta_time;
            cam_euler.1 -= mouse_move.1 * cam.rot_speed * delta_time;
        }

        if left || right {
            let forward = cam.transform.forward();
            let right = cam.transform.right();
            let vertical = self.input.axis(Up, Down);

            cam.transform.location += (forward * cam_move.0 + right * cam_move.1
                - Vec3::Y * vertical)
                * cam.speed
                * 0.2
                * delta_time;

            cam.speed += scroll * 1.5 * delta_time;
            cam.speed = cam.speed.max(0.);
        } else {
            cam.fov += scroll * cam.speed * 0.1 * delta_time;
        }

        cam_euler.1 = cam_euler.1.clamp(-1.5, 1.5);

        self.scene.camera.transform.rotation =
            Quat::from_euler(EulerRot::YXZ, cam_euler.0, cam_euler.1, cam_euler.2);

        self.input.init();

        self.draw_gui(delta_time);
    }

    fn draw_gui(&mut self, delta_time: f32) {
        let renderer = &self.renderer.as_ref().unwrap();
        let mut gui = renderer.rhi().gui_mut();

        gui.immediate_ui(|ui| {
            let ctx = ui.context();
            egui::CentralPanel::default()
                .frame(Frame::default().fill(Color32::TRANSPARENT))
                .show(&ctx, |ui| {
                    ui.heading("Render Statistics");
                    ui.label(format!("Frametime: {:.4}s", delta_time));
                    ui.label(format!("Framerate: {:.1}fps", 1f32 / delta_time));
                    ui.label(format!(
                        "Number of pipelines: {}",
                        self.asset_manager
                            .borrow()
                            .resource_manager()
                            .get_iter::<Material>()
                            .unwrap()
                            .count()
                    ));

                    ui.add_space(10f32);
                    ui.heading("Render Settings");
                    ui.label("Post Process:");
                    renderer.post_process_settings().draw_gui(ui);

                    //ui.text_edit_singleline(&mut name);
                    //ui.add(egui::Slider::new(&mut age, 0..=120).text("age"));
                    //if ui.button("Increment").clicked() {
                    // age += 1;
                    //}
                    //ui.label(format!("Hello '{name}', age {age}"));
                });
        });
    }

    fn mark_frame(&mut self, frame_type: &AppEvent) -> f32 {
        self.time_measurement.update(frame_type).as_secs_f32()
    }
}

impl ApplicationHandler<AppEvent> for Application {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let rhi = VKRHI::new(event_loop, self.asset_manager.clone());

        rhi.resource_manager_mut().allocate_shared_buffer::<Vertex>(
            1000000,
            BufferUsage::VERTEX_BUFFER
                | BufferUsage::TRANSFER_DST
                | BufferUsage::STORAGE_BUFFER
                | BufferUsage::SHADER_DEVICE_ADDRESS,
        );
        rhi.resource_manager_mut().allocate_shared_buffer::<Index>(
            1000000,
            BufferUsage::INDEX_BUFFER
                | BufferUsage::TRANSFER_DST
                | BufferUsage::STORAGE_BUFFER
                | BufferUsage::SHADER_DEVICE_ADDRESS,
        );
        self.update_scene_proxy(rhi.as_ref());

        self.renderer = Some(Rc::new(VKRenderer::new(rhi)));
        // TODO: This should be called on demand as well
        self.renderer.as_ref().unwrap().compile_materials();
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: AppEvent) {
        let delta_time = self.mark_frame(&event);
        match event {
            AppEvent::Tick => self.tick(delta_time),
            AppEvent::Render => self
                .renderer
                .as_ref()
                .unwrap()
                .rhi()
                .window()
                .request_redraw(),
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        self.input.update_with_window_event(&event);
        self.renderer
            .as_ref()
            .unwrap()
            .rhi()
            .gui_mut()
            .update(&event);
        match event {
            WindowEvent::Resized(size) => {
                self.update_aspect_ratio(size.width, size.height);
                self.renderer
                    .as_ref()
                    .unwrap()
                    .mutable_state()
                    .request_recreate_swapchain()
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                self.update_scene_proxy(self.renderer.clone().unwrap().rhi());
                self.renderer
                    .as_ref()
                    .unwrap()
                    .redraw(self.rhi_scene_proxy.as_ref().unwrap());
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        device_id: DeviceId,
        event: DeviceEvent,
    ) {
        self.input.update_with_device_event(device_id, &event);
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.input.update_with_gilrs(&mut self.gilrs);
    }

    fn exiting(&mut self, event_loop: &ActiveEventLoop) {
        self.renderer.as_ref().unwrap().rhi().shutdown();
    }
}

struct TimeMeasureSystem {
    last_times: [SystemTime; 2],
}

impl TimeMeasureSystem {
    pub fn new() -> Self {
        Self {
            last_times: [SystemTime::now(); 2],
        }
    }

    pub fn update(&mut self, frame_type: &AppEvent) -> Duration {
        let index: usize = match frame_type {
            AppEvent::Tick => 0,
            AppEvent::Render => 1,
        };
        let last = self.last_times[index];
        self.last_times[index] = SystemTime::now();
        self.last_times[index].duration_since(last).unwrap()
    }
}
