mod assets;
mod input;
mod renderer;
mod rhi;
mod scene;

use std::{
    cell::RefCell,
    collections::BTreeMap,
    ops::DerefMut,
    rc::Rc,
    sync::Arc,
    time::{Duration, SystemTime},
};

use asset_system::assets::AssetHandle;
use egui_winit_vulkano::{
    egui,
    egui::{Color32, Sense, Stroke, StrokeKind, Ui, Vec2, epaint, epaint::PathShape},
};
use emath::{Pos2, Rect, RectTransform};
use enum_iterator::all;
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
            asset_traits::{
                CameraInterface, Index, RHIInterface, RHISceneInterface, RendererInterface, Vertex,
            },
            material::Material,
        },
        input::InputAction,
        renderer::{
            VKRenderer,
            profiling::{Profiler, ProfilerCategory},
        },
        rhi::rhi_assets::vulkan_scene::VKScene,
        scene::{Scene, transform::Transform},
    },
};

pub struct Application {
    renderer: Option<Rc<VKRenderer>>,
    asset_manager: Arc<RefCell<AssetManager>>,
    rhi_scene_proxy: Option<VKScene>,
    scene: Scene,
    fallback_material: AssetHandle<Material>,
    input: InputMap<InputAction>,
    time_measurement: TimeMeasureSystem,
    gilrs: Gilrs,
}

impl Application {
    pub fn new() -> Self {
        let asset_manager = AssetManager::new();
        let fallback_material = asset_manager.borrow_mut().add_material(
            "FallbackMaterial",
            "Materials/basicMaterials",
            "FallbackMaterial",
        );
        let scene = Self::scene(asset_manager.borrow_mut().deref_mut());
        Self {
            renderer: None,
            rhi_scene_proxy: None,
            asset_manager,
            scene,
            fallback_material,
            input: Self::build_input_map(),
            time_measurement: TimeMeasureSystem::new(),
            gilrs: Gilrs::new().unwrap(),
        }
    }

    fn scene(asset_manager: &mut AssetManager) -> Scene {
        let num_materials = 4000;
        let num_instances = 50;

        let mut scene = Scene::new();
        let sphere = asset_manager.add_mesh("TestMesh1", "resources/assets/meshes/sphere.glb");
        let meshes = [
            sphere.clone(),
            sphere.clone(),
            sphere,
            asset_manager.add_mesh("TestMesh2", "resources/assets/meshes/Suzanne.glb"),
        ];

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
                    rotation: Quat::from_euler(
                        EulerRot::XYZ,
                        rng.random_range(0f32..6.28f32),
                        rng.random_range(-3.14f32..3.14f32) * 0.5f32,
                        0f32,
                    ),
                    ..Transform::default()
                };
                let mesh = &meshes[rng.random::<u32>() as usize % meshes.len()];
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
        // TODO: This is super hacky
        rhi.resource_manager_mut()
            .create_material(self.fallback_material.clone());

        // TODO: This should be more lazy
        self.rhi_scene_proxy = Some(VKScene::create(
            &self.scene,
            rhi,
            rhi.resource_manager_mut().deref_mut(),
        ));
    }

    // TODO: This should just be update_scene_proxy but that one is not optimized
    fn update_scene_proxy_camera(&mut self, rhi: &VKRHI) {
        self.rhi_scene_proxy
            .as_mut()
            .unwrap()
            .set_camera(self.scene.camera.rhi(rhi));
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
    }

    fn draw_gui(&mut self) {
        let renderer = &self.renderer.as_ref().unwrap();
        let mut gui = renderer.rhi().gui_mut();

        gui.immediate_ui(|ui| {
            let ctx = ui.context();
            egui::Window::new("GUI")
                .resizable(true)
                .vscroll(true)
                .default_size([200.0, 500.0])
                .show(&ctx, |ui| {
                    ui.heading("Render Statistics");
                    ui.label(format!(
                        "Screen Resolution:\t {}x{}",
                        renderer.swapchain_extent()[0],
                        renderer.swapchain_extent()[1]
                    ));
                    ui.label(format!(
                        "Framerate: {:.1}fps",
                        1f32 / self
                            .time_measurement
                            .last_duration(&AppEvent::Render)
                            .unwrap_or(Duration::ZERO)
                            .as_secs_f32()
                    ));
                    ui.label(format!(
                        "Materials in Scene:\t {}",
                        self.asset_manager
                            .borrow()
                            .resource_manager()
                            .get_iter::<Material>()
                            .unwrap()
                            .count()
                    ));
                    ui.label(format!(
                        "Visible Materials:\t {}",
                        renderer.scene_statistics().visible_materials
                    ));
                    ui.label(format!(
                        "Drawn Materials:\t {}",
                        renderer.scene_statistics().drawn_materials
                    ));
                    ui.label(format!(
                        "Culled Materials:\t {}",
                        renderer.scene_statistics().culled_materials
                    ));
                    ui.label(format!(
                        "Fallback Pixels:\t {}",
                        renderer.scene_statistics().fallback_pixels
                    ));
                    ui.label(format!(
                        "Fallback Percentage:\t {:.2}%",
                        renderer.scene_statistics().fallback_pixels as f32 * 100f32
                            / (renderer.swapchain_extent()[0] * renderer.swapchain_extent()[1])
                                as f32
                    ));

                    self.time_measurement
                        .paint_graph_to_gui(&AppEvent::Render, ui);
                    self.time_measurement.paint_detail_graphs_to_gui(ui);

                    ui.add_space(10f32);
                    ui.heading("Render Settings");
                    ui.label("Post Process:");
                    renderer.post_process_settings().draw_gui(ui);
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
        self.time_measurement.reset();
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: AppEvent) {
        match event {
            AppEvent::Tick => {
                let delta_time = self.mark_frame(&event);
                self.tick(delta_time)
            }
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
        let input_consumed = self
            .renderer
            .as_ref()
            .unwrap()
            .rhi()
            .gui_mut()
            .update(&event);
        if !input_consumed {
            self.input.update_with_window_event(&event);
        }
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
                //self.update_scene_proxy(self.renderer.clone().unwrap().rhi());
                self.update_scene_proxy_camera(self.renderer.clone().unwrap().rhi());
                self.draw_gui();
                self.renderer
                    .as_ref()
                    .unwrap()
                    .redraw(self.rhi_scene_proxy.as_ref().unwrap());
                self.mark_frame(&AppEvent::Render);
                self.time_measurement
                    .update_detail_graphs(self.renderer.as_ref().unwrap().profiler())
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
    graphs: [FrameTimeGraph; 2],

    detail_graphs: BTreeMap<ProfilerCategory, FrameTimeGraph>,
}

impl TimeMeasureSystem {
    pub fn new() -> Self {
        let detail_graphs = all::<ProfilerCategory>()
            .map(|category| (category, FrameTimeGraph::new(format!("{:?}", category))))
            .collect();
        Self {
            last_times: [SystemTime::now(); 2],
            graphs: [
                FrameTimeGraph::new("Tick"),
                FrameTimeGraph::new("Total Frametime"),
            ],
            detail_graphs,
        }
    }

    pub fn reset(&mut self) {
        self.last_times = [SystemTime::now(); 2];
    }

    pub fn update(&mut self, frame_type: &AppEvent) -> Duration {
        let index: usize = match frame_type {
            AppEvent::Tick => 0,
            AppEvent::Render => 1,
        };
        let last = self.last_times[index];
        self.last_times[index] = SystemTime::now();
        let duration = self.last_times[index].duration_since(last).unwrap();
        self.graphs[index].register_duration(duration);
        duration
    }

    pub fn paint_graph_to_gui(&self, frame_type: &AppEvent, ui: &mut Ui) {
        let index: usize = match frame_type {
            AppEvent::Tick => 0,
            AppEvent::Render => 1,
        };
        self.graphs[index].paint_to_gui(ui)
    }

    pub fn update_detail_graphs(&mut self, profiler: &Profiler) {
        profiler.records().last_durations().map(|durations| {
            durations.iter().for_each(|(category, duration)| {
                self.detail_graphs
                    .get_mut(category)
                    .unwrap()
                    .register_duration(*duration)
            })
        });
    }

    pub fn paint_detail_graphs_to_gui(&self, ui: &mut Ui) {
        self.detail_graphs
            .iter()
            .for_each(|(_, graph)| graph.paint_to_gui(ui))
    }

    pub fn last_duration(&self, frame_type: &AppEvent) -> Option<Duration> {
        let index: usize = match frame_type {
            AppEvent::Tick => 0,
            AppEvent::Render => 1,
        };
        self.graphs[index].last_duration()
    }
}

struct FrameTimeGraph {
    durations: Vec<Duration>,
    smoothed_durations: Vec<Duration>,
    current_index: usize,
    name: String,
    total_time: Duration,
    total_sample_count: usize,
}

impl FrameTimeGraph {
    const CAPACITY: usize = 200;
    const SMOOTHING: usize = 10;
    const HEIGHT: f32 = 100f32;
    const WIDTH: f32 = 150f32;
    pub fn new(name: impl Into<String>) -> Self {
        let mut durations = vec![];
        let mut smoothed_durations = vec![];
        durations.reserve_exact(Self::CAPACITY);
        smoothed_durations.reserve_exact(Self::CAPACITY);
        Self {
            durations,
            smoothed_durations,
            current_index: 0,
            name: name.into(),
            total_time: Duration::ZERO,
            total_sample_count: 0,
        }
    }

    pub fn paint_to_gui(&self, ui: &mut Ui) {
        if self.durations.is_empty() {
            return;
        }

        ui.collapsing(&self.name, |ui| {
            let current_time = self.last_duration().unwrap().as_secs_f32();
            ui.label(format!("Current: {:.2}ms", current_time * 1000f32));

            let (response, painter) =
                ui.allocate_painter(Vec2::new(Self::WIDTH, Self::HEIGHT), Sense::hover());

            let to_screen = RectTransform::from_to(
                Rect::from_min_size(Pos2::ZERO, response.rect.size()),
                response.rect,
            );

            let min = self.durations.iter().min().unwrap().as_secs_f32();
            let max = self.durations.iter().max().unwrap().as_secs_f32();

            ui.label(format!("Recent Min: {:.2}ms", min * 1000f32));
            ui.label(format!("Recent Max: {:.2}ms", max * 1000f32));

            let avg = self
                .durations
                .iter()
                .map(Duration::as_secs_f32)
                .sum::<f32>()
                / self.durations.len() as f32;

            ui.label(format!("Recent Average: {:.2}ms", avg * 1000f32));
            ui.label(format!(
                "Total Average: {:.2}ms",
                self.total_time.as_secs_f32() / self.total_sample_count as f32 * 1000f32
            ));

            let durations_path = Self::shape_from_vec(
                &self.durations,
                min,
                max,
                self.current_index,
                &to_screen,
                Stroke::new(1f32, Color32::CYAN.linear_multiply(0.3f32)),
            );
            let smoothed_durations_path = Self::shape_from_vec(
                &self.smoothed_durations,
                min,
                max,
                self.current_index,
                &to_screen,
                Stroke::new(2f32, Color32::RED.linear_multiply(0.7f32)),
            );

            painter.add(epaint::RectShape::stroke(
                response.rect,
                0,
                Stroke::new(1f32, Color32::GRAY),
                StrokeKind::Inside,
            ));

            painter.add(durations_path);
            painter.add(smoothed_durations_path);
        });
    }

    fn shape_from_vec(
        vec: &Vec<Duration>,
        min: f32,
        max: f32,
        current_index: usize,
        to_screen: &RectTransform,
        stroke: Stroke,
    ) -> PathShape {
        let sample_width = Self::WIDTH / Self::CAPACITY as f32;
        let samples = vec
            .iter()
            .chain(vec.iter())
            .skip(current_index)
            .take(vec.len())
            .enumerate()
            .map(|(i, duration)| {
                Pos2::new(
                    Self::WIDTH - (vec.len() - i) as f32 * sample_width,
                    (max - duration.as_secs_f32()) * Self::HEIGHT / (max - min),
                )
            })
            .map(|pos| to_screen * pos)
            .collect();

        PathShape::line(samples, stroke)
    }

    pub fn register_duration(&mut self, duration: Duration) {
        if self.current_index >= self.durations.len() {
            self.durations.push(duration);
            if self.durations.len() < Self::SMOOTHING {
                self.smoothed_durations.push(Duration::from_secs_f32(
                    self.durations.iter().map(|d| d.as_secs_f32()).sum::<f32>()
                        / self.durations.len() as f32,
                ));
            } else {
                self.smoothed_durations.push(Duration::from_secs_f32(
                    self.durations
                        .iter()
                        .skip(self.durations.len() - Self::SMOOTHING)
                        .map(|d| d.as_secs_f32())
                        .sum::<f32>()
                        / Self::SMOOTHING as f32,
                ));
            }
        } else {
            self.durations[self.current_index] = duration;
            self.smoothed_durations[self.current_index] = Duration::from_secs_f32(
                self.durations
                    .iter()
                    .chain(self.durations.iter())
                    .skip(Self::CAPACITY + self.current_index - Self::SMOOTHING)
                    .take(Self::SMOOTHING)
                    .map(|d| d.as_secs_f32())
                    .sum::<f32>()
                    / Self::SMOOTHING as f32,
            );
        }

        self.current_index = (self.current_index + 1) % Self::CAPACITY;

        self.total_time += duration;
        self.total_sample_count += 1;
    }

    pub fn last_duration(&self) -> Option<Duration> {
        if self.current_index == 0 {
            self.durations.last().cloned()
        } else {
            Some(self.durations[self.current_index - 1])
        }
    }
}
