mod assets;
mod input;
mod renderer;
mod rhi;
mod scene;

use std::{cell::RefCell, marker::PhantomData, ops::DerefMut, rc::Rc, sync::Arc};

use asset_system::{assets::AssetHandle, resource_management::ResourceManager};
use gilrs::Gilrs;
use glam::{EulerRot, Quat, Vec3};
use rhi::VKRHI;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::ActiveEventLoop,
    window::WindowId,
};
use winit_input_map::{input_map, InputCode, InputMap};

use crate::{
    application::{
        assets::{
            asset_traits::{RHIInterface, RHISceneInterface, RendererInterface},
            material::Material,
            material_instance::MaterialInstance,
            mesh::Mesh,
        },
        input::InputAction,
        renderer::VKRenderer,
        rhi::rhi_assets::vulkan_scene::VKScene,
        scene::{model::Model, transform::Transform, Scene},
    },
    AppEvent,
};

pub struct Application {
    renderer: Option<Rc<VKRenderer>>,
    asset_manager: Arc<RefCell<ResourceManager>>,
    rhi_scene_proxy: Option<VKScene>,
    scene: Scene,
    input: InputMap<InputAction>,
    gilrs: Gilrs,
}

impl Application {
    pub fn new() -> Self {
        let asset_manager = Arc::new(RefCell::new(ResourceManager::new()));
        let scene = Self::scene(asset_manager.borrow_mut().deref_mut());
        Self {
            renderer: None,
            rhi_scene_proxy: None,
            asset_manager,
            scene,
            input: Self::build_input_map(),
            gilrs: Gilrs::new().unwrap(),
        }
    }

    fn scene(asset_manager: &mut ResourceManager) -> Scene {
        let material = AssetHandle::<Material> {
            uuid: asset_manager.add(Material::new(
                "TestMat".into(),
                "Materials/basicMaterials".into(),
                "SingleColorUnlitMaterial".into(),
            )),
            _phantom: PhantomData,
        };
        let mut scene = Scene::new();
        scene.models.push(Model::new(
            "TestModel".into(),
            Transform::default(),
            AssetHandle::<Mesh> {
                uuid: asset_manager.add(Mesh::new(
                    "TestMesh".into(),
                    "resources/assets/meshes/sphere.glb",
                )),
                _phantom: PhantomData,
            },
            AssetHandle::<MaterialInstance> {
                uuid: asset_manager.add(MaterialInstance::new("TestMatInst".into(), material)),
                _phantom: PhantomData,
            },
        ));
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

    fn update_scene_proxy(&mut self) {
        // TODO: This should be more lazy
        let rhi = self.renderer.as_ref().unwrap().rhi();
        self.rhi_scene_proxy = Some(VKScene::create(
            &self.scene,
            rhi,
            rhi.resource_manager_mut().deref_mut(),
        ));
    }

    fn update_aspect_ratio(&mut self, x: u32, y: u32) {
        self.scene.camera.aspect = x as f32 / y as f32;
    }

    fn tick(&mut self) {
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
            cam.transform.location += (right * mouse_move.0 + up * mouse_move.1) * cam.speed;
        } else if left {
            let forward = cam.transform.forward();
            let forward_proj = forward.with_y(0.).normalize();

            cam.transform.location += forward_proj * mouse_move.1;
            cam_euler.0 += mouse_move.0 * 0.1;
        } else if right {
            cam_euler.0 += mouse_move.0 * cam.rot_speed;
            cam_euler.1 -= mouse_move.1 * cam.rot_speed;
        }

        if left || right {
            let forward = cam.transform.forward();
            let right = cam.transform.right();
            let vertical = self.input.axis(Up, Down);

            cam.transform.location +=
                (forward * cam_move.0 + right * cam_move.1 - Vec3::Y * vertical) * cam.speed * 0.2;

            cam.speed += scroll * 0.01;
            cam.speed = cam.speed.max(0.);
        } else {
            cam.fov += scroll * cam.speed * 0.1;
        }

        cam_euler.1 = cam_euler.1.clamp(-1.5, 1.5);

        self.scene.camera.transform.rotation =
            Quat::from_euler(EulerRot::YXZ, cam_euler.0, cam_euler.1, cam_euler.2);

        self.input.init();
    }
}

impl ApplicationHandler<AppEvent> for Application {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let rhi = VKRHI::new(event_loop, self.asset_manager.clone());
        self.renderer = Some(Rc::new(VKRenderer::new(rhi)));
        self.update_scene_proxy();
        // TODO: This should be called on demand as well
        self.renderer.as_ref().unwrap().compile_materials();
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: AppEvent) {
        match event {
            AppEvent::Tick => self.tick(),
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
                self.update_scene_proxy();
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
}
