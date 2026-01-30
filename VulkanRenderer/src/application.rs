mod assets;
mod input;
mod renderer;
mod scene;

use crate::AppEvent;
use crate::application::input::InputAction;
use crate::application::{
    assets::{
        asset_traits::{RHIInterface, RHISceneInterface},
        material::Material,
        material_instance::MaterialInstance,
        mesh::Mesh,
    },
    renderer::rhi_assets::vulkan_scene::VKScene,
    scene::{Scene, model::Model, transform::Transform},
};
use asset_system::{assets::AssetHandle, resource_management::ResourceManager};
use gilrs::Gilrs;
use glam::{EulerRot, Quat, Vec3};
use renderer::Renderer;
use std::{cell::RefCell, marker::PhantomData, ops::DerefMut, rc::Rc, sync::Arc};
use winit::event::{DeviceEvent, DeviceId};
use winit::{
    application::ApplicationHandler, event::WindowEvent, event_loop::ActiveEventLoop,
    window::WindowId,
};
use winit_input_map::InputCode;
use winit_input_map::{InputMap, input_map};

pub struct Application {
    renderer: Option<Rc<Renderer>>,
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
        let rhi = self.renderer.as_ref().unwrap();
        self.rhi_scene_proxy = Some(VKScene::create(
            &self.scene,
            rhi,
            rhi.resource_manager_mut().deref_mut(),
        ));
    }

    fn tick(&mut self) {
        use InputAction::*;

        let mouse_move = self.input.dir(MouseLeft, MouseRight, MouseUp, MouseDown);
        let cam_move = self.input.dir(Forward, Back, Right, Left);

        let mut cam_euler = self.scene.camera.transform.rotation.to_euler(EulerRot::YXZ);
        cam_euler.2 = 0.;

        if self.input.pressing(BothClick) {
            let right = self.scene.camera.transform.right();
            let up = self.scene.camera.transform.up();
            self.scene.camera.transform.location +=
                (right * mouse_move.0 + up * mouse_move.1) * 0.5;
        } else if self.input.pressing(LeftClick) {
            let forward = self.scene.camera.transform.forward();
            let forward_proj = forward.with_y(0.).normalize();

            self.scene.camera.transform.location += forward_proj * mouse_move.1;
            cam_euler.0 += mouse_move.0 * 0.1;
        } else if self.input.pressing(RightClick) {
            cam_euler.0 += mouse_move.0 * 0.1;
            cam_euler.1 -= mouse_move.1 * 0.1;
        }

        if self.input.pressing(LeftClick) || self.input.pressing(RightClick) {
            let forward = self.scene.camera.transform.forward();
            let right = self.scene.camera.transform.right();
            let vertical = self.input.axis(Up, Down);

            self.scene.camera.transform.location +=
                (forward * cam_move.0 + right * cam_move.1 - Vec3::Y * vertical) * 0.1;
        }

        cam_euler.1 = cam_euler.1.clamp(-1.5, 1.5);

        self.scene.camera.transform.rotation =
            Quat::from_euler(EulerRot::YXZ, cam_euler.0, cam_euler.1, cam_euler.2);

        self.input.init();
    }
}

impl ApplicationHandler<AppEvent> for Application {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.renderer = Some(Renderer::new(event_loop, self.asset_manager.clone()));
        self.update_scene_proxy();
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: AppEvent) {
        match event {
            AppEvent::Tick => self.tick(),
            AppEvent::Render => self.renderer.as_ref().unwrap().window().request_redraw(),
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        self.input.update_with_window_event(&event);
        self.renderer.as_ref().unwrap().gui_mut().update(&event);
        match event {
            WindowEvent::Resized(_) => self
                .renderer
                .as_ref()
                .unwrap()
                .mutable_state()
                .request_recreate_swapchain(),
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
