mod assets;
mod renderer;
mod scene;

use crate::application::assets::material::Material;
use crate::application::assets::material_instance::MaterialInstance;
use crate::application::assets::mesh::Mesh;
use crate::application::scene::model::Model;
use crate::application::scene::transform::Transform;
use crate::application::{renderer::rhi_assets::vulkan_scene::VKScene, scene::Scene};
use asset_system::assets::AssetHandle;
use asset_system::resource_management::ResourceManager;
use assets::asset_traits::SceneInterface;
use renderer::Renderer;
use std::marker::PhantomData;
use std::{rc::Rc, sync::Arc};
use std::cell::RefCell;
use std::ops::DerefMut;
use winit::{
    application::ApplicationHandler, event::WindowEvent, event_loop::ActiveEventLoop,
    window::WindowId,
};

pub struct Application {
    renderer: Option<Rc<Renderer>>,
    asset_manager: Option<Arc<RefCell<ResourceManager>>>,
    scene: Option<VKScene>,
}

impl Application {
    pub fn new() -> Self {
        Self {
            renderer: None,
            scene: None,
            asset_manager: None,
        }
    }

    fn scene(&mut self) -> Scene {
        let mut asset_manager_guard = self.asset_manager.as_ref().unwrap().borrow_mut();
        let asset_manager = asset_manager_guard.deref_mut();
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
                uuid: asset_manager.add(MaterialInstance::new(
                    "TestMatInst".into(),
                    material
                )),
                _phantom: PhantomData,
            },
        ));
        scene
    }
}

impl ApplicationHandler for Application {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.asset_manager = Some(Arc::new(RefCell::new(ResourceManager::new())));
        self.renderer = Some(Renderer::new(
            event_loop,
            self.asset_manager.as_ref().unwrap().clone(),
        ));
        self.scene = Some(self.scene().rhi::<VKScene>(self.renderer.as_ref().unwrap()));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        self.renderer.as_ref().unwrap().gui_mut().update(&event);
        match event {
            WindowEvent::ActivationTokenDone { .. } => {}
            WindowEvent::Resized(_) => self
                .renderer
                .as_ref()
                .unwrap()
                .mutable_state()
                .request_recreate_swapchain(),
            WindowEvent::Moved(_) => {}
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Destroyed => {}
            WindowEvent::DroppedFile(_) => {}
            WindowEvent::HoveredFile(_) => {}
            WindowEvent::HoveredFileCancelled => {}
            WindowEvent::Focused(_) => {}
            WindowEvent::KeyboardInput { .. } => {}
            WindowEvent::ModifiersChanged(_) => {}
            WindowEvent::Ime(_) => {}
            WindowEvent::CursorMoved { .. } => {}
            WindowEvent::CursorEntered { .. } => {}
            WindowEvent::CursorLeft { .. } => {}
            WindowEvent::MouseWheel { .. } => {}
            WindowEvent::MouseInput { .. } => {}
            WindowEvent::PinchGesture { .. } => {}
            WindowEvent::PanGesture { .. } => {}
            WindowEvent::DoubleTapGesture { .. } => {}
            WindowEvent::RotationGesture { .. } => {}
            WindowEvent::TouchpadPressure { .. } => {}
            WindowEvent::AxisMotion { .. } => {}
            WindowEvent::Touch(_) => {}
            WindowEvent::ScaleFactorChanged { .. } => {}
            WindowEvent::ThemeChanged(_) => {}
            WindowEvent::Occluded(_) => {}
            WindowEvent::RedrawRequested => {
                self.renderer
                    .as_ref()
                    .unwrap()
                    .redraw(self.scene.as_ref().unwrap());
            }
        }
    }
}
