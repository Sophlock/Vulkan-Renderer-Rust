mod renderer;
mod scene;
mod assets;

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::WindowId,
};

use renderer::Renderer;
use crate::application::assets::asset_traits::SceneInterface;
use crate::application::renderer::rhi_assets::vulkan_scene::VKScene;
use crate::application::scene::Scene;

pub struct Application {
    renderer: Option<Renderer>,
    scene: Option<VKScene>
}

impl Application {
    pub fn new() -> Self {
        Self {
            renderer: None,
            scene: None
        }
    }

    fn scene() -> Scene {
        let mut scene = Scene::new();
        scene
    }
}

impl ApplicationHandler for Application {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.renderer = Some(Renderer::new(event_loop));
        self.scene = Some(Self::scene().rhi::<VKScene>(self.renderer.as_ref().unwrap()));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        self.renderer.as_mut().unwrap().gui().update(&event);
        match event {
            WindowEvent::ActivationTokenDone { .. } => {}
            WindowEvent::Resized(_) => {
                self.renderer.as_mut().unwrap().recreate_swapchain()
            }
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
                self.renderer.as_mut().unwrap().redraw(self.scene.as_ref().unwrap());
            }
        }
    }
}
