extern crate winit;
mod application;

use application::Application;

use winit::event_loop::{
    ControlFlow,
    EventLoop
};

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = Application::new();
    event_loop.run_app(&mut app).unwrap()
}


