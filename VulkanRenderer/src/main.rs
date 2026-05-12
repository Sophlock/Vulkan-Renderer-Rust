extern crate winit;
mod application;

use application::Application;
use winit::event_loop::EventLoop;

pub enum AppEvent {
    Tick = 0,
    Render = 1,
}

fn main() {
    // Create the main event loop that drives the application
    let event_loop = EventLoop::<AppEvent>::with_user_event().build().unwrap();
    let event_loop_proxy = event_loop.create_proxy();

    // Continuously send events to the event loop
    std::thread::spawn(move || {
        loop {
            let _ = event_loop_proxy.send_event(AppEvent::Tick);
            let _ = event_loop_proxy.send_event(AppEvent::Render);
            std::thread::sleep(std::time::Duration::from_millis(16));
        }
    });

    // Create and run the application
    let mut app = Application::new();
    event_loop.run_app(&mut app).unwrap()
}
