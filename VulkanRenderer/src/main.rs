extern crate winit;
mod application;

use application::Application;
use winit::event_loop::EventLoop;

pub enum AppEvent {
    Tick,
    Render,
}

fn main() {
    let event_loop = EventLoop::<AppEvent>::with_user_event().build().unwrap();
    let event_loop_proxy = event_loop.create_proxy();

    std::thread::spawn(move || {
        loop {
            let _ = event_loop_proxy.send_event(AppEvent::Tick);
            let _ = event_loop_proxy.send_event(AppEvent::Render);
            std::thread::sleep(std::time::Duration::from_millis(16));
        }
    });

    let mut app = Application::new();
    event_loop.run_app(&mut app).unwrap()
}
