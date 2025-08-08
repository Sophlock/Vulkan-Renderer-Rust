use std::sync::Arc;
use vulkano::sync::fence::{FenceCreateFlags, FenceCreateInfo};
use vulkano::{
    device::Device,
    sync::{
        fence::Fence,
        semaphore::{Semaphore, SemaphoreCreateInfo},
    },
};

pub struct RenderSync {
    pub image_available_semaphore: Arc<Semaphore>,
    pub render_finished_semaphore: Semaphore,
    pub in_flight_fence: Fence,
}

impl RenderSync {
    pub fn create_sync_objects(device: &Arc<Device>, frames_in_flight: usize) -> Vec<Self> {
        (0..frames_in_flight).map(|_| {
            Self::new(&device)
        }).collect()
    }
    pub fn new(device: &Arc<Device>) -> Self {
        let image_available_semaphore =
            Semaphore::new(device.clone(), SemaphoreCreateInfo::default()).unwrap().into();
        let render_finished_semaphore =
            Semaphore::new(device.clone(), SemaphoreCreateInfo::default()).unwrap();
        let in_flight_fence = Fence::new(
            device.clone(),
            FenceCreateInfo {
                flags: FenceCreateFlags::SIGNALED,
                ..FenceCreateInfo::default()
            },
        )
        .unwrap();
        Self {
            image_available_semaphore,
            render_finished_semaphore,
            in_flight_fence,
        }
    }
}
