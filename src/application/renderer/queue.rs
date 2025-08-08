use std::{
    collections::HashSet,
    sync::Arc
};
use vulkano::{
    device::{
        physical::PhysicalDevice,
        Queue,
        QueueCreateInfo,
        QueueFlags
    },
    swapchain::Surface
};
use vulkano::device::QueueFamilyProperties;

pub struct QueueFamilyIndices {
    pub graphics_family: u32,
    pub present_family: u32,
    pub compute_family: u32,
}

pub struct QueueCollection {
    pub graphics_queue: Arc<Queue>,
    pub present_queue: Arc<Queue>,
    pub compute_queue: Arc<Queue>,
}

impl QueueFamilyIndices {
    fn new(shared_family: u32) -> Self {
        Self{
            graphics_family: shared_family,
            present_family: shared_family,
            compute_family: shared_family,
        }
    }

    pub fn find_queue_indices(physical_device: &PhysicalDevice, surface: &Surface) -> Self {
        let queues = physical_device.queue_family_properties().iter()
            .zip(0u32..);
        let ideal_queues = queues.clone()
            .filter(|(prop, index)| {
                let has_graphics_support = prop.queue_flags.contains(QueueFlags::GRAPHICS);
                let has_compute_support = prop.queue_flags.contains(QueueFlags::COMPUTE);
                let has_present_support = physical_device.surface_support(*index, &*surface).unwrap();
                has_graphics_support && has_compute_support && has_present_support
            });
        ideal_queues.last()
            .map(|(_, index)| { QueueFamilyIndices::new(index)})
            .unwrap_or_else(|| {
                QueueFamilyIndices {
                    graphics_family: Self::find_queue_index_with_flag(queues.clone(), QueueFlags::GRAPHICS).unwrap(),
                    present_family: Self::find_present_index(queues.clone(), &physical_device, surface).unwrap(),
                    compute_family: Self::find_queue_index_with_flag(queues.clone(), QueueFlags::COMPUTE).unwrap(),
                }
            })
    }

    pub fn generate_create_infos(&self) -> Vec<QueueCreateInfo> {
        [self.graphics_family, self.compute_family, self.present_family].iter()
            .copied()
            .collect::<HashSet<_>>().iter()
            .map(|index| QueueCreateInfo {
                queue_family_index: *index,
                queues: vec![1f32],
                ..QueueCreateInfo::default()
            })
        .collect()
    }

    pub fn has_shared_family(&self) -> bool {
        self.graphics_family == self.compute_family && self.graphics_family == self.present_family
    }

    fn find_queue_index_with_flag<'a>(queues: impl IntoIterator<Item=(&'a QueueFamilyProperties, u32)>, flag: QueueFlags) -> Option<u32> {
        queues.into_iter()
            .filter(|(prop, _)| prop.queue_flags.contains(flag))
            .last()
            .map(|(_, index)| index)
    }

    fn find_present_index<'a>(queues: impl IntoIterator<Item=(&'a QueueFamilyProperties, u32)>, physical_device: &PhysicalDevice, surface: &Surface) -> Option<u32> {
        queues.into_iter()
            .filter(|(_, index)| physical_device.surface_support(*index, &*surface).unwrap_or(false))
            .last()
            .map(|(_, index)| index)
    }
}

impl QueueCollection {
    pub fn new(queues: Vec<Arc<Queue>>, queue_family_indices: &QueueFamilyIndices) -> Self {
        Self {
            graphics_queue: Self::find_queue_of_family(&queues, queue_family_indices.graphics_family),
            present_queue: Self::find_queue_of_family(&queues, queue_family_indices.present_family),
            compute_queue: Self::find_queue_of_family(&queues, queue_family_indices.compute_family),
        }
    }
    
    fn find_queue_of_family(queues: &Vec<Arc<Queue>>, queue_family_index: u32) -> Arc<Queue> {
        queues.into_iter().filter(|queue| {
            queue.queue_family_index() == queue_family_index
        }).last().unwrap().clone()
    }

}