mod physical_device;
mod swapchain;
mod queue;
mod render_pass;

use crate::application::renderer::swapchain::Swapchain;
use queue::{
    QueueCollection,
    QueueFamilyIndices
};
use std::sync::Arc;
use vulkano::{
    device::{
        physical::PhysicalDevice,
        Device,
        DeviceCreateInfo,
        DeviceExtensions,
        DeviceFeatures
    },
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
    swapchain::Surface,
    VulkanLibrary
};
use vulkano::command_buffer::pool::{CommandPool, CommandPoolCreateFlags, CommandPoolCreateInfo};
use vulkano::render_pass::RenderPass;
use winit::{event_loop::ActiveEventLoop, window::Window};
use crate::application::renderer::render_pass::RenderPassBuilder;

pub struct Renderer {
    window: Arc<Window>,
    instance: Arc<Instance>,
    surface: Arc<Surface>,
    physical_device: Arc<PhysicalDevice>,
    device: Arc<Device>,
    queues: QueueCollection,
    swapchain: Swapchain,
    render_pass: Arc<RenderPass>,
    command_pool: CommandPool,
}

impl Renderer {
    pub fn new(event_loop: &ActiveEventLoop) -> Self {
        let window = Self::create_window(event_loop);
        let instance = Self::create_instance(&Surface::required_extensions(event_loop).unwrap());
        let surface = Self::create_surface(&instance, &window);
        let physical_device = Self::pick_physical_device(&instance, &surface);
        let queue_family_indices = QueueFamilyIndices::find_queue_indices(&physical_device, &surface);
        let (device, queues) = Self::create_logical_device(&physical_device, &queue_family_indices);
        let swapchain = Swapchain::new(&device, &physical_device, &surface, &window, &queue_family_indices);
        let render_pass = RenderPassBuilder::build_default_render_pass(&device, &physical_device, swapchain.format)
            .build();
        let command_pool = Self::create_command_pool(&device, &queue_family_indices);
        Self {
            window,
            instance,
            surface,
            physical_device,
            device,
            queues,
            swapchain,
            render_pass,
            command_pool,
        }
    }

    pub fn redraw(&self) {
        self.window.as_ref().request_redraw();
    }

    fn create_window(event_loop: &ActiveEventLoop) -> Arc<Window> {
        let window_attributes = Window::default_attributes()
            .with_title("Vulkan renderer")
            .with_inner_size(winit::dpi::LogicalSize::new(800.0, 600.0));

        let window = event_loop.create_window(window_attributes).unwrap().into();
        window
    }

    fn create_instance(window_extensions: &InstanceExtensions) -> Arc<Instance> {
        let library = VulkanLibrary::new().unwrap();
        let instance_extensions = window_extensions.clone();
        let instance_create_info = InstanceCreateInfo {
            enabled_extensions: instance_extensions,
            enabled_layers: vec![String::from("VK_LAYER_KHRONOS_validation")],
            ..InstanceCreateInfo::application_from_cargo_toml()
        };
        let instance = Instance::new(library, instance_create_info).unwrap();
        instance
    }

    fn create_surface(instance: &Arc<Instance>, window: &Arc<Window>) -> Arc<Surface> {
        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();
        surface
    }

    fn pick_physical_device(
        instance: &Arc<Instance>,
        surface: &Arc<Surface>,
    ) -> Arc<PhysicalDevice> {
        let physical_device = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|physical_device: &Arc<PhysicalDevice>| {
                physical_device::is_physical_device_suitable_for_surface(physical_device, surface)
            })
            .last()
            .expect("No suitable physical device found");

        physical_device
    }

    fn create_logical_device(
        physical_device: &Arc<PhysicalDevice>, queue_indices: &QueueFamilyIndices
    ) -> (Arc<Device>, QueueCollection) {
        let queue_create_infos = queue_indices.generate_create_infos();
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::default()
        };
        let device_features = DeviceFeatures {
            sampler_anisotropy: true,
            ..DeviceFeatures::default()
        };
        let device_create_info = DeviceCreateInfo {
            queue_create_infos,
            enabled_extensions: device_extensions,
            enabled_features: device_features,
            physical_devices: vec![physical_device.clone()].into(),
            ..DeviceCreateInfo::default()
        };

        let (device, queues) = Device::new(physical_device.clone(), device_create_info).unwrap();
        (device, QueueCollection::new(queues.collect()))
    }

    fn create_command_pool(device: &Arc<Device>, queue_family_indices: &QueueFamilyIndices) -> CommandPool {
        let create_info = CommandPoolCreateInfo {
            flags: CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index: queue_family_indices.graphics_family,
            ..CommandPoolCreateInfo::default()
        };
        CommandPool::new(device.clone(), create_info).unwrap()
    }
}
