pub mod buffer;
pub mod command_buffer;
pub mod device_helper;
mod layers;
mod physical_device;
pub mod pipeline;
mod queue;
pub mod render_pass;
mod render_sync;
pub mod rhi_assets;
pub mod shader_cursor;
pub mod shader_object;
pub mod shaders;
pub mod swapchain;
pub mod swapchain_resources;

use std::{
    cell::{Ref, RefCell, RefMut},
    ops::Deref,
    rc::Rc,
    sync::Arc,
};

use command_buffer::CommandBufferInterface;
use egui_winit_vulkano::{Gui, GuiConfig};
use physical_device::find_depth_format;
use queue::{QueueCollection, QueueFamilyIndices};
use rhi_assets::{vulkan_mesh::VKMesh, vulkan_texture::VKTexture};
use swapchain::Swapchain;
use vulkano::{
    VulkanLibrary,
    descriptor_set::allocator::{
        DescriptorSetAllocator, StandardDescriptorSetAllocator,
        StandardDescriptorSetAllocatorCreateInfo,
    },
    device::{Device, physical::PhysicalDevice},
    format::Format,
    image::{
        Image, ImageAspects, ImageCreateInfo, ImageLayout, ImageSubresourceRange, ImageTiling,
        ImageType, ImageUsage, SampleCount,
        view::{ImageView, ImageViewCreateInfo, ImageViewType},
    },
    instance::{
        Instance, InstanceCreateInfo, InstanceExtensions,
        debug::{
            DebugUtilsMessageSeverity, DebugUtilsMessenger, DebugUtilsMessengerCallback,
            DebugUtilsMessengerCreateInfo,
        },
    },
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
    },
    swapchain::Surface,
    sync::{GpuFuture, Sharing},
};
use winit::{event_loop::ActiveEventLoop, window::Window};

use super::assets::asset_traits::{
    RHICameraInterface, RHIInterface, RHIModelInterface, RHISceneInterface,
};
use crate::application::{
    assets::AssetManager::AssetManager,
    rhi::{
        rhi_assets::{
            RHIResourceManager, vulkan_camera::VKCamera, vulkan_material::VKMaterial,
            vulkan_material_instance::VKMaterialInstance, vulkan_model::VKModel,
            vulkan_scene::VKScene,
        },
        shaders::SlangCompiler,
        swapchain::SwapchainSupportDetails,
    },
};
use crate::application::rhi::shader_object::ShaderObjectQueue;

pub struct VKRHI {
    frames_in_flight: usize,
    window: Arc<Window>,
    instance: Arc<Instance>,
    debug_messenger: DebugUtilsMessenger,
    surface: Arc<Surface>,
    physical_device: Arc<PhysicalDevice>,
    device: Arc<Device>,
    queues: QueueCollection,
    queue_family_indices: QueueFamilyIndices,

    command_buffer_interface: CommandBufferInterface,
    gui: RefCell<Gui>,
    slang_compiler: SlangCompiler,
    buffer_allocator: Arc<dyn MemoryAllocator>,
    descriptor_allocator: Arc<dyn DescriptorSetAllocator>,
    resource_manager: RefCell<RHIResourceManager>,
    shader_object_update_queue: Arc<RefCell<ShaderObjectQueue>>,
}

impl VKRHI {
    pub fn new(
        event_loop: &ActiveEventLoop,
        asset_manager: Arc<RefCell<AssetManager>>,
    ) -> Rc<Self> {
        let window = Self::create_window(event_loop);
        let instance = Self::create_instance(&Surface::required_extensions(event_loop).unwrap());
        let debug_messenger = Self::create_debug_messenger(instance.clone());
        let surface = Self::create_surface(&instance, &window);
        let physical_device = Self::pick_physical_device(&instance, &surface);
        let queue_family_indices =
            QueueFamilyIndices::find_queue_indices(&physical_device, &surface);
        let (device, queues) = Self::create_logical_device(&physical_device, &queue_family_indices);

        let swapchain_support =
            SwapchainSupportDetails::query_swapchain_support(&physical_device, &surface);
        let frames_in_flight =
            Swapchain::decide_image_count(&swapchain_support.capabilities) as usize;
        let swapchain_format = Swapchain::choose_surface_format(&swapchain_support.formats).0;

        let command_buffer_interface =
            CommandBufferInterface::new(device.clone(), frames_in_flight);
        let gui = RefCell::new(Gui::new(
            event_loop,
            surface.clone(),
            queues.graphics_queue.clone(),
            swapchain_format,
            GuiConfig {
                is_overlay: true,
                ..GuiConfig::default()
            },
        ));
        let slang_compiler = SlangCompiler::new("resources/assets/materials/shaders".as_ref());
        let buffer_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo {
                update_after_bind: false,
                ..StandardDescriptorSetAllocatorCreateInfo::default()
            },
        ));
        let resource_manager = RefCell::new(RHIResourceManager::new(asset_manager));
        let shader_object_update_queue = ShaderObjectQueue::new();
        let result = Rc::new(Self {
            frames_in_flight,
            window,
            instance,
            debug_messenger,
            surface,
            physical_device,
            device,
            queues,
            queue_family_indices,
            command_buffer_interface,
            gui,
            slang_compiler,
            buffer_allocator,
            descriptor_allocator,
            resource_manager,
            shader_object_update_queue
        });
        result.resource_manager.borrow_mut().register_rhi(&result);
        result
    }

    fn create_window(event_loop: &ActiveEventLoop) -> Arc<Window> {
        let window_attributes = Window::default_attributes()
            .with_title("Vulkan rhi")
            .with_inner_size(winit::dpi::LogicalSize::new(1920.0, 1080.0));

        let window = event_loop.create_window(window_attributes).unwrap().into();
        window
    }

    fn create_instance(window_extensions: &InstanceExtensions) -> Arc<Instance> {
        let library = VulkanLibrary::new().unwrap();
        let instance_extensions = InstanceExtensions {
            ext_debug_utils: true,
            ..window_extensions.clone()
        };
        let instance_create_info = InstanceCreateInfo {
            enabled_extensions: instance_extensions,
            enabled_layers: vec![String::from("VK_LAYER_KHRONOS_validation")],
            ..InstanceCreateInfo::application_from_cargo_toml()
        };
        let instance = Instance::new(library, instance_create_info).unwrap();
        instance
    }

    fn create_debug_messenger(instance: Arc<Instance>) -> DebugUtilsMessenger {
        let callback = unsafe {
            DebugUtilsMessengerCallback::new(|severity, message_type, data| {
                println!("{:?} {:?}: {}", message_type, severity, data.message);
            })
        };
        let create_info = DebugUtilsMessengerCreateInfo {
            message_severity: DebugUtilsMessageSeverity::ERROR
                | DebugUtilsMessageSeverity::WARNING
                | DebugUtilsMessageSeverity::INFO
                | DebugUtilsMessageSeverity::VERBOSE,
            ..DebugUtilsMessengerCreateInfo::user_callback(callback)
        };
        DebugUtilsMessenger::new(instance, create_info).unwrap()
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
        physical_device: &Arc<PhysicalDevice>,
        queue_indices: &QueueFamilyIndices,
    ) -> (Arc<Device>, QueueCollection) {
        /*let queue_create_infos = queue_indices.generate_create_infos();
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::default()
        };
        let device_features = DeviceFeatures {
            sampler_anisotropy: true,
            compute_derivative_group_quads: true,
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
        (
            device,
            QueueCollection::new(queues.collect(), queue_indices),
        )*/
        device_helper::create_logical_device(physical_device, queue_indices)
    }

    pub fn create_gbuffer(
        &self,
        extent: [u32; 2],
        format: Format,
        usage: ImageUsage,
        aspects: ImageAspects,
    ) -> Arc<ImageView> {
        let image_create_info = ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format,
            extent: [extent[0], extent[1], 1],
            array_layers: 1,
            mip_levels: 1,
            samples: SampleCount::Sample1,
            tiling: ImageTiling::Optimal,
            usage,
            sharing: Sharing::Exclusive,
            initial_layout: ImageLayout::Undefined,
            ..ImageCreateInfo::default()
        };
        let allocation_info = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..AllocationCreateInfo::default()
        };
        let image_view_create_info = ImageViewCreateInfo {
            view_type: ImageViewType::Dim2d,
            format,
            subresource_range: ImageSubresourceRange {
                aspects,
                mip_levels: 0..1,
                array_layers: 0..1,
            },
            ..ImageViewCreateInfo::default()
        };
        let image = Image::new(
            self.buffer_allocator.clone(),
            image_create_info,
            allocation_info,
        )
        .unwrap();
        let image_view = ImageView::new(image, image_view_create_info).unwrap();
        image_view
    }

    pub fn create_depth_buffer(&self, extent: [u32; 2]) -> Arc<ImageView> {
        let depth_format = find_depth_format(self.physical_device());
        self.create_gbuffer(
            extent,
            depth_format,
            ImageUsage::DEPTH_STENCIL_ATTACHMENT,
            ImageAspects::DEPTH,
        )
    }

    pub fn gui_mut(&self) -> RefMut<Gui> {
        self.gui.borrow_mut()
    }

    pub fn window(&self) -> &Window {
        self.window.as_ref()
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn physical_device(&self) -> &Arc<PhysicalDevice> {
        &self.physical_device
    }

    pub fn surface(&self) -> &Arc<Surface> {
        &self.surface
    }

    pub fn queue_family_indices(&self) -> &QueueFamilyIndices {
        &self.queue_family_indices
    }

    pub fn queues(&self) -> &QueueCollection {
        &self.queues
    }

    pub fn command_buffer_interface(&self) -> &CommandBufferInterface {
        &self.command_buffer_interface
    }

    pub fn descriptor_allocator(&self) -> &Arc<dyn DescriptorSetAllocator> {
        &self.descriptor_allocator
    }

    pub fn buffer_allocator(&self) -> &Arc<dyn MemoryAllocator> {
        &self.buffer_allocator
    }

    pub fn in_flight_frames(&self) -> usize {
        self.frames_in_flight
    }

    pub fn slang_compiler(&self) -> &SlangCompiler {
        &self.slang_compiler
    }

    pub fn shutdown(&self) {
        unsafe {
            self.device.wait_idle().unwrap();
        }
    }
    
    pub fn shader_object_update_queue(&self) -> &Arc<RefCell<ShaderObjectQueue>> {
        &self.shader_object_update_queue
    }
}

impl RHIInterface for VKRHI {
    type MeshType = VKMesh;
    type TextureType = VKTexture;
    type MaterialType = VKMaterial;
    type MaterialInstanceType = VKMaterialInstance;
    type CameraType = VKCamera;
    type ModelType = VKModel;
    type SceneType = VKScene;

    fn resource_manager(&self) -> Ref<RHIResourceManager> {
        self.resource_manager.borrow()
    }

    fn resource_manager_mut(&self) -> RefMut<RHIResourceManager> {
        self.resource_manager.borrow_mut()
    }
}
