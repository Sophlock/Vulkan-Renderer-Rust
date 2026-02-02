mod buffer;
mod command_buffer;
mod layers;
mod physical_device;
pub(crate) mod pipeline;
mod queue;
pub mod render_pass;
mod render_sync;
pub mod rhi_assets;
mod shader_cursor;
mod shader_object;
mod shaders;
pub mod swapchain;

use std::{
    cell::{Ref, RefCell, RefMut},
    ops::Deref,
    rc::Rc,
    sync::Arc,
};

use asset_system::resource_management::ResourceManager;
use command_buffer::CommandBufferInterface;
use egui_winit_vulkano::{
    Gui, GuiConfig, egui,
    egui::{Color32, Frame},
};
use physical_device::find_depth_format;
use queue::{QueueCollection, QueueFamilyIndices};
use render_pass::RenderPassBuilder;
use rhi_assets::{vulkan_mesh::VKMesh, vulkan_texture::VKTexture};
use smallvec::smallvec;
use swapchain::Swapchain;
use vulkano::command_buffer::CommandBuffer;
use vulkano::{
    Validated, ValidationError, VulkanError, VulkanLibrary,
    command_buffer::{
        AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo,
        SubpassContents, SubpassEndInfo,
    },
    descriptor_set::allocator::{
        DescriptorSetAllocator, StandardDescriptorSetAllocator,
        StandardDescriptorSetAllocatorCreateInfo,
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, physical::PhysicalDevice,
    },
    format::{ClearValue, Format},
    image::{
        Image, ImageAspects, ImageCreateInfo, ImageLayout, ImageSubresourceRange, ImageTiling,
        ImageType, ImageUsage, SampleCount,
        view::{ImageView, ImageViewCreateInfo, ImageViewType},
    },
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
    },
    pipeline::{
        PipelineBindPoint,
        graphics::viewport::{Scissor, Viewport},
    },
    render_pass::{Framebuffer, RenderPass},
    swapchain::{Surface, SwapchainPresentInfo, present},
    sync::{GpuFuture, Sharing, future::FenceSignalFuture},
};
use winit::{dpi::PhysicalSize, event_loop::ActiveEventLoop, window::Window};

use super::assets::asset_traits::{
    RHICameraInterface, RHIInterface, RHIModelInterface, RHISceneInterface,
};
use crate::application::rhi::swapchain::SwapchainSupportDetails;
use crate::application::rhi::{
    rhi_assets::{
        RHIResourceManager, vulkan_camera::VKCamera, vulkan_material::VKMaterial,
        vulkan_material_instance::VKMaterialInstance, vulkan_model::VKModel, vulkan_scene::VKScene,
    },
    shaders::SlangCompiler,
};

pub struct VKRHI {
    frames_in_flight: usize,
    window: Arc<Window>,
    instance: Arc<Instance>,
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
}

impl VKRHI {
    pub fn new(
        event_loop: &ActiveEventLoop,
        asset_manager: Arc<RefCell<ResourceManager>>,
    ) -> Rc<Self> {
        let window = Self::create_window(event_loop);
        let instance = Self::create_instance(&Surface::required_extensions(event_loop).unwrap());
        let surface = Self::create_surface(&instance, &window);
        let physical_device = Self::pick_physical_device(&instance, &surface);
        let queue_family_indices =
            QueueFamilyIndices::find_queue_indices(&physical_device, &surface);
        let (device, queues) = Self::create_logical_device(&physical_device, &queue_family_indices);

        let swapchain_support = SwapchainSupportDetails::query_swapchain_support(&physical_device, &surface);
        let frames_in_flight = Swapchain::decide_image_count(
            &swapchain_support.capabilities,
        ) as usize;
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
        let result = Rc::new(Self {
            frames_in_flight,
            window,
            instance,
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
        physical_device: &Arc<PhysicalDevice>,
        queue_indices: &QueueFamilyIndices,
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
        (
            device,
            QueueCollection::new(queues.collect(), queue_indices),
        )
    }

    pub fn create_depth_buffer(&self, extent: [u32; 2]) -> Arc<ImageView> {
        let depth_format = find_depth_format(self.physical_device());
        let image_create_info = ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: depth_format,
            extent: [extent[0], extent[1], 1],
            array_layers: 1,
            mip_levels: 1,
            samples: SampleCount::Sample1,
            tiling: ImageTiling::Optimal,
            usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
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
            format: depth_format,
            subresource_range: ImageSubresourceRange {
                aspects: ImageAspects::DEPTH,
                mip_levels: 0..1,
                array_layers: 0..1,
            },
            ..ImageViewCreateInfo::default()
        };
        let alloc = Arc::new(StandardMemoryAllocator::new_default(self.device.clone()));
        let depth_image = Image::new(alloc, image_create_info, allocation_info).unwrap();
        let depth_image_view = ImageView::new(depth_image, image_view_create_info).unwrap();
        depth_image_view
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
