mod command_buffer;
mod image;
mod physical_device;
mod queue;
mod render_pass;
mod render_sync;
mod swapchain;

use command_buffer::CommandBufferInterface;
use physical_device::find_depth_format;
use queue::{QueueCollection, QueueFamilyIndices};
use render_pass::RenderPassBuilder;
use smallvec::smallvec;
use std::sync::Arc;
use swapchain::Swapchain;
use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo,
        SubpassContents, SubpassEndInfo,
    }, device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures,
    }, format::ClearValue, format::Format,
    image::{
        view::{ImageView, ImageViewCreateInfo, ImageViewType}, Image, ImageAspects, ImageCreateInfo, ImageLayout, ImageSubresourceRange,
        ImageTiling, ImageType, ImageUsage,
        SampleCount,
    },
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::graphics::viewport::{Scissor, Viewport},
    render_pass::{Framebuffer, RenderPass},
    swapchain::{present, Surface, SwapchainPresentInfo},
    sync::future::FenceSignalFuture,
    sync::Sharing,
    sync::GpuFuture,
    Validated,
    ValidationError,
    VulkanError,
    VulkanLibrary,
};
use vulkano::image::ImageFormatInfo;
use winit::{event_loop::ActiveEventLoop, window::Window};

pub struct Renderer {
    should_recreate_swapchain: bool,
    current_image: usize,
    frames_in_flight: usize,
    window: Arc<Window>,
    instance: Arc<Instance>,
    surface: Arc<Surface>,
    physical_device: Arc<PhysicalDevice>,
    device: Arc<Device>,
    queues: QueueCollection,
    queue_family_indices: QueueFamilyIndices,
    swapchain: Swapchain,
    depth_image_view: Arc<ImageView>,
    render_pass: Arc<RenderPass>,
    command_buffer_interface: CommandBufferInterface,
    framebuffers: Vec<Arc<Framebuffer>>,
    in_flight_future: Option<FenceSignalFuture<Box<dyn GpuFuture>>>,
}

impl Renderer {
    pub fn new(event_loop: &ActiveEventLoop) -> Self {
        let window = Self::create_window(event_loop);
        let instance = Self::create_instance(&Surface::required_extensions(event_loop).unwrap());
        let surface = Self::create_surface(&instance, &window);
        let physical_device = Self::pick_physical_device(&instance, &surface);
        let queue_family_indices =
            QueueFamilyIndices::find_queue_indices(&physical_device, &surface);
        let (device, queues) = Self::create_logical_device(&physical_device, &queue_family_indices);
        let swapchain = Swapchain::new(
            &device,
            &physical_device,
            &surface,
            &window,
            &queue_family_indices,
        );
        let frames_in_flight = swapchain.image_count.try_into().unwrap();
        let render_pass = RenderPassBuilder::build_default_render_pass(
            &device,
            &physical_device,
            swapchain.format,
        )
        .build();
        let depth_image_view = Self::create_depth_resources(
            &device,
            find_depth_format(&physical_device),
            swapchain.extent,
        );
        let command_buffer_interface =
            CommandBufferInterface::new(device.clone(), frames_in_flight);
        let framebuffers = swapchain.create_framebuffers(&render_pass, &depth_image_view);
        Self {
            should_recreate_swapchain: false,
            current_image: 0,
            frames_in_flight,
            window,
            instance,
            surface,
            physical_device,
            device,
            queues,
            queue_family_indices,
            swapchain,
            depth_image_view,
            render_pass,
            command_buffer_interface,
            framebuffers,
            in_flight_future: None,
        }
    }

    pub fn redraw(&mut self) {
        self.in_flight_future = self.draw_frame();
        self.window.as_ref().request_redraw();
    }

    fn draw_frame(&mut self) -> Option<FenceSignalFuture<Box<dyn GpuFuture>>> {
        if self.should_recreate_swapchain {
            //self.recreate_swapchain_internal();
        }
        self.current_image = (self.current_image + 1) % self.frames_in_flight;
        self.in_flight_future
            .as_ref()
            .map(|f| f.wait(None).unwrap());

        let acquire_image_result = self.swapchain.acquire_next_image();
        let (swapchain_image_index, suboptimal, image_available_future) = acquire_image_result
            .map_or_else(
                |e| match e {
                    Validated::Error(VulkanError::OutOfDate) => {
                        self.recreate_swapchain();
                        None
                    }
                    _ => panic!("Error acquiring swapchain image"),
                },
                |v| Some(v),
            )?;
        if suboptimal {
            self.recreate_swapchain();
        }
        let mut command_buffer = self
            .command_buffer_interface
            .primary_command_buffer(self.queue_family_indices.graphics_family);

        self.record_draw_command_buffer(&mut command_buffer, swapchain_image_index as usize)
            .unwrap();

        let draw_finished_future = image_available_future
            .then_execute(
                self.queues.graphics_queue.clone(),
                command_buffer.build().unwrap(),
            )
            .unwrap();

        let present_future = present(
            draw_finished_future,
            self.queues.present_queue.clone(),
            SwapchainPresentInfo::swapchain_image_index(
                self.swapchain.raw().clone(),
                swapchain_image_index,
            ),
        );

        let in_flight_future = present_future
            .boxed()
            .then_signal_fence_and_flush()
            .map_or_else(
                |e| match e {
                    Validated::Error(VulkanError::OutOfDate) => {
                        self.recreate_swapchain();
                        None
                    }
                    _ => panic!("Error presenting swapchain image"),
                },
                Some,
            );
        in_flight_future
    }

    fn record_draw_command_buffer(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_index: usize
    ) -> Result<(), Box<ValidationError>> {
        command_buffer
            .begin_render_pass(
                RenderPassBeginInfo {
                    render_area_offset: [0, 0],
                    render_area_extent: self.swapchain.extent,
                    clear_values: vec![
                        Some(ClearValue::Float([0.0, 0.0, 0.0, 1.0])),
                        Some(ClearValue::DepthStencil((1.0, 0))),
                    ],
                    render_pass: self.render_pass.clone(),
                    ..RenderPassBeginInfo::framebuffer(
                        self.framebuffers[image_index].clone(),
                    )
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..SubpassBeginInfo::default()
                },
            )?
            .set_viewport_with_count(smallvec![Viewport {
                offset: [0., 0.],
                extent: self.swapchain.extent.map(|u| u as f32),
                depth_range: 0.0f32..=1.0f32,
            }])?
            .set_scissor_with_count(smallvec![Scissor {
                offset: [0, 0],
                extent: self.swapchain.extent,
            }])?
            .end_render_pass(SubpassEndInfo::default())
            .map(|_| ())
    }

    pub fn recreate_swapchain(&mut self) {
        self.should_recreate_swapchain = true;
    }

    fn recreate_swapchain_internal(&mut self) {
        unsafe { self.device.wait_idle().unwrap() }
        self.swapchain = self.swapchain.recreate(
            &self.physical_device,
            &self.surface,
            &self.window,
            &self.queue_family_indices,
        );
        let image_format_properties = self.physical_device.image_format_properties(ImageFormatInfo {
            image_type: ImageType::Dim2d,
            tiling: ImageTiling::Optimal,
            usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT,
            ..ImageFormatInfo::default()
        });
        println!("{:?}", image_format_properties);
        if image_format_properties.unwrap().is_none() {
            return;
        }
        self.depth_image_view = Self::create_depth_resources(
            &self.device,
            self.swapchain.format,
            self.swapchain.extent,
        );
        self.framebuffers = self
            .swapchain
            .create_framebuffers(&self.render_pass, &self.depth_image_view);
        self.should_recreate_swapchain = false;
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

    fn create_depth_resources(
        device: &Arc<Device>,
        depth_format: Format,
        extent: [u32; 2],
    ) -> Arc<ImageView> {
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
        let alloc = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let depth_image = Image::new(alloc, image_create_info, allocation_info).unwrap();
        let depth_image_view = ImageView::new(depth_image, image_view_create_info).unwrap();
        depth_image_view
    }
}
