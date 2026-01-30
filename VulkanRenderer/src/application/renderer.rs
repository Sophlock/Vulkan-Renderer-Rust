mod buffer;
mod command_buffer;
mod layers;
mod physical_device;
mod pipeline;
mod queue;
mod render_pass;
mod render_sync;
pub mod rhi_assets;
mod shader_cursor;
mod shader_object;
mod shaders;
mod swapchain;

use std::{
    cell::{Ref, RefCell, RefMut},
    ops::Deref,
    rc::Rc,
    sync::Arc,
};

use asset_system::resource_management::ResourceManager;
use command_buffer::CommandBufferInterface;
use egui_winit_vulkano::{
    egui, egui::{Color32, Frame}, Gui,
    GuiConfig,
};
use physical_device::find_depth_format;
use queue::{QueueCollection, QueueFamilyIndices};
use render_pass::RenderPassBuilder;
use rhi_assets::{vulkan_mesh::VKMesh, vulkan_texture::VKTexture};
use smallvec::smallvec;
use swapchain::Swapchain;
use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo,
        SubpassContents, SubpassEndInfo,
    }, descriptor_set::allocator::{
        DescriptorSetAllocator, StandardDescriptorSetAllocator,
        StandardDescriptorSetAllocatorCreateInfo,
    }, device::{
        physical::PhysicalDevice, Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures,
    }, format::{ClearValue, Format},
    image::{
        view::{ImageView, ImageViewCreateInfo, ImageViewType}, Image, ImageAspects, ImageCreateInfo, ImageLayout, ImageSubresourceRange,
        ImageTiling, ImageType, ImageUsage,
        SampleCount,
    },
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
    },
    pipeline::{
        graphics::viewport::{Scissor, Viewport},
        PipelineBindPoint,
    },
    render_pass::{Framebuffer, RenderPass},
    swapchain::{present, Surface, SwapchainPresentInfo},
    sync::{future::FenceSignalFuture, GpuFuture, Sharing},
    Validated,
    ValidationError,
    VulkanError,
    VulkanLibrary,
};
use winit::{dpi::PhysicalSize, event_loop::ActiveEventLoop, window::Window};

use super::assets::asset_traits::{
    RHICameraInterface, RHIInterface, RHIModelInterface, RHISceneInterface,
};
use crate::application::renderer::{
    rhi_assets::{
        vulkan_camera::VKCamera, vulkan_material::VKMaterial, vulkan_material_instance::VKMaterialInstance,
        vulkan_model::VKModel, vulkan_scene::VKScene, RHIResourceManager,
    },
    shaders::SlangCompiler,
};

pub struct MutableRenderState {
    swapchain: Swapchain,
    depth_image_view: Arc<ImageView>,
    framebuffers: Vec<Arc<Framebuffer>>,
    should_recreate_swapchain: bool,
    in_flight_future: Option<FenceSignalFuture<Box<dyn GpuFuture>>>,
}

pub struct Renderer {
    frames_in_flight: usize,
    window: Arc<Window>,
    instance: Arc<Instance>,
    surface: Arc<Surface>,
    physical_device: Arc<PhysicalDevice>,
    device: Arc<Device>,
    queues: QueueCollection,
    queue_family_indices: QueueFamilyIndices,
    mutable_state: RefCell<MutableRenderState>,
    render_pass: Arc<RenderPass>,
    command_buffer_interface: CommandBufferInterface,
    gui: RefCell<Gui>,
    slang_compiler: SlangCompiler,
    buffer_allocator: Arc<dyn MemoryAllocator>,
    descriptor_allocator: Arc<dyn DescriptorSetAllocator>,
    resource_manager: RefCell<RHIResourceManager>,
}

impl Renderer {
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
        let gui = RefCell::new(Gui::new(
            event_loop,
            surface.clone(),
            queues.graphics_queue.clone(),
            swapchain.format,
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
            mutable_state: RefCell::new(MutableRenderState {
                swapchain,
                depth_image_view,
                framebuffers,
                should_recreate_swapchain: false,
                in_flight_future: None,
            }),
            render_pass,
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

    pub fn redraw(&self, scene: &VKScene) {
        self.mutable_state().in_flight_future = self.draw_frame(scene);
        self.window.as_ref().request_redraw();
    }

    fn draw_frame(&self, scene: &VKScene) -> Option<FenceSignalFuture<Box<dyn GpuFuture>>> {
        if self.mutable_state_const().should_recreate_swapchain {
            self.mutable_state().recreate_swapchain_internal(self);
        }
        self.mutable_state_const()
            .in_flight_future
            .as_ref()
            .map(|f| f.wait(None).unwrap());

        self.gui_mut().immediate_ui(|ui| {
            let ctx = ui.context();
            egui::CentralPanel::default()
                .frame(Frame::default().fill(Color32::TRANSPARENT))
                .show(&ctx, |ui| {
                    ui.heading("My egui Application");
                    ui.horizontal(|ui| {
                        ui.label("Your name: ");
                        //ui.text_edit_singleline(&mut name);
                    });
                    //ui.add(egui::Slider::new(&mut age, 0..=120).text("age"));
                    if ui.button("Increment").clicked() {
                        // age += 1;
                    }
                    //ui.label(format!("Hello '{name}', age {age}"));
                });
        });

        let acquire_image_result = self.mutable_state_const().swapchain.acquire_next_image();
        let (swapchain_image_index, suboptimal, image_available_future) = acquire_image_result
            .map_or_else(
                |e| match e {
                    Validated::Error(VulkanError::OutOfDate) => {
                        self.mutable_state().request_recreate_swapchain();
                        None
                    }
                    _ => panic!("Error acquiring swapchain image"),
                },
                |v| Some(v),
            )?;
        if suboptimal {
            self.mutable_state().request_recreate_swapchain();
        }
        let mut command_buffer = self
            .command_buffer_interface
            .primary_command_buffer(self.queue_family_indices.graphics_family);

        self.record_draw_command_buffer(&mut command_buffer, swapchain_image_index as usize, scene)
            .unwrap();

        let draw_finished_future = image_available_future
            .then_execute(
                self.queues.graphics_queue.clone(),
                command_buffer.build().unwrap(),
            )
            .unwrap();

        let gui_draw_future = self.gui_mut().draw_on_image(
            draw_finished_future,
            self.mutable_state_const()
                .swapchain
                .image_view(swapchain_image_index as usize)
                .clone(),
        );

        let present_future = present(
            gui_draw_future,
            self.queues.present_queue.clone(),
            SwapchainPresentInfo::swapchain_image_index(
                self.mutable_state_const().swapchain.raw().clone(),
                swapchain_image_index,
            ),
        );

        let in_flight_future = present_future
            .boxed()
            .then_signal_fence_and_flush()
            .map_or_else(
                |e| match e {
                    Validated::Error(VulkanError::OutOfDate) => {
                        self.mutable_state().request_recreate_swapchain();
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
        image_index: usize,
        scene: &VKScene,
    ) -> Result<(), Box<ValidationError>> {
        command_buffer
            .begin_render_pass(
                RenderPassBeginInfo {
                    render_area_offset: [0, 0],
                    render_area_extent: self.mutable_state_const().swapchain.extent,
                    clear_values: vec![
                        Some(ClearValue::Float([0.0, 0.0, 0.0, 1.0])),
                        Some(ClearValue::DepthStencil((1.0, 0))),
                    ],
                    render_pass: self.render_pass.clone(),
                    ..RenderPassBeginInfo::framebuffer(
                        self.mutable_state_const().framebuffers[image_index].clone(),
                    )
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..SubpassBeginInfo::default()
                },
            )?
            .set_viewport_with_count(smallvec![Viewport {
                offset: [0., 0.],
                extent: self
                    .mutable_state_const()
                    .swapchain
                    .extent
                    .map(|u| u as f32),
                depth_range: 0.0f32..=1.0f32,
            }])?
            .set_scissor_with_count(smallvec![Scissor {
                offset: [0, 0],
                extent: self.mutable_state_const().swapchain.extent,
            }])?;

        let resources = self.resource_manager();
        let rcs = resources.deref();

        scene
            .models()
            .iter()
            .map(|model| {
                let material_instance = model.material().get(rcs).unwrap();
                let material = material_instance.material().get(rcs).unwrap();
                let mesh = model.mesh().get(rcs).unwrap();

                let cursor = material_instance.shader_cursor();
                let model_cursor = cursor.field("gModelData").unwrap();
                model_cursor
                    .field("modelTransform")
                    .unwrap()
                    .write(&model.transform());
                model_cursor
                    .field("inverseTransposeModelTransform")
                    .unwrap()
                    .write(&model.transform().transpose().inverse());

                let view_cursor = cursor.field("gViewData").unwrap();
                view_cursor
                    .field("viewPosition")
                    .unwrap()
                    .write(&scene.camera().location());
                view_cursor
                    .field("viewProjection")
                    .unwrap()
                    .write(&scene.camera().view_projection());
                let ev = 1f32;
                view_cursor.field("exposureValue").unwrap().write(&ev);

                command_buffer
                    .bind_pipeline_graphics(material.pipeline().clone())?
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        material.pipeline_layout().clone(),
                        0,
                        material_instance.descriptor_sets()[image_index].clone(),
                    )?
                    .bind_vertex_buffers(0, mesh.vertex().clone())?
                    .bind_index_buffer(mesh.index().reinterpret_ref::<[u32]>().clone())?;
                unsafe { command_buffer.draw_indexed(mesh.index().len() as u32, 1, 0, 0, 0) }
                    .map(|_| ())
            })
            .reduce(Result::or)
            .unwrap_or(Ok(()))?;

        command_buffer
            .end_render_pass(SubpassEndInfo::default())
            .map(|_| ())
    }

    fn create_window(event_loop: &ActiveEventLoop) -> Arc<Window> {
        let window_attributes = Window::default_attributes()
            .with_title("Vulkan renderer")
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

    pub fn gui_mut(&self) -> RefMut<Gui> {
        self.gui.borrow_mut()
    }

    pub fn mutable_state_const(&self) -> Ref<MutableRenderState> {
        self.mutable_state.borrow()
    }
    pub fn mutable_state(&self) -> RefMut<MutableRenderState> {
        self.mutable_state.borrow_mut()
    }

    pub fn window(&self) -> &Window {
        self.window.as_ref()
    }
}

impl RHIInterface for Renderer {
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

impl MutableRenderState {
    pub fn request_recreate_swapchain(&mut self) {
        self.should_recreate_swapchain = true;
    }

    fn recreate_swapchain_internal(&mut self, rhi: &Renderer) {
        //unsafe { self.device.wait_idle().unwrap() }
        if rhi.window.inner_size() == PhysicalSize::new(0, 0) {
            return;
        }
        self.swapchain = self.swapchain.recreate(
            &rhi.physical_device,
            &rhi.surface,
            &rhi.window,
            &rhi.queue_family_indices,
        );
        self.depth_image_view = Renderer::create_depth_resources(
            &rhi.device,
            find_depth_format(&rhi.physical_device),
            self.swapchain.extent,
        );
        self.framebuffers = self
            .swapchain
            .create_framebuffers(&rhi.render_pass, &self.depth_image_view);
        self.should_recreate_swapchain = false;
    }
}
