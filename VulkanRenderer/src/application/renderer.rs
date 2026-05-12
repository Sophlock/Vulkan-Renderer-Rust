mod full_screen_pass;
mod post_processing;
pub mod profiling;
mod visibility_buffer_data;
mod visibility_buffer_generation;
mod visibility_buffer_shading;

use std::{
    cell::{Ref, RefCell, RefMut},
    collections::HashMap,
    rc::Rc,
    sync::{Arc, RwLock},
};

use vulkano::{
    Validated, VulkanError,
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    format::Format,
    image::{ImageAspects, ImageLayout, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::GraphicsPipeline,
    render_pass::RenderPass,
    swapchain::{SwapchainPresentInfo, present},
    sync::{AccessFlags, GpuFuture, PipelineStages, future::FenceSignalFuture},
};
use winit::dpi::PhysicalSize;

use crate::application::{
    assets::asset_traits::{RHICameraInterface, RHIResource, RHISceneInterface, RendererInterface},
    renderer::{
        full_screen_pass::FullScreenPass,
        post_processing::{PostProcessPass, PostProcessSettings},
        profiling::{Profiler, ProfilerStage},
        visibility_buffer_data::{MutatingData, VisibilityBufferData, VisibilityBufferGlobalData},
        visibility_buffer_generation::{
            VisibilityBufferProcessingPass, VisibilityBufferRasterizer,
        },
        visibility_buffer_shading::VisibilityBufferShadePass,
    },
    rhi::{
        VKRHI,
        render_pass::RenderPassBuilder,
        rhi_assets::{vulkan_material::VKMaterial, vulkan_scene::VKScene},
        swapchain::Swapchain,
        swapchain_resources::{
            SwapchainFramebuffer, SwapchainFramebufferCreateInfo, SwapchainImage,
        },
    },
};

/// Data inside the renderer that may be mutated.
/// Some of the members do not need to be mutable anymore but are left in here for now to avoid breaking changes.
pub struct MutableRenderState {
    /// The swapchain (i.e., the source of presentable images)
    swapchain: Swapchain,
    depth_image_view: Arc<RwLock<SwapchainImage>>,
    /// The render target where the raw render output is drawn to
    color_render_target: Arc<RwLock<SwapchainImage>>,
    /// The render target that post processing renders into
    pp_render_target: Arc<RwLock<SwapchainImage>>,
    rt_framebuffer: Arc<RwLock<SwapchainFramebuffer>>,
    /// True if a swapchain recreation request was queued
    should_recreate_swapchain: bool,
    /// The future that represents the previous frame being presented
    in_flight_future: Option<FenceSignalFuture<Box<dyn GpuFuture>>>,
    
    /// Fullscreen rectangle pass to transfer the rendered image from compute to graphics.
    /// This is needed because it may not always be possible to write to the swapchain images from compute
    fullscreen_pass: FullScreenPass,
    
    /// Rasterization step of the Visibility Buffer
    vis_buffer_rasterizer: VisibilityBufferRasterizer,
    /// Processing step of the Visibility Buffer
    vis_buffer_processing: VisibilityBufferProcessingPass,
    /// All data needed for the Visibility Buffer
    vis_buffer_data: Arc<VisibilityBufferData>,
    /// Shading step of the Visibility Buffer
    vis_buffer_shade: VisibilityBufferShadePass,
    /// A buffer of data that changes often, e.g., the camera data.
    mutating_data: Subbuffer<MutatingData>,
}

/// Renderer struct that is responsible for drawing scenes
pub struct VKRenderer {
    /// Render Hardware Interface
    rhi: Rc<VKRHI>,
    /// Internally mutable data
    mutable_state: RefCell<MutableRenderState>,
    //render_pass: Arc<RenderPass>,
    /// Material compiler for forward rendering. Deprecated
    material_compiler: RefCell<MaterialCompiler>,
    /// Post processing pass
    post_process: PostProcessPass,
    /// Profiler for measuring GPU times
    profiler: Profiler,
    /// System to record data about the scene (e.g., number of visible materials)
    scene_statistics: RefCell<SceneStatistics>,
}

/// Material compiler for forward rendering. Not currently used and to be considered deprecated.
struct MaterialCompiler {
    compiled_materials: HashMap<usize, CompiledMaterial>,
}

/// A compiled material for forward rendering. Not currently used and to be considered deprecated.
struct CompiledMaterial {
    pipeline: Arc<GraphicsPipeline>,
}

impl VKRenderer {
    pub fn new(rhi: Rc<VKRHI>) -> Self {
        let swapchain = Swapchain::new(rhi.as_ref());
        let render_pass =
            RenderPassBuilder::build_default_render_pass(rhi.as_ref(), Format::R32G32B32A32_SFLOAT)
                .build();
        let depth_image_view = swapchain.create_depth_buffer(rhi.as_ref());

        let color_render_target = swapchain.create_gbuffer(
            rhi.as_ref(),
            Format::R32G32B32A32_SFLOAT,
            ImageUsage::COLOR_ATTACHMENT
                | ImageUsage::SAMPLED
                | ImageUsage::STORAGE
                | ImageUsage::TRANSFER_DST,
            ImageAspects::COLOR,
        );
        let pp_render_target = swapchain.create_gbuffer(
            rhi.as_ref(),
            Format::R32G32B32A32_SFLOAT,
            ImageUsage::STORAGE | ImageUsage::SAMPLED,
            ImageAspects::COLOR,
        );
        let rt_framebuffer = swapchain.create_framebuffer(
            render_pass.clone(),
            SwapchainFramebufferCreateInfo {
                attachments: vec![color_render_target.clone(), depth_image_view.clone()],
                layers: 1,
                ..SwapchainFramebufferCreateInfo::default()
            },
        );

        let post_process = PostProcessPass::new(
            rhi.slang_compiler(),
            rhi.as_ref(),
            color_render_target.clone(),
            pp_render_target.clone(),
        );

        let fullscreen_pass = FullScreenPass::new(
            rhi.as_ref(),
            Format::R32G32B32A32_SFLOAT,
            swapchain.format,
            ImageLayout::PresentSrc,
            PipelineStages::COLOR_ATTACHMENT_OUTPUT,
            AccessFlags::COLOR_ATTACHMENT_WRITE,
            ImageLayout::ShaderReadOnlyOptimal,
            None,
            pp_render_target.clone(),
            &swapchain,
        );

        let mutating_data = Buffer::new_sized(
            rhi.buffer_allocator().clone(),
            BufferCreateInfo {
                usage: BufferUsage::SHADER_DEVICE_ADDRESS
                    | BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_DST,
                ..BufferCreateInfo::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..AllocationCreateInfo::default()
            },
        )
        .unwrap();
        let vis_buffer_global_data =
            VisibilityBufferGlobalData::new(rhi.as_ref(), mutating_data.clone());
        let vis_buffer_data = Arc::new(VisibilityBufferData::new(
            rhi.as_ref(),
            &swapchain,
            VisibilityBufferShadePass::MAX_SEQUENCE_COUNT,
            vis_buffer_global_data,
            color_render_target.clone(),
        ));
        let vis_buffer_rasterizer =
            VisibilityBufferRasterizer::new(rhi.clone(), &swapchain, vis_buffer_data.as_ref());
        let vis_buffer_processing =
            VisibilityBufferProcessingPass::new(rhi.as_ref(), &vis_buffer_data);
        let vis_buffer_shade = VisibilityBufferShadePass::new(rhi.clone(), vis_buffer_data.clone());

        let profiler = Profiler::new(rhi.device().clone());

        Self {
            rhi,
            mutable_state: RefCell::new(MutableRenderState {
                swapchain,
                depth_image_view,
                color_render_target,
                pp_render_target,
                rt_framebuffer,
                should_recreate_swapchain: false,
                in_flight_future: None,
                fullscreen_pass,
                vis_buffer_rasterizer,
                vis_buffer_processing,
                vis_buffer_data,
                vis_buffer_shade,
                mutating_data,
            }),
            //render_pass,
            material_compiler: RefCell::new(MaterialCompiler::new()),
            post_process,
            profiler,
            scene_statistics: RefCell::new(SceneStatistics::default()),
        }
    }

    /// Draw and present a new frame. If needed, waits for the previous frame to be finished
    pub fn redraw(&self, scene: &VKScene) {
        // Essentially just delegates to draw_frame and updates the in flight future
        self.mutable_state().in_flight_future = self.draw_frame(scene);
    }

    /// Perform drawing and present operations. Returns a future representing a successful present or None in case of any errors
    fn draw_frame(&self, scene: &VKScene) -> Option<FenceSignalFuture<Box<dyn GpuFuture>>> {
        // Wait for the previous frame to finish
        self.mutable_state_const()
            .in_flight_future
            .as_ref()
            .map(|f| f.wait(None).unwrap());

        // Update profiler measurements on CPU and reset query pool
        self.profiler.read_results();

        // Update scene statistics
        self.update_scene_statistics();

        // Recreate swapchain if needed
        if self.mutable_state_const().should_recreate_swapchain {
            self.mutable_state()
                .recreate_swapchain_internal(self.rhi.as_ref());
        }

        // Update camera matrix and screen data
        self.update_mutating_data(scene);

        // Acquire swapchain image
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
        // Request a recreation for the next frame if the swapchain is not optimal
        if suboptimal {
            self.mutable_state().request_recreate_swapchain();
        }

        let swapchain_extent = self.mutable_state_const().swapchain.extent;

        // Render the scene using the visibility buffer setup
        let draw_finished_future = self.draw_frame_visbuffer(
            scene,
            swapchain_image_index,
            swapchain_extent,
            image_available_future,
        );

        // Command buffer for post-processing
        let mut compute_command_buffer = self
            .rhi
            .command_buffer_interface()
            .primary_command_buffer(self.rhi.queue_family_indices().compute_family);

        // Do post-processing
        self.post_process
            .record_command_buffer(
                &mut compute_command_buffer,
                swapchain_image_index as usize,
                swapchain_extent,
            )
            .unwrap();

        // Submit to compute queue
        let post_process_finished_future = draw_finished_future
            .then_signal_semaphore()
            .then_execute(
                self.rhi.queues().compute_queue.clone(),
                compute_command_buffer.build().unwrap(),
            )
            .unwrap();

        // Command buffer for drawing back into the swapchain
        let mut command_buffer = self
            .rhi
            .command_buffer_interface()
            .primary_command_buffer(self.rhi.queue_family_indices().graphics_family);

        // Fullscreen pass to render image into the swapchain image
        self.mutable_state_const()
            .fullscreen_pass
            .record_command_buffer(
                &mut command_buffer,
                swapchain_extent,
                swapchain_image_index as usize,
            )
            .unwrap();

        // Submit to graphics queue
        let final_output_finished_future = post_process_finished_future
            .then_signal_semaphore()
            .then_execute(
                self.rhi.queues().graphics_queue.clone(),
                command_buffer.build().unwrap(),
            )
            .unwrap();

        // Render UI onto swapchain image
        let gui_draw_future = self.rhi.gui_mut().draw_on_image(
            final_output_finished_future,
            self.mutable_state_const()
                .swapchain
                .image_view(swapchain_image_index as usize)
                .read()
                .unwrap()
                .image_view()
                .clone(),
        );

        // Present
        let present_future = present(
            gui_draw_future,
            self.rhi.queues().present_queue.clone(),
            SwapchainPresentInfo::swapchain_image_index(
                self.mutable_state_const().swapchain.raw().clone(),
                swapchain_image_index,
            ),
        );

        // Return the future representing the end of the frame
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

    fn draw_frame_forward(
        &self,
        scene: &VKScene,
        image_index: usize,
        before_future: impl GpuFuture,
    ) /*-> Result<impl GpuFuture, Box<ValidationError>>*/
    {
        // Command buffer for a graphics queue. This will do all the rasterization work
        /*let mut command_buffer = self
            .rhi
            .command_buffer_interface()
            .primary_command_buffer(self.rhi.queue_family_indices().graphics_family);

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
                        self.mutable_state_const()
                            .rt_framebuffer
                            .read()
                            .unwrap()
                            .framebuffer()
                            .clone(),
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

        let resources = self.rhi.resource_manager();
        let compiler = self.material_compiler.borrow();
        let rcs = resources.deref();

        let data = &self.mutable_state_const().vis_buffer_data;

        command_buffer
            .bind_vertex_buffers(0, data.global_data.vertices.clone())?
            .bind_vertex_buffers(1, data.global_data.instances.clone())?
            .bind_index_buffer(data.global_data.indices.clone())?;

        let mut sorted = scene
            .models()
            .iter()
            .map(|model| {
                let material_instance = model.get(rcs).unwrap().material();
                let material = material_instance.get(rcs).unwrap().material();
                let mesh = model.get(rcs).unwrap().mesh();
                (model, material_instance, material, mesh)
            })
            .collect::<Vec<_>>();

        sorted.sort_unstable_by_key(|(_, material_instance, material, mesh)| {
            (material.id(), material_instance.id(), mesh.id())
        });

        sorted
            .chunk_by(|(_, _, material1, _), (_, _, material2, _)| material1.id() == material2.id())
            .into_iter()
            .for_each(|models| {
                let material = models[0].2.get(rcs).unwrap();
                let compiled_material = compiler.find_compiled_material(material).unwrap();

                command_buffer
                    .bind_pipeline_graphics(compiled_material.pipeline.clone())
                    .unwrap();

                models
                    .chunk_by(|(_, instance1, _, _), (_, instance2, _, _)| {
                        instance1.id() == instance2.id()
                    })
                    .for_each(|models| {
                        let material_instance = models[0].1.get(rcs).unwrap();

                        command_buffer.bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            material.pipeline_layout().clone(),
                            0,
                            material_instance.descriptor_sets()[image_index].clone(),
                        ).unwrap();

                        models.chunk_by(|(_, _, _, mesh1), (_, _, _, mesh2)| mesh1.id() == mesh2.id()).for_each(|models| {

                        });
                    })
            });

        scene
            .models()
            .iter()
            .map(|model_handle| {
                let model = model_handle.get(rcs).unwrap();
                let material_instance = model.material().get(rcs).unwrap();
                let material = material_instance.material().get(rcs).unwrap();
                let compiled_material = compiler.find_compiled_material(material).unwrap();
                let mesh = model.mesh().get(rcs).unwrap();

                let cursor = material_instance.shader_cursor();
                let model_cursor = cursor.field("gModelData").unwrap();
                model_cursor
                    .field("modelTransform")
                    .unwrap()
                    .write(model.transform().as_ref());
                model_cursor
                    .field("inverseTransposeModelTransform")
                    .unwrap()
                    .write(model.transform().transpose().inverse().as_ref());

                let view_cursor = cursor.field("gViewData").unwrap();
                view_cursor
                    .field("viewPosition")
                    .unwrap()
                    .write(scene.camera().location().as_ref());
                view_cursor
                    .field("viewProjection")
                    .unwrap()
                    .write(scene.camera().view_projection().as_ref());

                command_buffer
                    .bind_pipeline_graphics(compiled_material.pipeline.clone())?
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

        command_buffer.end_render_pass(SubpassEndInfo::default())?;

        let draw_finished_future = before_future
            .then_execute(
                self.rhi.queues().graphics_queue.clone(),
                command_buffer.build().unwrap(),
            )
            .unwrap();

        Ok(draw_finished_future)*/
        unimplemented!()
    }

    /// Render the scene using the visibility buffer setup
    /// Returns a future representing the finished rendering operation
    /// Everything will be executed after before_future.
    /// Note that this will transition from before_future's queue into the graphics queue and return a future on the compute queue
    fn draw_frame_visbuffer(
        &self,
        scene: &VKScene,
        swapchain_image_index: u32,
        swapchain_extent: [u32; 2],
        before_future: impl GpuFuture + 'static,
    ) -> impl GpuFuture + 'static {
        // Command buffer for a graphics queue. This will do all the rasterization work
        let mut command_buffer = self
            .rhi
            .command_buffer_interface()
            .primary_command_buffer(self.rhi.queue_family_indices().graphics_family);

        // Flush any pending writes to shader parameters
        self.rhi
            .as_ref()
            .shader_object_update_queue()
            .borrow_mut()
            .flush_writes(&mut command_buffer);

        // Rasterize the visibility buffer
        self.profiler
            .write(&mut command_buffer, ProfilerStage::PreVisbufferRaster)
            .unwrap();
        self.mutable_state_const()
            .vis_buffer_rasterizer
            .record_command_buffer(
                &mut command_buffer,
                swapchain_image_index as usize,
                swapchain_extent,
                scene,
                &self.mutable_state_const().vis_buffer_data,
            )
            .unwrap();
        self.profiler
            .write(&mut command_buffer, ProfilerStage::PostVisbufferRaster)
            .unwrap();

        // Submit to graphics queue
        let vis_buffer_generated_future = before_future
            .then_execute(
                self.rhi.queues().graphics_queue.clone(),
                command_buffer.build().unwrap(),
            )
            .unwrap();

        // Command buffer for compute queue. This will perform processing on the visibility buffer
        // and execute device generated commands to shade the final image.
        let mut compute_command_buffer = self
            .rhi
            .command_buffer_interface()
            .primary_command_buffer(self.rhi.queue_family_indices().compute_family);

        // Clear outdated visibility buffer data
        self.mutable_state_const()
            .vis_buffer_data
            .clear(&mut compute_command_buffer)
            .unwrap();

        self.profiler
            .write(
                &mut compute_command_buffer,
                ProfilerStage::PreVisbufferProcess,
            )
            .unwrap();

        // Perform visibility buffer processing
        self.mutable_state_const()
            .vis_buffer_processing
            .record_command_buffer(
                &mut compute_command_buffer,
                swapchain_image_index as usize,
                swapchain_extent,
                self.profiler(),
            )
            .unwrap();

        self.profiler
            .write(
                &mut compute_command_buffer,
                ProfilerStage::PostVisbufferProcess,
            )
            .unwrap();

        // Shade the visibility buffer
        self.mutable_state_const()
            .vis_buffer_shade
            .record_command_buffer(&mut compute_command_buffer, swapchain_image_index as usize)
            .unwrap();

        self.profiler
            .write(
                &mut compute_command_buffer,
                ProfilerStage::PostVisbufferShade,
            )
            .unwrap();

        // Submit to compute queue
        let draw_finished_future = vis_buffer_generated_future
            .then_execute(
                self.rhi.queues().compute_queue.clone(),
                compute_command_buffer.build().unwrap(),
            )
            .unwrap();

        draw_finished_future
    }

    fn update_scene_statistics(&self) {
        let mut_state = self.mutable_state_const();
        let data = &mut_state.vis_buffer_data;

        let mut statistics = self.scene_statistics.borrow_mut();

        statistics.visible_materials = *data.drawn_index_counter_buffer.read().unwrap();
        statistics.culled_materials = *data.culled_index_counter_buffer.read().unwrap();
        statistics.drawn_materials = *data.final_material_count_buffer.read().unwrap();
        statistics.fallback_pixels = *data.offset_accumulator_buffer.read().unwrap()
            - *data.no_fallback_texel_count_buffer.read().unwrap();
    }

    pub fn compile_materials(&self) {
        self.material_compiler
            .borrow_mut()
            .compile_materials(&self.rhi);
    }

    pub fn mutable_state_const(&self) -> Ref<MutableRenderState> {
        self.mutable_state.borrow()
    }
    pub fn mutable_state(&self) -> RefMut<MutableRenderState> {
        self.mutable_state.borrow_mut()
    }

    /// Update camera and screen data
    fn update_mutating_data(&self, scene: &VKScene) {
        let data = MutatingData {
            screen_size: self.mutable_state_const().swapchain.extent,
            view_matrix: scene.camera().view_projection().to_cols_array_2d(),
            view_position: scene.camera().location().into(),
        };
        let state = self.mutable_state_const();
        let mut write = state.mutating_data.write().unwrap();
        write.screen_size = data.screen_size;
        write.view_matrix = data.view_matrix;
        write.view_position = data.view_position;
    }

    pub fn post_process_settings(&self) -> RefMut<PostProcessSettings> {
        self.post_process.settings_mut()
    }

    pub fn profiler(&self) -> &Profiler {
        &self.profiler
    }

    pub fn scene_statistics(&self) -> Ref<SceneStatistics> {
        self.scene_statistics.borrow()
    }

    pub fn swapchain_extent(&self) -> [u32; 2] {
        self.mutable_state_const().swapchain.extent
    }
}
impl RendererInterface for VKRenderer {
    type RHI = VKRHI;
    fn rhi(&self) -> &VKRHI {
        &self.rhi
    }
}

impl MutableRenderState {
    pub fn request_recreate_swapchain(&mut self) {
        self.should_recreate_swapchain = true;
    }

    fn recreate_swapchain_internal(&mut self, rhi: &VKRHI) {
        //unsafe { self.device.wait_idle().unwrap() }
        if rhi.window().inner_size() == PhysicalSize::new(0, 0) {
            return;
        }
        self.swapchain.recreate(
            &rhi.physical_device(),
            &rhi.surface(),
            &rhi.window(),
            &rhi.queue_family_indices(),
        );

        self.should_recreate_swapchain = false;
    }
}

#[derive(Default)]
pub struct SceneStatistics {
    pub visible_materials: u32,
    pub drawn_materials: u32,
    pub culled_materials: u32,
    pub fallback_pixels: u32,
}

impl MaterialCompiler {
    fn new() -> Self {
        Self {
            compiled_materials: HashMap::new(),
        }
    }

    /*fn compile_material(
        &self,
        material: &VKMaterial,
        device: &Arc<Device>,
        render_pass: &Arc<RenderPass>,
    ) -> CompiledMaterial {
        let pipeline = graphics_pipeline()
            .input_assembly(None, None)
            .vertex_shader(
                device.clone(),
                bytes_to_words(material.vert_spirv().as_slice())
                    .unwrap()
                    .deref(),
            )
            .vertex_input::<Vertex>()
            .rasterizer(None, None, None, None, None, None)
            .skip_multisample()
            .fragment_shader(
                device.clone(),
                bytes_to_words(material.frag_spirv().as_slice())
                    .unwrap()
                    .deref(),
            )
            .opaque_color_blend()
            .default_depth_test()
            .build_pipeline(
                device.clone(),
                material.shader_object_layout().pipeline_layout().clone(),
                PipelineSubpassType::BeginRenderPass(render_pass.clone().first_subpass()),
                [
                    DynamicState::ViewportWithCount,
                    DynamicState::ScissorWithCount,
                ]
                .into(),
            );
        CompiledMaterial { pipeline }
    }*/

    fn find_compiled_material(&self, material: &VKMaterial) -> Option<&CompiledMaterial> {
        self.compiled_materials.get(&material.uuid())
    }

    pub fn compile_materials(&mut self, rhi: &VKRHI/*, render_pass: &Arc<RenderPass>*/) {
        /*let resource_manager = rhi.resource_manager();
        self.compiled_materials = resource_manager
            .resource_iterator::<VKMaterial>()
            .unwrap()
            .enumerate()
            .map(|(i, material)| {
                if i % 10 == 0 {
                    println!("Created {} ordinary materials", i + 1);
                }
                (
                    material.uuid(),
                    self.compile_material(material, rhi.device(), render_pass),
                )
            })
            .collect();*/
    }
}
