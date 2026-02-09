mod full_screen_pass;

use std::{
    cell::{Ref, RefCell, RefMut},
    collections::HashMap,
    ops::Deref,
    rc::Rc,
    sync::Arc,
};

use egui_winit_vulkano::{
    egui,
    egui::{Color32, Frame},
};
use shader_slang::ComponentType;
use smallvec::smallvec;
use vulkano::sync::{AccessFlags, PipelineStages};
use vulkano::{
    Validated, ValidationError, VulkanError,
    command_buffer::{
        AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo,
        SubpassContents, SubpassEndInfo,
    },
    device::Device,
    format::{ClearValue, Format},
    image::{
        ImageAspects, ImageLayout, ImageUsage,
        sampler::{Sampler, SamplerCreateInfo},
        view::ImageView,
    },
    pipeline::{
        ComputePipeline, DynamicState, GraphicsPipeline, PipelineBindPoint,
        graphics::{
            subpass::PipelineSubpassType,
            viewport::{Scissor, Viewport},
        },
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    shader::{ShaderStages, spirv::bytes_to_words},
    swapchain::{SwapchainPresentInfo, present},
    sync::{GpuFuture, future::FenceSignalFuture},
};
use winit::dpi::PhysicalSize;

use crate::application::renderer::full_screen_pass::FullScreenPass;
use crate::application::{
    assets::asset_traits::{
        RHICameraInterface, RHIInterface, RHIModelInterface, RHIResource, RHISceneInterface,
        RendererInterface, Vertex,
    },
    rhi::{
        VKRHI,
        pipeline::{compute_pipeline, graphics_pipeline},
        render_pass::RenderPassBuilder,
        rhi_assets::{vulkan_material::VKMaterial, vulkan_scene::VKScene},
        shader_cursor::ShaderCursor,
        shader_object::{ShaderObject, ShaderObjectLayout},
        shaders::SlangCompiler,
        swapchain::Swapchain,
    },
};

pub struct MutableRenderState {
    swapchain: Swapchain,
    depth_image_view: Arc<ImageView>,
    color_render_target: Arc<ImageView>,
    pp_render_target: Arc<ImageView>,
    rt_framebuffer: Arc<Framebuffer>,
    should_recreate_swapchain: bool,
    in_flight_future: Option<FenceSignalFuture<Box<dyn GpuFuture>>>,
    fullscreen_pass: FullScreenPass,
}

pub struct VKRenderer {
    rhi: Rc<VKRHI>,
    mutable_state: RefCell<MutableRenderState>,
    render_pass: Arc<RenderPass>,
    material_compiler: RefCell<MaterialCompiler>,
    post_process: PostProcessPass,
}

struct MaterialCompiler {
    compiled_materials: HashMap<usize, CompiledMaterial>,
}

struct CompiledMaterial {
    pipeline: Arc<GraphicsPipeline>,
}

struct PostProcessPass {
    shader_object_layout: Arc<ShaderObjectLayout>,
    shader_object: ShaderObject,
    pipeline: Arc<ComputePipeline>,
    sampler: Arc<Sampler>,
}

impl VKRenderer {
    pub fn new(rhi: Rc<VKRHI>) -> Self {
        let swapchain = Swapchain::new(rhi.as_ref());
        let render_pass =
            RenderPassBuilder::build_default_render_pass(rhi.as_ref(), Format::R32G32B32A32_SFLOAT)
                .build();
        let depth_image_view = rhi.create_depth_buffer(swapchain.extent);

        let color_render_target = rhi.create_gbuffer(
            swapchain.extent,
            Format::R32G32B32A32_SFLOAT,
            ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
            ImageAspects::COLOR,
        );
        let pp_render_target = rhi.create_gbuffer(
            swapchain.extent,
            Format::R32G32B32A32_SFLOAT,
            ImageUsage::STORAGE | ImageUsage::SAMPLED | ImageUsage::INPUT_ATTACHMENT,
            ImageAspects::COLOR,
        );
        let rt_framebuffer = Framebuffer::new(
            render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![color_render_target.clone(), depth_image_view.clone()],
                extent: swapchain.extent,
                layers: 1,
                ..FramebufferCreateInfo::default()
            },
        )
        .unwrap();

        let post_process = PostProcessPass::new(rhi.slang_compiler(), rhi.as_ref());

        let global_cursor = ShaderCursor::new(&post_process.shader_object);
        let cursor = global_cursor.field("gComputeInput").unwrap();
        cursor
            .field("input")
            .unwrap()
            .write_image_view_sampler(color_render_target.clone(), post_process.sampler.clone());
        cursor.field("screenSize").unwrap().write(&swapchain.extent);
        cursor.field("exposureValue").unwrap().write(&1.0f32);
        cursor
            .field("result")
            .unwrap()
            .write_image_view(pp_render_target.clone());

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
            swapchain.image_view_iter().cloned(),
            swapchain.extent,
        );

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
            }),
            render_pass,
            material_compiler: RefCell::new(MaterialCompiler::new()),
            post_process,
        }
    }

    pub fn redraw(&self, scene: &VKScene) {
        self.mutable_state().in_flight_future = self.draw_frame(scene);
    }

    fn draw_frame(&self, scene: &VKScene) -> Option<FenceSignalFuture<Box<dyn GpuFuture>>> {
        if self.mutable_state_const().should_recreate_swapchain {
            self.mutable_state()
                .recreate_swapchain_internal(self.rhi.as_ref(), &self.render_pass);
        }
        self.mutable_state_const()
            .in_flight_future
            .as_ref()
            .map(|f| f.wait(None).unwrap());

        self.rhi.gui_mut().immediate_ui(|ui| {
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
            .rhi
            .command_buffer_interface()
            .primary_command_buffer(self.rhi.queue_family_indices().graphics_family);

        self.record_draw_command_buffer(&mut command_buffer, swapchain_image_index as usize, scene)
            .unwrap();

        let draw_finished_future = image_available_future
            .then_execute(
                self.rhi.queues().graphics_queue.clone(),
                command_buffer.build().unwrap(),
            )
            .unwrap();

        let mut compute_command_buffer = self
            .rhi
            .command_buffer_interface()
            .primary_command_buffer(self.rhi.queue_family_indices().compute_family);

        self.record_post_process_command_buffer(
            &mut compute_command_buffer,
            swapchain_image_index as usize,
        )
        .unwrap();

        let post_process_finished_future = draw_finished_future
            .then_execute(
                self.rhi.queues().compute_queue.clone(),
                compute_command_buffer.build().unwrap(),
            )
            .unwrap();

        let mut command_buffer = self
            .rhi
            .command_buffer_interface()
            .primary_command_buffer(self.rhi.queue_family_indices().graphics_family);

        self.mutable_state_const()
            .fullscreen_pass
            .record_command_buffer(
                &mut command_buffer,
                self.mutable_state_const().swapchain.extent,
                swapchain_image_index as usize,
            )
            .unwrap();

        let final_output_finished_future = post_process_finished_future
            .then_execute(
                self.rhi.queues().graphics_queue.clone(),
                command_buffer.build().unwrap(),
            )
            .unwrap();

        let gui_draw_future = self.rhi.gui_mut().draw_on_image(
            final_output_finished_future,
            self.mutable_state_const()
                .swapchain
                .image_view(swapchain_image_index as usize)
                .clone(),
        );

        let present_future = present(
            gui_draw_future,
            self.rhi.queues().present_queue.clone(),
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
                        self.mutable_state_const().rt_framebuffer.clone(),
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

        scene
            .models()
            .iter()
            .map(|model| {
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
                let ev = 1f32;
                view_cursor.field("exposureValue").unwrap().write(&ev);

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

        command_buffer
            .end_render_pass(SubpassEndInfo::default())
            .map(|_| ())
    }

    fn record_post_process_command_buffer(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_index: usize,
    ) -> Result<(), Box<ValidationError>> {
        let render_target = self
            .mutable_state()
            .swapchain
            .image_view(image_index)
            .clone();

        /*let global_cursor = ShaderCursor::new(&self.post_process.shader_object);
        let cursor = global_cursor.field("gComputeInput").unwrap();
        cursor
            .field("result")
            .unwrap()
            .write_image_view(render_target.clone());*/

        let extent = self.mutable_state().swapchain.extent;
        let groups = [extent[0] / 16, extent[1] / 16, 1];

        command_buffer
            .bind_pipeline_compute(self.post_process.pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.post_process
                    .shader_object_layout
                    .pipeline_layout()
                    .clone(),
                0,
                self.post_process.shader_object.descriptor_sets()[image_index].clone(),
            )?;
        unsafe { command_buffer.dispatch(groups) }?;
        /* command_buffer.copy_image(CopyImageInfo::images(
            self.mutable_state().pp_render_target.image().clone(),
            render_target.image().clone(),
        ))?;*/
        Ok(())
    }

    pub fn compile_materials(&self) {
        self.material_compiler
            .borrow_mut()
            .compile_materials(&self.rhi, &self.render_pass);
    }

    pub fn mutable_state_const(&self) -> Ref<MutableRenderState> {
        self.mutable_state.borrow()
    }
    pub fn mutable_state(&self) -> RefMut<MutableRenderState> {
        self.mutable_state.borrow_mut()
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

    fn recreate_swapchain_internal(&mut self, rhi: &VKRHI, render_pass: &Arc<RenderPass>) {
        //unsafe { self.device.wait_idle().unwrap() }
        if rhi.window().inner_size() == PhysicalSize::new(0, 0) {
            return;
        }
        self.swapchain = self.swapchain.recreate(
            &rhi.physical_device(),
            &rhi.surface(),
            &rhi.window(),
            &rhi.queue_family_indices(),
        );
        self.depth_image_view = rhi.create_depth_buffer(self.swapchain.extent);

        self.color_render_target = rhi.create_gbuffer(
            self.swapchain.extent,
            Format::R32G32B32A32_SFLOAT,
            ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
            ImageAspects::COLOR,
        );
        self.pp_render_target = rhi.create_gbuffer(
            self.swapchain.extent,
            Format::R32G32B32A32_SFLOAT,
            ImageUsage::STORAGE | ImageUsage::SAMPLED,
            ImageAspects::COLOR,
        );
        self.rt_framebuffer = Framebuffer::new(
            render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![
                    self.color_render_target.clone(),
                    self.depth_image_view.clone(),
                ],
                extent: self.swapchain.extent,
                layers: 1,
                ..FramebufferCreateInfo::default()
            },
        )
        .unwrap();

        self.fullscreen_pass.recreate_framebuffers(self.swapchain.image_view_iter().cloned(), self.swapchain.extent);
        self.should_recreate_swapchain = false;
    }
}

impl MaterialCompiler {
    fn new() -> Self {
        Self {
            compiled_materials: HashMap::new(),
        }
    }

    fn compile_material(
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
    }

    fn find_compiled_material(&self, material: &VKMaterial) -> Option<&CompiledMaterial> {
        self.compiled_materials.get(&material.uuid())
    }

    pub fn compile_materials(&mut self, rhi: &VKRHI, render_pass: &Arc<RenderPass>) {
        let resource_manager = rhi.resource_manager();
        self.compiled_materials = resource_manager
            .resource_iterator::<VKMaterial>()
            .unwrap()
            .map(|material| {
                (
                    material.uuid(),
                    self.compile_material(material, rhi.device(), render_pass),
                )
            })
            .collect();
    }
}

impl PostProcessPass {
    fn new(compiler: &SlangCompiler, rhi: &VKRHI) -> Self {
        let module = compiler
            .session()
            .load_module("Compute/compute_test")
            .unwrap();
        let entry = module.find_entry_point_by_name("compute_test").unwrap();
        let module_component: ComponentType = module.into();
        let composed = compiler
            .session()
            .create_composite_component_type(&[module_component, entry.into()])
            .unwrap();
        let linked = composed.link().unwrap();
        let spirv = linked.entry_point_code(0, 0).unwrap();

        let shader_object_layout =
            ShaderObjectLayout::new(linked, &[], rhi.device(), ShaderStages::COMPUTE);
        let shader_object = ShaderObject::new(
            shader_object_layout.clone(),
            rhi.descriptor_allocator(),
            rhi.buffer_allocator(),
            rhi.in_flight_frames() as u32,
        );

        let pipeline = compute_pipeline()
            .shader(
                rhi.device().clone(),
                bytes_to_words(spirv.as_slice()).unwrap().deref(),
            )
            .build_pipeline(
                rhi.device().clone(),
                shader_object_layout.pipeline_layout().clone(),
            );

        let sampler = Sampler::new(
            rhi.device().clone(),
            SamplerCreateInfo::simple_repeat_linear_no_mipmap(),
        )
        .unwrap();

        Self {
            shader_object_layout,
            shader_object,
            pipeline,
            sampler,
        }
    }
}
