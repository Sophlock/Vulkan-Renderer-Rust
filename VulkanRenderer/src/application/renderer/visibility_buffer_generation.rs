use crate::application::assets::asset_traits::{
    Instance, RHICameraInterface, RHIInterface, RHIModelInterface, RHISceneInterface, Vertex,
};
use crate::application::rhi::VKRHI;
use crate::application::rhi::buffer::buffer_from_slice;
use crate::application::rhi::pipeline::{compute_pipeline, graphics_pipeline};
use crate::application::rhi::render_pass::RenderPassBuilder;
use crate::application::rhi::rhi_assets::vulkan_scene::VKScene;
use crate::application::rhi::shader_cursor::ShaderCursor;
use crate::application::rhi::shader_object::{ShaderObject, ShaderObjectLayout};
use smallvec::smallvec;
use std::mem::offset_of;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::Arc;
use vulkano::ValidationError;
use vulkano::buffer::{BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo,
    SubpassContents, SubpassEndInfo,
};
use vulkano::format::{ClearValue, Format};
use vulkano::image::view::ImageView;
use vulkano::image::{ImageAspects, ImageUsage};
use vulkano::memory::allocator::MemoryTypeFilter;
use vulkano::pipeline::graphics::subpass::PipelineSubpassType;
use vulkano::pipeline::graphics::vertex_input::{
    VertexBufferDescription, VertexInputRate, VertexMemberInfo,
};
use vulkano::pipeline::graphics::viewport::{Scissor, Viewport};
use vulkano::pipeline::{ComputePipeline, DynamicState, GraphicsPipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};
use vulkano::shader::ShaderStages;
use vulkano::shader::spirv::bytes_to_words;

pub struct VisibilityBufferProcessingPass {
    vis_buffer_scan: VisBufferStep,
    shader_cull: VisBufferStep,
}

struct VisBufferStep {
    shader_object: Arc<ShaderObject>,
    pipeline: Arc<ComputePipeline>,
    dispatch: [u32; 3],
}

pub struct VisibilityBufferRasterizer {
    shader_object: Arc<ShaderObject>,
    pipeline: Arc<GraphicsPipeline>,
    render_pass: Arc<RenderPass>,
    rhi: Rc<VKRHI>,
    visibility_buffer: Arc<ImageView>,
    depth_buffer: Arc<ImageView>,
    rt_framebuffer: Arc<Framebuffer>,

    // TODO: Remove
    instance_buffer: Subbuffer<[Instance]>,
}

impl VisibilityBufferProcessingPass {
    pub fn new(rhi: &VKRHI, target_extent: [u32; 2], num_materials: u32) -> Self {
        let vis_buffer_scan = VisBufferStep::new(
            rhi,
            "Engine/VisibilityBuffer/visBufferScan",
            "countTexels",
            [target_extent[0] / 16, target_extent[1] / 16, 1],
        );
        let shader_cull = VisBufferStep::new(
            rhi,
            "Engine/VisibilityBuffer/visBufferShaderCull",
            "cullShaders",
            [num_materials / 16, 1, 1],
        );
        Self {
            vis_buffer_scan,
            shader_cull,
        }
    }

    pub fn record_command_buffer(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_index: usize,
    ) -> Result<(), Box<ValidationError>> {
        self.vis_buffer_scan
            .record_command_buffer(command_buffer, image_index)?;
        self.shader_cull
            .record_command_buffer(command_buffer, image_index)?;
        Ok(())
    }
}

impl VisBufferStep {
    fn new(rhi: &VKRHI, module: &str, entry_point: &str, dispatch: [u32; 3]) -> Self {
        let session = rhi.slang_compiler().session();
        let module = session.load_module(module).unwrap();
        let entry = module.find_entry_point_by_name(entry_point).unwrap();
        let linked = session
            .create_composite_component_type(&[module.into(), entry.into()])
            .unwrap()
            .link()
            .unwrap();
        let shader_object_layout =
            ShaderObjectLayout::new(linked.clone(), &[], rhi.device(), ShaderStages::COMPUTE);
        let pipeline_layout = shader_object_layout.pipeline_layout().clone();
        let shader_object = ShaderObject::new(
            shader_object_layout,
            rhi.descriptor_allocator(),
            rhi.buffer_allocator(),
            rhi.in_flight_frames() as u32,
        );
        let pipeline = compute_pipeline()
            .shader(
                rhi.device().clone(),
                bytes_to_words(linked.entry_point_code(0, 0).unwrap().as_slice())
                    .unwrap()
                    .deref(),
            )
            .build_pipeline(rhi.device().clone(), pipeline_layout);
        Self {
            shader_object,
            pipeline,
            dispatch,
        }
    }

    fn record_command_buffer(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_index: usize,
    ) -> Result<(), Box<ValidationError>> {
        command_buffer
            .bind_pipeline_compute(self.pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.shader_object.pipeline_layout().clone(),
                0,
                self.shader_object.descriptor_sets()[image_index].clone(),
            )?;
        unsafe { command_buffer.dispatch(self.dispatch) }?;
        Ok(())
    }
}

impl VisibilityBufferRasterizer {
    pub fn new(rhi: Rc<VKRHI>, extent: [u32; 2]) -> Self {
        let render_pass =
            RenderPassBuilder::build_default_render_pass(rhi.as_ref(), Format::R32G32B32A32_UINT)
                .build();
        let module = "Engine/VisibilityBuffer/visBufferGenerator";
        let vert_entry_point = "vertexMain";
        let frag_entry_point = "fragmentMain";
        let session = rhi.slang_compiler().session();
        let module = session.load_module(module).unwrap();
        let vert_entry = module.find_entry_point_by_name(vert_entry_point).unwrap();
        let frag_entry = module.find_entry_point_by_name(frag_entry_point).unwrap();
        let linked = session
            .create_composite_component_type(&[module.into(), vert_entry.into(), frag_entry.into()])
            .unwrap()
            .link()
            .unwrap();
        let shader_object_layout = ShaderObjectLayout::new(
            linked.clone(),
            &[],
            rhi.device(),
            ShaderStages::all_graphics(),
        );
        let pipeline_layout = shader_object_layout.pipeline_layout().clone();
        let shader_object = ShaderObject::new(
            shader_object_layout,
            rhi.descriptor_allocator(),
            rhi.buffer_allocator(),
            rhi.in_flight_frames() as u32,
        );
        let pipeline = unsafe {
            graphics_pipeline()
                .input_assembly(None, None)
                .vertex_shader(
                    rhi.device().clone(),
                    bytes_to_words(linked.entry_point_code(0, 0).unwrap().as_slice())
                        .unwrap()
                        .deref(),
                )
                .vertex_buffer_description(&[
                    VertexBufferDescription {
                        members: [(
                            String::from("vertexInput.position"),
                            VertexMemberInfo {
                                offset: offset_of!(Vertex, position) as u32,
                                format: Format::R32G32B32_SFLOAT,
                                num_elements: 1,
                                stride: 0,
                            },
                        )]
                        .iter()
                        .cloned()
                        .collect(),
                        stride: size_of::<Vertex>() as u32,
                        input_rate: VertexInputRate::Vertex,
                    },
                    VertexBufferDescription {
                        members: [(
                            String::from("instanceInput.transform"),
                            VertexMemberInfo {
                                offset: offset_of!(Instance, model_matrix) as u32,
                                format: Format::R32G32B32A32_SFLOAT,
                                num_elements: 4,
                                stride: size_of::<[f32; 4]>() as u32,
                            },
                        )]
                        .iter()
                        .cloned()
                        .collect(),
                        stride: size_of::<Instance>() as u32,
                        input_rate: VertexInputRate::Instance { divisor: 1 },
                    },
                ])
                //.vertex_input::<Vertex>()
                .rasterizer(None, None, None, None, None, None)
                .skip_multisample()
                .fragment_shader(
                    rhi.device().clone(),
                    bytes_to_words(linked.entry_point_code(1, 0).unwrap().as_slice())
                        .unwrap()
                        .deref(),
                )
                .opaque_color_blend()
                .default_depth_test()
                .build_pipeline_unchecked(
                    rhi.device().clone(),
                    pipeline_layout,
                    PipelineSubpassType::BeginRenderPass(render_pass.clone().first_subpass()),
                    [
                        DynamicState::ViewportWithCount,
                        DynamicState::ScissorWithCount,
                    ]
                    .into(),
                )
        };

        let visibility_buffer = rhi.create_gbuffer(
            extent,
            Format::R32G32B32A32_UINT,
            ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
            ImageAspects::COLOR,
        );
        let depth_buffer = rhi.create_depth_buffer(extent);

        let rt_framebuffer = Framebuffer::new(
            render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![visibility_buffer.clone(), depth_buffer.clone()],
                extent,
                ..FramebufferCreateInfo::default()
            },
        )
        .unwrap();

        let instance_buffer = buffer_from_slice(
            rhi.buffer_allocator().clone(),
            rhi.command_buffer_interface(),
            rhi.queues().graphics_queue.clone(),
            &[Instance {
                model_matrix: glam::Mat4::default().to_cols_array_2d(),
            }],
            BufferUsage::VERTEX_BUFFER,
            MemoryTypeFilter::PREFER_DEVICE,
        ).unwrap();

        Self {
            shader_object,
            pipeline,
            render_pass,
            rhi,
            visibility_buffer,
            depth_buffer,
            rt_framebuffer,
            instance_buffer
        }
    }

    pub fn record_command_buffer(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_index: usize,
        extent: [u32; 2],
        scene: &VKScene,
    ) -> Result<(), Box<ValidationError>> {
        command_buffer
            .begin_render_pass(
                RenderPassBeginInfo {
                    render_area_offset: [0, 0],
                    render_area_extent: extent,
                    clear_values: vec![
                        Some(ClearValue::Uint([0, 0, 0, 0])),
                        Some(ClearValue::DepthStencil((1.0, 0))),
                    ],
                    render_pass: self.render_pass.clone(),
                    ..RenderPassBeginInfo::framebuffer(self.rt_framebuffer.clone())
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..SubpassBeginInfo::default()
                },
            )?
            .set_viewport_with_count(smallvec![Viewport {
                offset: [0., 0.],
                extent: extent.map(|u| u as f32),
                depth_range: 0.0f32..=1.0f32,
            }])?
            .set_scissor_with_count(smallvec![Scissor {
                offset: [0, 0],
                extent,
            }])?;

        let cursor = ShaderCursor::new(self.shader_object.clone());

        let view_cursor = cursor.field("gViewData").unwrap();
        view_cursor
            .field("viewProjection")
            .unwrap()
            .write(scene.camera().view_projection().as_ref());

        command_buffer
            .bind_pipeline_graphics(self.pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.shader_object.pipeline_layout().clone(),
                0,
                self.shader_object.descriptor_sets()[image_index].clone(),
            )?;

        let resources = self.rhi.resource_manager();
        let rcs = resources.deref();

        scene
            .models()
            .iter()
            .map(|model| {
                let mesh = model.mesh().get(rcs).unwrap();
                command_buffer
                    .bind_vertex_buffers(0, mesh.vertex().clone())?
                    .bind_vertex_buffers(1, self.instance_buffer.clone())?
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
}
