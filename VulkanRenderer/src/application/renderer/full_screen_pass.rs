use crate::application::rhi::buffer::buffer_from_slice;
use crate::application::rhi::command_buffer::CommandBufferInterface;
use crate::application::rhi::pipeline::graphics_pipeline;
use crate::application::rhi::shader_cursor::ShaderCursor;
use crate::application::rhi::shader_object::{ShaderObject, ShaderObjectLayout};
use crate::application::rhi::shaders::SlangCompiler;
use crate::application::rhi::VKRHI;
use shader_slang::{Blob, ComponentType};
use smallvec::smallvec;
use std::ops::Deref;
use std::sync::Arc;
use vulkano::buffer::{BufferContents, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo,
    SubpassContents, SubpassEndInfo,
};
use vulkano::device::{Device, Queue};
use vulkano::format::{ClearValue, Format};
use vulkano::image::sampler::{Sampler, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{ImageLayout, SampleCount};
use vulkano::memory::allocator::{MemoryAllocator, MemoryTypeFilter};
use vulkano::pipeline::graphics::subpass::PipelineSubpassType;
use vulkano::pipeline::graphics::vertex_input;
use vulkano::pipeline::graphics::viewport::{Scissor, Viewport};
use vulkano::pipeline::{DynamicState, GraphicsPipeline, PipelineBindPoint};
use vulkano::render_pass::{
    AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp, Framebuffer,
    FramebufferCreateInfo, RenderPass, RenderPassCreateInfo, SubpassDependency, SubpassDescription,
};
use vulkano::shader::spirv::bytes_to_words;
use vulkano::shader::ShaderStages;
use vulkano::sync::{AccessFlags, PipelineStages};
use vulkano::ValidationError;

pub struct FullScreenPass {
    render_pass: Arc<RenderPass>,
    vertex_buffer: Subbuffer<[FullScreenPassVertex]>,
    index_buffer: Subbuffer<[u32]>,
    shader_object_layout: Arc<ShaderObjectLayout>,
    shader_object: ShaderObject,
    pipeline: Arc<GraphicsPipeline>,
    framebuffers: Vec<Arc<Framebuffer>>,
}

#[derive(BufferContents, Copy, Clone, vertex_input::Vertex)]
#[repr(C)]
struct FullScreenPassVertex {
    #[name("input_position_0")]
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
    #[name("input_textureCoordinate_0")]
    #[format(R32G32_SFLOAT)]
    pub texture_coordinates: [f32; 2],
}

impl FullScreenPass {
    pub fn new(
        rhi: &VKRHI,
        input_format: Format,
        output_format: Format,
        output_final_layout: ImageLayout,
        previous_stages: PipelineStages,
        previous_access: AccessFlags,
        input_initial_layout: ImageLayout,
        output_initial_layout: Option<ImageLayout>,
        source_image: Arc<ImageView>,
        target_images: impl Iterator<Item = Arc<ImageView>>,
        image_extent: [u32; 2],
    ) -> Self {
        let render_pass = Self::create_render_pass(
            rhi.device().clone(),
            input_format,
            output_format,
            output_final_layout,
            previous_stages,
            previous_access,
            input_initial_layout,
            output_initial_layout,
        );

        let framebuffers = target_images
            .map(|target| {
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![target /*, source_image.clone()*/],
                        extent: image_extent,
                        layers: 1,
                        ..FramebufferCreateInfo::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        let queue = &rhi.queues().graphics_queue;
        let vertex_buffer = Self::create_vertex_buffer(
            rhi.buffer_allocator().clone(),
            rhi.command_buffer_interface(),
            queue.clone(),
        );
        let index_buffer = Self::create_index_buffer(
            rhi.buffer_allocator().clone(),
            rhi.command_buffer_interface(),
            queue.clone(),
        );
        let (vert, frag, linked) = Self::compile_shader(rhi.slang_compiler());
        let shader_object_layout =
            ShaderObjectLayout::new(linked, &[], rhi.device(), ShaderStages::all_graphics());
        let pipeline = Self::create_graphics_pipeline(
            rhi.device(),
            bytes_to_words(vert.as_slice()).unwrap().deref(),
            bytes_to_words(frag.as_slice()).unwrap().deref(),
            &shader_object_layout,
            render_pass.clone(),
        );

        let shader_object = ShaderObject::new(
            shader_object_layout.clone(),
            rhi.descriptor_allocator(),
            rhi.buffer_allocator(),
            framebuffers.len() as u32,
        );

        let sampler = Sampler::new(
            rhi.device().clone(),
            SamplerCreateInfo {
                ..SamplerCreateInfo::default()
            },
        )
        .unwrap();

        ShaderCursor::new(&shader_object)
            .field("gInput")
            .unwrap()
            .field("colorInput")
            .unwrap()
            .write_image_view_sampler(source_image, sampler);

        Self {
            render_pass,
            vertex_buffer,
            index_buffer,
            shader_object_layout,
            shader_object,
            pipeline,
            framebuffers,
        }
    }

    pub fn record_command_buffer(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        extent: [u32; 2],
        image_index: usize,
    ) -> Result<(), Box<ValidationError>> {
        command_buffer
            .begin_render_pass(
                RenderPassBeginInfo {
                    render_area_offset: [0, 0],
                    render_area_extent: extent,
                    clear_values: vec![
                        Some(ClearValue::Float([0.0, 0.0, 0.0, 1.0]))
                    ],
                    render_pass: self.render_pass.clone(),
                    ..RenderPassBeginInfo::framebuffer(self.framebuffers[image_index].clone())
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
            }])?
            .bind_pipeline_graphics(self.pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.shader_object_layout.pipeline_layout().clone(),
                0,
                self.shader_object.descriptor_sets()[image_index].clone(),
            )?
            .bind_vertex_buffers(0, self.vertex_buffer.clone())?
            .bind_index_buffer(self.index_buffer.clone())?;

        unsafe { command_buffer.draw_indexed(6, 1, 0, 0, 0) }?;
        command_buffer.end_render_pass(SubpassEndInfo::default())?;
        Ok(())
    }

    fn create_render_pass(
        device: Arc<Device>,
        input_format: Format,
        output_format: Format,
        output_final_layout: ImageLayout,
        previous_stages: PipelineStages,
        previous_access: AccessFlags,
        input_initial_layout: ImageLayout,
        output_initial_layout: Option<ImageLayout>,
    ) -> Arc<RenderPass> {
        RenderPass::new(
            device,
            RenderPassCreateInfo {
                attachments: vec![
                    AttachmentDescription {
                        format: output_format,
                        load_op: AttachmentLoadOp::Clear,
                        store_op: AttachmentStoreOp::Store,
                        initial_layout: output_initial_layout.unwrap_or(ImageLayout::Undefined),
                        final_layout: output_final_layout,
                        ..AttachmentDescription::default()
                    },
                    /*AttachmentDescription {
                        format: input_format,
                        load_op: AttachmentLoadOp::Load,
                        store_op: AttachmentStoreOp::DontCare,
                        initial_layout: input_initial_layout,
                        final_layout: ImageLayout::ShaderReadOnlyOptimal,
                        ..AttachmentDescription::default()
                    },*/
                ],
                subpasses: vec![SubpassDescription {
                    /*input_attachments: vec![Some(AttachmentReference {
                        attachment: 1,
                        layout: ImageLayout::ShaderReadOnlyOptimal,
                        aspects: ImageAspects::COLOR,
                        ..AttachmentReference::default()
                    })],*/
                    color_attachments: vec![Some(AttachmentReference {
                        attachment: 0,
                        layout: ImageLayout::ColorAttachmentOptimal,
                        ..AttachmentReference::default()
                    })],
                    ..SubpassDescription::default()
                }],
                dependencies: vec![
                    /*SubpassDependency {
                        src_subpass: None,
                        dst_subpass: Some(0),
                        src_stages: previous_stages,
                        dst_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
                        src_access: previous_access,
                        dst_access: AccessFlags::COLOR_ATTACHMENT_WRITE
                            | AccessFlags::COLOR_ATTACHMENT_READ,
                        ..SubpassDependency::default()
                    }*/
                    SubpassDependency {
                        src_subpass: None,
                        dst_subpass: Some(0),
                        src_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT
                            | PipelineStages::EARLY_FRAGMENT_TESTS,
                        dst_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT
                            | PipelineStages::EARLY_FRAGMENT_TESTS,
                        src_access: AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        dst_access: AccessFlags::COLOR_ATTACHMENT_WRITE
                            | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        ..SubpassDependency::default()
                    },
                ],
                ..RenderPassCreateInfo::default()
            },
        )
        .unwrap()
    }

    fn create_vertex_buffer(
        allocator: Arc<dyn MemoryAllocator>,
        command_buffer_interface: &CommandBufferInterface,
        queue: Arc<Queue>,
    ) -> Subbuffer<[FullScreenPassVertex]> {
        let vertices = [
            FullScreenPassVertex {
                position: [-1f32, -1f32, 0f32],
                texture_coordinates: [0f32, 0f32],
            },
            FullScreenPassVertex {
                position: [1f32, -1f32, 0f32],
                texture_coordinates: [1f32, 0f32],
            },
            FullScreenPassVertex {
                position: [-1f32, 1f32, 0f32],
                texture_coordinates: [0f32, 1f32],
            },
            FullScreenPassVertex {
                position: [1f32, 1f32, 0f32],
                texture_coordinates: [1f32, 1f32],
            },
        ];
        buffer_from_slice(
            allocator,
            command_buffer_interface,
            queue,
            vertices.as_slice(),
            BufferUsage::VERTEX_BUFFER,
            MemoryTypeFilter::PREFER_DEVICE,
        )
        .unwrap()
    }

    fn create_index_buffer(
        allocator: Arc<dyn MemoryAllocator>,
        command_buffer_interface: &CommandBufferInterface,
        queue: Arc<Queue>,
    ) -> Subbuffer<[u32]> {
        let indices = [0, 1, 2, 2, 1, 3];
        buffer_from_slice(
            allocator,
            command_buffer_interface,
            queue,
            indices.as_slice(),
            BufferUsage::INDEX_BUFFER,
            MemoryTypeFilter::PREFER_DEVICE,
        )
        .unwrap()
    }

    fn create_graphics_pipeline(
        device: &Arc<Device>,
        vert_spirv: &[u32],
        frag_spirv: &[u32],
        layout: &Arc<ShaderObjectLayout>,
        render_pass: Arc<RenderPass>,
    ) -> Arc<GraphicsPipeline> {
        graphics_pipeline()
            .input_assembly(None, None)
            .vertex_shader(device.clone(), vert_spirv)
            .vertex_input::<FullScreenPassVertex>()
            .rasterizer(None, None, None, None, None, None)
            .skip_multisample()
            .fragment_shader(device.clone(), frag_spirv)
            .opaque_color_blend()
            .skip_depth_test()
            .build_pipeline(
                device.clone(),
                layout.pipeline_layout().clone(),
                PipelineSubpassType::BeginRenderPass(render_pass.first_subpass()),
                [
                    DynamicState::ViewportWithCount,
                    DynamicState::ScissorWithCount,
                ]
                .into(),
            )
    }

    fn compile_shader(compiler: &SlangCompiler) -> (Blob, Blob, ComponentType) {
        let module = compiler
            .session()
            .load_module("Engine/Utils/FullscreenPass/fullscreenPass")
            .unwrap();
        let vert_entry = module.find_entry_point_by_name("vertexMain").unwrap();
        let frag_entry = module.find_entry_point_by_name("fragmentMain").unwrap();
        let composed = compiler
            .session()
            .create_composite_component_type(&[module.into(), vert_entry.into(), frag_entry.into()])
            .unwrap();
        let linked = composed.link().unwrap();
        (
            linked.entry_point_code(0, 0).unwrap(),
            linked.entry_point_code(1, 0).unwrap(),
            linked,
        )
    }

    pub fn recreate_framebuffers(
        &mut self,
        target_images: impl Iterator<Item = Arc<ImageView>>,
        image_extent: [u32; 2],
    ) {
        self.framebuffers = target_images
            .map(|target| {
                Framebuffer::new(
                    self.render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![target /*, source_image.clone()*/],
                        extent: image_extent,
                        layers: 1,
                        ..FramebufferCreateInfo::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
    }
}
