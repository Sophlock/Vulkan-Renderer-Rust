use crate::application::rhi::swapchain::Swapchain;
use crate::application::rhi::swapchain_resources::{
    SwapchainFramebuffer, SwapchainFramebufferCreateInfo, SwapchainImage,
};
use crate::application::{
    assets::asset_traits::{
        Instance, RHICameraInterface, RHIInterface, RHIModelInterface, RHISceneInterface, Vertex,
    },
    rhi::{
        VKRHI,
        buffer::buffer_from_slice,
        pipeline::{compute_pipeline, graphics_pipeline},
        render_pass::RenderPassBuilder,
        rhi_assets::vulkan_scene::VKScene,
        shader_cursor::ShaderCursor,
        shader_object::{ShaderObject, ShaderObjectLayout},
    },
};
use ash::vk::DeviceAddress;
use smallvec::smallvec;
use std::sync::RwLock;
use std::{mem::offset_of, ops::Deref, rc::Rc, sync::Arc};
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo};
use vulkano::command_buffer::CopyBufferInfo;
use vulkano::memory::allocator::AllocationCreateInfo;
use vulkano::{
    ValidationError,
    buffer::{BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo,
        SubpassContents, SubpassEndInfo,
    },
    format::{ClearValue, Format},
    image::{ImageAspects, ImageUsage, view::ImageView},
    memory::allocator::MemoryTypeFilter,
    pipeline::{
        ComputePipeline, DynamicState, GraphicsPipeline, PipelineBindPoint,
        graphics::{
            subpass::PipelineSubpassType,
            vertex_input::{VertexBufferDescription, VertexInputRate, VertexMemberInfo},
            viewport::{Scissor, Viewport},
        },
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    shader::{ShaderStages, spirv::bytes_to_words},
};

pub struct VisibilityBufferProcessingPass {
    vis_buffer_scan: VisBufferStep,
    shader_cull: VisBufferStep,
    num_materials: u32,
}

struct VisBufferStep {
    shader_object: Arc<ShaderObject>,
    pipeline: Arc<ComputePipeline>,
}

pub struct VisibilityBufferRasterizer {
    shader_object: Arc<ShaderObject>,
    pipeline: Arc<GraphicsPipeline>,
    render_pass: Arc<RenderPass>,
    rhi: Rc<VKRHI>,
    depth_buffer: Arc<RwLock<SwapchainImage>>,
    rt_framebuffer: Arc<RwLock<SwapchainFramebuffer>>,

    // TODO: Remove
    instance_buffer: Subbuffer<[Instance]>,
}

#[derive(Copy, Clone, BufferContents)]
#[repr(C)]
pub struct PipelineBindParameter {
    pub pipeline_address: DeviceAddress,
}

#[derive(Copy, Clone, BufferContents)]
#[repr(C)]
pub struct ComputeDispatchParameter {
    pub dispatch: [u32; 3],
}

#[derive(Clone)]
pub struct VisibilityBufferData {
    // The packed visibility buffer
    pub visibility_buffer: Arc<RwLock<SwapchainImage>>,

    // Stores the number of texels for each material
    pub material_fragment_count_buffer: Subbuffer<[u32]>,

    // Used to atomically increment an index counter. After the culling step, this will hold the number of materials to be shaded
    pub index_counter_buffer: Subbuffer<u32>,

    // Stores for each shading step the index of the material to be used
    pub material_indices_buffer: Subbuffer<[u32]>,

    // Holds the pipeline bind data
    pub pipeline_bind_commands: Subbuffer<[PipelineBindParameter]>,

    // Holds the compute dispatch data
    pub compute_dispatch_commands: Subbuffer<[ComputeDispatchParameter]>,

    // Used to clear data before writing
    clear_buffer: Subbuffer<[u32]>,
}

impl VisibilityBufferProcessingPass {
    pub fn new(rhi: &VKRHI, num_materials: u32, data: &VisibilityBufferData) -> Self {
        let vis_buffer_scan =
            VisBufferStep::new(rhi, "Engine/VisibilityBuffer/visBufferScan", "countTexels");
        let shader_cull = VisBufferStep::new(
            rhi,
            "Engine/VisibilityBuffer/visBufferShaderCull",
            "cullShaders",
        );

        let cursor = ShaderCursor::new(vis_buffer_scan.shader_object.clone());
        let input_cursor = cursor.field("gInput").unwrap();
        input_cursor.field("visBuffer").unwrap().write_swapchain_image(data.visibility_buffer.clone());
        input_cursor.field("materialFragmentCounts").unwrap().write_buffer(data.material_fragment_count_buffer.clone());
        // TODO
        let global_cursor = cursor.field("gGlobalData").unwrap();

        let cursor = ShaderCursor::new(shader_cull.shader_object.clone());
        let input_cursor = cursor.field("gInput").unwrap();
        input_cursor.field("texelCounts").unwrap().write_buffer(data.material_fragment_count_buffer.clone());
        input_cursor.field("index").unwrap().write_buffer(data.index_counter_buffer.clone());
        input_cursor.field("materialIndices").unwrap().write_buffer(data.material_indices_buffer.clone());
        input_cursor.field("pipelineBindParameters").unwrap().write_buffer(data.pipeline_bind_commands.clone());
        input_cursor.field("computeDispatchParameters").unwrap().write_buffer(data.compute_dispatch_commands.clone());
        // TODO
        let global_cursor = cursor.field("gGlobalData").unwrap();

        Self {
            vis_buffer_scan,
            shader_cull,
            num_materials,
        }
    }

    pub fn record_command_buffer(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_index: usize,
        swapchain_extent: [u32; 2],
    ) -> Result<(), Box<ValidationError>> {
        self.vis_buffer_scan.record_command_buffer(
            command_buffer,
            image_index,
            [swapchain_extent[0] / 16, swapchain_extent[1] / 16, 1],
        )?;
        self.shader_cull.record_command_buffer(
            command_buffer,
            image_index,
            [self.num_materials / 16, 1, 1],
        )?;
        Ok(())
    }
}

impl VisBufferStep {
    fn new(rhi: &VKRHI, module: &str, entry_point: &str) -> Self {
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
        }
    }

    fn record_command_buffer(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_index: usize,
        dispatch: [u32; 3],
    ) -> Result<(), Box<ValidationError>> {
        command_buffer
            .bind_pipeline_compute(self.pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.shader_object.pipeline_layout().clone(),
                0,
                self.shader_object.descriptor_sets()[image_index].clone(),
            )?;
        unsafe { command_buffer.dispatch(dispatch) }?;
        Ok(())
    }
}

impl VisibilityBufferRasterizer {
    pub fn new(rhi: Rc<VKRHI>, swapchain: &Swapchain, data: &VisibilityBufferData) -> Self {
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

        let depth_buffer = swapchain.create_depth_buffer(rhi.as_ref());

        let rt_framebuffer = swapchain.create_framebuffer(
            render_pass.clone(),
            SwapchainFramebufferCreateInfo {
                attachments: vec![data.visibility_buffer.clone(), depth_buffer.clone()],
                ..SwapchainFramebufferCreateInfo::default()
            },
        );

        let instance_buffer = buffer_from_slice(
            rhi.buffer_allocator().clone(),
            rhi.command_buffer_interface(),
            rhi.queues().graphics_queue.clone(),
            &[Instance {
                model_matrix: glam::Mat4::default().to_cols_array_2d(),
            }],
            BufferUsage::VERTEX_BUFFER,
            MemoryTypeFilter::PREFER_DEVICE,
        )
        .unwrap();

        Self {
            shader_object,
            pipeline,
            render_pass,
            rhi,
            depth_buffer,
            rt_framebuffer,
            instance_buffer,
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
                    ..RenderPassBeginInfo::framebuffer(
                        self.rt_framebuffer.read().unwrap().framebuffer().clone(),
                    )
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

impl VisibilityBufferData {
    pub fn new(
        rhi: &VKRHI,
        swapchain: &Swapchain,
        max_sequence_count: u32,
        num_materials: u32,
    ) -> Self {
        let visibility_buffer = swapchain.create_gbuffer(
            rhi,
            Format::R32G32B32A32_UINT,
            ImageUsage::COLOR_ATTACHMENT | ImageUsage::SAMPLED,
            ImageAspects::COLOR,
        );

        let material_fragment_count_buffer = Self::create_slice_buffer(
            rhi,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
            num_materials,
        );

        let index_counter_buffer = Buffer::new_sized(
            rhi.buffer_allocator().clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST,
                ..BufferCreateInfo::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap();

        let material_indices_buffer =
            Self::create_slice_buffer(rhi, BufferUsage::STORAGE_BUFFER, max_sequence_count);

        let pipeline_bind_commands =
            Self::create_slice_buffer(rhi, BufferUsage::INDIRECT_BUFFER, max_sequence_count);

        let compute_dispatch_commands =
            Self::create_slice_buffer(rhi, BufferUsage::INDIRECT_BUFFER, max_sequence_count);

        let clear_buffer = buffer_from_slice(
            rhi.buffer_allocator().clone(),
            rhi.command_buffer_interface(),
            rhi.queues().compute_queue.clone(),
            (0..max_sequence_count)
                .map(|_| 0u32)
                .collect::<Vec<_>>()
                .as_slice(),
            BufferUsage::TRANSFER_SRC,
            MemoryTypeFilter::PREFER_DEVICE,
        ).unwrap();

        Self {
            visibility_buffer,
            material_fragment_count_buffer,
            index_counter_buffer,
            material_indices_buffer,
            pipeline_bind_commands,
            compute_dispatch_commands,
            clear_buffer,
        }
    }

    fn create_slice_buffer<T: BufferContents>(
        rhi: &VKRHI,
        usage: BufferUsage,
        length: u32,
    ) -> Subbuffer<[T]> {
        Buffer::new_slice(
            rhi.buffer_allocator().clone(),
            BufferCreateInfo {
                usage,
                ..BufferCreateInfo::default()
            },
            AllocationCreateInfo::default(),
            length.into(),
        )
        .unwrap()
    }

    pub fn clear(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> Result<(), Box<ValidationError>> {
        command_buffer.copy_buffer(CopyBufferInfo::buffers(
            self.clear_buffer.clone(),
            self.material_fragment_count_buffer.clone(),
        ))?;
        command_buffer.copy_buffer(CopyBufferInfo::buffers(
            self.clear_buffer.clone(),
            self.index_counter_buffer.clone(),
        ))?;
        Ok(())
    }
}
