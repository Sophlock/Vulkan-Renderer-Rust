use std::{
    mem::offset_of,
    ops::Deref,
    rc::Rc,
    sync::{Arc, RwLock},
};

use smallvec::smallvec;
use vulkano::{
    DeviceAddress, ValidationError,
    buffer::BufferContents,
    command_buffer::{
        AutoCommandBufferBuilder, ClearColorImageInfo, CopyBufferInfo, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo,
    },
    device::DeviceOwned,
    format::{ClearValue, Format},
    image::ImageLayout,
    pipeline::{
        ComputePipeline, DynamicState, GraphicsPipeline, PipelineBindPoint,
        graphics::{
            subpass::PipelineSubpassType,
            vertex_input::{VertexBufferDescription, VertexInputRate, VertexMemberInfo},
            viewport::{Scissor, Viewport},
        },
    },
    render_pass::RenderPass,
    shader::{ShaderStages, spirv::bytes_to_words},
};

use crate::application::{
    assets::asset_traits::{RHICameraInterface, RHISceneInterface, Vertex},
    renderer::{
        profiling::{Profiler, ProfilerStage},
        visibility_buffer_data::{InstanceData, VisibilityBufferData},
        visibility_buffer_shading::VisibilityBufferShadePass,
    },
    rhi::{
        VKRHI,
        pipeline::{compute_pipeline, graphics_pipeline},
        render_pass::RenderPassBuilder,
        rhi_assets::vulkan_scene::VKScene,
        shader_cursor::ShaderCursor,
        shader_object::{ShaderObject, ShaderObjectLayout},
        swapchain::Swapchain,
        swapchain_resources::{
            SwapchainFramebuffer, SwapchainFramebufferCreateInfo, SwapchainImage,
        },
    },
};

pub struct VisibilityBufferProcessingPass {
    texel_count: VisBufferStep,
    naive_shader_cull: VisBufferStep,

    fill_commands: VisBufferStep,

    shader_cull: VisBufferStep,
    resolve_unsure: VisBufferStep,
    drawn_offset: VisBufferStep,
    culled_offset: VisBufferStep,
    texel_bin: VisBufferStep,
    generate_commands: VisBufferStep,

    num_materials: u32,
    data: Arc<VisibilityBufferData>,
}

struct VisBufferStep {
    shader_object: Arc<ShaderObject>,
    pipeline: Arc<ComputePipeline>,
    data: Arc<VisibilityBufferData>,
}

pub struct VisibilityBufferRasterizer {
    shader_object: Arc<ShaderObject>,
    pipeline: Arc<GraphicsPipeline>,
    render_pass: Arc<RenderPass>,
    rhi: Rc<VKRHI>,
    depth_buffer: Arc<RwLock<SwapchainImage>>,
    rt_framebuffer: Arc<RwLock<SwapchainFramebuffer>>,
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

#[derive(Copy, Clone, BufferContents)]
#[repr(C)]
pub struct VisBufferPushConstant {
    pub this_material_index: u32,
}

impl VisibilityBufferProcessingPass {
    pub fn new(rhi: &VKRHI, data: &Arc<VisibilityBufferData>) -> Self {
        Self {
            texel_count: Self::texel_count_shader(rhi, data),
            naive_shader_cull: Self::naive_shader_cull_shader(rhi, data),
            fill_commands: Self::fill_all_pipelines_shader(rhi, data),
            shader_cull: Self::shader_cull_shader(rhi, data),
            resolve_unsure: Self::resolve_unsure_shader(rhi, data),
            drawn_offset: Self::drawn_pipeline_offset_shader(rhi, data),
            culled_offset: Self::culled_pipeline_offset_shader(rhi, data),
            texel_bin: Self::texel_bin_shader(rhi, data),
            generate_commands: Self::generate_commands_shader(rhi, data),
            num_materials: data.global_data.num_materials(),
            data: data.clone(),
        }
    }

    pub fn record_command_buffer(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_index: usize,
        swapchain_extent: [u32; 2],
        profiler: &Profiler,
    ) -> Result<(), Box<ValidationError>> {
        command_buffer.clear_color_image(ClearColorImageInfo {
            image_layout: ImageLayout::General,
            ..ClearColorImageInfo::image(
                self.data
                    .final_render_target
                    .read()
                    .unwrap()
                    .image_view()
                    .image()
                    .clone(),
            )
        })?;

        if cfg!(feature = "no_cull_visbuffer") {
            self.record_filled_command_buffer(
                command_buffer,
                image_index,
                swapchain_extent,
                profiler,
            )
        } else if cfg!(feature = "binned_visbuffer") {
            self.record_binned_command_buffer(
                command_buffer,
                image_index,
                swapchain_extent,
                profiler,
            )
        } else {
            self.record_naive_culling_command_buffer(
                command_buffer,
                image_index,
                swapchain_extent,
                profiler,
            )
        }
    }

    fn record_naive_culling_command_buffer(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_index: usize,
        swapchain_extent: [u32; 2],
        profiler: &Profiler,
    ) -> Result<(), Box<ValidationError>> {
        // Count the texel for each pipeline
        self.texel_count.record_command_buffer(
            command_buffer,
            image_index,
            [
                swapchain_extent[0] / 16 + 1,
                swapchain_extent[1] / 16 + 1,
                1,
            ],
        )?;

        profiler.write(command_buffer, ProfilerStage::PostTexelCount)?;

        // Cull all pipelines that are invisible
        self.naive_shader_cull.record_command_buffer(
            command_buffer,
            image_index,
            [self.num_materials / 16 + 1, 1, 1],
        )?;

        profiler.write(command_buffer, ProfilerStage::PostEmptyCull)?;

        command_buffer.copy_buffer(CopyBufferInfo::buffers(
            self.data.drawn_index_counter_buffer.clone(),
            self.data.final_material_count_buffer.clone(),
        ))?;

        Ok(())
    }

    fn record_filled_command_buffer(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_index: usize,
        swapchain_extent: [u32; 2],
        profiler: &Profiler,
    ) -> Result<(), Box<ValidationError>> {
        self.fill_commands.record_command_buffer(
            command_buffer,
            image_index,
            [self.num_materials / 16 + 1, 1, 1],
        )?;

        Ok(())
    }

    fn record_binned_command_buffer(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_index: usize,
        swapchain_extent: [u32; 2],
        profiler: &Profiler,
    ) -> Result<(), Box<ValidationError>> {
        // Count the texel for each pipeline
        self.texel_count.record_command_buffer(
            command_buffer,
            image_index,
            [
                swapchain_extent[0] / 16 + 1,
                swapchain_extent[1] / 16 + 1,
                1,
            ],
        )?;

        profiler.write(command_buffer, ProfilerStage::PostTexelCount)?;

        self.shader_cull.record_command_buffer(
            command_buffer,
            image_index,
            [self.num_materials / 16 + 1, 1, 1],
        )?;

        profiler.write(command_buffer, ProfilerStage::PostEmptyCull)?;

        self.resolve_unsure.record_command_buffer(
            command_buffer,
            image_index,
            [self.num_materials / 16 + 1, 1, 1],
        )?;

        profiler.write(command_buffer, ProfilerStage::PostCull)?;

        self.drawn_offset.record_command_buffer(
            command_buffer,
            image_index,
            [
                (VisibilityBufferShadePass::MAX_SEQUENCE_COUNT - 1) / 16 + 1,
                1,
                1,
            ],
        )?;

        command_buffer.copy_buffer(CopyBufferInfo::buffers(
            self.data.offset_accumulator_buffer.clone(),
            self.data.no_fallback_texel_count_buffer.clone(),
        ))?;

        self.culled_offset.record_command_buffer(
            command_buffer,
            image_index,
            [self.num_materials / 16 + 1, 1, 1],
        )?;

        profiler.write(command_buffer, ProfilerStage::PostPrefixSum)?;

        self.texel_bin.record_command_buffer(
            command_buffer,
            image_index,
            [
                swapchain_extent[0] / 16 + 1,
                swapchain_extent[1] / 16 + 1,
                1,
            ],
        )?;

        profiler.write(command_buffer, ProfilerStage::PostTexelBin)?;

        self.generate_commands.record_command_buffer(
            command_buffer,
            image_index,
            [VisibilityBufferShadePass::MAX_SEQUENCE_COUNT / 16 + 1, 1, 1],
        )?;

        Ok(())
    }

    fn texel_count_shader(rhi: &VKRHI, data: &Arc<VisibilityBufferData>) -> VisBufferStep {
        let count_texel = VisBufferStep::new(
            rhi,
            "Engine/VisibilityBuffer/visBufferTexelCount",
            "countTexels",
            data.clone(),
        );

        let cursor = ShaderCursor::new(count_texel.shader_object.clone());
        let input_cursor = cursor.field("gInput").unwrap();
        input_cursor
            .field("visBuffer")
            .unwrap()
            .write_swapchain_image(data.visibility_buffer.clone());
        input_cursor
            .field("materialFragmentCounts")
            .unwrap()
            .write_buffer(data.material_fragment_count_buffer.clone());
        input_cursor
            .field("relativePerMaterialOffset")
            .unwrap()
            .write_swapchain_image(data.relative_per_material_offsets_buffer.clone());

        data.global_data
            .write_to_shader_cursor(&mut cursor.field("gGlobalData").unwrap());

        count_texel
    }

    fn naive_shader_cull_shader(rhi: &VKRHI, data: &Arc<VisibilityBufferData>) -> VisBufferStep {
        let shader_cull_naive = VisBufferStep::new(
            rhi,
            "Engine/VisibilityBuffer/visBufferShaderCullNaive",
            "cullShaders",
            data.clone(),
        );

        let cursor = ShaderCursor::new(shader_cull_naive.shader_object.clone());
        let input_cursor = cursor.field("gInput").unwrap();
        input_cursor
            .field("texelCounts")
            .unwrap()
            .write_buffer(data.material_fragment_count_buffer.clone());
        input_cursor
            .field("index")
            .unwrap()
            .write_buffer(data.drawn_index_counter_buffer.clone());
        input_cursor
            .field("pipelineBindParameters")
            .unwrap()
            .write_buffer(data.pipeline_bind_commands.clone());
        input_cursor
            .field("computeDispatchParameters")
            .unwrap()
            .write_buffer(data.compute_dispatch_commands.clone());
        input_cursor
            .field("shadePushConstants")
            .unwrap()
            .write_buffer(data.push_constants.clone());

        data.global_data
            .write_to_shader_cursor(&mut cursor.field("gGlobalData").unwrap());

        shader_cull_naive
    }

    fn fill_all_pipelines_shader(rhi: &VKRHI, data: &Arc<VisibilityBufferData>) -> VisBufferStep {
        let fill_all = VisBufferStep::new(
            rhi,
            "Engine/VisibilityBuffer/visBufferFillAll",
            "fillIndirectCommandsStreams",
            data.clone(),
        );

        let cursor = ShaderCursor::new(fill_all.shader_object.clone());
        let input_cursor = cursor.field("gInput").unwrap();
        input_cursor
            .field("pipelineBindParameters")
            .unwrap()
            .write_buffer(data.pipeline_bind_commands.clone());
        input_cursor
            .field("computeDispatchParameters")
            .unwrap()
            .write_buffer(data.compute_dispatch_commands.clone());
        input_cursor
            .field("shadePushConstants")
            .unwrap()
            .write_buffer(data.push_constants.clone());

        data.global_data
            .write_to_shader_cursor(&mut cursor.field("gGlobalData").unwrap());

        fill_all
    }

    fn shader_cull_shader(rhi: &VKRHI, data: &Arc<VisibilityBufferData>) -> VisBufferStep {
        let shader_cull = VisBufferStep::new(
            rhi,
            "Engine/VisibilityBuffer/visBufferShaderCull",
            "cullShaders",
            data.clone(),
        );

        let cursor = ShaderCursor::new(shader_cull.shader_object.clone());
        let input_cursor = cursor.field("gInput").unwrap();
        input_cursor
            .field("texelCounts")
            .unwrap()
            .write_buffer(data.material_fragment_count_buffer.clone());
        input_cursor
            .field("drawn")
            .unwrap()
            .field("currentIndex")
            .unwrap()
            .write_buffer(data.drawn_index_counter_buffer.clone());
        input_cursor
            .field("drawn")
            .unwrap()
            .field("materialIndices")
            .unwrap()
            .write_buffer(data.drawn_material_indices_buffer.clone());
        input_cursor
            .field("culled")
            .unwrap()
            .field("currentIndex")
            .unwrap()
            .write_buffer(data.culled_index_counter_buffer.clone());
        input_cursor
            .field("culled")
            .unwrap()
            .field("materialIndices")
            .unwrap()
            .write_buffer(data.culled_material_indices_buffer.clone());
        input_cursor
            .field("unsure")
            .unwrap()
            .field("currentIndex")
            .unwrap()
            .write_buffer(data.unsure_index_counter_buffer.clone());
        input_cursor
            .field("unsure")
            .unwrap()
            .field("materialIndices")
            .unwrap()
            .write_buffer(data.unsure_material_indices_buffer.clone());

        input_cursor
            .field("minDrawnPixelFootprint")
            .unwrap()
            .write(&0.0001f32);
        input_cursor
            .field("maxCulledPixelFootprint")
            .unwrap()
            .write(&0.0015f32);

        data.global_data
            .write_to_shader_cursor(&mut cursor.field("gGlobalData").unwrap());

        shader_cull
    }

    fn resolve_unsure_shader(rhi: &VKRHI, data: &Arc<VisibilityBufferData>) -> VisBufferStep {
        let resolve_unsure = VisBufferStep::new(
            rhi,
            "Engine/VisibilityBuffer/visBufferResolveUnsure",
            "resolveUnsure",
            data.clone(),
        );

        let cursor = ShaderCursor::new(resolve_unsure.shader_object.clone());
        let input_cursor = cursor.field("gInput").unwrap();

        input_cursor
            .field("unsureCount")
            .unwrap()
            .write_buffer(data.unsure_index_counter_buffer.clone());
        input_cursor
            .field("unsureIndices")
            .unwrap()
            .write_buffer(data.unsure_material_indices_buffer.clone());
        input_cursor
            .field("currentDrawnIndex")
            .unwrap()
            .write_buffer(data.drawn_index_counter_buffer.clone());
        input_cursor
            .field("drawnIndices")
            .unwrap()
            .write_buffer(data.drawn_material_indices_buffer.clone());
        input_cursor
            .field("culled")
            .unwrap()
            .field("currentIndex")
            .unwrap()
            .write_buffer(data.culled_index_counter_buffer.clone());
        input_cursor
            .field("culled")
            .unwrap()
            .field("materialIndices")
            .unwrap()
            .write_buffer(data.culled_material_indices_buffer.clone());

        resolve_unsure
    }

    fn drawn_pipeline_offset_shader(
        rhi: &VKRHI,
        data: &Arc<VisibilityBufferData>,
    ) -> VisBufferStep {
        let prefix_sum = VisBufferStep::new(
            rhi,
            "Engine/VisibilityBuffer/visBufferMaterialPrefixSum",
            "computeOffsets",
            data.clone(),
        );

        let cursor = ShaderCursor::new(prefix_sum.shader_object.clone());
        let input_cursor = cursor.field("gInput").unwrap();
        input_cursor
            .field("texelCounts")
            .unwrap()
            .write_buffer(data.material_fragment_count_buffer.clone());

        input_cursor
            .field("materialIndices")
            .unwrap()
            .write_buffer(data.drawn_material_indices_buffer.clone());
        input_cursor
            .field("materialCount")
            .unwrap()
            .write_buffer(data.drawn_index_counter_buffer.clone());

        input_cursor
            .field("currentOffset")
            .unwrap()
            .write_buffer(data.offset_accumulator_buffer.clone());
        input_cursor
            .field("outPerMaterialOffset")
            .unwrap()
            .write_buffer(data.per_material_offset_buffer.clone());

        data.global_data
            .write_to_shader_cursor(&mut cursor.field("gGlobalData").unwrap());

        prefix_sum
    }

    fn culled_pipeline_offset_shader(
        rhi: &VKRHI,
        data: &Arc<VisibilityBufferData>,
    ) -> VisBufferStep {
        let prefix_sum = VisBufferStep::new(
            rhi,
            "Engine/VisibilityBuffer/visBufferMaterialPrefixSum",
            "computeOffsets",
            data.clone(),
        );

        let cursor = ShaderCursor::new(prefix_sum.shader_object.clone());
        let input_cursor = cursor.field("gInput").unwrap();
        input_cursor
            .field("texelCounts")
            .unwrap()
            .write_buffer(data.material_fragment_count_buffer.clone());

        input_cursor
            .field("materialIndices")
            .unwrap()
            .write_buffer(data.culled_material_indices_buffer.clone());
        input_cursor
            .field("materialCount")
            .unwrap()
            .write_buffer(data.culled_index_counter_buffer.clone());

        input_cursor
            .field("currentOffset")
            .unwrap()
            .write_buffer(data.offset_accumulator_buffer.clone());
        input_cursor
            .field("outPerMaterialOffset")
            .unwrap()
            .write_buffer(data.per_material_offset_buffer.clone());

        data.global_data
            .write_to_shader_cursor(&mut cursor.field("gGlobalData").unwrap());

        prefix_sum
    }

    fn texel_bin_shader(rhi: &VKRHI, data: &Arc<VisibilityBufferData>) -> VisBufferStep {
        let texel_bin = VisBufferStep::new(
            rhi,
            "Engine/VisibilityBuffer/visBufferTexelBin",
            "binTexels",
            data.clone(),
        );

        let cursor = ShaderCursor::new(texel_bin.shader_object.clone());
        let input_cursor = cursor.field("gInput").unwrap();
        input_cursor
            .field("visBuffer")
            .unwrap()
            .write_swapchain_image(data.visibility_buffer.clone());
        input_cursor
            .field("relativePerMaterialOffsets")
            .unwrap()
            .write_swapchain_image(data.relative_per_material_offsets_buffer.clone());
        input_cursor
            .field("perMaterialOffsets")
            .unwrap()
            .write_buffer(data.per_material_offset_buffer.clone());
        input_cursor
            .field("outBinnedTexels")
            .unwrap()
            .write_buffer(data.binned_texel_buffer.clone());

        data.global_data
            .write_to_shader_cursor(&mut cursor.field("gGlobalData").unwrap());

        texel_bin
    }

    fn generate_commands_shader(rhi: &VKRHI, data: &Arc<VisibilityBufferData>) -> VisBufferStep {
        let generate_commands = VisBufferStep::new(
            rhi,
            "Engine/VisibilityBuffer/visBufferGenerateCommandsStreams",
            "generateCommandsStreams",
            data.clone(),
        );

        let cursor = ShaderCursor::new(generate_commands.shader_object.clone());
        let input_cursor = cursor.field("gInput").unwrap();
        input_cursor
            .field("drawnMaterials")
            .unwrap()
            .write_buffer(data.drawn_material_indices_buffer.clone());
        input_cursor
            .field("drawnMaterialCount")
            .unwrap()
            .write_buffer(data.drawn_index_counter_buffer.clone());
        input_cursor
            .field("texelCounts")
            .unwrap()
            .write_buffer(data.material_fragment_count_buffer.clone());

        input_cursor
            .field("texelsWithoutFallback")
            .unwrap()
            .write_buffer(data.no_fallback_texel_count_buffer.clone());
        input_cursor
            .field("totalTexelCount")
            .unwrap()
            .write_buffer(data.offset_accumulator_buffer.clone());
        input_cursor
            .field("finalMaterialCount")
            .unwrap()
            .write_buffer(data.final_material_count_buffer.clone());
        input_cursor
            .field("perMaterialOffsets")
            .unwrap()
            .write_buffer(data.per_material_offset_buffer.clone());

        input_cursor
            .field("pipelineBindParameters")
            .unwrap()
            .write_buffer(data.pipeline_bind_commands.clone());
        input_cursor
            .field("computeDispatchParameters")
            .unwrap()
            .write_buffer(data.compute_dispatch_commands.clone());
        input_cursor
            .field("shadePushConstants")
            .unwrap()
            .write_buffer(data.push_constants.clone());

        data.global_data
            .write_to_shader_cursor(&mut cursor.field("gGlobalData").unwrap());

        generate_commands
    }
}

impl VisBufferStep {
    fn new(rhi: &VKRHI, module: &str, entry_point: &str, data: Arc<VisibilityBufferData>) -> Self {
        let session = rhi.slang_compiler().session();
        let module = session.load_module(module).unwrap();
        let entry = module.find_entry_point_by_name(entry_point).unwrap();
        let linked = session
            .create_composite_component_type(&[module.into(), entry.into()])
            .unwrap()
            .link()
            .unwrap();
        let shader_object_layout = ShaderObjectLayout::new_with_push_constants(
            linked.clone(),
            &[],
            rhi.device(),
            ShaderStages::COMPUTE,
            // Global data is passed as push constant
            vec![/*PushConstantRange {
                stages: ShaderStages::COMPUTE,
                offset: 0,
                size: size_of::<DeviceAddress>() as u32,
            }*/],
        );
        let pipeline_layout = shader_object_layout.pipeline_layout().clone();
        let shader_object = ShaderObject::new(
            shader_object_layout,
            rhi.descriptor_allocator(),
            rhi.buffer_allocator(),
            rhi.in_flight_frames() as u32,
            rhi.shader_object_update_queue().clone(),
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
            data,
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
            rhi.shader_object_update_queue().clone(),
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
                                offset: offset_of!(InstanceData, model_transform) as u32,
                                format: Format::R32G32B32A32_SFLOAT,
                                num_elements: 4,
                                stride: size_of::<[f32; 4]>() as u32,
                            },
                        )]
                        .iter()
                        .cloned()
                        .collect(),
                        stride: size_of::<InstanceData>() as u32,
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

        Self {
            shader_object,
            pipeline,
            render_pass,
            rhi,
            depth_buffer,
            rt_framebuffer,
        }
    }

    pub fn record_command_buffer(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_index: usize,
        extent: [u32; 2],
        scene: &VKScene,
        data: &VisibilityBufferData,
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

        command_buffer
            .bind_vertex_buffers(0, data.global_data.vertices.clone())?
            .bind_vertex_buffers(1, data.global_data.instances.clone())?
            .bind_index_buffer(data.global_data.indices.clone())?;

        unsafe {
            command_buffer.draw_indexed_indirect(data.global_data.draw_indirect_commands.clone())
        }?;

        command_buffer
            .end_render_pass(SubpassEndInfo::default())
            .map(|_| ())
    }
}

impl PipelineBindParameter {
    pub fn pipeline(pipeline: &Arc<ComputePipeline>) -> Self {
        Self {
            pipeline_address: pipeline.indirect_device_address().get(),
        }
    }
}
