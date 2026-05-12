use std::{rc::Rc, sync::Arc};

use vulkano::{
    DeviceSize, ValidationError,
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    device::DeviceOwned,
    device_generated_commands::{
        GeneratedCommandsInfo, GeneratedCommandsPipeline, IndirectCommandsLayout,
        IndirectCommandsLayoutCreateInfo, IndirectCommandsLayoutToken,
        IndirectCommandsLayoutTokenPushConstant, IndirectCommandsLayoutUsageFlags,
        IndirectCommandsStream, IndirectCommandsTokenType,
    },
    memory::allocator::AllocationCreateInfo,
    pipeline::PipelineBindPoint,
    shader::ShaderStages,
};

use crate::application::assets::asset_traits::RHIInterface;
use crate::application::rhi::rhi_assets::vulkan_material::VKMaterial;
use crate::application::{
    renderer::{
        visibility_buffer_data::VisibilityBufferData,
        visibility_buffer_generation::{
            ComputeDispatchParameter, PipelineBindParameter, VisBufferPushConstant,
        },
    },
    rhi::{VKRHI, shader_cursor::ShaderCursor},
};

/// Shading step of the visibility buffer
pub struct VisibilityBufferShadePass {
    rhi: Rc<VKRHI>,

    /// Indirect commands layout. This defines what kinds of indirect commands are executed
    #[cfg(not(feature = "renderdoc_compatibility"))]
    commands_layout: Arc<IndirectCommandsLayout>,

    /// Preprocess buffer for indirect commands (this is mostly an implementation detail of DGC)
    #[cfg(not(feature = "renderdoc_compatibility"))]
    preprocess_buffer: Subbuffer<[u8]>,

    data: Arc<VisibilityBufferData>,
}

impl VisibilityBufferShadePass {
    /// Upper bound for the number of rendered materials/indirect commands sequences
    #[cfg(not(feature = "no_cull_visbuffer"))]
    pub const MAX_SEQUENCE_COUNT: u32 = 2000u32;

    #[cfg(feature = "no_cull_visbuffer")]
    pub const MAX_SEQUENCE_COUNT: u32 = 100000u32;

    #[cfg(feature = "renderdoc_compatibility")]
    pub fn new(rhi: Rc<VKRHI>, data: Arc<VisibilityBufferData>) -> Self {
        Self { rhi, data }
    }

    #[cfg(not(feature = "renderdoc_compatibility"))]
    pub fn new(rhi: Rc<VKRHI>, data: Arc<VisibilityBufferData>) -> Self {
        let commands_layout = IndirectCommandsLayout::new(
            rhi.device().clone(),
            IndirectCommandsLayoutCreateInfo {
                flags: IndirectCommandsLayoutUsageFlags::EXPLICIT_PREPROCESS
                    // Unordered sequences should make the execution more parallel but it does not seem to have a measurable impact
                    | IndirectCommandsLayoutUsageFlags::UNORDERED_SEQUENCES,
                pipeline_bind_point: PipelineBindPoint::Compute,
                tokens: vec![
                    // First token binds the pipeline
                    IndirectCommandsLayoutToken {
                        token_type: IndirectCommandsTokenType::Pipeline,
                        stream: 0,
                        ..IndirectCommandsLayoutToken::default()
                    },
                    // Then we pass the material IDs as push constants
                    IndirectCommandsLayoutToken {
                        token_type: IndirectCommandsTokenType::PushConstant,
                        pushconstant_data: Some(IndirectCommandsLayoutTokenPushConstant {
                            pipeline_layout: data
                                .global_data
                                .shader_object()
                                .pipeline_layout()
                                .clone(),
                            shader_stage_flags: ShaderStages::COMPUTE,
                            offset: 0,
                            size: size_of::<VisBufferPushConstant>() as u32,
                        }),
                        stream: 1,
                        ..IndirectCommandsLayoutToken::default()
                    },
                    // Then we dispatch compute shaders to write to the output target
                    IndirectCommandsLayoutToken {
                        token_type: IndirectCommandsTokenType::Dispatch,
                        stream: 2,
                        ..IndirectCommandsLayoutToken::default()
                    },
                ],
                stream_strides: vec![
                    size_of::<PipelineBindParameter>() as u32,
                    size_of::<VisBufferPushConstant>() as u32,
                    size_of::<ComputeDispatchParameter>() as u32,
                ],
                ..IndirectCommandsLayoutCreateInfo::default()
            },
        )
        .unwrap();

        // Allocate the preprocess buffer according to the memory requirements
        let requirements = commands_layout.memory_requirements(
            &GeneratedCommandsPipeline::Dynamic(),
            Self::MAX_SEQUENCE_COUNT,
        );
        let preprocess_buffer = Subbuffer::new(
            Buffer::new(
                rhi.buffer_allocator().clone(),
                BufferCreateInfo {
                    usage: BufferUsage::INDIRECT_BUFFER | BufferUsage::TRANSFER_DST,
                    ..BufferCreateInfo::default()
                },
                AllocationCreateInfo::default(),
                requirements.layout,
            )
            .unwrap(),
        );

        // Write data into the shader objects
        let cursor = ShaderCursor::new(data.global_data.shader_object().clone());
        cursor
            .field("visBuffer")
            .unwrap()
            .write_swapchain_image(data.visibility_buffer.clone());
        cursor
            .field("outputRT")
            .unwrap()
            .write_swapchain_image(data.final_render_target.clone());

        let bin_cursor = cursor.field("gBinInput").unwrap();
        bin_cursor
            .field("texelCounts")
            .unwrap()
            .write_buffer(data.material_fragment_count_buffer.clone());
        bin_cursor
            .field("offsets")
            .unwrap()
            .write_buffer(data.per_material_offset_buffer.clone());
        bin_cursor
            .field("binnedTexels")
            .unwrap()
            .write_buffer(data.binned_texel_buffer.clone());

        data.global_data
            .write_to_shader_cursor(&mut cursor.field("gGlobalData").unwrap());

        Self {
            rhi,
            commands_layout,
            preprocess_buffer,
            data,
        }
    }

    /// When supporting render doc, no commands are recorded
    #[cfg(feature = "renderdoc_compatibility")]
    pub fn record_command_buffer(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_index: usize,
    ) -> Result<(), Box<ValidationError>> {
        Ok(())
    }

    #[cfg(not(any(feature = "renderdoc_compatibility", feature = "no_indirect")))]
    pub fn record_command_buffer(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_index: usize,
    ) -> Result<(), Box<ValidationError>> {
        // Bind the global descriptor set (this will be used by all pipelines dispatched from the execute indirect commands below)
        let shader_object = self.data.global_data.shader_object();
        command_buffer.bind_descriptor_sets(
            PipelineBindPoint::Compute,
            shader_object.pipeline_layout().clone(),
            0,
            shader_object.descriptor_sets()[image_index].clone(),
        )?;

        // In the naive implementation, we use the number of materials as the (static) sequence count
        // In other implementations, we use the max sequence count as the static upper bound and material count buffer as the actual, GPU-driven, count
        let sequence_count = if cfg!(feature = "no_cull_visbuffer") {
            self.data.global_data.num_materials()
        } else {
            Self::MAX_SEQUENCE_COUNT
        };
        let sequence_count_buffer = if cfg!(feature = "no_cull_visbuffer") {
            None
        } else {
            Some(self.data.final_material_count_buffer.clone())
        };

        // Build the commands info using the commands streams and sequence counts
        let commands_info = GeneratedCommandsInfo {
            streams: vec![
                IndirectCommandsStream {
                    buffer: self.data.pipeline_bind_commands.clone().reinterpret(),
                },
                IndirectCommandsStream {
                    buffer: self.data.push_constants.clone().reinterpret(),
                },
                IndirectCommandsStream {
                    buffer: self.data.compute_dispatch_commands.clone().reinterpret(),
                },
            ],
            sequence_count,
            sequence_count_buffer,
            // We bind pipelines using pipeline tokens
            ..GeneratedCommandsInfo::dynamic_pipeline(
                self.commands_layout.clone(),
                self.preprocess_buffer.clone(),
            )
        };

        // Explicit preprocessing step
        unsafe { command_buffer.preprocess_generated_commands(commands_info.clone()) }?;

        // Execute the generated commands
        unsafe { command_buffer.execute_generated_commands(true, commands_info) }?;

        Ok(())
    }

    #[cfg(feature = "no_indirect")]
    pub fn record_command_buffer(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_index: usize,
    ) -> Result<(), Box<ValidationError>> {
        // Bind the global descriptor set. This will be used by all pipelines
        let shader_object = self.data.global_data.shader_object();
        command_buffer.bind_descriptor_sets(
            PipelineBindPoint::Compute,
            shader_object.pipeline_layout().clone(),
            0,
            shader_object.descriptor_sets()[image_index].clone(),
        )?;

        // The below mimics what the indirect commands layout does, except that is has to use all pipelines

        // For every material/pipeline
        self.data
            .global_data
            .pipelines
            .iter()
            .enumerate()
            .for_each(|(index, pipeline)| {
                // Bind the pipeline
                command_buffer
                    .bind_pipeline_compute(pipeline.clone())
                    .unwrap();

                // Pass the ID/index as a push constant
                command_buffer
                    .push_constants(shader_object.pipeline_layout().clone(), 0, index as u32)
                    .unwrap();

                // Dispatch a compute shader, indirectly and based on the computed dispatch size
                unsafe {
                    command_buffer.dispatch_indirect(
                        self.data
                            .compute_dispatch_commands
                            .clone()
                            .slice(index as DeviceSize..=index as DeviceSize)
                            .reinterpret(),
                    )
                }
                .unwrap();
            });

        Ok(())
    }
}
