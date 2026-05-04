use std::{rc::Rc, sync::Arc};

use vulkano::{
    ValidationError,
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
use crate::application::{
    renderer::{
        visibility_buffer_data::VisibilityBufferData,
        visibility_buffer_generation::{
            ComputeDispatchParameter, PipelineBindParameter, VisBufferPushConstant,
        },
    },
    rhi::{VKRHI, shader_cursor::ShaderCursor},
};

pub struct VisibilityBufferShadePass {
    rhi: Rc<VKRHI>,
    #[cfg(not(feature = "renderdoc_compatibility"))]
    commands_layout: Arc<IndirectCommandsLayout>,
    #[cfg(not(feature = "renderdoc_compatibility"))]
    preprocess_buffer: Subbuffer<[u8]>,
    data: Arc<VisibilityBufferData>,
}

impl VisibilityBufferShadePass {
    pub const MAX_SEQUENCE_COUNT: u32 = 1000u32;

    #[cfg(feature = "renderdoc_compatibility")]
    pub fn new(rhi: Rc<VKRHI>, data: Arc<VisibilityBufferData>) -> Self {
        Self {
            rhi,
            data,
        }
    }

    #[cfg(not(feature = "renderdoc_compatibility"))]
    pub fn new(rhi: Rc<VKRHI>, data: Arc<VisibilityBufferData>) -> Self {
        let commands_layout = IndirectCommandsLayout::new(
            rhi.device().clone(),
            IndirectCommandsLayoutCreateInfo {
                flags: IndirectCommandsLayoutUsageFlags::EXPLICIT_PREPROCESS | IndirectCommandsLayoutUsageFlags::UNORDERED_SEQUENCES,
                pipeline_bind_point: PipelineBindPoint::Compute,
                tokens: vec![
                    IndirectCommandsLayoutToken {
                        token_type: IndirectCommandsTokenType::Pipeline,
                        stream: 0,
                        ..IndirectCommandsLayoutToken::default()
                    },
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


    #[cfg(feature = "renderdoc_compatibility")]
    pub fn record_command_buffer(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_index: usize,
    ) -> Result<(), Box<ValidationError>> {
        Ok(())
    }

    #[cfg(not(feature = "renderdoc_compatibility"))]
    pub fn record_command_buffer(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_index: usize,
    ) -> Result<(), Box<ValidationError>> {
        let shader_object = self.data.global_data.shader_object();
        command_buffer.bind_descriptor_sets(
            PipelineBindPoint::Compute,
            shader_object.pipeline_layout().clone(),
            0,
            shader_object.descriptor_sets()[image_index].clone(),
        )?;

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
            sequence_count: Self::MAX_SEQUENCE_COUNT,
            sequence_count_buffer: Some(self.data.final_material_count_buffer.clone()),
            ..GeneratedCommandsInfo::dynamic_pipeline(
                self.commands_layout.clone(),
                self.preprocess_buffer.clone(),
            )
        };

        unsafe { command_buffer.preprocess_generated_commands(commands_info.clone()) }?;

        unsafe { command_buffer.execute_generated_commands(true, commands_info) }?;

        Ok(())
    }
}
