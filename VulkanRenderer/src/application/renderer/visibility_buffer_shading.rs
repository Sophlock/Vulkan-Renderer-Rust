use std::{rc::Rc, sync::Arc};

use ash::vk::{IndirectCommandsLayoutTokenNV, IndirectCommandsTokenTypeNV};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    device::DeviceOwned,
    memory::allocator::AllocationCreateInfo,
    pipeline::{Pipeline, PipelineBindPoint},
};

use crate::application::{
    renderer::{
        device_generated_commands::{
            GeneratedCommandsInfo, IndirectCommandsLayout, IndirectCommandsLayoutCreateInfo,
            execute_generated_commands,
        },
        visibility_buffer_data::VisibilityBufferData,
        visibility_buffer_generation::{ComputeDispatchParameter, PipelineBindParameter},
    },
    rhi::VKRHI,
};

pub struct VisibilityBufferShadePass {
    rhi: Rc<VKRHI>,
    commands_layout: Arc<IndirectCommandsLayout>,
    preprocess_buffer: Subbuffer<[u8]>,
    data: VisibilityBufferData,
}

impl VisibilityBufferShadePass {
    pub const MAX_SEQUENCE_COUNT: u32 = 1000u32;
    pub fn new(rhi: Rc<VKRHI>, data: VisibilityBufferData) -> Self {
        let commands_layout = IndirectCommandsLayout::new(
            rhi.device().clone(),
            IndirectCommandsLayoutCreateInfo {
                flags: Default::default(),
                pipeline_bind_point: PipelineBindPoint::Compute,
                tokens: vec![
                    IndirectCommandsLayoutTokenNV::default()
                        .token_type(IndirectCommandsTokenTypeNV::PIPELINE)
                        .stream(0),
                    IndirectCommandsLayoutTokenNV::default()
                        .token_type(IndirectCommandsTokenTypeNV::DISPATCH)
                        .stream(1),
                ],
                strides: vec![
                    size_of::<PipelineBindParameter>() as u32,
                    size_of::<ComputeDispatchParameter>() as u32,
                ],
                ..IndirectCommandsLayoutCreateInfo::default()
            },
        )
        .unwrap();

        let requirements = commands_layout.memory_requirements(Self::MAX_SEQUENCE_COUNT);
        let preprocess_buffer = Subbuffer::new(
            Buffer::new(
                rhi.buffer_allocator().clone(),
                BufferCreateInfo {
                    usage: BufferUsage::INDIRECT_BUFFER,
                    ..BufferCreateInfo::default()
                },
                AllocationCreateInfo::default(),
                requirements.layout,
            )
            .unwrap(),
        );

        Self {
            rhi,
            commands_layout,
            preprocess_buffer,
            data,
        }
    }

    pub fn run(
        &self,
        command_buffer: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> Arc<PrimaryAutoCommandBuffer> {
        let built_command_buffer = command_buffer.build().unwrap();
        let commands_info = GeneratedCommandsInfo {
            streams: vec![
                self.data.pipeline_bind_commands.clone().reinterpret(),
                self.data.compute_dispatch_commands.clone().reinterpret(),
            ],
            max_sequences: Self::MAX_SEQUENCE_COUNT,
            sequence_count_buffer: Some(self.data.index_counter_buffer.clone()),
            ..GeneratedCommandsInfo::layout(
                self.commands_layout.clone(),
                self.preprocess_buffer.clone(),
            )
        };
        unsafe { execute_generated_commands(&built_command_buffer, true, commands_info) };
        built_command_buffer
    }
}
