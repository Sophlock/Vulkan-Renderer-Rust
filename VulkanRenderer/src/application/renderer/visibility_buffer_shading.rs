use crate::application::renderer::device_generated_commands::{
    execute_generated_commands, map_pipeline_bind_point, GeneratedCommandsInfo,
    IndirectCommandsLayout, IndirectCommandsLayoutCreateInfo,
};
use crate::application::rhi::device_helper::{ash_device, ash_instance};
use crate::application::rhi::VKRHI;
use ash::vk::{
    DeviceAddress,
    IndirectCommandsLayoutTokenNV, IndirectCommandsTokenTypeNV
    , PipelineIndirectDeviceAddressInfoNV,
};
use std::rc::Rc;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::device::DeviceOwned;
use vulkano::instance::InstanceOwned;
use vulkano::memory::allocator::AllocationCreateInfo;
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use vulkano::sync::{now, GpuFuture};
use vulkano::VulkanObject;

pub struct VisibilityBufferShadePass {
    rhi: Rc<VKRHI>,
    commands_layout: Arc<IndirectCommandsLayout>,
    preprocess_buffer: Subbuffer<[u8]>,
    sequence_count_buffer: Subbuffer<u32>,
    pipeline_bind_commands: Subbuffer<[PipelineBindParameter]>,
    compute_dispatch_commands: Subbuffer<[ComputeDispatchParameter]>,
}

#[derive(Copy, Clone, BufferContents)]
#[repr(C)]
struct PipelineBindParameter {
    pub pipeline_address: DeviceAddress,
}

#[derive(Copy, Clone, BufferContents)]
#[repr(C)]
struct ComputeDispatchParameter {
    pub dispatch: [u32; 3],
}

impl VisibilityBufferShadePass {
    const MAX_SEQUENCE_COUNT: u32 = 1000u32;
    pub fn new(rhi: Rc<VKRHI>) -> Self {
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

        let sequence_count_buffer = Buffer::new_sized(
            rhi.buffer_allocator().clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDIRECT_BUFFER,
                ..BufferCreateInfo::default()
            },
            AllocationCreateInfo::default(),
        ).unwrap();

        let pipeline_bind_commands = Buffer::new_slice(
            rhi.buffer_allocator().clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDIRECT_BUFFER,
                ..BufferCreateInfo::default()
            },
            AllocationCreateInfo::default(),
            Self::MAX_SEQUENCE_COUNT.into(),
        )
        .unwrap();

        let compute_dispatch_commands = Buffer::new_slice(
            rhi.buffer_allocator().clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDIRECT_BUFFER,
                ..BufferCreateInfo::default()
            },
            AllocationCreateInfo::default(),
            Self::MAX_SEQUENCE_COUNT.into(),
        )
        .unwrap();

        Self {
            rhi,
            commands_layout,
            preprocess_buffer,
            sequence_count_buffer,
            pipeline_bind_commands,
            compute_dispatch_commands
        }
    }

    pub fn run(&self, command_buffer: &Arc<PrimaryAutoCommandBuffer>) {
        let commands_info = GeneratedCommandsInfo {
            streams: vec![
                self.pipeline_bind_commands.clone().reinterpret(),
                self.compute_dispatch_commands.clone().reinterpret(),
            ],
            max_sequences: Self::MAX_SEQUENCE_COUNT,
            sequence_count_buffer: Some(self.sequence_count_buffer.clone()),
            ..GeneratedCommandsInfo::layout(
                self.commands_layout.clone(),
                self.preprocess_buffer.clone(),
            )
        };
        unsafe { execute_generated_commands(command_buffer, false, commands_info) };
        let _ = now(self.rhi.device().clone())
            .then_execute(
                self.rhi.queues().compute_queue.clone(),
                command_buffer.clone(),
            )
            .unwrap();
    }
}

impl PipelineBindParameter {
    pub fn pipeline(
        pipeline: &Arc<impl Pipeline + VulkanObject<Handle = ash::vk::Pipeline>>,
    ) -> Self {
        let address_info = PipelineIndirectDeviceAddressInfoNV::default()
            .pipeline(pipeline.handle())
            .pipeline_bind_point(map_pipeline_bind_point(pipeline.bind_point()));

        let instance = unsafe { ash_instance(pipeline.device().instance()) };
        let device = unsafe { ash_device(pipeline.device()) };
        let dgc_device =
            ash::nv::device_generated_commands_compute::Device::new(&instance, &device);
        let address = unsafe { dgc_device.get_pipeline_indirect_device_address(&address_info) };
        Self {
            pipeline_address: address,
        }
    }
}
