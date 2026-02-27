use std::{ptr::null, sync::Arc};

use ash::{
    nv, vk,
    vk::{
        GeneratedCommandsInfoNV, GeneratedCommandsMemoryRequirementsInfoNV,
        IndirectCommandsLayoutTokenNV, IndirectCommandsLayoutUsageFlagsNV,
        IndirectCommandsStreamNV, MemoryRequirements2,
    },
};
use nv::device_generated_commands as dgc;
use vulkano::{
    VulkanObject,
    buffer::Subbuffer,
    device::{Device, DeviceOwned},
    memory::{MemoryRequirements, allocator::DeviceLayout},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
};

use crate::application::rhi::device_helper::{ash_device, ash_instance};

pub struct IndirectCommandsLayout {
    handle: vk::IndirectCommandsLayoutNV,
    device: Arc<Device>,
}

pub struct IndirectCommandsLayoutCreateInfo<'a> {
    pub flags: IndirectCommandsLayoutUsageFlagsNV,
    pub pipeline_bind_point: PipelineBindPoint,
    pub tokens: Vec<IndirectCommandsLayoutTokenNV<'a>>,
    pub strides: Vec<u32>,
}

pub struct GeneratedCommandsInfo<PipelineType: Pipeline> {
    pub pipeline: Option<Arc<PipelineType>>,
    pub indirect_commands_layout: Arc<IndirectCommandsLayout>,
    pub streams: Vec<Subbuffer<[u8]>>,
    pub max_sequences: u32,
    pub preprocess_buffer: Subbuffer<[u8]>,
    pub sequence_count_buffer: Option<Subbuffer<u32>>,
}

struct GeneratedCommandsInfoStorage<'a> {
    inner: GeneratedCommandsInfoNV<'a>,
    streams: Vec<IndirectCommandsStreamNV>,
}

impl IndirectCommandsLayout {
    pub fn new(
        device: Arc<Device>,
        create_info: IndirectCommandsLayoutCreateInfo,
    ) -> Result<Arc<Self>, vk::Result> {
        let dgc_device = Self::make_dgc_device(&device);
        let create_info = create_info.to_vk();
        let mut handle = vk::IndirectCommandsLayoutNV::null();
        let result = unsafe {
            (dgc_device.fp().create_indirect_commands_layout_nv)(
                dgc_device.device(),
                &create_info as *const _,
                null(),
                (&mut handle) as *const _ as *mut _,
            )
        };
        if result == vk::Result::SUCCESS {
            Ok(Self { handle, device }.into())
        } else {
            Err(result)
        }
    }

    pub fn memory_requirements(&self, max_sequence_count: u32) -> MemoryRequirements {
        let info = GeneratedCommandsMemoryRequirementsInfoNV::default()
            .pipeline_bind_point(vk::PipelineBindPoint::COMPUTE)
            .indirect_commands_layout(self.handle)
            .max_sequences_count(max_sequence_count);
        let device = Self::make_dgc_device(&self.device);
        let mut requirements_raw = MemoryRequirements2::default();
        unsafe {
            (device.fp().get_generated_commands_memory_requirements_nv)(
                device.device(),
                &info as *const _,
                (&mut requirements_raw) as *mut _,
            )
        };
        MemoryRequirements {
            layout: DeviceLayout::from_size_alignment(
                requirements_raw.memory_requirements.size,
                requirements_raw.memory_requirements.alignment,
            )
            .unwrap(),
            memory_type_bits: requirements_raw.memory_requirements.memory_type_bits,
            prefers_dedicated_allocation: false,
            requires_dedicated_allocation: false,
        }
    }

    fn make_dgc_device(device: &Arc<Device>) -> dgc::Device {
        let instance = device.instance();
        let vk_instance = unsafe { ash_instance(instance) };
        let vk_device = unsafe { ash_device(device) };
        dgc::Device::new(&vk_instance, &vk_device)
    }
}

impl Drop for IndirectCommandsLayout {
    fn drop(&mut self) {
        let dgc_device = Self::make_dgc_device(&self.device);
        unsafe {
            (dgc_device.fp().destroy_indirect_commands_layout_nv)(
                dgc_device.device(),
                self.handle,
                null(),
            );
        }
    }
}

impl IndirectCommandsLayoutCreateInfo<'_> {
    fn to_vk(&self) -> vk::IndirectCommandsLayoutCreateInfoNV {
        vk::IndirectCommandsLayoutCreateInfoNV::default()
            .flags(self.flags)
            .pipeline_bind_point(map_pipeline_bind_point(self.pipeline_bind_point))
            .tokens(self.tokens.as_slice())
            .stream_strides(self.strides.as_slice())
    }
}

impl Default for IndirectCommandsLayoutCreateInfo<'_> {
    fn default() -> Self {
        Self {
            flags: IndirectCommandsLayoutUsageFlagsNV::empty(),
            pipeline_bind_point: PipelineBindPoint::Graphics,
            tokens: vec![],
            strides: vec![],
        }
    }
}

pub fn map_pipeline_bind_point(pipeline_bind_point: PipelineBindPoint) -> vk::PipelineBindPoint {
    match pipeline_bind_point {
        PipelineBindPoint::Compute => vk::PipelineBindPoint::COMPUTE,
        PipelineBindPoint::Graphics => vk::PipelineBindPoint::GRAPHICS,
        PipelineBindPoint::RayTracing => vk::PipelineBindPoint::RAY_TRACING_KHR,
        _ => panic!("Unsupported pipeline bind point"),
    }
}

impl GeneratedCommandsInfo<ComputePipeline> {
    pub fn layout(
        indirect_commands_layout: Arc<IndirectCommandsLayout>,
        preprocess_buffer: Subbuffer<[u8]>,
    ) -> Self {
        Self {
            pipeline: None,
            indirect_commands_layout,
            streams: vec![],
            max_sequences: 0,
            preprocess_buffer,
            sequence_count_buffer: None,
        }
    }
}

impl<PipelineType: Pipeline + VulkanObject<Handle = vk::Pipeline>>
    GeneratedCommandsInfo<PipelineType>
{
    pub fn pipeline(
        pipeline: Arc<PipelineType>,
        indirect_commands_layout: Arc<IndirectCommandsLayout>,
        preprocess_buffer: Subbuffer<[u8]>,
    ) -> Self {
        Self {
            pipeline: Some(pipeline),
            indirect_commands_layout,
            streams: vec![],
            max_sequences: 0,
            preprocess_buffer,
            sequence_count_buffer: None,
        }
    }

    fn to_vk(&self) -> GeneratedCommandsInfoStorage {
        let stream_vec = self
            .streams
            .iter()
            .map(|b| {
                IndirectCommandsStreamNV::default()
                    .buffer(b.buffer().handle())
                    .offset(b.offset())
            })
            .collect::<Vec<_>>();
        GeneratedCommandsInfoStorage::new(
            stream_vec,
            vk::GeneratedCommandsInfoNV::default()
                .pipeline(
                    self.pipeline
                        .as_ref()
                        .map(|p| p.handle())
                        .unwrap_or(vk::Pipeline::null()),
                )
                .pipeline_bind_point(map_pipeline_bind_point(
                    self.pipeline
                        .as_ref()
                        .map(|p| p.bind_point())
                        .unwrap_or(PipelineBindPoint::Compute),
                ))
                .indirect_commands_layout(self.indirect_commands_layout.handle)
                //.streams(stream_vec.as_slice())
                .sequences_count(self.max_sequences)
                .preprocess_buffer(self.preprocess_buffer.buffer().handle())
                .preprocess_offset(self.preprocess_buffer.offset())
                .preprocess_size(self.preprocess_buffer.size())
                .sequences_count_buffer(
                    self.sequence_count_buffer
                        .as_ref()
                        .map(|b| b.buffer().handle())
                        .unwrap_or(vk::Buffer::null()),
                )
                .sequences_count_offset(
                    self.sequence_count_buffer
                        .as_ref()
                        .map(|b| b.offset())
                        .unwrap_or(0),
                ),
        )
    }
}

impl<'a> GeneratedCommandsInfoStorage<'a> {
    fn new(streams: Vec<IndirectCommandsStreamNV>, inner: GeneratedCommandsInfoNV<'a>) -> Self {
        Self { inner, streams }
    }

    fn get(&self) -> GeneratedCommandsInfoNV {
        self.inner.streams(self.streams.as_slice())
    }
}

pub unsafe fn execute_generated_commands<
    CommandBufferType: VulkanObject<Handle = vk::CommandBuffer>,
    PipelineType: Pipeline + VulkanObject<Handle = vk::Pipeline>,
>(
    command_buffer: &CommandBufferType,
    is_preprocessed: bool,
    generated_commands_info: GeneratedCommandsInfo<PipelineType>,
) {
    let raw_commands_info = generated_commands_info.to_vk();
    let dgc_device = IndirectCommandsLayout::make_dgc_device(
        &generated_commands_info.indirect_commands_layout.device,
    );
    unsafe {
        (dgc_device.fp().cmd_execute_generated_commands_nv)(
            command_buffer.handle(),
            is_preprocessed.into(),
            &raw_commands_info.get() as *const _,
        )
    };
}
