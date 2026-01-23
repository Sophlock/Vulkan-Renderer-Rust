use crate::application::renderer::shader_cursor::{ShaderOffset, ShaderSize};
use crate::{
    application::assets::asset_traits::Vertex, application::renderer::pipeline::graphics_pipeline,
};
use shader_slang::reflection::{TypeLayout, VariableLayout};
use shader_slang::{BindingType, ParameterCategory};
use std::collections::BTreeMap;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::descriptor_set::DescriptorSet;
use vulkano::descriptor_set::allocator::{
    DescriptorSetAllocator, StandardDescriptorSetAllocator,
    StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::descriptor_set::layout::{
    DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType,
};
use vulkano::descriptor_set::pool::{
    DescriptorPool, DescriptorPoolCreateFlags, DescriptorPoolCreateInfo, DescriptorSetAllocateInfo,
};
use vulkano::memory::allocator::{
    AllocationCreateInfo, DeviceLayout, MemoryAllocator, MemoryTypeFilter,
};
use vulkano::shader::ShaderStages;
use vulkano::sync::Sharing;
use vulkano::{
    DeviceSize, descriptor_set,
    device::Device,
    pipeline::{
        DynamicState, GraphicsPipeline, PipelineLayout, graphics::subpass::PipelineSubpassType,
        layout::PipelineLayoutCreateInfo,
    },
    render_pass::RenderPass,
};

pub struct ShaderObjectLayout {
    pipeline_layout: Arc<PipelineLayout>,
    descriptor_set_layout: Arc<DescriptorSetLayout>,
    descriptor_pool: DescriptorPool,
    existential_sizes: Vec<ShaderSize>,
    existential_offsets: Vec<ShaderOffset>,
    type_layout: *const TypeLayout
}

pub struct ShaderObject {
    layout: Arc<ShaderObjectLayout>,
    pipeline: Arc<GraphicsPipeline>,
    descriptor_sets: Vec<Arc<DescriptorSet>>,
    uniform_buffer: Option<Subbuffer<[u8]>>,
    type_layout: *const TypeLayout
}

impl ShaderObjectLayout {
    pub fn new(
        variable_layout: &VariableLayout,
        existential_objects: &[&TypeLayout],
        in_flight_frames: u32,
        device: &Arc<Device>,
    ) -> Arc<Self> {
        // TODO: This currently does not handle ParameterBlocks!
        // TODO: We don't need to support all shader stage flags

        let type_layout = variable_layout.type_layout().unwrap();

        let (existential_sizes, existential_offsets) =
            Self::build_sizes_offsets(type_layout, existential_objects);

        let ordinary_data = if type_layout.size(ParameterCategory::Uniform) > 0 {
            Some(DescriptorSetLayoutBinding {
                descriptor_count: 1,
                stages: ShaderStages::all_graphics(),
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
            })
        } else {
            None
        };

        let bindings =
            Self::bindings_for_layout(type_layout, type_layout.binding_range_count())
                .chain(ordinary_data)
                .chain(existential_objects.iter().zip(&existential_sizes).flat_map(
                    |(layout, size)| Self::bindings_for_layout(layout, size.binding_size as i64),
                ))
                .enumerate()
                .map(|(i, binding)| (i as u32, binding))
                .collect::<BTreeMap<_, _>>();

        let pool_sizes = (&bindings)
            .iter()
            .map(|(_, binding)| (binding.descriptor_type, in_flight_frames))
            .collect();

        let descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo {
                bindings,
                ..DescriptorSetLayoutCreateInfo::default()
            },
        )
        .unwrap();

        let descriptor_pool = DescriptorPool::new(
            device.clone(),
            DescriptorPoolCreateInfo {
                flags: DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
                max_sets: in_flight_frames,
                pool_sizes,
                ..DescriptorPoolCreateInfo::default()
            },
        )
        .unwrap();

        let pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![descriptor_set_layout.clone()],
                ..PipelineLayoutCreateInfo::default()
            },
        )
        .unwrap();

        Self {
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            existential_sizes,
            existential_offsets,
            type_layout
        }
        .into()
    }

    fn map_descriptor_type(binding_type: BindingType) -> DescriptorType {
        match binding_type {
            BindingType::Sampler => DescriptorType::Sampler,
            BindingType::Texture => DescriptorType::SampledImage,
            BindingType::ConstantBuffer | BindingType::ParameterBlock => {
                DescriptorType::UniformBuffer
            }
            BindingType::CombinedTextureSampler => DescriptorType::CombinedImageSampler,
            BindingType::InlineUniformData => DescriptorType::InlineUniformBlock,
            BindingType::RayTracingAccelerationStructure => DescriptorType::AccelerationStructure,
            BindingType::MutableTeture => DescriptorType::StorageImage,
            _ => DescriptorType::UniformBuffer, /*BindingType::TypedBuffer => {}
                                                BindingType::RawBuffer => {}
                                                BindingType::InputRenderTarget => {}
                                                BindingType::VaryingInput => {}
                                                BindingType::VaryingOutput => {}
                                                BindingType::ExistentialValue => {}
                                                BindingType::PushConstant => {}
                                                BindingType::MutableFlag => {}
                                                BindingType::MutableTypedBuffer => {}
                                                BindingType::MutableRawBuffer => {}
                                                BindingType::BaseMask => {}
                                                BindingType::ExtMask => {}
                                                BindingType::Unknown => {}*/
                                                // TODO: Missing: eUniformTexelBuffer, eStorageTexelBuffer, eUniformBuffer, eStorageBuffer, eUniformBufferDynamic, eStorageBufferDynamic, eInputAttachment, eMutableEXT
        }
    }

    fn bindings_for_layout(
        layout: &TypeLayout,
        size: i64,
    ) -> impl Iterator<Item = DescriptorSetLayoutBinding> + Clone {
        (0..size).map(|i| {
            let descriptor_type = Self::map_descriptor_type(layout.binding_range_type(i));
            DescriptorSetLayoutBinding {
                descriptor_count: layout.binding_range_binding_count(i) as u32,
                stages: ShaderStages::all_graphics(),
                ..DescriptorSetLayoutBinding::descriptor_type(descriptor_type)
            }
        })
    }

    fn build_sizes_offsets(
        type_layout: &TypeLayout,
        existential_layouts: &[&TypeLayout],
    ) -> (Vec<ShaderSize>, Vec<ShaderOffset>) {
        let sizes = existential_layouts
            .iter()
            .map(|layout| ShaderSize {
                byte_size: layout.size(ParameterCategory::Uniform),
                binding_size: layout.binding_range_count() as u32,
            })
            .collect::<Vec<_>>();

        let initial = ShaderOffset {
            byte_offset: type_layout
                .element_var_layout()
                .unwrap()
                .type_layout()
                .unwrap()
                .size(ParameterCategory::Uniform),
            binding_offset: type_layout.binding_range_count() as u32,
        };

        let offsets = (&sizes)
            .iter()
            .scan(initial, |offset, size| {
                let current = offset.clone();
                offset.byte_offset += size.byte_size;
                offset.binding_offset += size.binding_size;
                Some(current)
            })
            .collect::<Vec<_>>();

        (sizes, offsets)
    }

    // TODO: I don't like the pointer here
    pub fn type_layout(&self) -> *const TypeLayout {
        self.type_layout
    }

    pub fn ordinary_data_size(&self) -> usize {
        let last_size = self
            .existential_sizes
            .last()
            .map(|s| s.byte_size)
            .unwrap_or(
                unsafe { &*self.type_layout() }
                    .element_var_layout()
                    .unwrap()
                    .type_layout()
                    .unwrap()
                    .size(ParameterCategory::Uniform),
            );
        let last_offset = self
            .existential_offsets
            .last()
            .map(|s| s.byte_offset)
            .unwrap_or(0);
        last_size + last_offset
    }

    pub fn binding_size(&self) -> u32 {
        let last_size = self
            .existential_sizes
            .last()
            .map(|s| s.binding_size)
            .unwrap_or(
                unsafe { &*self.type_layout() }
                    .element_var_layout()
                    .unwrap()
                    .type_layout()
                    .unwrap()
                    .binding_range_count() as u32,
            );
        let last_offset = self
            .existential_offsets
            .last()
            .map(|s| s.binding_offset)
            .unwrap_or(0);
        last_size + last_offset
    }
}

impl ShaderObject {
    pub fn new(
        device: &Arc<Device>,
        render_pass: Arc<RenderPass>,
        layout: Arc<ShaderObjectLayout>,
        descriptor_allocator: &Arc<dyn DescriptorSetAllocator>,
        buffer_allocator: &Arc<dyn MemoryAllocator>,
        in_flight_frames: u32,
        vert_spriv: &[u32],
        frag_spriv: &[u32],
    ) -> Self {
        let type_layout = unsafe {&*layout.type_layout()}.element_var_layout().unwrap().type_layout().unwrap();

        let buffer_info = BufferCreateInfo {
            sharing: Sharing::Exclusive,
            usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
            ..BufferCreateInfo::default()
        };
        let alloc_info = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..AllocationCreateInfo::default()
        };

        let ordinary_size = layout.ordinary_data_size();
        let uniform_buffer = if ordinary_size > 0 {
            Some(
                Buffer::new_slice::<u8>(
                    buffer_allocator.clone(),
                    buffer_info,
                    alloc_info,
                    ordinary_size as DeviceSize,
                )
                .unwrap(),
            )
        } else {
            None
        };

        let descriptor_sets = (0..in_flight_frames)
            .map(|_| {
                DescriptorSet::new(
                    descriptor_allocator.clone(),
                    layout.descriptor_set_layout.clone(),
                    [],
                    [],
                )
                .unwrap()
            })
            .collect();

        let pipeline = graphics_pipeline()
            .input_assembly(None, None)
            .vertex_shader(device.clone(), vert_spriv)
            .vertex_input::<Vertex>()
            .rasterizer(None, None, None, None, None, None)
            .skip_multisample()
            .fragment_shader(device.clone(), frag_spriv)
            .opaque_color_blend()
            .default_depth_test()
            .build_pipeline(
                device.clone(),
                layout.pipeline_layout.clone(),
                PipelineSubpassType::BeginRenderPass(render_pass.first_subpass()),
                [
                    DynamicState::ViewportWithCount,
                    DynamicState::ScissorWithCount,
                ]
                .into(),
            );
        Self {
            layout,
            pipeline,
            descriptor_sets,
            uniform_buffer,
            type_layout
        }
    }
}
