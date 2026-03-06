use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex, RwLock},
};

use shader_slang::{BindingType, ComponentType, ParameterCategory, reflection::TypeLayout};
use vulkano::{
    DeviceSize,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    descriptor_set::{
        DescriptorImageViewInfo, DescriptorSet, WriteDescriptorSet,
        allocator::DescriptorSetAllocator,
        layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
            DescriptorType,
        },
    },
    device::{Device, DeviceOwned},
    image::{ImageLayout, sampler::Sampler, view::ImageView},
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
    pipeline::{PipelineLayout, layout::PipelineLayoutCreateInfo},
    shader::ShaderStages,
    sync::Sharing,
};
use vulkano::pipeline::layout::PushConstantRange;
use crate::application::rhi::{
    rhi_assets::vulkan_texture::VKTexture,
    shader_cursor::{ShaderOffset, ShaderSize},
    shader_object::BoundImageType::ImageSampler,
    swapchain_resources::SwapchainImage,
};

pub struct ShaderObjectLayout {
    pipeline_layout: Arc<PipelineLayout>,
    descriptor_set_layout: Arc<DescriptorSetLayout>,
    existential_sizes: Vec<ShaderSize>,
    existential_offsets: Vec<ShaderOffset>,
    type_layout: *const TypeLayout,
    linked_program: ComponentType,
}

pub struct ShaderObject {
    layout: Arc<ShaderObjectLayout>,
    descriptor_sets: Vec<Arc<DescriptorSet>>,
    uniform_buffer: Option<Subbuffer<[u8]>>,
    type_layout: *const TypeLayout,
    swapchain_resources: Mutex<BoundSwapchainResources>,
}

enum BoundImageType {
    Image(Arc<RwLock<SwapchainImage>>),
    ImageSampler(Arc<RwLock<SwapchainImage>>, Arc<Sampler>),
}

struct BoundSwapchainResources {
    bound_images: BTreeMap<(u32, u32), BoundImageType>,
}

impl ShaderObjectLayout {
    pub fn new(
        linked_program: ComponentType,
        existential_objects: &[&TypeLayout],
        device: &Arc<Device>,
        shader_stages: ShaderStages
    ) -> Arc<Self> {
        Self::new_with_push_constants(linked_program, existential_objects, device, shader_stages, vec![])
    }
    pub fn new_with_push_constants(
        linked_program: ComponentType,
        existential_objects: &[&TypeLayout],
        device: &Arc<Device>,
        shader_stages: ShaderStages,
        push_constant_ranges: Vec<PushConstantRange> 
    ) -> Arc<Self> {
        // TODO: This currently does not handle ParameterBlocks!

        let variable_layout = linked_program
            .layout(0)
            .unwrap()
            .global_params_var_layout()
            .unwrap();

        let type_layout = variable_layout.type_layout().unwrap();

        let inner_type_layout = type_layout.element_type_layout().unwrap();
        for field in inner_type_layout.fields() {
            println!(
                "Field: {:?}, Type: {:?}",
                field.name(),
                field.type_layout().unwrap().name()
            );
            for i in 0..field.type_layout().unwrap().binding_range_count() {
                println!(
                    "\t{} binding of type {:?}",
                    field.type_layout().unwrap().binding_range_binding_count(i),
                    field.type_layout().unwrap().binding_range_type(i)
                );
            }
            for sub_field in field.type_layout().unwrap().fields() {
                println!(
                    "\tField: {:?}, Type: {:?}, Binding ranges: {}, Category: {:?}, Size: {}",
                    sub_field.name(),
                    sub_field.type_layout().unwrap().name(),
                    sub_field.type_layout().unwrap().binding_range_count(),
                    sub_field.type_layout().unwrap().parameter_category(),
                    sub_field
                        .type_layout()
                        .unwrap()
                        .size(ParameterCategory::Uniform)
                );
                for category in sub_field.categories() {
                    println!(
                        "\t\tCategory {:?} has size {}",
                        category,
                        sub_field.type_layout().unwrap().size(category)
                    );
                }
                for i in 0..sub_field.type_layout().unwrap().binding_range_count() {
                    println!(
                        "\t\t{} binding of type {:?}",
                        sub_field
                            .type_layout()
                            .unwrap()
                            .binding_range_binding_count(i),
                        sub_field.type_layout().unwrap().binding_range_type(i)
                    );
                }
                for sub_sub_field in sub_field.type_layout().unwrap().fields() {
                    println!(
                        "\t\tField: {:?}, Type: {:?}, Binding ranges: {}",
                        sub_sub_field.name(),
                        sub_sub_field.type_layout().unwrap().name(),
                        sub_sub_field.type_layout().unwrap().binding_range_count()
                    );
                }
            }
        }
        println!(
            "Shader has {} descriptor sets",
            type_layout.descriptor_set_count()
        );
        for i in 0..type_layout.descriptor_set_count() {
            let range_count = type_layout.descriptor_set_descriptor_range_count(i);
            println!("Descriptor set {} has {} descriptor ranges", i, range_count);
            for j in 0..range_count {
                println!(
                    "\tRange {} has {} bindings of binding type {:?} and category {:?}",
                    j,
                    type_layout.descriptor_set_descriptor_range_descriptor_count(i, j),
                    type_layout.descriptor_set_descriptor_range_type(i, j),
                    type_layout.descriptor_set_descriptor_range_category(i, j)
                );
            }
        }
        println!("Categories:");
        for category in type_layout.categories() {
            println!(
                "Category {:?} has size {}",
                category,
                type_layout.size(category)
            );
        }

        let (existential_sizes, existential_offsets) =
            Self::build_sizes_offsets(type_layout, existential_objects);

        let ordinary_data = if inner_type_layout.size(ParameterCategory::Uniform) > 0 {
            Some(DescriptorSetLayoutBinding {
                descriptor_count: 1,
                stages: shader_stages,
                ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
            })
        } else {
            None
        };

        let bindings =
            ordinary_data
                .iter()
                .cloned()
                .chain(Self::bindings_for_layout(
                    inner_type_layout,
                    inner_type_layout.binding_range_count(),
                    shader_stages,
                ))
                .chain(existential_objects.iter().zip(&existential_sizes).flat_map(
                    |(layout, size)| {
                        Self::bindings_for_layout(layout, size.binding_size as i64, shader_stages)
                    },
                ))
                .enumerate()
                .map(|(i, binding)| (i as u32, binding))
                .collect::<BTreeMap<_, _>>();

        let descriptor_set_layout = DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo {
                bindings,
                ..DescriptorSetLayoutCreateInfo::default()
            },
        )
        .unwrap();

        let pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![descriptor_set_layout.clone()],
                push_constant_ranges,
                ..PipelineLayoutCreateInfo::default()
            },
        )
        .unwrap();

        Self {
            pipeline_layout,
            descriptor_set_layout,
            existential_sizes,
            existential_offsets,
            type_layout,
            linked_program,
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
            BindingType::InputRenderTarget => DescriptorType::InputAttachment,
            BindingType::RawBuffer => DescriptorType::StorageBuffer,
            BindingType::MutableRawBuffer => DescriptorType::StorageBuffer,
            BindingType::PushConstant => panic!("Push constants cannot be mapped to a descriptor type!"),
            _ => DescriptorType::UniformBuffer, //panic!("Unknown slang binding type {:?}", binding_type),
                                                /*BindingType::TypedBuffer => {}
                                                BindingType::RawBuffer => {}
                                                BindingType::InputRenderTarget => {}
                                                BindingType::VaryingInput => {}
                                                BindingType::VaryingOutput => {}
                                                BindingType::ExistentialValue => {}
                                                BindingType::MutableFlag => {}
                                                BindingType::MutableTypedBuffer => {}
                                                BindingType::BaseMask => {}
                                                BindingType::ExtMask => {}
                                                BindingType::Unknown => {}*/
                                                // TODO: Missing: eUniformTexelBuffer, eStorageTexelBuffer, eUniformBuffer, eStorageBuffer, eUniformBufferDynamic, eStorageBufferDynamic, eInputAttachment, eMutableEXT
        }
    }

    fn bindings_for_layout(
        layout: &TypeLayout,
        size: i64,
        shader_stages: ShaderStages,
    ) -> impl Iterator<Item = DescriptorSetLayoutBinding> + Clone {
        (0..size)
            .filter(|i| layout.binding_range_type(*i) != BindingType::PushConstant)
            .map(move |i| {
            let descriptor_type = Self::map_descriptor_type(layout.binding_range_type(i));
            DescriptorSetLayoutBinding {
                descriptor_count: layout.binding_range_binding_count(i) as u32,
                stages: shader_stages,
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
            binding_array_element: 0,
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
    pub fn type_layout(&self) -> &TypeLayout {
        unsafe { &*self.type_layout }
    }

    pub fn ordinary_data_size(&self) -> usize {
        let last_size = self
            .existential_sizes
            .last()
            .map(|s| s.byte_size)
            .unwrap_or(
                self.type_layout()
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
                self.type_layout()
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

    pub fn pipeline_layout(&self) -> &Arc<PipelineLayout> {
        &self.pipeline_layout
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.pipeline_layout.device()
    }
}

impl ShaderObject {
    pub fn new(
        layout: Arc<ShaderObjectLayout>,
        descriptor_allocator: &Arc<dyn DescriptorSetAllocator>,
        buffer_allocator: &Arc<dyn MemoryAllocator>,
        in_flight_frames: u32,
    ) -> Arc<Self> {
        let type_layout = layout
            .type_layout()
            .element_var_layout()
            .unwrap()
            .type_layout()
            .unwrap();

        let buffer_info = BufferCreateInfo {
            sharing: Sharing::Exclusive,
            usage: BufferUsage::UNIFORM_BUFFER | BufferUsage::TRANSFER_DST,
            ..BufferCreateInfo::default()
        };
        let alloc_info = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..AllocationCreateInfo::default()
        };

        // TODO: Expose option to make this buffer be host visible and use staging setup
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

        let initial_writes = uniform_buffer
            .clone()
            .map(|buffer| WriteDescriptorSet::buffer(0, buffer));

        let descriptor_sets = (0..in_flight_frames)
            .map(|_| {
                DescriptorSet::new(
                    descriptor_allocator.clone(),
                    layout.descriptor_set_layout.clone(),
                    initial_writes.clone(),
                    [],
                )
                .unwrap()
            })
            .collect();

        Self {
            descriptor_sets,
            uniform_buffer,
            type_layout,
            layout,
            swapchain_resources: Mutex::new(BoundSwapchainResources::default()),
        }
        .into()
    }

    fn device(&self) -> &Arc<Device> {
        self.layout.device()
    }

    pub fn layout(&self) -> &Arc<ShaderObjectLayout> {
        &self.layout
    }

    pub fn pipeline_layout(&self) -> &Arc<PipelineLayout> {
        &self.layout.pipeline_layout
    }

    pub fn type_layout(&self) -> &TypeLayout {
        unsafe { &*self.type_layout }
    }

    pub fn existential_to_offset(&self, existential: usize) -> ShaderOffset {
        self.layout.existential_offsets[existential]
    }

    pub fn write_data<T: BufferContents + Clone>(&self, offset: ShaderOffset, data: &T) {
        let mut content = self.uniform_buffer.as_ref().unwrap().write().unwrap();
        let pos = (&mut content[offset.byte_offset] as *mut u8).cast::<T>();
        unsafe { *pos = data.clone() };
    }

    pub fn write_texture(&self, offset: ShaderOffset, texture: &VKTexture) {
        self.write_image_view(offset, texture.image_view().clone());
    }

    pub fn write_image_view(&self, offset: ShaderOffset, view: Arc<ImageView>) {
        let write = WriteDescriptorSet::image_view_with_layout(
            offset.binding_offset,
            DescriptorImageViewInfo {
                image_view: view,
                image_layout: ImageLayout::ShaderReadOnlyOptimal, // TODO: Is this always correct?
            },
        );
        self.perform_descriptor_write([write].iter().cloned());
    }

    pub fn write_sampler(&self, offset: ShaderOffset, sampler: Arc<Sampler>) {
        let write = WriteDescriptorSet::sampler(offset.binding_offset, sampler);
        self.perform_descriptor_write([write].iter().cloned());
    }

    pub fn write_image_view_sampler(
        &self,
        offset: ShaderOffset,
        view: Arc<ImageView>,
        sampler: Arc<Sampler>,
    ) {
        let write = WriteDescriptorSet::image_view_with_layout_sampler(
            offset.binding_offset,
            DescriptorImageViewInfo {
                image_view: view,
                image_layout: ImageLayout::ShaderReadOnlyOptimal, // TODO: Is this always correct?
            },
            sampler,
        );
        self.perform_descriptor_write([write].iter().cloned());
    }

    pub fn write_buffer<T: ?Sized>(&self, offset: ShaderOffset, buffer: Subbuffer<T>) {
        let write = WriteDescriptorSet::buffer(offset.binding_offset, buffer);
        self.perform_descriptor_write([write].iter().cloned());
    }

    pub fn write_swapchain_image(
        self: &Arc<Self>,
        offset: ShaderOffset,
        image: Arc<RwLock<SwapchainImage>>,
    ) {
        self.write_image_view(offset, image.read().unwrap().image_view().clone());
        self.register_swapchain_image(offset, BoundImageType::Image(image));
    }

    pub fn write_swapchain_image_sampler(
        self: &Arc<Self>,
        offset: ShaderOffset,
        image: Arc<RwLock<SwapchainImage>>,
        sampler: Arc<Sampler>,
    ) {
        self.write_image_view_sampler(
            offset,
            image.read().unwrap().image_view().clone(),
            sampler.clone(),
        );
        self.register_swapchain_image(offset, BoundImageType::ImageSampler(image, sampler));
    }

    fn perform_descriptor_write<T>(&self, writes: T)
    where
        T: Iterator<Item = WriteDescriptorSet>,
        T: Clone,
    {
        self.descriptor_sets
            .iter()
            // TODO: This should use the safe checked version but this requires manual layout transitions
            .for_each(|set| {
                if let Err(error) = unsafe {set.update_by_ref(writes.clone(), [])} {
                    println!("Warning: Descriptor write error occurred: {}\nNote that this might just be a bug in Vulkano!", error);
                    unsafe { set.update_by_ref_unchecked(writes.clone(), []) }
                }
                });
    }

    fn register_swapchain_image(self: &Arc<Self>, offset: ShaderOffset, image: BoundImageType) {
        let mut resources = self.swapchain_resources.lock().unwrap();
        let position = (0u32, offset.binding_offset);

        image
            .image()
            .write()
            .unwrap()
            .register_shader_object(position, self.clone());

        let bound = resources.bound_images.insert(position, image);
        if let Some(bound) = bound {
            bound
                .image()
                .write()
                .unwrap()
                .unregister_shader_object(position, self);
        }
    }

    pub fn reload_swapchain_image(self: &Arc<Self>, position: &(u32, u32)) {
        // TODO: Non swapchain images should unbind swapchain images as well when they are written
        let resources = self.swapchain_resources.lock().unwrap();
        let image = resources.bound_images.get(position).unwrap();
        match image {
            BoundImageType::Image(image) => {
                self.write_image_view(
                    ShaderOffset {
                        binding_offset: position.1,
                        ..ShaderOffset::default()
                    },
                    image.read().unwrap().image_view().clone(),
                );
            }
            ImageSampler(image, sampler) => {
                self.write_image_view_sampler(
                    ShaderOffset {
                        binding_offset: position.1,
                        ..ShaderOffset::default()
                    },
                    image.read().unwrap().image_view().clone(),
                    sampler.clone(),
                );
            }
        }
    }

    pub fn descriptor_sets(&self) -> &[Arc<DescriptorSet>] {
        self.descriptor_sets.as_slice()
    }
}

impl BoundImageType {
    fn image(&self) -> &Arc<RwLock<SwapchainImage>> {
        match self {
            BoundImageType::Image(image) => image,
            BoundImageType::ImageSampler(image, _) => image,
        }
    }
}

impl Default for BoundSwapchainResources {
    fn default() -> Self {
        Self {
            bound_images: Default::default(),
        }
    }
}
