use std::{
    ops::{Add, Deref},
    sync::{Arc, RwLock},
};
use std::collections::HashMap;
use crate::application::{
    assets::asset_traits::{Index, RHIInterface, RHIModelInterface, Vertex},
    renderer::visibility_buffer_generation::{
        ComputeDispatchParameter, PipelineBindParameter, VisBufferPushConstant,
    },
    rhi::{
        VKRHI,
        buffer::buffer_from_slice,
        pipeline::compute_pipeline,
        rhi_assets::{
            vulkan_material::VKMaterial, vulkan_material_instance::VKMaterialInstance,
            vulkan_mesh::VKMesh, vulkan_model::VKModel,
        },
        shader_cursor::ShaderCursor,
        shader_object::{ShaderObject, ShaderObjectLayout},
        swapchain::Swapchain,
        swapchain_resources::SwapchainImage,
    },
};
use shader_slang::{ComponentType, structs::specialization_arg::SpecializationArg, Blob};
use vulkano::command_buffer::{DrawIndexedIndirectCommand, DrawIndirectCommand};
use vulkano::{
    DeviceAddress, DeviceSize, ValidationError,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CopyBufferInfo, PrimaryAutoCommandBuffer},
    device_generated_commands::{ComputePipelineIndirectBufferInfo, IndirectCommandsLayout},
    format::Format,
    image::{ImageAspects, ImageUsage},
    memory::{
        DeviceAlignment,
        allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    },
    pipeline::{
        ComputePipeline, PipelineCreateFlags, PipelineLayout, compute::ComputePipelineCreateInfo,
        layout::PushConstantRange,
    },
    shader::{ShaderStages, spirv::bytes_to_words},
    sync::{GpuFuture, now},
};
use vulkano::device::Device;
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};

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

    // Holds the push constants
    pub push_constants: Subbuffer<[VisBufferPushConstant]>,

    // Used to clear data before writing
    clear_buffer: Subbuffer<[u32]>,

    // Global data (buffers etc.)
    pub global_data: VisibilityBufferGlobalData,

    // Final output rt
    pub final_render_target: Arc<RwLock<SwapchainImage>>,

    // Buffer with the global data pointers to be used as a push constant
    pub global_data_buffer: Subbuffer<VisBufferGlobalDataPointers>,
}

#[derive(Clone)]
pub struct VisibilityBufferGlobalData {
    pub instances: Subbuffer<[InstanceData]>,
    pub materials: Subbuffer<[MaterialData]>,
    pub material_instances: Subbuffer<[MaterialInstanceData]>,
    pub meshes: Subbuffer<[MeshData]>,
    pub indices: Subbuffer<[u32]>,
    pub vertices: Subbuffer<[Vertex]>,
    pub mutating_data: Subbuffer<MutatingData>,
    pipelines: Vec<Arc<ComputePipeline>>,
    shader_object: Arc<ShaderObject>,
    material_count: u32,
    pub draw_indirect_commands: Subbuffer<[DrawIndexedIndirectCommand]>,
}

#[derive(Copy, Clone, BufferContents)]
#[repr(C)]
pub struct InstanceData {
    pub mesh_index: u32,
    pub material_index: u32,
    pub model_transform: [[f32; 4]; 4],
    pub inverse_transpose_model_transform: [[f32; 4]; 4],
}

#[derive(Copy, Clone, BufferContents)]
#[repr(C)]
pub struct MaterialData {
    pub pipeline_address: u64,
}

#[derive(Copy, Clone, BufferContents)]
#[repr(C)]
pub struct MaterialInstanceData {
    pub material_index: u32,
}

#[derive(Copy, Clone, BufferContents)]
#[repr(C)]
pub struct MeshData {
    pub first_primitive: u32,
    pub primitive_count: u32,
    pub first_vertex: u32,
    pub vertex_count: u32,
}

#[derive(Copy, Clone, BufferContents)]
#[repr(C)]
pub struct MutatingData {
    pub screen_size: [u32; 2],
    pub view_matrix: [[f32; 4]; 4],
    pub view_position: [f32; 3],
}

#[derive(Copy, Clone, BufferContents)]
#[repr(C)]
pub struct VisBufferGlobalDataPointers {
    instances: DeviceAddress,
    materials: DeviceAddress,
    material_instances: DeviceAddress,
    meshes: DeviceAddress,
    index_buffer: DeviceAddress,
    vertex_buffer: DeviceAddress,
    mutating_data: DeviceAddress,
}

impl VisibilityBufferData {
    pub fn new(
        rhi: &VKRHI,
        swapchain: &Swapchain,
        max_sequence_count: u32,
        global_data: VisibilityBufferGlobalData,
        final_render_target: Arc<RwLock<SwapchainImage>>,
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
            global_data.num_materials(),
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

        let pipeline_bind_commands = Self::create_slice_buffer(
            rhi,
            BufferUsage::STORAGE_BUFFER | BufferUsage::INDIRECT_BUFFER,
            max_sequence_count,
        );

        let compute_dispatch_commands = Self::create_slice_buffer(
            rhi,
            BufferUsage::STORAGE_BUFFER | BufferUsage::INDIRECT_BUFFER,
            max_sequence_count,
        );

        let push_constants = Self::create_slice_buffer(
            rhi,
            BufferUsage::STORAGE_BUFFER | BufferUsage::INDIRECT_BUFFER,
            max_sequence_count,
        );

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
        )
            .unwrap();

        let global_data_buffer = buffer_from_slice(
            rhi.buffer_allocator().clone(),
            rhi.command_buffer_interface(),
            rhi.queues().compute_queue.clone(),
            &[global_data.buffer_pointers()],
            BufferUsage::SHADER_DEVICE_ADDRESS,
            MemoryTypeFilter::PREFER_DEVICE,
        )
            .unwrap()
            .reinterpret();

        Self {
            visibility_buffer,
            material_fragment_count_buffer,
            index_counter_buffer,
            material_indices_buffer,
            pipeline_bind_commands,
            compute_dispatch_commands,
            push_constants,
            clear_buffer,
            global_data,
            final_render_target,
            global_data_buffer,
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

    pub fn global_data_buffer_address(&self) -> DeviceAddress {
        self.global_data_buffer.device_address().unwrap().get()
    }
}

impl VisibilityBufferGlobalData {
    pub fn new(rhi: &VKRHI, mutating_data: Subbuffer<MutatingData>) -> Self {
        let resources = rhi.resource_manager();

        let mut instances = resources
            .resource_iterator::<VKModel>()
            .unwrap()
            .map(|instance| InstanceData {
                mesh_index: resources.index(instance.mesh().id()).unwrap() as u32,
                material_index: resources.index(instance.material().id()).unwrap() as u32,
                model_transform: instance.transform().to_cols_array_2d(),
                inverse_transpose_model_transform: instance
                    .transform()
                    .inverse()
                    .transpose()
                    .to_cols_array_2d(),
            })
            .collect::<Vec<_>>();

        instances.sort_unstable_by_key(|instance| instance.mesh_index);

        let draw_indirect_commands = instances
            .chunk_by(|left, right| left.mesh_index == right.mesh_index)
            .scan(0u32, |offset, slice| {
                let mesh = resources
                    .resource_iterator::<VKMesh>()
                    .unwrap()
                    .nth(slice[0].mesh_index as usize)
                    .unwrap();
                let first_instance = *offset;
                *offset += slice.len() as u32;
                Some(DrawIndexedIndirectCommand {
                    index_count: mesh.index_size() as u32,
                    instance_count: slice.len() as u32,
                    first_index: mesh.index_offset() as u32,
                    vertex_offset: mesh.vertex_offset() as u32,
                    first_instance,
                })
            })
            .collect::<Vec<_>>();

        let first_linked = Self::create_linked_program(
            rhi,
            resources.resource_iterator().unwrap().next().unwrap(),
        );
        let shader_object = Self::create_shader_object(rhi, first_linked);

        let pipelines = Self::compile_pipelines(rhi, shader_object.pipeline_layout());

        let materials = pipelines
            .iter()
            .map(|pipeline| MaterialData {
                pipeline_address: PipelineBindParameter::pipeline(pipeline).pipeline_address,
            })
            .collect::<Vec<_>>();

        let material_instances = resources
            .resource_iterator::<VKMaterialInstance>()
            .unwrap()
            .map(|instance| MaterialInstanceData {
                material_index: resources.index(instance.material().id()).unwrap() as u32,
            })
            .collect::<Vec<_>>();

        let meshes = resources
            .resource_iterator::<VKMesh>()
            .unwrap()
            .map(|mesh| MeshData {
                first_primitive: mesh.index_offset() as u32 / 3,
                primitive_count: mesh.index_size() as u32 / 3,
                first_vertex: mesh.vertex_offset() as u32,
                vertex_count: mesh.vertex_size() as u32,
            })
            .collect::<Vec<_>>();

        let material_count = materials.len() as u32;

        let draw_indirect_commands = buffer_from_slice(
            rhi.buffer_allocator().clone(),
            rhi.command_buffer_interface(),
            rhi.queues().compute_queue.clone(),
            draw_indirect_commands.as_slice(),
            BufferUsage::INDIRECT_BUFFER,
            MemoryTypeFilter::PREFER_DEVICE,
        )
            .unwrap();

        Self {
            instances: Self::make_buffer(
                rhi,
                instances.as_slice(),
                BufferUsage::VERTEX_BUFFER | BufferUsage::SHADER_DEVICE_ADDRESS,
            ),
            materials: Self::make_buffer(
                rhi,
                materials.as_slice(),
                BufferUsage::SHADER_DEVICE_ADDRESS,
            ),
            meshes: Self::make_buffer(rhi, meshes.as_slice(), BufferUsage::SHADER_DEVICE_ADDRESS),
            material_instances: Self::make_buffer(
                rhi,
                material_instances.as_slice(),
                BufferUsage::SHADER_DEVICE_ADDRESS,
            ),
            indices: resources
                .shared_buffer::<Index>()
                .unwrap()
                .clone()
                .reinterpret(),
            vertices: resources.shared_buffer().unwrap().clone(),
            mutating_data,
            pipelines,
            shader_object,
            material_count,
            draw_indirect_commands,
        }
    }

    pub fn write_to_shader_cursor(&self, shader_cursor: &mut ShaderCursor) {
        shader_cursor
            .field("instances")
            .unwrap()
            //    .write_address(self.instances.device_address().unwrap());
            .write_buffer(self.instances.clone());
        shader_cursor
            .field("materials")
            .unwrap()
            //    .write_address(self.materials.device_address().unwrap());
            .write_buffer(self.materials.clone());
        shader_cursor
            .field("materialInstances")
            .unwrap()
            //    .write_address(self.material_instances.device_address().unwrap());
            .write_buffer(self.material_instances.clone());
        shader_cursor
            .field("meshes")
            .unwrap()
            //    .write_address(self.meshes.device_address().unwrap());
            .write_buffer(self.meshes.clone());
        shader_cursor
            .field("indexBuffer")
            .unwrap()
            //    .write_address(self.indices.device_address().unwrap());
            .write_buffer(self.indices.clone());
        shader_cursor
            .field("vertexBuffer")
            .unwrap()
            //    .write_address(self.vertices.device_address().unwrap());
            .write_buffer(self.vertices.clone());
        shader_cursor
            .field("mutData")
            .unwrap()
            //    .write_address(self.mutating_data.device_address().unwrap());
            .write_buffer(self.mutating_data.clone());
    }

    pub fn buffer_pointers(&self) -> VisBufferGlobalDataPointers {
        VisBufferGlobalDataPointers {
            instances: self.instances.device_address().unwrap().get(),
            materials: self.materials.device_address().unwrap().get(),
            material_instances: self.material_instances.device_address().unwrap().get(),
            meshes: self.meshes.device_address().unwrap().get(),
            index_buffer: self.indices.device_address().unwrap().get(),
            vertex_buffer: self.vertices.device_address().unwrap().get(),
            mutating_data: self.mutating_data.device_address().unwrap().get(),
        }
    }

    fn make_buffer<T: BufferContents + Copy>(
        rhi: &VKRHI,
        data: &[T],
        usage: BufferUsage,
    ) -> Subbuffer<[T]> {
        buffer_from_slice(
            rhi.buffer_allocator().clone(),
            rhi.command_buffer_interface(),
            rhi.queues().compute_queue.clone(),
            data,
            BufferUsage::STORAGE_BUFFER | usage,
            MemoryTypeFilter::PREFER_DEVICE,
        )
            .unwrap()
    }

    fn create_linked_program(rhi: &VKRHI, material: &VKMaterial) -> ComponentType {
        let compiler = rhi.slang_compiler();
        let module = compiler
            .session()
            .load_module("Engine/VisibilityBuffer/visBufferComputeShade")
            .unwrap();
        let entry = module.find_entry_point_by_name("shadeVisBuffer").unwrap();
        let module_component: ComponentType = module.into();
        let material_module = compiler
            .session()
            .load_module(material.module_name())
            .unwrap();
        let material_module_component: ComponentType = material_module.into();
        let material_reflection = material_module_component
            .layout(0)
            .unwrap()
            .find_type_by_name(material.material_name())
            .unwrap();
        let composed = compiler
            .session()
            .create_composite_component_type(&[module_component, entry.into()])
            .unwrap();
        let specialized = composed
            .specialize(&[SpecializationArg::new(material_reflection)])
            .unwrap();
        specialized.link().unwrap()
    }

    fn create_shader_object(rhi: &VKRHI, linked: ComponentType) -> Arc<ShaderObject> {
        // We are assuming that all dynamically bound pipelines have the same layout and no (relevant) existential objects
        let layout = ShaderObjectLayout::new_with_push_constants(
            linked,
            &[],
            rhi.device(),
            ShaderStages::COMPUTE,
            vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                offset: 0,
                size: size_of::<VisBufferPushConstant>() as u32,
            }],
        );
        ShaderObject::new(
            layout,
            rhi.descriptor_allocator(),
            rhi.buffer_allocator(),
            rhi.in_flight_frames() as u32,
            rhi.shader_object_update_queue().clone(),
        )
    }

    fn make_pipeline_create_info(
        rhi: &VKRHI,
        material: &VKMaterial,
        pipeline_layout: Arc<PipelineLayout>,
        spirv_cache: &mut HashMap<String, Blob>
    ) -> ComputePipelineCreateInfo {
        let material_key = format!("{}_{}", material.module_name(), material.module_name());

        let spirv = if let Some(spirv) = spirv_cache.get(&material_key) {
            spirv.clone()
        } else {
            let linked = Self::create_linked_program(rhi, material);
            let spirv = linked.entry_point_code(0, 0).unwrap();
            spirv_cache.insert(material_key, spirv.clone());
            spirv
        };

        compute_pipeline()
            .shader(
                rhi.device().clone(),
                bytes_to_words(spirv.as_slice()).unwrap().deref(),
            )
            .build_create_info_with_flags(
                pipeline_layout.clone(),
                PipelineCreateFlags::INDIRECT_BINDABLE,
            )
    }

    fn compile_pipelines(
        rhi: &VKRHI,
        pipeline_layout: &Arc<PipelineLayout>,
    ) -> Vec<Arc<ComputePipeline>> {
        let resources = rhi.resource_manager();

        // TODO: This is just for testing the visibility buffer performance!
        let mut spirv_cache = HashMap::new();

        let pipeline_create_infos = resources
            .resource_iterator::<VKMaterial>()
            .unwrap()
            .map(|material| Self::make_pipeline_create_info(rhi, material, pipeline_layout.clone(), &mut spirv_cache))
            .collect::<Vec<_>>();

        let layouts = pipeline_create_infos
            .iter()
            .enumerate()
            .map(|(i, create_info)| {
                if i % 10 == 0 {
                    println!("Queried memory requirements of {} pipelines", i + 1);
                }
                IndirectCommandsLayout::pipeline_indirect_memory_requirements(
                    rhi.device(),
                    create_info,
                )
                    .layout
            });

        let alignment = layouts
            .clone()
            .map(|layout| layout.alignment())
            .fold(DeviceAlignment::default(), DeviceAlignment::max);

        let sizes = layouts.map(|layout| {
            layout
                .size()
                .checked_next_multiple_of(alignment.as_devicesize())
                .unwrap()
        });

        let total_size = sizes.clone().fold(0, DeviceSize::add);

        let indirect_metadata_buffer = Subbuffer::new(
            Buffer::new(
                rhi.buffer_allocator().clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_DST
                        | BufferUsage::INDIRECT_BUFFER
                        | BufferUsage::SHADER_DEVICE_ADDRESS,
                    ..BufferCreateInfo::default()
                },
                AllocationCreateInfo::default(),
                DeviceLayout::new(total_size.try_into().unwrap(), alignment).unwrap(),
            )
                .unwrap(),
        );

        let create_infos_with_indirect = sizes
            .scan(0u64, |state, size| {
                let offset = *state;
                *state += size;
                Some((offset, size))
            })
            .map(|(offset, size)| {
                indirect_metadata_buffer
                    .clone()
                    .slice(offset..(offset + size))
            })
            .zip(pipeline_create_infos.iter().cloned())
            .map(|(buffer, create_info)| ComputePipelineCreateInfo {
                indirect_buffer_info: Some(ComputePipelineIndirectBufferInfo::buffer(buffer)),
                ..create_info
            });

        let num_materials = pipeline_create_infos.len();
        let pipelines = create_infos_with_indirect
            .enumerate()
            .map(|(i, create_info)| {
                if i % 10 == 0 {
                    println!(
                        "Created {} visibility buffer materials of {}",
                        i + 1,
                        num_materials
                    );
                }
                ComputePipeline::new(rhi.device().clone(), None, create_info).unwrap()
            })
            .collect::<Vec<_>>();

        let mut command_buffer = rhi
            .command_buffer_interface()
            .primary_command_buffer(rhi.queue_family_indices().compute_family);

        for pipeline in pipelines.iter() {
            command_buffer
                .update_pipeline_indirect_buffer(pipeline.clone())
                .unwrap();
        }

        now(rhi.device().clone())
            .then_execute(
                rhi.queues().compute_queue.clone(),
                command_buffer.build().unwrap(),
            )
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        pipelines
    }

    pub fn shader_object(&self) -> &Arc<ShaderObject> {
        &self.shader_object
    }

    pub fn num_materials(&self) -> u32 {
        self.material_count
    }
}
