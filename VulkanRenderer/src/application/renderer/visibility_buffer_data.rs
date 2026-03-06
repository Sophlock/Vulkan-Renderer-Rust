use std::{
    ops::Deref,
    sync::{Arc, RwLock},
};

use crate::application::{
    assets::asset_traits::{Index, RHIInterface, RHIModelInterface, Vertex},
    renderer::visibility_buffer_generation::{ComputeDispatchParameter, PipelineBindParameter},
    rhi::{
        VKRHI,
        buffer::buffer_from_slice,
        pipeline::compute_pipeline,
        rhi_assets::{
            vulkan_material::VKMaterial, vulkan_material_instance::VKMaterialInstance,
            vulkan_mesh::VKMesh, vulkan_model::VKModel,
        },
        shader_cursor::ShaderCursor,
        swapchain::Swapchain,
        swapchain_resources::SwapchainImage,
    },
};
use shader_slang::{ComponentType, structs::specialization_arg::SpecializationArg};
use vulkano::descriptor_set::layout::{
    DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType,
};
use vulkano::{
    ValidationError,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CopyBufferInfo, PrimaryAutoCommandBuffer},
    format::Format,
    image::{ImageAspects, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        ComputePipeline, PipelineLayout,
        layout::{PipelineLayoutCreateInfo, PushConstantRange},
    },
    shader::{ShaderStages, spirv::bytes_to_words},
};
use crate::application::rhi::shader_object::{ShaderObject, ShaderObjectLayout};

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

    // Global data (buffers etc.)
    pub global_data: VisibilityBufferGlobalData,

    // Final output rt
    pub final_render_target: Arc<RwLock<SwapchainImage>>
}

#[derive(Clone)]
pub struct VisibilityBufferGlobalData {
    pub instances: Subbuffer<[InstanceData]>,
    pub materials: Subbuffer<[MaterialData]>,
    pub material_instances: Subbuffer<[MaterialInstanceData]>,
    pub meshes: Subbuffer<[MeshData]>,
    pub indices: Subbuffer<[u32]>,
    pub vertices: Subbuffer<[Vertex]>,
    pub screen_size: [u32; 2],
    pipelines: Vec<Arc<ComputePipeline>>,
    shader_object: Arc<ShaderObject>,
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

impl VisibilityBufferData {
    pub fn new(
        rhi: &VKRHI,
        swapchain: &Swapchain,
        max_sequence_count: u32,
        num_materials: u32,
        global_data: VisibilityBufferGlobalData,
        final_render_target: Arc<RwLock<SwapchainImage>>
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

        Self {
            visibility_buffer,
            material_fragment_count_buffer,
            index_counter_buffer,
            material_indices_buffer,
            pipeline_bind_commands,
            compute_dispatch_commands,
            clear_buffer,
            global_data,
            final_render_target
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

impl VisibilityBufferGlobalData {
    pub fn new(rhi: &VKRHI, screen_size: [u32; 2]) -> Self {
        let resources = rhi.resource_manager();

        let instances = resources
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

        let first_linked = Self::create_linked_program(rhi, resources.resource_iterator().unwrap().next().unwrap());
        let shader_object = Self::create_shader_object(rhi, first_linked);
        let pipelines = resources
            .resource_iterator::<VKMaterial>()
            .unwrap()
            .map(|material| Self::compile_pipeline(rhi, material, shader_object.layout().clone()))
            .collect::<Vec<_>>();

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
            screen_size,
            pipelines,
            shader_object
        }
    }

    pub fn write_to_shader_cursor(&self, shader_cursor: &mut ShaderCursor) {
        shader_cursor
            .field("instances")
            .unwrap()
            .write_address(self.instances.device_address().unwrap());
        //.write_buffer(self.instances.clone());
        shader_cursor
            .field("materials")
            .unwrap()
            .write_address(self.materials.device_address().unwrap());
        //.write_buffer(self.materials.clone());
        shader_cursor
            .field("materialInstances")
            .unwrap()
            .write_address(self.material_instances.device_address().unwrap());
        //.write_buffer(self.material_instances.clone());
        shader_cursor
            .field("meshes")
            .unwrap()
            .write_address(self.meshes.device_address().unwrap());
        //.write_buffer(self.meshes.clone());
        shader_cursor
            .field("indexBuffer")
            .unwrap()
            .write_address(self.indices.device_address().unwrap());
        //.write_buffer(self.indices.clone());
        shader_cursor
            .field("vertexBuffer")
            .unwrap()
            .write_address(self.vertices.device_address().unwrap());
        //.write_buffer(self.vertices.clone());
        shader_cursor
            .field("screenSize")
            .unwrap()
            .write(&self.screen_size);
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

    fn create_pipeline_layout(rhi: &VKRHI) -> Arc<PipelineLayout> {
        let bindings = [
            (
                0u32,
                DescriptorSetLayoutBinding {
                    stages: ShaderStages::COMPUTE,
                    ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageImage)
                },
            ),
            (
                1u32,
                DescriptorSetLayoutBinding {
                    stages: ShaderStages::COMPUTE,
                    ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageImage)
                },
            ),
        ]
        .iter()
        .cloned()
        .collect();
        let create_info = DescriptorSetLayoutCreateInfo {
            bindings,
            ..DescriptorSetLayoutCreateInfo::default()
        };
        let descriptor = DescriptorSetLayout::new(rhi.device().clone(), create_info).unwrap();
        let push = PushConstantRange {
            stages: ShaderStages::COMPUTE,
            offset: 0,
            size: 32,
        };
        let create_info = PipelineLayoutCreateInfo {
            set_layouts: vec![descriptor],
            push_constant_ranges: vec![push],
            ..PipelineLayoutCreateInfo::default()
        };
        PipelineLayout::new(rhi.device().clone(), create_info).unwrap()
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
        let layout = ShaderObjectLayout::new(linked, &[], rhi.device(), ShaderStages::COMPUTE);
        ShaderObject::new(layout, rhi.descriptor_allocator(), rhi.buffer_allocator(), rhi.in_flight_frames() as u32)
    }

    fn compile_pipeline(
        rhi: &VKRHI,
        material: &VKMaterial,
        shader_object_layout: Arc<ShaderObjectLayout>,
    ) -> Arc<ComputePipeline> {
        let linked = Self::create_linked_program(rhi, material);
        let spirv = linked.entry_point_code(0, 0).unwrap();

        compute_pipeline()
            .shader(
                rhi.device().clone(),
                bytes_to_words(spirv.as_slice()).unwrap().deref(),
            )
            .build_pipeline(rhi.device().clone(), shader_object_layout.pipeline_layout().clone())
    }

    pub fn shader_object(&self) -> &Arc<ShaderObject> {
        &self.shader_object
    }
}
