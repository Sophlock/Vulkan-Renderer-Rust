use std::{
    mem::offset_of,
    ops::Deref,
    rc::Rc,
    sync::{Arc, RwLock},
};

use crate::application::{
    assets::asset_traits::{
        RHICameraInterface, RHIInterface, RHIModelInterface, RHIResource, RHISceneInterface, Vertex,
    },
    renderer::{
        device_generated_commands::map_pipeline_bind_point,
        visibility_buffer_data::{InstanceData, VisibilityBufferData},
    },
    rhi::{
        VKRHI,
        device_helper::{ash_device, ash_instance},
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
use ash::vk::{DeviceAddress, PipelineIndirectDeviceAddressInfoNV};
use smallvec::smallvec;
use vulkano::pipeline::layout::PushConstantRange;
use vulkano::{
    ValidationError, VulkanObject,
    buffer::BufferContents,
    command_buffer::{
        AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo,
        SubpassContents, SubpassEndInfo,
    },
    device::DeviceOwned,
    format::{ClearValue, Format},
    pipeline::{
        ComputePipeline, DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint,
        graphics::{
            subpass::PipelineSubpassType,
            vertex_input::{VertexBufferDescription, VertexInputRate, VertexMemberInfo},
            viewport::{Scissor, Viewport},
        },
    },
    render_pass::RenderPass,
    shader::{ShaderStages, spirv::bytes_to_words},
};

pub struct VisibilityBufferProcessingPass {
    vis_buffer_scan: VisBufferStep,
    shader_cull: VisBufferStep,
    num_materials: u32,
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
    pub global_data_address: DeviceAddress,
    pub this_pipeline_address: DeviceAddress,
}

impl VisibilityBufferProcessingPass {
    pub fn new(rhi: &VKRHI, data: &Arc<VisibilityBufferData>) -> Self {
        let vis_buffer_scan = VisBufferStep::new(
            rhi,
            "Engine/VisibilityBuffer/visBufferScan",
            "countTexels",
            data.clone(),
        );
        let shader_cull = VisBufferStep::new(
            rhi,
            "Engine/VisibilityBuffer/visBufferShaderCull",
            "cullShaders",
            data.clone(),
        );

        let cursor = ShaderCursor::new(vis_buffer_scan.shader_object.clone());
        let input_cursor = cursor.field("gInput").unwrap();
        input_cursor
            .field("visBuffer")
            .unwrap()
            .write_swapchain_image(data.visibility_buffer.clone());
        input_cursor
            .field("materialFragmentCounts")
            .unwrap()
            .write_buffer(data.material_fragment_count_buffer.clone());

        //data.global_data
        //    .write_to_shader_cursor(&mut cursor.field("gGlobalData").unwrap());

        let cursor = ShaderCursor::new(shader_cull.shader_object.clone());
        let input_cursor = cursor.field("gInput").unwrap();
        input_cursor
            .field("texelCounts")
            .unwrap()
            .write_buffer(data.material_fragment_count_buffer.clone());
        input_cursor
            .field("index")
            .unwrap()
            .write_buffer(data.index_counter_buffer.clone());
        input_cursor
            .field("materialIndices")
            .unwrap()
            .write_buffer(data.material_indices_buffer.clone());
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

        //data.global_data
        //    .write_to_shader_cursor(&mut cursor.field("gGlobalData").unwrap());

        Self {
            vis_buffer_scan,
            shader_cull,
            num_materials: data.global_data.num_materials(),
        }
    }

    pub fn record_command_buffer(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_index: usize,
        swapchain_extent: [u32; 2],
    ) -> Result<(), Box<ValidationError>> {
        self.vis_buffer_scan.record_command_buffer(
            command_buffer,
            image_index,
            [swapchain_extent[0] / 16, swapchain_extent[1] / 16, 1],
        )?;
        self.shader_cull.record_command_buffer(
            command_buffer,
            image_index,
            [self.num_materials / 16, 1, 1],
        )?;
        Ok(())
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
            vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                offset: 0,
                size: size_of::<DeviceAddress>() as u32,
            }],
        );
        let pipeline_layout = shader_object_layout.pipeline_layout().clone();
        let shader_object = ShaderObject::new(
            shader_object_layout,
            rhi.descriptor_allocator(),
            rhi.buffer_allocator(),
            rhi.in_flight_frames() as u32,
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
            )?
            .push_constants(
                self.shader_object
                    .pipeline_layout()
                    .clone(),
                0,
                self.data.global_data_buffer_address(),
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

        let resources = self.rhi.resource_manager();
        let rcs = resources.deref();

        command_buffer
            .bind_vertex_buffers(0, data.global_data.vertices.clone())?
            .bind_vertex_buffers(1, data.global_data.instances.clone())?
            .bind_index_buffer(data.global_data.indices.clone())?;

        scene
            .models()
            .iter()
            .map(|model_handle| {
                let model = model_handle.get(rcs).unwrap();
                let mesh = model.mesh().get(rcs).unwrap();
                unsafe {
                    command_buffer.draw_indexed(
                        mesh.index_size() as u32,
                        1,
                        mesh.index_offset() as u32,
                        mesh.vertex_offset() as i32,
                        rcs.index(model.uuid()).unwrap() as u32,
                    )
                }
                .map(|_| ())
            })
            .reduce(Result::or)
            .unwrap_or(Ok(()))?;

        command_buffer
            .end_render_pass(SubpassEndInfo::default())
            .map(|_| ())
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
