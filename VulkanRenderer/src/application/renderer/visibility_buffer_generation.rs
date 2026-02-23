use crate::application::assets::asset_traits::Vertex;
use crate::application::rhi::VKRHI;
use crate::application::rhi::pipeline::{compute_pipeline, graphics_pipeline};
use crate::application::rhi::shader_object::{ShaderObject, ShaderObjectLayout};
use std::ops::Deref;
use std::sync::Arc;
use vulkano::ValidationError;
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer};
use vulkano::pipeline::graphics::subpass::PipelineSubpassType;
use vulkano::pipeline::{ComputePipeline, DynamicState, GraphicsPipeline, PipelineBindPoint};
use vulkano::render_pass::RenderPass;
use vulkano::shader::ShaderStages;
use vulkano::shader::spirv::bytes_to_words;

pub struct VisibilityBufferGenerationPass {
    vis_buffer_scan: VisBufferStep,
    shader_cull: VisBufferStep,
}

struct VisBufferStep {
    shader_object: ShaderObject,
    pipeline: Arc<ComputePipeline>,
    dispatch: [u32; 3],
}

pub struct VisBufferRasterizer {
    shader_object: ShaderObject,
    pipeline: Arc<GraphicsPipeline>,
}

impl VisibilityBufferGenerationPass {
    pub fn new(rhi: &VKRHI, target_extent: [u32; 2], num_materials: u32) -> Self {
        let vis_buffer_scan = VisBufferStep::new(
            rhi,
            "visBufferScan",
            "countTexels",
            [target_extent[0] / 16, target_extent[1] / 16, 1],
        );
        let shader_cull = VisBufferStep::new(
            rhi,
            "visBufferShaderCull",
            "cullShaders",
            [num_materials / 16, 1, 1],
        );
        Self {
            vis_buffer_scan,
            shader_cull,
        }
    }

    pub fn record_command_buffer(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_index: usize,
    ) -> Result<(), Box<ValidationError>> {
        self.vis_buffer_scan
            .record_command_buffer(command_buffer, image_index)?;
        self.shader_cull
            .record_command_buffer(command_buffer, image_index)?;
        Ok(())
    }
}

impl VisBufferStep {
    fn new(rhi: &VKRHI, module: &str, entry_point: &str, dispatch: [u32; 3]) -> Self {
        let session = rhi.slang_compiler().session();
        let module = session.load_module(module).unwrap();
        let entry = module.find_entry_point_by_name(entry_point).unwrap();
        let linked = session
            .create_composite_component_type(&[module.into(), entry.into()])
            .unwrap()
            .link()
            .unwrap();
        let shader_object_layout =
            ShaderObjectLayout::new(linked.clone(), &[], rhi.device(), ShaderStages::COMPUTE);
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
            dispatch,
        }
    }

    fn record_command_buffer(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_index: usize,
    ) -> Result<(), Box<ValidationError>> {
        command_buffer
            .bind_pipeline_compute(self.pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.shader_object.pipeline_layout().clone(),
                0,
                self.shader_object.descriptor_sets()[image_index].clone(),
            )?;
        unsafe { command_buffer.dispatch(self.dispatch) }?;
        Ok(())
    }
}

impl VisBufferRasterizer {
    pub fn new(rhi: &VKRHI, render_pass: &Arc<RenderPass>) -> Self {
        let module = "visBufferGenerator";
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
        let pipeline = graphics_pipeline()
            .input_assembly(None, None)
            .vertex_shader(
                rhi.device().clone(),
                bytes_to_words(linked.entry_point_code(0, 0).unwrap().as_slice())
                    .unwrap()
                    .deref(),
            )
            .vertex_input::<Vertex>()
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
            .build_pipeline(
                rhi.device().clone(),
                pipeline_layout,
                PipelineSubpassType::BeginRenderPass(render_pass.clone().first_subpass()),
                [
                    DynamicState::ViewportWithCount,
                    DynamicState::ScissorWithCount,
                ]
                .into(),
            );
        Self {
            shader_object,
            pipeline
        }
    }
}
