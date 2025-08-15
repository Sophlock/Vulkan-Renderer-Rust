use crate::{
    application::assets::mesh::Vertex,
    application::renderer::pipeline::graphics_pipeline
};
use std::sync::Arc;
use vulkano::{
    pipeline::{
        DynamicState,
        GraphicsPipeline,
        graphics::subpass::PipelineSubpassType,
        layout::PipelineLayoutCreateInfo,
        PipelineLayout
    },
    render_pass::RenderPass,
    device::Device,
};

pub struct ShaderObjectLayout {
    pipeline_layout: Arc<PipelineLayout>,
}

pub struct ShaderObject {
    layout: Arc<ShaderObjectLayout>,
    pipeline: Arc<GraphicsPipeline>,
}

impl ShaderObjectLayout {
    pub fn new(device: &Arc<Device>) -> Arc<Self> {
        let pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                flags: Default::default(),
                set_layouts: vec![],
                push_constant_ranges: vec![],
                ..PipelineLayoutCreateInfo::default()
            },
        )
        .unwrap();
        Self { pipeline_layout }.into()
    }
}

impl ShaderObject {
    pub fn new(
        device: &Arc<Device>,
        render_pass: Arc<RenderPass>,
        layout: Arc<ShaderObjectLayout>,
        vert_spriv: &[u32],
        frag_spriv: &[u32]
    ) -> Self {
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
                [DynamicState::ViewportWithCount, DynamicState::ScissorWithCount].into(),
            );
        Self { layout, pipeline }
    }
}
