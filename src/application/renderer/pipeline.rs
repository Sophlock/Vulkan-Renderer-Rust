use smallvec::smallvec;
use std::collections::HashSet;
use std::{ops::RangeInclusive, sync::Arc};
use vulkano::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition, VertexInputState};
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::{DynamicState, GraphicsPipeline};
use vulkano::{
    device::Device,
    image::SampleCount,
    pipeline::{
        PipelineLayout, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{
                AttachmentBlend, ColorBlendAttachmentState, ColorBlendState, ColorComponents,
            },
            depth_stencil::{CompareOp, DepthState, DepthStencilState, StencilState},
            multisample::MultisampleState,
            rasterization::{CullMode, FrontFace, PolygonMode, RasterizationState},
        },
    },
    shader::spirv::ExecutionModel,
    shader::{ShaderModule, ShaderModuleCreateInfo},
};
use vulkano::pipeline::graphics::subpass::PipelineSubpassType;

pub struct EmptyGraphicsPipeline {}

pub struct InputAssemblyStateGraphicsPipeline {
    previous: EmptyGraphicsPipeline,
    input_assembly: InputAssemblyState,
}

pub struct VertexShaderGraphicsPipeline {
    previous: InputAssemblyStateGraphicsPipeline,
    vertex_shader: PipelineShaderStageCreateInfo,
}

pub struct VertexInputStateGraphicsPipeline {
    previous: VertexShaderGraphicsPipeline,
    vertex_input: VertexInputState,
}

// TODO: Add optional shader stages

pub struct RasterizerGraphicsPipeline {
    previous: VertexInputStateGraphicsPipeline,
    rasterizer_state: RasterizationState,
}

pub struct MultisampleGraphicsPipeline {
    previous: RasterizerGraphicsPipeline,
    multisample_state: MultisampleState,
}

pub struct FragmentShaderGraphicsPipeline {
    previous: MultisampleGraphicsPipeline,
    fragment_shader: PipelineShaderStageCreateInfo,
}

pub struct ColorBlendGraphicsPipeline {
    previous: FragmentShaderGraphicsPipeline,
    color_blend_state: ColorBlendState,
}

pub struct DepthStencilGraphicsPipeline {
    previous: ColorBlendGraphicsPipeline,
    depth_stencil_state: DepthStencilState,
}

pub fn graphics_pipeline() -> EmptyGraphicsPipeline {
    EmptyGraphicsPipeline {}
}

impl EmptyGraphicsPipeline {
    pub fn input_assembly(
        self,
        topology: Option<PrimitiveTopology>,
        primitive_restart: Option<bool>,
    ) -> InputAssemblyStateGraphicsPipeline {
        let input_assembly = InputAssemblyState {
            topology: topology.unwrap_or(PrimitiveTopology::TriangleList),
            primitive_restart_enable: primitive_restart.unwrap_or(false),
            ..InputAssemblyState::default()
        };
        InputAssemblyStateGraphicsPipeline {
            previous: self,
            input_assembly,
        }
    }
}

impl InputAssemblyStateGraphicsPipeline {
    pub fn vertex_shader(
        self,
        device: Arc<Device>,
        vertex_shader: &[u32],
    ) -> VertexShaderGraphicsPipeline {
        let shader_module = unsafe {
            ShaderModule::new(device, ShaderModuleCreateInfo::new(vertex_shader)).unwrap()
        };
        let create_info = PipelineShaderStageCreateInfo::new(
            shader_module
                .single_entry_point_with_execution(ExecutionModel::Vertex)
                .unwrap(),
        );

        VertexShaderGraphicsPipeline {
            previous: self,
            vertex_shader: create_info,
        }
    }
}

impl VertexShaderGraphicsPipeline {
    pub fn vertex_input<V: Vertex>(self) -> VertexInputStateGraphicsPipeline {
        let vertex_input = V::per_vertex()
            .definition(&self.vertex_shader.entry_point)
            .unwrap();
        VertexInputStateGraphicsPipeline {
            previous: self,
            vertex_input,
        }
    }
}

impl VertexInputStateGraphicsPipeline {
    pub fn rasterizer(
        self,
        polygon_mode: Option<PolygonMode>,
        cull_mode: Option<CullMode>,
        front_face: Option<FrontFace>,
        depth_clamp: Option<bool>,
        rasterizer_discard: Option<bool>,
        line_width: Option<f32>,
    ) -> RasterizerGraphicsPipeline {
        let rasterizer_state = RasterizationState {
            depth_clamp_enable: depth_clamp.unwrap_or(false),
            rasterizer_discard_enable: rasterizer_discard.unwrap_or(false),
            polygon_mode: polygon_mode.unwrap_or(PolygonMode::Fill),
            cull_mode: cull_mode.unwrap_or(CullMode::Back),
            front_face: front_face.unwrap_or(FrontFace::CounterClockwise),
            line_width: line_width.unwrap_or(1.0),
            ..RasterizationState::default()
        };
        RasterizerGraphicsPipeline {
            previous: self,
            rasterizer_state,
        }
    }
}

impl RasterizerGraphicsPipeline {
    pub fn multisample(
        self,
        rasterization_samples: SampleCount,
        sample_shading: Option<f32>,
        alpha_to_coverage_enable: bool,
        alpha_to_one_enable: bool,
    ) -> MultisampleGraphicsPipeline {
        let multisample_state = MultisampleState {
            rasterization_samples,
            sample_shading,
            alpha_to_coverage_enable,
            alpha_to_one_enable,
            ..MultisampleState::default()
        };
        MultisampleGraphicsPipeline {
            previous: self,
            multisample_state,
        }
    }

    pub fn skip_multisample(self) -> MultisampleGraphicsPipeline {
        MultisampleGraphicsPipeline {
            previous: self,
            multisample_state: MultisampleState::default(),
        }
    }
}

impl MultisampleGraphicsPipeline {
    pub fn fragment_shader(
        self,
        device: Arc<Device>,
        fragment_shader: &[u32],
    ) -> FragmentShaderGraphicsPipeline {
        let shader_module = unsafe {
            ShaderModule::new(device, ShaderModuleCreateInfo::new(fragment_shader)).unwrap()
        };
        let create_info = PipelineShaderStageCreateInfo::new(
            shader_module
                .single_entry_point_with_execution(ExecutionModel::Fragment)
                .unwrap(),
        );
        FragmentShaderGraphicsPipeline {
            previous: self,
            fragment_shader: create_info,
        }
    }
}

impl FragmentShaderGraphicsPipeline {
    pub fn color_blend(
        self,
        blend: Option<AttachmentBlend>,
        color_write_mask: ColorComponents,
        color_write_enable: bool,
        blend_constant: Option<[f32; 4]>,
    ) -> ColorBlendGraphicsPipeline {
        let attachment = ColorBlendAttachmentState {
            blend,
            color_write_mask,
            color_write_enable,
        };
        let color_blend_state = ColorBlendState {
            flags: Default::default(),
            logic_op: None,
            attachments: vec![attachment],
            blend_constants: blend_constant.unwrap_or([0.0, 0.0, 0.0, 0.0]),
            ..ColorBlendState::default()
        };
        ColorBlendGraphicsPipeline {
            previous: self,
            color_blend_state,
        }
    }

    pub fn opaque_color_blend(self) -> ColorBlendGraphicsPipeline {
        self.color_blend(None, ColorComponents::all(), true, None)
    }
}

impl ColorBlendGraphicsPipeline {
    pub fn depth_stencil(
        self,
        depth: Option<DepthState>,
        depth_bounds: Option<RangeInclusive<f32>>,
        stencil: Option<StencilState>,
    ) -> DepthStencilGraphicsPipeline {
        let depth_stencil_state = DepthStencilState {
            depth,
            depth_bounds,
            stencil,
            ..DepthStencilState::default()
        };
        DepthStencilGraphicsPipeline {
            previous: self,
            depth_stencil_state,
        }
    }

    pub fn skip_depth_test(self) -> DepthStencilGraphicsPipeline {
        self.depth_stencil(None, None, None)
    }

    pub fn default_depth_test(self) -> DepthStencilGraphicsPipeline {
        self.depth_stencil(
            Some(DepthState {
                write_enable: true,
                compare_op: CompareOp::LessOrEqual,
            }),
            None,
            None,
        )
    }
}

impl DepthStencilGraphicsPipeline {
    pub fn build_create_info(
        self,
        layout: Arc<PipelineLayout>,
        subpass: PipelineSubpassType,
        dynamic_state: HashSet<DynamicState>,
    ) -> GraphicsPipelineCreateInfo {
        let stages = smallvec![
            self.previous
                .previous
                .previous
                .previous
                .previous
                .previous
                .vertex_shader,
            self.previous.previous.fragment_shader
        ];
        GraphicsPipelineCreateInfo {
            stages,
            vertex_input_state: Some(
                self.previous
                    .previous
                    .previous
                    .previous
                    .previous
                    .vertex_input,
            ),
            input_assembly_state: Some(
                self.previous
                    .previous
                    .previous
                    .previous
                    .previous
                    .previous
                    .previous
                    .input_assembly,
            ),
            tessellation_state: None,
            viewport_state: Some(ViewportState::default()),
            rasterization_state: Some(self.previous.previous.previous.previous.rasterizer_state),
            multisample_state: Some(self.previous.previous.previous.multisample_state),
            depth_stencil_state: Some(self.depth_stencil_state),
            color_blend_state: Some(self.previous.color_blend_state),
            dynamic_state: dynamic_state.iter().copied().collect(),
            subpass: Some(subpass),
            base_pipeline: None,
            discard_rectangle_state: None,
            fragment_shading_rate_state: None,
            ..GraphicsPipelineCreateInfo::layout(layout)
        }
    }

    pub fn build_pipeline(
        self,
        device: Arc<Device>,
        layout: Arc<PipelineLayout>,
        subpass: PipelineSubpassType,
        dynamic_state: HashSet<DynamicState>,
    ) -> Arc<GraphicsPipeline> {
        GraphicsPipeline::new(device, None, self.build_create_info(layout, subpass, dynamic_state)).unwrap()
    }
}
