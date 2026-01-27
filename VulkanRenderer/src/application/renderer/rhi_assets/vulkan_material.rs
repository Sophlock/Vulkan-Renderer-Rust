use crate::application::assets::asset_traits::{
    MaterialInterface, RHIMaterialInterface, RHIResource, Vertex,
};
use crate::application::renderer::Renderer;
use crate::application::renderer::pipeline::graphics_pipeline;
use crate::{
    application::renderer::shader_object::{ShaderObject, ShaderObjectLayout},
    application::renderer::shaders::SlangCompiler,
};
use asset_system::resource_management::Resource;
use shader_slang::structs::specialization_arg::SpecializationArg;
use shader_slang::{Blob, ComponentType, Error, IUnknown, LayoutRules};
use std::{ops::Deref, sync::Arc};
use vulkano::descriptor_set::allocator::DescriptorSetAllocator;
use vulkano::memory::allocator::MemoryAllocator;
use vulkano::pipeline::graphics::subpass::PipelineSubpassType;
use vulkano::pipeline::{DynamicState, GraphicsPipeline, PipelineLayout};
use vulkano::{device::Device, render_pass::RenderPass, shader::spirv::bytes_to_words};

pub struct VKMaterial {
    vert_spirv: Blob,
    frag_spirv: Blob,
    shader_object_layout: Arc<ShaderObjectLayout>,
    pipeline: Arc<GraphicsPipeline>,
    uuid: usize,
}

impl VKMaterial {
    fn new(
        compiler: &SlangCompiler,
        device: &Arc<Device>,
        render_pass: Arc<RenderPass>,
        in_flight_frames: usize,
        module_name: &str,
        material_name: &str,
    ) -> shader_slang::Result<Self> {
        let module = compiler.session().load_module(module_name)?;
        let module_component: ComponentType = module.into();
        let composed = Self::append_raster_entry_points(&module_component, compiler)?;

        let material_reflection = module_component
            .layout(0)?
            .find_type_by_name(material_name)
            .unwrap();
        let specialized = composed.specialize(&[SpecializationArg::new(material_reflection)])?;

        let existential_objects = [specialized
            .layout(0)?
            .type_layout(material_reflection, LayoutRules::Default)
            .unwrap()];

        let linked = specialized.link()?;
        let vert_spirv = linked.entry_point_code(0, 0)?;
        let frag_spirv = linked.entry_point_code(1, 0)?;
        let shader_object_layout = ShaderObjectLayout::new(
            specialized.layout(0)?.global_params_var_layout().unwrap(),
            existential_objects.as_slice(),
            in_flight_frames as u32,
            device,
        );

        let pipeline = graphics_pipeline()
            .input_assembly(None, None)
            .vertex_shader(
                device.clone(),
                bytes_to_words(vert_spirv.as_slice()).unwrap().deref(),
            )
            .vertex_input::<Vertex>()
            .rasterizer(None, None, None, None, None, None)
            .skip_multisample()
            .fragment_shader(
                device.clone(),
                bytes_to_words(frag_spirv.as_slice()).unwrap().deref(),
            )
            .opaque_color_blend()
            .default_depth_test()
            .build_pipeline(
                device.clone(),
                shader_object_layout.pipeline_layout().clone(),
                PipelineSubpassType::BeginRenderPass(render_pass.first_subpass()),
                [
                    DynamicState::ViewportWithCount,
                    DynamicState::ScissorWithCount,
                ]
                .into(),
            );

        Ok(Self {
            shader_object_layout,
            pipeline,
            vert_spirv,
            frag_spirv,
            uuid: 0,
        })
    }

    fn append_raster_entry_points(
        component: &ComponentType,
        compiler: &SlangCompiler,
    ) -> shader_slang::Result<ComponentType> {
        let raster_module = compiler.session().load_module("Core/mainRaster")?;
        let vertex_main = raster_module
            .find_entry_point_by_name("vertexMain")
            .unwrap();
        let fragment_main = raster_module
            .find_entry_point_by_name("fragmentMain")
            .unwrap();
        compiler.session().create_composite_component_type(&[
            component.clone(),
            raster_module.into(),
            vertex_main.into(),
            fragment_main.into(),
        ])
    }
    
    pub fn shader_object_layout(&self) -> &Arc<ShaderObjectLayout> {
        &self.shader_object_layout
    }
    
    pub fn pipeline(&self) -> &Arc<GraphicsPipeline> {
        &self.pipeline
    }
    
    pub fn pipeline_layout(&self) -> &Arc<PipelineLayout> {
        self.shader_object_layout.pipeline_layout()
    }
}

impl Resource for VKMaterial {
    fn set_uuid(&mut self, uuid: usize) {
        self.uuid = uuid;
    }
}

impl RHIResource for VKMaterial {
    fn uuid_mut(&mut self) -> &mut usize {
        &mut self.uuid
    }
}

impl RHIMaterialInterface for VKMaterial {
    type RHI = Renderer;

    fn create<T: MaterialInterface>(source: &T, rhi: &Self::RHI) -> Self {
        VKMaterial::new(
            &rhi.slang_compiler,
            &rhi.device,
            rhi.render_pass.clone(),
            rhi.frames_in_flight,
            source.module(),
            source.material(),
        )
        .unwrap()
    }
}
