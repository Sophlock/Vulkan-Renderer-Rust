use std::sync::Arc;

use asset_system::resource_management::Resource;
use shader_slang::{
    Blob, ComponentType, LayoutRules, structs::specialization_arg::SpecializationArg,
};
use vulkano::{device::Device, pipeline::PipelineLayout, shader::ShaderStages};

use crate::application::{
    assets::asset_traits::{
        MaterialInterface, RHIMaterialInterface, RHIResource, RendererInterface,
    },
    rhi::{
        VKRHI, rhi_assets::RHIResourceManager, shader_object::ShaderObjectLayout,
        shaders::SlangCompiler,
    },
};

pub struct VKMaterial {
    vert_spirv: Blob,
    frag_spirv: Blob,
    shader_object_layout: Arc<ShaderObjectLayout>,
    uuid: usize,
}

impl VKMaterial {
    fn new(
        compiler: &SlangCompiler,
        device: &Arc<Device>,
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
            linked,
            existential_objects.as_slice(),
            device,
            ShaderStages::all_graphics(),
        );

        Ok(Self {
            shader_object_layout,
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

    pub fn pipeline_layout(&self) -> &Arc<PipelineLayout> {
        self.shader_object_layout.pipeline_layout()
    }

    // TODO: These should belong to the compiled material
    pub fn vert_spirv(&self) -> &Blob {
        &self.vert_spirv
    }

    pub fn frag_spirv(&self) -> &Blob {
        &self.frag_spirv
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
    type RHI = VKRHI;

    fn create<T: MaterialInterface>(
        source: &T,
        rhi: &Self::RHI,
        _: &mut RHIResourceManager,
    ) -> Self {
        VKMaterial::new(
            &rhi.slang_compiler,
            &rhi.device,
            source.module(),
            source.material(),
        )
        .unwrap()
    }
}
