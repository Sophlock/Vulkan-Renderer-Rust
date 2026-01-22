use crate::application::assets::asset_traits::{MaterialInterface, RHIMaterialInterface};
use crate::application::renderer::Renderer;
use crate::{
    application::renderer::shader_object::{ShaderObject, ShaderObjectLayout},
    application::renderer::shaders::SlangCompiler,
};
use shader_slang::{Blob, ComponentType, Error, IUnknown};
use std::{ops::Deref, sync::Arc};
use vulkano::{device::Device, render_pass::RenderPass, shader::spirv::bytes_to_words};

struct VulkanMaterial {
    shader_object: ShaderObject,
    vert_spirv: Blob,
    frag_spirv: Blob,
}

impl VulkanMaterial {
    pub fn new(
        compiler: &SlangCompiler,
        device: &Arc<Device>,
        render_pass: Arc<RenderPass>,
        module_name: &str,
        material_name: &str,
    ) -> shader_slang::Result<Self> {
        let module = compiler.session().load_module(module_name)?;
        let module_component: ComponentType = module.into();
        // TODO: Specialize
        let material_reflection = module_component
            .layout(0)?
            .find_type_by_name(material_name)
            .unwrap();
        let composed = Self::append_raster_entry_points(&module_component, compiler)?;
        let linked = composed.link()?;
        let vert_spirv = linked.entry_point_code(0, 0)?;
        let frag_spirv = linked.entry_point_code(1, 0)?;
        let shader_object_layout = ShaderObjectLayout::new(device);

        let shader_object = ShaderObject::new(
            device,
            render_pass,
            shader_object_layout,
            bytes_to_words(vert_spirv.as_slice()).unwrap().deref(),
            bytes_to_words(frag_spirv.as_slice()).unwrap().deref(),
        );

        Ok(Self {
            shader_object,
            vert_spirv,
            frag_spirv,
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
}

impl RHIMaterialInterface for VulkanMaterial {
    type RHI = Renderer;

    fn create<T: MaterialInterface>(source: &T, rhi: &Self::RHI) -> Self {
        VulkanMaterial::new(&rhi.slang_compiler, &rhi.device, rhi.render_pass.clone(), source.module(), source.material()).unwrap()
    }
}
