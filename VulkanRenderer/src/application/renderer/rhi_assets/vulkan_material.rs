use crate::application::assets::asset_traits::{MaterialInterface, RHIMaterialInterface, RHIResource};
use crate::application::renderer::Renderer;
use crate::{
    application::renderer::shader_object::{ShaderObject, ShaderObjectLayout},
    application::renderer::shaders::SlangCompiler,
};
use shader_slang::structs::specialization_arg::SpecializationArg;
use shader_slang::{Blob, ComponentType, Error, IUnknown, LayoutRules};
use std::{ops::Deref, sync::Arc};
use vulkano::{device::Device, render_pass::RenderPass, shader::spirv::bytes_to_words};
use vulkano::descriptor_set::allocator::DescriptorSetAllocator;
use vulkano::memory::allocator::MemoryAllocator;
use crate::application::resource_management::Resource;

pub struct VKMaterial {
    shader_object: ShaderObject,
    vert_spirv: Blob,
    frag_spirv: Blob,
    uuid: usize
}

impl VKMaterial {
    pub fn new(
        compiler: &SlangCompiler,
        device: &Arc<Device>,
        render_pass: Arc<RenderPass>,
        descriptor_allocator: &Arc<dyn DescriptorSetAllocator>,
        buffer_allocator: &Arc<dyn MemoryAllocator>,
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

        let existential_objects = [specialized.layout(0)?.type_layout(material_reflection, LayoutRules::Default).unwrap()];

        let linked = specialized.link()?;
        let vert_spirv = linked.entry_point_code(0, 0)?;
        let frag_spirv = linked.entry_point_code(1, 0)?;
        let shader_object_layout =
            ShaderObjectLayout::new(specialized.layout(0)?.global_params_var_layout().unwrap(),
                                    existential_objects.as_slice(),
                                    in_flight_frames as u32,
                                    device);

        let shader_object = ShaderObject::new(
            device,
            render_pass,
            shader_object_layout,
            descriptor_allocator,
            buffer_allocator,
            in_flight_frames as u32,
            bytes_to_words(vert_spirv.as_slice()).unwrap().deref(),
            bytes_to_words(frag_spirv.as_slice()).unwrap().deref(),
        );

        Ok(Self {
            shader_object,
            vert_spirv,
            frag_spirv,
            uuid: 0
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
            &rhi.descriptor_allocator,
            &rhi.buffer_allocator,
            rhi.frames_in_flight,
            source.module(),
            source.material(),
        )
        .unwrap()
    }
}
