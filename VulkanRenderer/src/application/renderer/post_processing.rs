use std::{
    cell::{RefCell, RefMut},
    ops::Deref,
    sync::{Arc, RwLock},
};

use egui_winit_vulkano::{egui, egui::Ui};
use shader_slang::ComponentType;
use vulkano::{
    ValidationError,
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    image::sampler::{Sampler, SamplerCreateInfo},
    pipeline::{ComputePipeline, PipelineBindPoint},
    shader::{ShaderStages, spirv::bytes_to_words},
};

use crate::application::rhi::{
    VKRHI,
    pipeline::compute_pipeline,
    shader_cursor::ShaderCursor,
    shader_object::{ShaderObject, ShaderObjectLayout},
    shaders::SlangCompiler,
    swapchain_resources::SwapchainImage,
};

/// Post processing pass
/// 
/// Currently mostly placeholder 
pub struct PostProcessPass {
    shader_object_layout: Arc<ShaderObjectLayout>,
    shader_object: Arc<ShaderObject>,
    pipeline: Arc<ComputePipeline>,
    sampler: Arc<Sampler>,
    settings: RefCell<PostProcessSettings>,
}

impl PostProcessPass {
    pub fn new(
        compiler: &SlangCompiler,
        rhi: &VKRHI,
        source: Arc<RwLock<SwapchainImage>>,
        target: Arc<RwLock<SwapchainImage>>,
    ) -> Self {
        let module = compiler
            .session()
            .load_module("Compute/postProcess")
            .unwrap();
        let entry = module.find_entry_point_by_name("postProcessMain").unwrap();
        let module_component: ComponentType = module.into();
        let composed = compiler
            .session()
            .create_composite_component_type(&[module_component, entry.into()])
            .unwrap();
        let linked = composed.link().unwrap();
        let spirv = linked.entry_point_code(0, 0).unwrap();

        let shader_object_layout =
            ShaderObjectLayout::new(linked, &[], rhi.device(), ShaderStages::COMPUTE);
        let shader_object = ShaderObject::new(
            shader_object_layout.clone(),
            rhi.descriptor_allocator(),
            rhi.buffer_allocator(),
            rhi.in_flight_frames() as u32,
            rhi.shader_object_update_queue().clone(),
        );

        let pipeline = compute_pipeline()
            .shader(
                rhi.device().clone(),
                bytes_to_words(spirv.as_slice()).unwrap().deref(),
            )
            .build_pipeline(
                rhi.device().clone(),
                shader_object_layout.pipeline_layout().clone(),
            );

        let sampler = Sampler::new(
            rhi.device().clone(),
            SamplerCreateInfo::simple_repeat_linear_no_mipmap(),
        )
        .unwrap();

        Self::write_framebuffer_descriptors(shader_object.clone(), source, target, sampler.clone());

        let settings = RefCell::new(PostProcessSettings::new(shader_object.clone()));

        Self {
            shader_object_layout,
            shader_object,
            pipeline,
            sampler,
            settings,
        }
    }

    pub fn record_command_buffer(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_index: usize,
        extent: [u32; 2],
    ) -> Result<(), Box<ValidationError>> {
        let groups = [extent[0] / 16 + 1, extent[1] / 16 + 1, 1];

        command_buffer
            .bind_pipeline_compute(self.pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.shader_object_layout.pipeline_layout().clone(),
                0,
                self.shader_object.descriptor_sets()[image_index].clone(),
            )?;
        unsafe { command_buffer.dispatch(groups) }?;

        Ok(())
    }

    fn write_framebuffer_descriptors(
        shader_object: Arc<ShaderObject>,
        source: Arc<RwLock<SwapchainImage>>,
        target: Arc<RwLock<SwapchainImage>>,
        sampler: Arc<Sampler>,
    ) {
        let global_cursor = ShaderCursor::new(shader_object);
        let cursor = global_cursor.field("gPostProcessData").unwrap();
        cursor
            .field("input")
            .unwrap()
            .write_swapchain_image_sampler(source, sampler);
        cursor
            .field("result")
            .unwrap()
            .write_swapchain_image(target);
    }

    pub fn settings_mut(&self) -> RefMut<PostProcessSettings> {
        self.settings.borrow_mut()
    }
}

pub struct PostProcessSettings {
    pub exposure_value: f32,
    pub shader_object: Arc<ShaderObject>,
}

impl PostProcessSettings {
    pub fn new(shader_object: Arc<ShaderObject>) -> Self {
        let result = Self {
            exposure_value: 3f32,
            shader_object,
        };
        result.write_settings();
        result
    }

    fn write_settings(&self) {
        let global_cursor = ShaderCursor::new(self.shader_object.clone());
        let cursor = global_cursor
            .field("gPostProcessData")
            .unwrap()
            .field("settings")
            .unwrap();
        cursor
            .field("exposureValue")
            .unwrap()
            .write(&self.exposure_value);
    }

    pub fn draw_gui(&mut self, gui: &mut Ui) {
        let global_cursor = ShaderCursor::new(self.shader_object.clone());
        let cursor = global_cursor
            .field("gPostProcessData")
            .unwrap()
            .field("settings")
            .unwrap();

        if gui
            .add(egui::Slider::new(&mut self.exposure_value, 0f32..=20f32))
            .changed()
        {
            cursor
                .field("exposureValue")
                .unwrap()
                .write(&self.exposure_value);
        }
    }
}
