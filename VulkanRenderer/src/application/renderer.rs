use std::cell::{Ref, RefCell, RefMut};
use std::ops::Deref;
use std::rc::Rc;
use std::sync::Arc;
use egui_winit_vulkano::egui;
use egui_winit_vulkano::egui::{Color32, Frame};
use smallvec::smallvec;
use vulkano::image::view::ImageView;
use vulkano::render_pass::{Framebuffer, RenderPass};
use vulkano::sync::future::FenceSignalFuture;
use vulkano::sync::GpuFuture;
use vulkano::{Validated, ValidationError, VulkanError};
use vulkano::command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo};
use vulkano::format::ClearValue;
use vulkano::pipeline::graphics::viewport::{Scissor, Viewport};
use vulkano::pipeline::PipelineBindPoint;
use vulkano::swapchain::{present, SwapchainPresentInfo};
use winit::dpi::PhysicalSize;
use crate::application::assets::asset_traits::{RHICameraInterface, RHIInterface, RHIModelInterface, RHISceneInterface, RendererInterface};
use crate::application::rhi::{VKRHI, swapchain::Swapchain};
use crate::application::rhi::render_pass::RenderPassBuilder;
use crate::application::rhi::rhi_assets::vulkan_scene::VKScene;

pub struct MutableRenderState {
    swapchain: Swapchain,
    depth_image_view: Arc<ImageView>,
    framebuffers: Vec<Arc<Framebuffer>>,
    should_recreate_swapchain: bool,
    in_flight_future: Option<FenceSignalFuture<Box<dyn GpuFuture>>>,
}

pub struct VKRenderer {
    rhi: Rc<VKRHI>,
    mutable_state: RefCell<MutableRenderState>,
    render_pass: Arc<RenderPass>,
}

impl VKRenderer {
    pub fn new(rhi: Rc<VKRHI>) -> Self {
        let swapchain = Swapchain::new(rhi.as_ref());
        let render_pass = RenderPassBuilder::build_default_render_pass(
            rhi.as_ref(),
            swapchain.format,
        ).build();
        let depth_image_view = rhi.create_depth_buffer(swapchain.extent);
        let framebuffers = swapchain.create_framebuffers(&render_pass, &depth_image_view);

        Self {
            rhi,
            mutable_state: RefCell::new(MutableRenderState {
                swapchain,
                depth_image_view,
                framebuffers,
                should_recreate_swapchain: false,
                in_flight_future: None,
            }),
            render_pass,
        }
    }

    pub fn redraw(&self, scene: &VKScene) {
        self.mutable_state().in_flight_future = self.draw_frame(scene);
    }

    fn draw_frame(&self, scene: &VKScene) -> Option<FenceSignalFuture<Box<dyn GpuFuture>>> {
        if self.mutable_state_const().should_recreate_swapchain {
            self.mutable_state().recreate_swapchain_internal(self.rhi.as_ref(), &self.render_pass);
        }
        self.mutable_state_const()
            .in_flight_future
            .as_ref()
            .map(|f| f.wait(None).unwrap());

        self.rhi.gui_mut().immediate_ui(|ui| {
            let ctx = ui.context();
            egui::CentralPanel::default()
                .frame(Frame::default().fill(Color32::TRANSPARENT))
                .show(&ctx, |ui| {
                    ui.heading("My egui Application");
                    ui.horizontal(|ui| {
                        ui.label("Your name: ");
                        //ui.text_edit_singleline(&mut name);
                    });
                    //ui.add(egui::Slider::new(&mut age, 0..=120).text("age"));
                    if ui.button("Increment").clicked() {
                        // age += 1;
                    }
                    //ui.label(format!("Hello '{name}', age {age}"));
                });
        });

        let acquire_image_result = self.mutable_state_const().swapchain.acquire_next_image();
        let (swapchain_image_index, suboptimal, image_available_future) = acquire_image_result
            .map_or_else(
                |e| match e {
                    Validated::Error(VulkanError::OutOfDate) => {
                        self.mutable_state().request_recreate_swapchain();
                        None
                    }
                    _ => panic!("Error acquiring swapchain image"),
                },
                |v| Some(v),
            )?;
        if suboptimal {
            self.mutable_state().request_recreate_swapchain();
        }
        let mut command_buffer = self.rhi
            .command_buffer_interface()
            .primary_command_buffer(self.rhi.queue_family_indices().graphics_family);

        self.record_draw_command_buffer(&mut command_buffer, swapchain_image_index as usize, scene)
            .unwrap();

        let draw_finished_future = image_available_future
            .then_execute(
                self.rhi.queues().graphics_queue.clone(),
                command_buffer.build().unwrap(),
            )
            .unwrap();

        let gui_draw_future = self.rhi.gui_mut().draw_on_image(
            draw_finished_future,
            self.mutable_state_const()
                .swapchain
                .image_view(swapchain_image_index as usize)
                .clone(),
        );

        let present_future = present(
            gui_draw_future,
            self.rhi.queues().present_queue.clone(),
            SwapchainPresentInfo::swapchain_image_index(
                self.mutable_state_const().swapchain.raw().clone(),
                swapchain_image_index,
            ),
        );

        let in_flight_future = present_future
            .boxed()
            .then_signal_fence_and_flush()
            .map_or_else(
                |e| match e {
                    Validated::Error(VulkanError::OutOfDate) => {
                        self.mutable_state().request_recreate_swapchain();
                        None
                    }
                    _ => panic!("Error presenting swapchain image"),
                },
                Some,
            );
        in_flight_future
    }

    fn record_draw_command_buffer(
        &self,
        command_buffer: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        image_index: usize,
        scene: &VKScene,
    ) -> Result<(), Box<ValidationError>> {
        command_buffer
            .begin_render_pass(
                RenderPassBeginInfo {
                    render_area_offset: [0, 0],
                    render_area_extent: self.mutable_state_const().swapchain.extent,
                    clear_values: vec![
                        Some(ClearValue::Float([0.0, 0.0, 0.0, 1.0])),
                        Some(ClearValue::DepthStencil((1.0, 0))),
                    ],
                    render_pass: self.render_pass.clone(),
                    ..RenderPassBeginInfo::framebuffer(
                        self.mutable_state_const().framebuffers[image_index].clone(),
                    )
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..SubpassBeginInfo::default()
                },
            )?
            .set_viewport_with_count(smallvec![Viewport {
                offset: [0., 0.],
                extent: self
                    .mutable_state_const()
                    .swapchain
                    .extent
                    .map(|u| u as f32),
                depth_range: 0.0f32..=1.0f32,
            }])?
            .set_scissor_with_count(smallvec![Scissor {
                offset: [0, 0],
                extent: self.mutable_state_const().swapchain.extent,
            }])?;

        let resources = self.rhi.resource_manager();
        let rcs = resources.deref();

        scene
            .models()
            .iter()
            .map(|model| {
                let material_instance = model.material().get(rcs).unwrap();
                let material = material_instance.material().get(rcs).unwrap();
                let mesh = model.mesh().get(rcs).unwrap();

                let cursor = material_instance.shader_cursor();
                let model_cursor = cursor.field("gModelData").unwrap();
                model_cursor
                    .field("modelTransform")
                    .unwrap()
                    .write(&model.transform());
                model_cursor
                    .field("inverseTransposeModelTransform")
                    .unwrap()
                    .write(&model.transform().transpose().inverse());

                let view_cursor = cursor.field("gViewData").unwrap();
                view_cursor
                    .field("viewPosition")
                    .unwrap()
                    .write(&scene.camera().location());
                view_cursor
                    .field("viewProjection")
                    .unwrap()
                    .write(&scene.camera().view_projection());
                let ev = 1f32;
                view_cursor.field("exposureValue").unwrap().write(&ev);

                command_buffer
                    .bind_pipeline_graphics(material.pipeline().clone())?
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        material.pipeline_layout().clone(),
                        0,
                        material_instance.descriptor_sets()[image_index].clone(),
                    )?
                    .bind_vertex_buffers(0, mesh.vertex().clone())?
                    .bind_index_buffer(mesh.index().reinterpret_ref::<[u32]>().clone())?;
                unsafe { command_buffer.draw_indexed(mesh.index().len() as u32, 1, 0, 0, 0) }
                    .map(|_| ())
            })
            .reduce(Result::or)
            .unwrap_or(Ok(()))?;

        command_buffer
            .end_render_pass(SubpassEndInfo::default())
            .map(|_| ())
    }

    pub fn mutable_state_const(&self) -> Ref<MutableRenderState> {
        self.mutable_state.borrow()
    }
    pub fn mutable_state(&self) -> RefMut<MutableRenderState> {
        self.mutable_state.borrow_mut()
    }
    
    pub fn render_pass(&self) -> &Arc<RenderPass> {
        &self.render_pass
    }
}
impl RendererInterface for VKRenderer {
    type RHI = VKRHI;
    fn rhi(&self) -> &VKRHI {
        &self.rhi
    }
}

impl MutableRenderState {
    pub fn request_recreate_swapchain(&mut self) {
        self.should_recreate_swapchain = true;
    }

    fn recreate_swapchain_internal(&mut self, rhi: &VKRHI, render_pass: &Arc<RenderPass>) {
        //unsafe { self.device.wait_idle().unwrap() }
        if rhi.window().inner_size() == PhysicalSize::new(0, 0) {
            return;
        }
        self.swapchain = self.swapchain.recreate(
            &rhi.physical_device(),
            &rhi.surface(),
            &rhi.window(),
            &rhi.queue_family_indices(),
        );
        self.depth_image_view = rhi.create_depth_buffer(self.swapchain.extent);
        self.framebuffers = self
            .swapchain
            .create_framebuffers(render_pass, &self.depth_image_view);
        self.should_recreate_swapchain = false;
    }
}