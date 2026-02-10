use std::sync::Arc;

use vulkano::{
    device::{physical::PhysicalDevice, Device},
    format::Format,
    image::{ImageLayout, SampleCount},
    render_pass::{
        AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp,
        RenderPass, RenderPassCreateInfo, SubpassDependency, SubpassDescription,
    },
    sync::{AccessFlags, PipelineStages},
};

use super::physical_device::find_depth_format;
use crate::application::rhi::VKRHI;

pub struct RenderPassBuilder {
    create_info: RenderPassCreateInfo,
    device: Arc<Device>,
}

impl RenderPassBuilder {
    pub fn new(device: &Arc<Device>) -> Self {
        Self {
            create_info: RenderPassCreateInfo::default(),
            device: device.clone(),
        }
    }

    pub fn build(self) -> Arc<RenderPass> {
        RenderPass::new(self.device, self.create_info).unwrap()
    }

    pub fn add_attachment(&mut self, attachment: AttachmentDescription) -> &mut Self {
        self.create_info.attachments.push(attachment);
        self
    }

    pub fn add_subpass(&mut self, subpass: SubpassDescription) -> &mut Self {
        self.create_info.subpasses.push(subpass);
        self
    }

    pub fn add_dependency(&mut self, dependency: SubpassDependency) -> &mut Self {
        self.create_info.dependencies.push(dependency);
        self
    }

    pub fn add_color_attachment(&mut self, format: Format) -> &mut Self {
        self.add_attachment(AttachmentDescription {
            format,
            samples: SampleCount::Sample1,
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::Store,
            initial_layout: ImageLayout::Undefined,
            final_layout: ImageLayout::ShaderReadOnlyOptimal,
            ..AttachmentDescription::default()
        })
    }

    pub fn add_depth_attachment(&mut self, physical_device: &PhysicalDevice) -> &mut Self {
        self.add_attachment(AttachmentDescription {
            format: find_depth_format(physical_device),
            samples: SampleCount::Sample1,
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::DontCare,
            initial_layout: ImageLayout::Undefined,
            final_layout: ImageLayout::DepthStencilAttachmentOptimal,
            ..AttachmentDescription::default()
        })
    }

    pub fn add_graphics_subpass(
        &mut self,
        color_attachments: Vec<AttachmentReference>,
        depth_stencil_attachment: AttachmentReference,
        preserve_attachments: Vec<u32>,
    ) -> &mut Self {
        self.add_subpass(SubpassDescription {
            color_attachments: color_attachments.iter().map(|x| Some(x.clone())).collect(),
            depth_stencil_attachment: Some(depth_stencil_attachment),
            preserve_attachments,
            ..SubpassDescription::default()
        })
    }

    pub fn add_depth_dependency(&mut self) -> &mut Self {
        self.add_dependency(SubpassDependency {
            src_subpass: None,
            dst_subpass: Some(0),
            src_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT
                | PipelineStages::EARLY_FRAGMENT_TESTS,
            dst_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT
                | PipelineStages::EARLY_FRAGMENT_TESTS,
            src_access: AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            dst_access: AccessFlags::COLOR_ATTACHMENT_WRITE
                | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ..SubpassDependency::default()
        })
    }

    pub fn build_default_render_pass(rhi: &VKRHI, color_format: Format) -> Self {
        Self::new(rhi.device())
            .add_color_attachment(color_format)
            .add_depth_attachment(rhi.physical_device())
            .add_graphics_subpass(
                vec![AttachmentReference {
                    attachment: 0,
                    layout: ImageLayout::ColorAttachmentOptimal,
                    ..AttachmentReference::default()
                }],
                AttachmentReference {
                    attachment: 1,
                    layout: ImageLayout::DepthStencilAttachmentOptimal,
                    ..AttachmentReference::default()
                },
                vec![],
            )
            .add_depth_dependency()
            .clone()
    }
}

impl Clone for RenderPassBuilder {
    fn clone(&self) -> Self {
        Self {
            create_info: self.create_info.clone(),
            device: self.device.clone(),
        }
    }
}
