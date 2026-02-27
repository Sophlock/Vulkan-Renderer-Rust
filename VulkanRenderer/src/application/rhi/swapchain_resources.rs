use crate::application::rhi::VKRHI;
use std::sync::{Arc, RwLock};
use vulkano::format::Format;
use vulkano::image::view::{ImageView, ImageViewCreateInfo};
use vulkano::image::{Image, ImageAspects, ImageCreateInfo, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryAllocator};
use vulkano::render_pass::{
    Framebuffer, FramebufferCreateFlags, FramebufferCreateInfo, RenderPass,
};

pub struct SwapchainImage {
    image_view: Arc<ImageView>,
    buffer_allocator: Arc<dyn MemoryAllocator>,
    allocation_info: AllocationCreateInfo,
}

pub struct SwapchainFramebuffer {
    framebuffer: Arc<Framebuffer>,
    attachments: Vec<Arc<RwLock<SwapchainImage>>>,
}

pub struct SwapchainFramebufferCreateInfo {
    pub flags: FramebufferCreateFlags,
    pub attachments: Vec<Arc<RwLock<SwapchainImage>>>,
    pub layers: u32,
}

/*pub struct SwapchainBuffer {
    buffer: 
}*/

impl SwapchainImage {
    pub fn new(
        buffer_allocator: Arc<dyn MemoryAllocator>,
        image_create_info: ImageCreateInfo,
        image_view_create_info: ImageViewCreateInfo,
        allocation_info: AllocationCreateInfo,
    ) -> Self {
        let image = Image::new(
            buffer_allocator.clone(),
            image_create_info,
            allocation_info.clone(),
        )
        .unwrap();
        let image_view = ImageView::new(image, image_view_create_info).unwrap();
        Self {
            image_view,
            buffer_allocator,
            allocation_info,
        }
    }

    pub fn new_gbuffer(
        rhi: &VKRHI,
        extent: [u32; 2],
        format: Format,
        usage: ImageUsage,
        aspects: ImageAspects,
    ) -> Self {
        Self {
            image_view: rhi.create_gbuffer(extent, format, usage, aspects),
            buffer_allocator: rhi.buffer_allocator().clone(),
            allocation_info: AllocationCreateInfo::default(),
        }
    }

    pub fn new_depth_buffer(rhi: &VKRHI, extent: [u32; 2]) -> Self {
        Self {
            image_view: rhi.create_depth_buffer(extent),
            buffer_allocator: rhi.buffer_allocator().clone(),
            allocation_info: AllocationCreateInfo::default(),
        }
    }

    pub fn recreate(&mut self, new_extent: [u32; 2]) {
        let image = self.image_view.image();

        let image_create_info = ImageCreateInfo {
            flags: image.flags(),
            image_type: image.image_type(),
            format: image.format(),
            view_formats: image.view_formats().into(),
            extent: [new_extent[0], new_extent[1], 1],
            array_layers: image.array_layers(),
            mip_levels: image.mip_levels(),
            samples: image.samples(),
            tiling: image.tiling(),
            usage: image.usage(),
            stencil_usage: image.stencil_usage(),
            sharing: image.sharing().clone(),
            initial_layout: image.initial_layout(),
            external_memory_handle_types: image.external_memory_handle_types(),
            ..ImageCreateInfo::default()
        };
        let image_view_create_info = ImageViewCreateInfo {
            view_type: self.image_view.view_type(),
            format: self.image_view.format(),
            component_mapping: self.image_view.component_mapping(),
            subresource_range: self.image_view.subresource_range().clone(),
            usage: self.image_view.usage(),
            sampler_ycbcr_conversion: self.image_view.sampler_ycbcr_conversion().cloned(),
            ..ImageViewCreateInfo::default()
        };

        let image = Image::new(
            self.buffer_allocator.clone(),
            image_create_info,
            self.allocation_info.clone(),
        )
        .unwrap();
        self.image_view = ImageView::new(image, image_view_create_info).unwrap();
    }
    
    pub fn image_view(&self) -> &Arc<ImageView> {
        &self.image_view
    }
}

impl SwapchainFramebuffer {
    pub fn new(
        render_pass: Arc<RenderPass>,
        extent: [u32; 2],
        create_info: SwapchainFramebufferCreateInfo,
    ) -> Self {
        Self {
            framebuffer: Framebuffer::new(
                render_pass,
                FramebufferCreateInfo {
                    flags: create_info.flags,
                    attachments: create_info
                        .attachments
                        .iter()
                        .map(|image| image.read().unwrap().image_view.clone())
                        .collect(),
                    extent,
                    layers: create_info.layers,
                    ..FramebufferCreateInfo::default()
                },
            )
            .unwrap(),
            attachments: create_info.attachments,
        }
    }
    
    pub fn recreate(&mut self, new_extent: [u32; 2]) {
        self.framebuffer = Framebuffer::new(
            self.framebuffer.render_pass().clone(),
            FramebufferCreateInfo {
                flags: self.framebuffer.flags(),
                attachments: self
                    .attachments
                    .iter()
                    .map(|image| image.read().unwrap().image_view.clone())
                    .collect(),
                extent: new_extent,
                layers: self.framebuffer.layers(),
                ..FramebufferCreateInfo::default()
            },
        )
            .unwrap();
    }
    
    pub fn framebuffer(&self) -> &Arc<Framebuffer> {
        &self.framebuffer
    }
}

impl Default for SwapchainFramebufferCreateInfo {
    fn default() -> Self {
        Self {
            flags: FramebufferCreateFlags::empty(),
            attachments: vec![],
            layers: 0,
        }
    }
}
