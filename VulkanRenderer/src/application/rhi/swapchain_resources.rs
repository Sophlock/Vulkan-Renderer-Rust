use std::sync::{Arc, Mutex, RwLock};

use vulkano::{
    format::Format,
    image::{
        Image, ImageAspects, ImageCreateInfo, ImageUsage,
        view::{ImageView, ImageViewCreateInfo},
    },
    memory::allocator::{AllocationCreateInfo, MemoryAllocator},
    render_pass::{Framebuffer, FramebufferCreateFlags, FramebufferCreateInfo, RenderPass},
};

use crate::application::rhi::{VKRHI, shader_object::ShaderObject};

pub struct SwapchainImage {
    image_view: Arc<ImageView>,
    buffer_allocator: Option<Arc<dyn MemoryAllocator>>,
    allocation_info: AllocationCreateInfo,
    bindings: Mutex<BoundShaderObjectCollection>,
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

struct BoundShaderObjectCollection {
    shader_objects: Vec<((u32, u32), Arc<ShaderObject>)>,
}

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
            buffer_allocator: Some(buffer_allocator),
            allocation_info,
            bindings: Default::default(),
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
            buffer_allocator: Some(rhi.buffer_allocator().clone()),
            allocation_info: AllocationCreateInfo::default(),
            bindings: Default::default(),
        }
    }

    pub fn new_depth_buffer(rhi: &VKRHI, extent: [u32; 2]) -> Self {
        Self {
            image_view: rhi.create_depth_buffer(extent),
            buffer_allocator: Some(rhi.buffer_allocator().clone()),
            allocation_info: AllocationCreateInfo::default(),
            bindings: Default::default(),
        }
    }

    pub fn recreate(this: &RwLock<Self>, new_extent: [u32; 2]) {
        let read = this.read().unwrap();
        let image_view = &read.image_view;
        let image = image_view.image();

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
            view_type: image_view.view_type(),
            format: image_view.format(),
            component_mapping: image_view.component_mapping(),
            subresource_range: image_view.subresource_range().clone(),
            usage: image_view.usage(),
            sampler_ycbcr_conversion: image_view.sampler_ycbcr_conversion().cloned(),
            ..ImageViewCreateInfo::default()
        };

        let image = Image::new(
            this.read()
                .unwrap()
                .buffer_allocator
                .clone()
                .expect("Cannot recreate external image!"),
            image_create_info,
            this.read().unwrap().allocation_info.clone(),
        )
        .unwrap();
        let image_view = ImageView::new(image, image_view_create_info).unwrap();
        drop(read);
        this.write().unwrap().image_view = image_view;
        this.read().unwrap().updated();
    }

    pub fn from_external(image_view: Arc<ImageView>) -> Self {
        Self {
            image_view,
            buffer_allocator: None,
            allocation_info: AllocationCreateInfo::default(),
            bindings: Default::default(),
        }
    }

    pub fn update_external(this: &RwLock<Self>, image_view: Arc<ImageView>) {
        if this.read().unwrap().buffer_allocator.is_some() {
            panic!("Cannot external update a swapchain image that is internally managed!");
        }
        this.write().unwrap().image_view = image_view;
        this.read().unwrap().updated();
    }

    pub fn image_view(&self) -> &Arc<ImageView> {
        &self.image_view
    }

    pub fn register_shader_object(
        &mut self,
        position: (u32, u32),
        shader_object: Arc<ShaderObject>,
    ) {
        self.bindings
            .lock()
            .unwrap()
            .shader_objects
            .push((position, shader_object));
    }

    pub fn unregister_shader_object(
        &mut self,
        position: (u32, u32),
        shader_object: &Arc<ShaderObject>,
    ) {
        let mut bindings = self.bindings.lock().unwrap();
        if let Some(index) = bindings
            .shader_objects
            .iter()
            .position(|x| x.0 == position && Arc::ptr_eq(&x.1, shader_object))
        {
            bindings.shader_objects.swap_remove(index);
        }
    }

    fn updated(&self) {
        self.reload_shader_objects();
    }

    fn reload_shader_objects(&self) {
        self.bindings
            .lock()
            .unwrap()
            .shader_objects
            .iter()
            .for_each(|(position, shader_object)| {
                shader_object.reload_swapchain_image(position);
            })
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

impl Default for BoundShaderObjectCollection {
    fn default() -> Self {
        Self {
            shader_objects: Vec::new(),
        }
    }
}
