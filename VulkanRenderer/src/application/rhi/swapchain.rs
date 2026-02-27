use std::sync::{Mutex, RwLock, Weak};
use std::{cmp::max, sync::Arc};
use vulkano::{
    Validated, VulkanError,
    device::physical::PhysicalDevice,
    format::Format,
    image::{
        Image, ImageAspects, ImageSubresourceRange, ImageUsage,
        view::{ImageView, ImageViewCreateInfo, ImageViewType},
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    swapchain::{
        ColorSpace, CompositeAlpha, PresentMode, Surface, SurfaceCapabilities, SurfaceInfo,
        Swapchain as VKSwapchain, SwapchainAcquireFuture, SwapchainCreateInfo, acquire_next_image,
    },
    sync::Sharing,
};
use winit::window::Window;

use crate::application::rhi::swapchain_resources::{SwapchainFramebuffer, SwapchainFramebufferCreateInfo, SwapchainImage};
use crate::application::rhi::{VKRHI, queue::QueueFamilyIndices};

pub struct Swapchain {
    swapchain: Arc<VKSwapchain>,
    pub format: Format,
    pub color_space: ColorSpace,
    pub extent: [u32; 2],
    pub image_count: u32,
    image_views: Vec<Arc<RwLock<SwapchainImage>>>,
    resources: Mutex<SwapchainResourceCollection>,
}

struct SwapchainResourceCollection {
    images: Vec<Weak<RwLock<SwapchainImage>>>,
    framebuffers: Vec<Weak<RwLock<SwapchainFramebuffer>>>,
}

impl Swapchain {
    pub fn new(rhi: &VKRHI) -> Self {
        let create_info = Self::create_info(
            rhi.physical_device(),
            rhi.surface(),
            rhi.window(),
            rhi.queue_family_indices(),
        );
        let (swapchain, images) = VKSwapchain::new(
            rhi.device().clone(),
            rhi.surface().clone(),
            create_info.clone(),
        )
        .unwrap();
        let mut result = Self::from_raw(swapchain, images, create_info);
        //result.persistent_image_views = result.image_views.iter().map(|view| Arc::new(RwLock::new()))
        result
    }

    pub fn acquire_next_image(
        &self,
    ) -> Result<(u32, bool, SwapchainAcquireFuture), Validated<VulkanError>> {
        acquire_next_image(self.swapchain.clone(), None)
    }

    pub fn image_view(&self, index: usize) -> &Arc<RwLock<SwapchainImage>> {
        &self.image_views[index]
    }

    pub fn image_view_iter(&self) -> impl Iterator<Item = &Arc<RwLock<SwapchainImage>>> {
        self.image_views.iter()
    }

    pub fn recreate(
        &mut self,
        physical_device: &PhysicalDevice,
        surface: &Arc<Surface>,
        window: &Window,
        queue_family_indices: &QueueFamilyIndices,
    ) {
        let create_info = Self::create_info(physical_device, surface, window, queue_family_indices);
        let (swapchain, images) = self.swapchain.recreate(create_info.clone()).unwrap();
        let new_swapchain = Self::from_raw(swapchain, images, create_info);
        self.swapchain = new_swapchain.swapchain;
        self.format = new_swapchain.format;
        self.color_space = new_swapchain.color_space;
        self.extent = new_swapchain.extent;

        assert_eq!(self.image_count, new_swapchain.image_count);

        self.image_views.iter().zip(new_swapchain.image_views).for_each(|(persistent, new)| {
            SwapchainImage::update_external(persistent, new.read().unwrap().image_view().clone());
        });

        self.resources
            .lock()
            .unwrap()
            .recreate_all(new_swapchain.extent);
    }

    fn create_info(
        physical_device: &PhysicalDevice,
        surface: &Arc<Surface>,
        window: &Window,
        queue_family_indices: &QueueFamilyIndices,
    ) -> SwapchainCreateInfo {
        let swapchain_support =
            SwapchainSupportDetails::query_swapchain_support(&physical_device, &surface);
        let (format, color_space) = Self::choose_surface_format(&swapchain_support.formats);
        let present_mode = Self::choose_present_mode(&swapchain_support.present_modes);
        let extent = Self::choose_extent(&swapchain_support.capabilities, window);
        let image_count = Self::decide_image_count(&swapchain_support.capabilities);

        let create_info = SwapchainCreateInfo {
            min_image_count: image_count,
            image_format: format,
            image_color_space: color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            image_sharing: if queue_family_indices.has_shared_family() {
                Sharing::Exclusive
            } else {
                Sharing::Concurrent(Default::default())
            },
            pre_transform: swapchain_support.capabilities.current_transform,
            composite_alpha: CompositeAlpha::Opaque,
            present_mode,
            clipped: true,
            ..SwapchainCreateInfo::default()
        };
        create_info
    }

    fn from_raw(
        swapchain: Arc<VKSwapchain>,
        images: Vec<Arc<Image>>,
        create_info: SwapchainCreateInfo,
    ) -> Self {
        let image_views = images
            .iter()
            .map(|image| {
                let image_view_create_info = ImageViewCreateInfo {
                    view_type: ImageViewType::Dim2d,
                    format: create_info.image_format,
                    subresource_range: ImageSubresourceRange {
                        aspects: ImageAspects::COLOR,
                        mip_levels: 0..1,
                        array_layers: 0..image.array_layers(),
                    },
                    usage: create_info.image_usage,
                    sampler_ycbcr_conversion: None,
                    ..ImageViewCreateInfo::default()
                };
                Arc::new(RwLock::new(SwapchainImage::from_external(ImageView::new(image.clone(), image_view_create_info).unwrap())))
            })
            .collect();
        let resources = Mutex::new(SwapchainResourceCollection::new());
        Self {
            swapchain,
            format: create_info.image_format,
            color_space: create_info.image_color_space,
            extent: create_info.image_extent,
            image_views,
            image_count: create_info.min_image_count,
            resources,
        }
    }

    pub fn choose_surface_format(formats: &Vec<(Format, ColorSpace)>) -> (Format, ColorSpace) {
        formats
            .iter()
            .filter(|(format, colorspace)| {
                format == &Format::R8G8B8_SRGB && colorspace == &ColorSpace::SrgbNonLinear
            })
            .last()
            .unwrap_or(&formats[0])
            .clone()
    }

    fn choose_present_mode(present_modes: &Vec<PresentMode>) -> PresentMode {
        if present_modes
            .iter()
            .any(|mode| mode == &PresentMode::Mailbox)
        {
            PresentMode::Mailbox
        } else {
            PresentMode::Fifo
        }
    }

    fn choose_extent(surface_capabilities: &SurfaceCapabilities, window: &Window) -> [u32; 2] {
        surface_capabilities
            .current_extent
            .unwrap_or_else(|| window.inner_size().into())
    }

    pub fn decide_image_count(surface_capabilities: &SurfaceCapabilities) -> u32 {
        let desired_image_count = surface_capabilities.min_image_count + 1;
        surface_capabilities
            .max_image_count
            .map_or(desired_image_count, |max_image_count| {
                max(max_image_count, desired_image_count)
            })
    }

    pub fn create_gbuffer(
        &self,
        rhi: &VKRHI,
        format: Format,
        usage: ImageUsage,
        aspects: ImageAspects,
    ) -> Arc<RwLock<SwapchainImage>> {
        let image = Arc::new(RwLock::new(SwapchainImage::new_gbuffer(
            rhi, self.extent, format, usage, aspects,
        )));
        self.resources.lock().unwrap().register_image(&image);
        image
    }

    pub fn create_depth_buffer(
        &self,
        rhi: &VKRHI,
    ) -> Arc<RwLock<SwapchainImage>> {
        let image = Arc::new(RwLock::new(SwapchainImage::new_depth_buffer(rhi, self.extent)));
        self.resources.lock().unwrap().register_image(&image);
        image
    }

    pub fn create_framebuffer(
        &self,
        render_pass: Arc<RenderPass>,
        create_info: SwapchainFramebufferCreateInfo
    ) -> Arc<RwLock<SwapchainFramebuffer>> {
        let framebuffer = Arc::new(RwLock::new(SwapchainFramebuffer::new(render_pass, self.extent, create_info)));
        self.resources.lock().unwrap().register_framebuffer(&framebuffer);
        framebuffer
    }

    pub fn raw(&self) -> &Arc<VKSwapchain> {
        &self.swapchain
    }
}

impl SwapchainResourceCollection {
    fn new() -> Self {
        Self {
            images: Vec::new(),
            framebuffers: Vec::new(),
        }
    }

    fn register_image(&mut self, image: &Arc<RwLock<SwapchainImage>>) {
        self.images.push(Arc::downgrade(image));
    }

    fn register_framebuffer(&mut self, framebuffer: &Arc<RwLock<SwapchainFramebuffer>>) {
        self.framebuffers.push(Arc::downgrade(framebuffer));
    }

    fn recreate_all(&mut self, new_extent: [u32; 2]) {
        self.images = self
            .images
            .iter()
            .cloned()
            .filter(|image| image.upgrade().is_some())
            .collect();

        self.framebuffers = self
            .framebuffers
            .iter()
            .cloned()
            .filter(|framebuffer| framebuffer.upgrade().is_some())
            .collect();

        self.images.iter().for_each(|weak_image| {
            if let Some(image) = weak_image.upgrade() {
                SwapchainImage::recreate(image.as_ref(), new_extent);
            }
        });

        self.framebuffers.iter().for_each(|framebuffer| {
            if let Some(framebuffer) = framebuffer.upgrade() {
                framebuffer.write().unwrap().recreate(new_extent);
            }
        })
    }
}

pub struct SwapchainSupportDetails {
    pub capabilities: SurfaceCapabilities,
    pub formats: Vec<(Format, ColorSpace)>,
    pub present_modes: Vec<PresentMode>,
}

impl SwapchainSupportDetails {
    pub fn query_swapchain_support(
        physical_device: &PhysicalDevice,
        surface: &Surface,
    ) -> SwapchainSupportDetails {
        SwapchainSupportDetails {
            capabilities: physical_device
                .surface_capabilities(surface, SurfaceInfo::default())
                .unwrap(),
            formats: physical_device
                .surface_formats(surface, SurfaceInfo::default())
                .unwrap(),
            present_modes: physical_device
                .surface_present_modes(surface, SurfaceInfo::default())
                .unwrap(),
        }
    }
}
