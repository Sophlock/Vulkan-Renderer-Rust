use std::{
    cmp::max,
    sync::Arc
};
use vulkano::{device::{
    Device,
    physical::PhysicalDevice
}, image::{
    view::{ImageViewCreateInfo, ImageViewType},
    ImageAspects,
    ImageSubresourceRange,
    ImageUsage,
    view::ImageView,
    Image
}, swapchain::{ColorSpace, CompositeAlpha, PresentMode, Surface, SurfaceCapabilities, SurfaceInfo, SwapchainCreateInfo}, format::Format, swapchain::Swapchain as VKSwapchain, VulkanError, Validated};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};
use vulkano::swapchain::{acquire_next_image, AcquireNextImageInfo, AcquiredImage, SwapchainAcquireFuture};
use vulkano::sync::semaphore::Semaphore;
use vulkano::sync::Sharing;
use winit::window::Window;
use crate::application::renderer::queue::QueueFamilyIndices;

pub struct Swapchain {
    swapchain: Arc<VKSwapchain>,
    pub format: Format,
    pub color_space: ColorSpace,
    pub extent: [u32; 2],
    pub image_count: u32,
    images: Vec<Arc<Image>>,
    image_views: Vec<Arc<ImageView>>,
}

impl Swapchain {
    pub fn new(device: &Arc<Device>, physical_device: &PhysicalDevice, surface: &Arc<Surface>, window: &Window, queue_family_indices: &QueueFamilyIndices) -> Self {
        let create_info = Self::create_info(physical_device, surface, window, queue_family_indices);
        let (swapchain, images) = VKSwapchain::new(device.clone(), surface.clone(), create_info.clone()).unwrap();
        Self::from_raw(swapchain, images, create_info)
    }
    
    pub fn acquire_next_image(&self) -> Result<(u32, bool, SwapchainAcquireFuture), Validated<VulkanError>> {
        acquire_next_image(self.swapchain.clone(), None)
    }
    
    pub fn recreate(&self, physical_device: &PhysicalDevice, surface: &Arc<Surface>, window: &Window, queue_family_indices: &QueueFamilyIndices) -> Self {
        let create_info = Self::create_info(physical_device, surface, window, queue_family_indices);
        let (swapchain, images) = self.swapchain.recreate(create_info.clone()).unwrap();
        Self::from_raw(swapchain, images, create_info)
    }
    
    fn create_info(physical_device: &PhysicalDevice, surface: &Arc<Surface>, window: &Window, queue_family_indices: &QueueFamilyIndices) -> SwapchainCreateInfo {
        let swapchain_support = SwapchainSupportDetails::query_swapchain_support(&physical_device, &surface);
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
            image_sharing: if queue_family_indices.has_shared_family() {Sharing::Exclusive} else {Sharing::Concurrent(Default::default()) } ,
            pre_transform: swapchain_support.capabilities.current_transform,
            composite_alpha: CompositeAlpha::Opaque,
            present_mode,
            clipped: true,
            ..SwapchainCreateInfo::default()
        };
        create_info
    }
    
    fn from_raw(swapchain: Arc<VKSwapchain>, images: Vec<Arc<Image>>, create_info: SwapchainCreateInfo) -> Self {
        let image_views = images.iter().map(|image| {
            let image_view_create_info = ImageViewCreateInfo {
                view_type: ImageViewType::Dim2d,
                format: create_info.image_format,
                subresource_range: ImageSubresourceRange {
                    aspects: ImageAspects::COLOR,
                    mip_levels: 0..1,
                    array_layers: 0..image.array_layers(),
                },
                usage: ImageUsage::COLOR_ATTACHMENT,
                sampler_ycbcr_conversion: None,
                ..ImageViewCreateInfo::default()
            };
            ImageView::new(image.clone(), image_view_create_info).unwrap()
        }).collect();
        Self {
            swapchain,
            format: create_info.image_format,
            color_space: create_info.image_color_space,
            extent: create_info.image_extent,
            images,
            image_views,
            image_count: create_info.min_image_count
        }
    }

    fn choose_surface_format(formats: &Vec<(Format, ColorSpace)>) -> (Format, ColorSpace) {
        formats.iter().filter(|(format, colorspace)| {
            format == &Format::R8G8B8_SRGB && colorspace == &ColorSpace::SrgbNonLinear
        }).last().unwrap_or(&formats[0]).clone()
    }

    fn choose_present_mode(present_modes: &Vec<PresentMode>) -> PresentMode {
        if present_modes.iter().any(|mode| mode == &PresentMode::Mailbox) {
            PresentMode::Mailbox
        } else { PresentMode::Fifo }
    }

    fn choose_extent(surface_capabilities: &SurfaceCapabilities, window: &Window) -> [u32; 2] {
        surface_capabilities.current_extent
            .unwrap_or_else(|| {
                window.inner_size().into()
            })
    }

    fn decide_image_count(surface_capabilities: &SurfaceCapabilities) -> u32 {
        let desired_image_count = surface_capabilities.min_image_count + 1;
        surface_capabilities.max_image_count
            .map_or(desired_image_count, |max_image_count| { max(max_image_count, desired_image_count) })
    }
    
    pub fn create_framebuffers(&self, render_pass: &Arc<RenderPass>, depth_image_view: &Arc<ImageView>) -> Vec<Arc<Framebuffer>> {
        let frame_buffers = self.image_views.iter().map(|image_view| {
            let create_info = FramebufferCreateInfo{
                attachments: vec![image_view.clone(), depth_image_view.clone()],
                extent: self.extent,
                layers: 1,
                ..FramebufferCreateInfo::default()
            };
            Framebuffer::new(render_pass.clone(), create_info)
        }).flatten();
        frame_buffers.collect()
    }
    
    pub fn raw(&self) -> &Arc<VKSwapchain> {
        &self.swapchain
    }
}

struct SwapchainSupportDetails {
    capabilities: SurfaceCapabilities,
    formats: Vec<(Format, ColorSpace)>,
    present_modes: Vec<PresentMode>,
}

impl SwapchainSupportDetails {
    fn query_swapchain_support(physical_device: &PhysicalDevice, surface: &Surface) -> SwapchainSupportDetails {
        SwapchainSupportDetails {
            capabilities: physical_device.surface_capabilities(surface, SurfaceInfo::default()).unwrap(),
            formats: physical_device.surface_formats(surface, SurfaceInfo::default()).unwrap(),
            present_modes: physical_device.surface_present_modes(surface, SurfaceInfo::default()).unwrap(),
        }
    }
}