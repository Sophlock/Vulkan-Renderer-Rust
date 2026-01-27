use std::{cmp::max, sync::Arc};

use asset_system::resource_management::Resource;
use smallvec::smallvec;
use vulkano::{
    Validated, ValidationError, VulkanError,
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, CopyBufferToImageInfo, ImageBlit,
        PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
    },
    format::Format,
    image::{
        Image, ImageAspects, ImageCreateInfo, ImageLayout, ImageSubresourceLayers,
        ImageSubresourceRange, ImageTiling, ImageType, ImageUsage, SampleCount,
        sampler::Filter,
        view::{ImageView, ImageViewCreateInfo, ImageViewType},
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    sync::{GpuFuture, Sharing},
};

use crate::application::{
    assets::asset_traits::{RHIResource, RHITextureInterface, TextureInterface},
    renderer::Renderer,
};

pub struct VKTexture {
    image: Arc<ImageView>,
    uuid: usize,
}

impl Resource for VKTexture {
    fn set_uuid(&mut self, uuid: usize) {
        self.uuid = uuid;
    }
}

impl RHIResource for VKTexture {
    fn uuid_mut(&mut self) -> &mut usize {
        &mut self.uuid
    }
}

impl RHITextureInterface for VKTexture {
    type RHI = Renderer;

    fn create<T: TextureInterface>(source: &T, rhi: &Self::RHI) -> Self {
        let image_create_info = ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::B8G8R8A8_SRGB,
            extent: source.size(),
            array_layers: 1,
            mip_levels: 1,
            samples: SampleCount::Sample1,
            tiling: ImageTiling::Optimal,
            usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
            sharing: Sharing::Exclusive,
            initial_layout: ImageLayout::Undefined,
            ..ImageCreateInfo::default()
        };
        let allocation_info = AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..AllocationCreateInfo::default()
        };
        let image_view_create_info = ImageViewCreateInfo {
            view_type: ImageViewType::Dim2d,
            format: Format::B8G8R8A8_SRGB,
            subresource_range: ImageSubresourceRange {
                aspects: ImageAspects::COLOR,
                mip_levels: 0..1,
                array_layers: 0..1,
            },
            ..ImageViewCreateInfo::default()
        };
        let alloc = Arc::new(StandardMemoryAllocator::new_default(rhi.device.clone()));

        let staging_buffer = Buffer::from_iter(
            alloc.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                sharing: Sharing::Exclusive,
                ..BufferCreateInfo::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..AllocationCreateInfo::default()
            },
            source.pixels().iter().copied(),
        )
        .unwrap();

        let image = Image::new(alloc, image_create_info, allocation_info).unwrap();

        let mut cb = rhi
            .command_buffer_interface
            .primary_command_buffer(rhi.queues.graphics_queue.queue_family_index());

        Self::copy_buffer_to_image(staging_buffer, &image, &mut cb).unwrap();

        let mip_levels = ((max(source.size()[0], source.size()[1]) as f32)
            .log2()
            .floor() as u32)
            + 1;
        Self::generate_mips(
            &image,
            [source.size()[0], source.size()[1]],
            mip_levels,
            1,
            &mut cb,
        )
        .unwrap();

        let image_view = ImageView::new(image, image_view_create_info).unwrap();

        cb.build()
            .unwrap()
            .execute(rhi.queues.graphics_queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        Self {
            image: image_view,
            uuid: 0,
        }
    }
}

impl VKTexture {
    pub fn copy_buffer_to_image(
        src_buffer: Subbuffer<[u8]>,
        dst_image: &Arc<Image>,
        cb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> Result<(), Validated<VulkanError>> {
        cb.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
            src_buffer,
            dst_image.clone(),
        ))?;
        Ok(())
    }

    // TODO: There is a safety check missing regarding image Format
    fn generate_mips(
        image: &Arc<Image>,
        size: [u32; 2],
        mip_levels: u32,
        array_layers: u32,
        cb: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> Result<(), Box<ValidationError>> {
        let mut width = size[0];
        let mut height = size[1];
        for mip in 0..mip_levels {
            let next_width = max(width / 2, 1);
            let next_height = max(height / 2, 1);

            let region = ImageBlit {
                src_subresource: ImageSubresourceLayers {
                    aspects: ImageAspects::COLOR,
                    mip_level: mip - 1,
                    array_layers: 0..array_layers,
                },
                src_offsets: [[0, 0, 0], [width, height, 1]],
                dst_subresource: ImageSubresourceLayers {
                    aspects: ImageAspects::COLOR,
                    mip_level: mip,
                    array_layers: 0..array_layers,
                },
                dst_offsets: [[0, 0, 0], [next_width, next_height, 1]],
                ..ImageBlit::default()
            };
            let blit = BlitImageInfo {
                src_image_layout: ImageLayout::TransferSrcOptimal,
                dst_image_layout: ImageLayout::TransferDstOptimal,
                regions: smallvec![region],
                filter: Filter::Linear,
                ..BlitImageInfo::images(image.clone(), image.clone())
            };

            cb.blit_image(blit)?;

            width = next_width;
            height = next_height;
        }
        Ok(())
    }

    pub fn layer_count(view_type: ImageViewType) -> u32 {
        if view_type == ImageViewType::Cube || view_type == ImageViewType::CubeArray {
            6
        } else {
            1
        }
    }

    /*pub fn transition_image_layout(
        image: &Arc<Image>,
        format: Format,
        initial_layout: ImageLayout,
        final_layout: ImageLayout,
        mip_levels: u32,
        cb: &AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        let array_layers = 1; // TODO

        let aspects = if final_layout == ImageLayout::DepthStencilAttachmentOptimal {
            ImageAspects::DEPTH
        }
        // TODO: Stencil component
        else {
            ImageAspects::COLOR
        };

        let (src_access, dst_access, src_stages, dst_stages) = match (initial_layout, final_layout)
        {
            (ImageLayout::Undefined, ImageLayout::TransferDstOptimal) => Some((
                AccessFlags::empty(),
                AccessFlags::TRANSFER_WRITE,
                PipelineStages::TOP_OF_PIPE,
                PipelineStages::ALL_TRANSFER,
            )),
            (ImageLayout::TransferDstOptimal, ImageLayout::ShaderReadOnlyOptimal) => Some((
                AccessFlags::TRANSFER_WRITE,
                AccessFlags::SHADER_READ,
                PipelineStages::ALL_TRANSFER,
                PipelineStages::FRAGMENT_SHADER,
            )),
            (ImageLayout::Undefined, ImageLayout::DepthStencilAttachmentOptimal) => Some((
                AccessFlags::empty(),
                AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                PipelineStages::TOP_OF_PIPE,
                PipelineStages::EARLY_FRAGMENT_TESTS,
            )),
            (_, _) => None,
        }
        .unwrap();

        let barrier = ImageMemoryBarrier {
            src_stages,
            src_access,
            dst_stages,
            dst_access,
            old_layout: initial_layout,
            new_layout: final_layout,
            subresource_range: ImageSubresourceRange {
                aspects,
                mip_levels: 0..mip_levels,
                array_layers: 0..array_layers,
            },
            ..ImageMemoryBarrier::image(image.clone())
        };

        cb.
    }*/

    pub fn image_view(&self) -> &Arc<ImageView> {
        &self.image
    }
}
