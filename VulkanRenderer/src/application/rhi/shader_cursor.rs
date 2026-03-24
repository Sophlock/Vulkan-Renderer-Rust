use std::sync::{Arc, RwLock};

use shader_slang::{ParameterCategory, TypeKind, reflection::TypeLayout};
use vulkano::{
    NonNullDeviceAddress,
    buffer::{BufferContents, Subbuffer},
    image::{sampler::Sampler, view::ImageView},
};

use crate::application::rhi::{
    rhi_assets::vulkan_texture::VKTexture, shader_object::ShaderObject,
    swapchain_resources::SwapchainImage,
};

pub struct ShaderCursor {
    shader_object: Arc<ShaderObject>,
    offset: ShaderOffset,
    type_layout: *const TypeLayout,
}

pub struct ShaderSize {
    pub byte_size: usize,
    pub binding_size: u32,
}

#[derive(Copy, Clone)]
pub struct ShaderOffset {
    pub byte_offset: usize,
    pub binding_offset: u32,
    pub binding_array_element: u32,
}

impl ShaderCursor {
    pub fn new(source: Arc<ShaderObject>) -> Self {
        Self {
            type_layout: source.type_layout(),
            shader_object: source,
            offset: ShaderOffset::default(),
        }
    }

    pub fn field(&self, name: &str) -> Option<ShaderCursor> {
        self.field_index(self.type_layout().find_field_index_by_name(name) as u32)
    }

    pub fn field_index(&self, index: u32) -> Option<ShaderCursor> {
        let field = self.type_layout().field_by_index(index)?;
        if field.type_layout()?.kind() == TypeKind::Interface {
            // TODO: All of this is bad architecture because the C++ version said so
            let offset = self
                .shader_object
                .existential_to_offset(field.offset(ParameterCategory::ExistentialObjectParam));
            Some(ShaderCursor {
                shader_object: self.shader_object.clone(),
                offset,
                type_layout: field.type_layout()?.pending_data_type_layout()?,
            })
        } else {
            Some(ShaderCursor {
                shader_object: self.shader_object.clone(),
                offset: ShaderOffset {
                    byte_offset: self.offset.byte_offset + field.offset(ParameterCategory::Uniform),
                    binding_offset: self.offset.binding_offset
                        + self.type_layout().field_binding_range_offset(index as i64) as u32,
                    ..self.offset
                },
                type_layout: field.type_layout()?,
            })
        }
    }

    pub fn at(&self, index: u32) -> Option<ShaderCursor> {
        let element = self.type_layout().element_type_layout()?;
        Some(ShaderCursor {
            shader_object: self.shader_object.clone(),
            offset: ShaderOffset {
                byte_offset: self.offset.byte_offset
                    + (index as usize) * element.stride(ParameterCategory::Uniform),
                binding_array_element: self.offset.binding_array_element
                    * (self.type_layout().element_count()? as u32)
                    + index,
                ..self.offset
            },
            type_layout: element,
        })
    }

    pub fn type_layout(&self) -> &TypeLayout {
        unsafe { &*self.type_layout }
    }

    pub fn offset(&self) -> ShaderOffset {
        self.offset
    }

    pub fn write<T: BufferContents + Clone>(&mut self, data: &T) {
        self.shader_object.write_data(self.offset, data);
    }

    pub fn write_texture(&mut self, texture: &VKTexture) {
        self.shader_object.write_texture(self.offset, texture);
    }

    pub fn write_image_view(&mut self, view: Arc<ImageView>) {
        self.shader_object.write_image_view(self.offset, view);
    }

    pub fn write_sampler(&mut self, sampler: Arc<Sampler>) {
        self.shader_object.write_sampler(self.offset, sampler);
    }

    pub fn write_image_view_sampler(&mut self, view: Arc<ImageView>, sampler: Arc<Sampler>) {
        self.shader_object
            .write_image_view_sampler(self.offset, view, sampler);
    }

    pub fn write_buffer<T: ?Sized>(&mut self, buffer: Subbuffer<T>) {
        self.shader_object.write_buffer(self.offset, buffer);
    }

    pub fn write_swapchain_image(&mut self, image: Arc<RwLock<SwapchainImage>>) {
        self.shader_object.write_swapchain_image(self.offset, image);
    }

    pub fn write_swapchain_image_sampler(
        &mut self,
        image: Arc<RwLock<SwapchainImage>>,
        sampler: Arc<Sampler>,
    ) {
        self.shader_object
            .write_swapchain_image_sampler(self.offset, image, sampler);
    }

    pub fn write_address(&mut self, address: NonNullDeviceAddress) {
        self.write(&address.get())
    }
}

impl Default for ShaderOffset {
    fn default() -> Self {
        Self {
            byte_offset: 0,
            binding_offset: 1,
            binding_array_element: 0,
        }
    }
}
