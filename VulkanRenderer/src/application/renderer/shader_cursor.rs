use shader_slang::reflection::TypeLayout;
use crate::application::renderer::shader_object::ShaderObject;

pub struct ShaderCursor<'a> {
    shader_object: &'a ShaderObject,
    offset: ShaderOffset
}

pub struct ShaderSize {
    pub byte_size: usize,
    pub binding_size: u32,
}

#[derive(Copy, Clone)]
pub struct ShaderOffset {
    pub byte_offset: usize,
    pub binding_offset: u32,
}

impl<'a> ShaderCursor<'a> {
    /*pub fn field(name: &str) -> ShaderCursor {

    }

    pub fn type_layout(&self) -> TypeLayout {
        self.shader_object.type_layout
    }*/
}