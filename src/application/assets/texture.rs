use crate::application::assets::asset_traits::TextureInterface;

pub struct Texture {
    pixels: [u32; 0]
}

impl Texture {
    pub fn new() -> Self {
        Self{
            pixels: []
        }
    }
}

impl TextureInterface for Texture {
    fn pixels(&self) -> &[u32] {
        &self.pixels
    }
}