use std::ops::Deref;

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

pub trait TextureInterface {
    fn pixels(&self) -> &[u32];
}

impl TextureInterface for Texture {
    fn pixels(&self) -> &[u32] {
        &self.pixels
    }
}