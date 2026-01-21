use crate::application::assets::asset_traits::TextureInterface;
use image::{DynamicImage, GenericImageView, ImageError, ImageReader};
use std::path::Path;

pub struct Texture {
    image: DynamicImage,
}

impl Texture {
    pub fn new(filepath: impl AsRef<Path>) -> Result<Self, ImageError> {
        let image = ImageReader::open(filepath)?.decode()?;
        Ok(Self { image })
    }
}

impl TextureInterface for Texture {
    fn pixels(&self) -> &[u8] {
        self.image.as_bytes()
    }
    fn size(&self) -> [u32; 3] {
        [self.image.width(), self.image.height(), 1]
    }
}
