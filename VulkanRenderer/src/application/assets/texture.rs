use crate::application::assets::asset_traits::TextureInterface;
use image::{DynamicImage, GenericImageView, ImageError, ImageReader};
use std::path::Path;
use crate::application::assets::{Asset, AssetMetadata};
use crate::application::resource_management::Resource;

pub struct Texture {
    image: DynamicImage,
    metadata: AssetMetadata,
}

impl Texture {
    pub fn new(filepath: impl AsRef<Path>, name: String) -> Result<Self, ImageError> {
        let image = ImageReader::open(filepath)?.decode()?;
        Ok(Self { image, metadata: AssetMetadata::new(name) })
    }
}

impl Resource for Texture {
    fn set_uuid(&mut self, uuid: usize) {
        self.metadata.uuid = uuid;
    }
}

impl Asset for Texture {
    fn asset_metadata(&self) -> &AssetMetadata {
        &self.metadata
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
