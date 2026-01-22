pub mod mesh;
pub mod texture;
pub mod asset_traits;

use std::sync::Arc;

pub trait Asset : Sized {
    fn asset_metadata(&self) -> &AssetMetadata;
    fn uuid(&self) -> usize {
        self.asset_metadata().uuid
    }

    fn name(&self) -> &String {
        &self.asset_metadata().name
    }
}

pub struct AssetMetadata {
    uuid: usize,
    name: String,
}

impl AssetMetadata {
    pub fn new(name: String) -> Self {
        Self { uuid: 0, name }
    }

    pub fn uuid(&self) -> usize {
        self.uuid
    }
}

pub type AssetHandle<T: Asset> = Arc<T>;