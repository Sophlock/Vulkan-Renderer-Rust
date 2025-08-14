pub mod mesh;

use std::sync::Arc;

trait Asset {
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

pub type AssetHandle<T: Asset> = Arc<T>;