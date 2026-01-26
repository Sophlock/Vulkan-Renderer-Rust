use crate::application::assets::{Asset, AssetMetadata};
use crate::application::assets::asset_traits::MaterialInterface;
use crate::application::resource_management::Resource;

pub struct Material {
    module_name: String,
    material_name: String,
    asset_metadata: AssetMetadata
}

impl Resource for Material {
    fn set_uuid(&mut self, uuid: usize) {
        self.asset_metadata.uuid = uuid;
    }
}

impl MaterialInterface for Material {
    fn module(&self) -> &str {
        self.module_name.as_str()
    }

    fn material(&self) -> &str {
        self.material_name.as_str()
    }
}

impl Asset for Material {
    fn asset_metadata(&self) -> &AssetMetadata {
        &self.asset_metadata
    }
}