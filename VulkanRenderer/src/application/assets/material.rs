use asset_system::{Asset, assets::AssetMetadata};

use crate::application::assets::asset_traits::MaterialInterface;

#[derive(Asset)]
pub struct Material {
    module_name: String,
    material_name: String,
    asset_metadata: AssetMetadata,
}

impl Material {
    pub fn new(name: String, module_name: String, material_name: String) -> Self {
        Self {
            module_name,
            material_name,
            asset_metadata: AssetMetadata::new(name),
        }
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
