use asset_system::{
    Asset,
    assets::{AssetHandle, AssetMetadata},
};

use crate::application::assets::{asset_traits::MaterialInstanceInterface, material::Material};

#[derive(Asset)]
pub struct MaterialInstance {
    material: AssetHandle<Material>,
    asset_metadata: AssetMetadata,
}

impl MaterialInstanceInterface for MaterialInstance {
    type MaterialType = Material;

    fn material(&self) -> AssetHandle<Self::MaterialType> {
        self.material.clone()
    }
}
