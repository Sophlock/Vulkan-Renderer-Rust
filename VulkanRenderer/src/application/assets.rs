use std::marker::PhantomData;
use crate::application::resource_management::{Resource, ResourceManager};

pub mod mesh;
pub mod texture;
pub mod asset_traits;
pub mod material;

pub trait Asset : Resource + Sized {
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

pub struct AssetHandle<T: Asset> {
    uuid: usize,
    _phantom: PhantomData<T>,
}

impl<T: Asset + 'static> AssetHandle<T> {
    pub fn get<'a>(&self, manager: &'a ResourceManager) -> Option<&'a T> {
        manager.get(self.uuid)
    }
}

impl<T: Asset> Clone for AssetHandle<T> {
    fn clone(&self) -> Self {
        Self {
            uuid: self.uuid,
            _phantom: Default::default(),
        }
    }
}