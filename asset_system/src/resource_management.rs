use std::{
    any::{Any, TypeId},
    collections::HashMap,
};

struct ResourcePosition {
    type_id: TypeId,
    index: usize,
}

trait VecLike {
    fn size(&self) -> usize;

    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;
}

struct TypedMultiMap {
    map: HashMap<TypeId, Box<dyn VecLike>>,
}

impl TypedMultiMap {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    fn size<T: 'static>(&self) -> usize {
        self.map
            .get(&Self::type_id::<T>())
            .map(|v| v.size())
            .unwrap_or(0)
    }

    fn get_vec<T: 'static>(&self) -> Option<&Vec<T>> {
        self.map
            .get(&Self::type_id::<T>())
            .map(|v| v.as_any().downcast_ref::<Vec<T>>())
            .flatten()
    }

    fn get_vec_mut<T: 'static>(&mut self) -> Option<&mut Vec<T>> {
        self.map
            .get_mut(&Self::type_id::<T>())
            .map(|v| v.as_any_mut().downcast_mut::<Vec<T>>())
            .flatten()
    }

    fn get_add_vec<T: 'static>(&mut self) -> &mut Vec<T> {
        if self.get_vec::<T>().is_none() {
            self.map
                .insert(Self::type_id::<T>(), Box::new(Vec::<T>::new()));
        }

        self.get_vec_mut::<T>().unwrap()
    }

    fn type_id<T: 'static>() -> TypeId {
        TypeId::of::<T>()
    }
}

impl<T: 'static> VecLike for Vec<T> {
    fn size(&self) -> usize {
        self.len()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub trait Resource {
    fn set_uuid(&mut self, uuid: usize);
}

pub struct ResourceManager {
    data: TypedMultiMap,
    id_pos_map: HashMap<usize, ResourcePosition>,
    next_id: usize,
}

impl ResourceManager {
    pub fn new() -> Self {
        Self {
            data: TypedMultiMap::new(),
            id_pos_map: HashMap::new(),
            next_id: 1,
        }
    }

    pub fn get<T: Resource + 'static>(&self, uuid: usize) -> Option<&T> {
        let pos = self.id_pos_map.get(&uuid)?;
        if pos.type_id == TypedMultiMap::type_id::<T>() {
            self.data.get_vec::<T>()?.get(pos.index)
        } else {
            None
        }
    }

    pub fn get_mut<T: Resource + 'static>(&mut self, uuid: usize) -> Option<&mut T> {
        let pos = self.id_pos_map.get(&uuid)?;
        if pos.type_id == TypedMultiMap::type_id::<T>() {
            self.data.get_vec_mut::<T>()?.get_mut(pos.index)
        } else {
            None
        }
    }

    pub fn get_iter<T: Resource + 'static>(&self) -> Option<impl Iterator<Item = &T>> {
        Some(self.data.get_vec::<T>()?.iter())
    }

    pub fn add<T: Resource + 'static>(&mut self, mut data: T) -> usize {
        let id = self.next_id;
        data.set_uuid(id);
        self.next_id += 1;

        let vec = self.data.get_add_vec::<T>();
        let index = vec.len();
        vec.push(data);

        self.id_pos_map.insert(
            id,
            ResourcePosition {
                type_id: TypeId::of::<T>(),
                index,
            },
        );
        id
    }
}
