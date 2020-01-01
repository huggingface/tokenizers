use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Mutex;

/// Provides a simple multithread cache that will try to retrieve values
/// but won't block if someone else is already using it.
/// The goal is clearly not the accuracy of the content, both get and set
/// are not guaranteed to actually get or set.
#[derive(Default)]
pub struct Cache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    map: Mutex<HashMap<K, V>>,
}

impl<K, V> Cache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    pub fn new() -> Self {
        Cache {
            map: Mutex::new(HashMap::new()),
        }
    }

    pub fn get_values<I>(&self, keys_iter: I) -> Option<Vec<Option<V>>>
    where
        I: Iterator<Item = K>,
    {
        let mut lock = self.map.try_lock();
        if let Ok(ref mut cache) = lock {
            Some(keys_iter.map(|k| cache.get(&k).cloned()).collect())
        } else {
            None
        }
    }

    pub fn set_values<I, J>(&self, keys_iter: I, values_iter: J)
    where
        I: Iterator<Item = K>,
        J: Iterator<Item = Option<V>>,
    {
        let mut lock = self.map.try_lock();
        if let Ok(ref mut cache) = lock {
            for (key, value) in keys_iter.zip(values_iter).filter(|(_, v)| v.is_some()) {
                cache.insert(key, value.unwrap());
            }
        }
    }
}
