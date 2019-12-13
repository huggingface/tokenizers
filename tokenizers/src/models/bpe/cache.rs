use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Mutex;

///
/// # Cache
///
/// Provides a simple multithread cache that will try to retrieve values
/// but won't block if someone else is already using it.
/// The goal is clearly not the accuracy of the content, both get and set
/// are not guaranteed to actually get or set.
///
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

    pub fn get_values(&self, keys: &[K]) -> Vec<Option<V>> {
        let mut lock = self.map.try_lock();
        if let Ok(ref mut cache) = lock {
            keys.iter().map(|k| cache.get(k).cloned()).collect()
        } else {
            keys.iter().map(|_| None).collect()
        }
    }

    pub fn set_values(&self, keys: Vec<K>, values: Vec<V>) {
        let mut lock = self.map.try_lock();
        if let Ok(ref mut cache) = lock {
            for (key, value) in keys.into_iter().zip(values) {
                cache.insert(key, value);
            }
        }
    }
}
