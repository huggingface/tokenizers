use evmap::{ReadHandle, WriteHandle};
use evmap::shallow_copy::ShallowCopy;
use std::hash::Hash;
use std::sync::Mutex;

/// The default capacity for a new `Cache`.
pub static DEFAULT_CACHE_CAPACITY: usize = 10_000;

/// Provides a simple multithread cache that will try to retrieve values
/// but won't block if someone else is already using it.
/// The goal is clearly not the accuracy of the content, both get and set
/// are not guaranteed to actually get or set.
pub struct Cache<K, V>
where
    K: Eq + Hash + Clone,
    V: Eq + ShallowCopy + Clone,
{
    map_r: ReadHandle<K, V>,
    map_w: Mutex<WriteHandle<K, V>>,
    pub capacity: usize,
}

impl<K, V> Default for Cache<K, V>
where
    K: Eq + Hash + Clone,
    V: Eq + ShallowCopy + Clone,
{
    fn default() -> Self {
        Self::new(DEFAULT_CACHE_CAPACITY)
    }
}

impl<K, V> Cache<K, V>
where
    K: Eq + Hash + Clone,
    V: Eq + ShallowCopy + Clone,
{
    /// Create new `Cache` with the given capacity.
    pub fn new(capacity: usize) -> Self {
        let (map_r, mut raw_map_w) = evmap::Options::default().with_capacity(capacity).construct();
        let map_w = Mutex::new(raw_map_w);
        Self {
            map_r,
            map_w,
            capacity: capacity,
        }
    }

    /// Create a fresh `Cache` with the same configuration.
    pub fn fresh(&self) -> Self {
        Self::new(self.capacity)
    }

    /// Try clearing the cache.
    pub fn try_clear(&self) {
        if let Ok(ref mut w) = self.map_w.try_lock() {
            w.purge();
            w.refresh();
        }
    }

    pub fn get_values<I>(&self, keys_iter: I) -> Option<Vec<Option<V>>>
    where
        I: Iterator<Item = K>,
    {
        Some(keys_iter.map(|k| self.map_r.get_and(&k, |v| v[0].clone())).collect())
    }

    pub fn set_values<I, J>(&self, keys_iter: I, values_iter: J)
    where
        I: Iterator<Item = K>,
        J: Iterator<Item = Option<V>>,
    {
        if let Ok(ref mut w) = self.map_w.try_lock() {
            for (key, value) in keys_iter.zip(values_iter).filter(|(_, v)| v.is_some()) {
                // If already at capacity, don't add any more values.
                if w.len() >= self.capacity {
                    break;
                }
                w.update(key, value.unwrap());
            }
            w.flush();
        }
    }
}
