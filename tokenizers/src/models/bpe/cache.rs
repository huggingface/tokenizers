use std::collections::HashMap;
use std::hash::Hash;
use std::sync::RwLock;

/// The default capacity for a `BPE`'s internal cache.
pub static DEFAULT_CACHE_CAPACITY: usize = 10_000;

/// Provides a simple multithread cache to speed up BPE tokenization that will try to read values
/// concurrently but won't block if another thread is writing.
/// The goal is clearly not the accuracy of the content, both get and set
/// are not guaranteed to actually get or set.
pub(super) struct Cache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    map: RwLock<HashMap<K, V>>,
    pub capacity: usize,
}

impl<K, V> Default for Cache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    fn default() -> Self {
        Self::new(DEFAULT_CACHE_CAPACITY)
    }
}

impl<K, V> Cache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    /// Create new `Cache` with the given capacity.
    pub(super) fn new(capacity: usize) -> Self {
        let map = RwLock::new(HashMap::with_capacity(capacity));
        Cache { map, capacity }
    }

    /// Create a fresh `Cache` with the same configuration.
    pub(super) fn fresh(&self) -> Self {
        Self::new(self.capacity)
    }

    /// Clear the cache.
    pub(super) fn clear(&self) {
        self.map.write().unwrap().clear();
    }

    pub(super) fn get_values<I>(&self, keys_iter: I) -> Option<Vec<Option<V>>>
    where
        I: Iterator<Item = K>,
    {
        if let Ok(ref mut cache) = self.map.try_read() {
            Some(keys_iter.map(|k| cache.get(&k).cloned()).collect())
        } else {
            None
        }
    }

    pub(super) fn set_values<I, J>(&self, keys_iter: I, values_iter: J)
    where
        I: Iterator<Item = K>,
        J: Iterator<Item = Option<V>>,
    {
        // Before trying to acquire a write lock, we check if we are already at
        // capacity with a read handler.
        if let Ok(ref mut cache) = self.map.try_read() {
            if cache.len() >= self.capacity {
                // At capacity, so do nothing.
                return;
            }
        } else {
            // If we couldn't acquire a read handle then we probably won't be able to acquire
            // a write handle one quadrillionth of a second later.
            return;
        }
        // Not at capacity, so try acquiring a write handle.
        if let Ok(ref mut cache) = self.map.try_write() {
            for (key, value) in keys_iter.zip(values_iter).filter(|(_, v)| v.is_some()) {
                // If already at capacity, don't add any more values.
                if cache.len() >= self.capacity {
                    break;
                }
                cache.insert(key, value.unwrap());
            }
        }
    }
}
