use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::RwLock;

/// The default capacity for a `BPE`'s internal cache.
pub static DEFAULT_CACHE_CAPACITY: usize = 10_000;

/// Provides a simple multithread cache to speed up BPE tokenization that will try to read values
/// concurrently but won't block if another thread is writing.
/// The goal is clearly not the accuracy of the content, both get and set
/// are not guaranteed to actually get or set.
#[derive(Debug)]
pub(crate) struct Cache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    map: RwLock<HashMap<K, V>>,
    pub capacity: usize,
}

// We dont really care about Cache comparison, so let's make them always equal
impl<K, V> PartialEq for Cache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    fn eq(&self, _other: &Cache<K, V>) -> bool {
        true
    }
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
    pub(crate) fn new(capacity: usize) -> Self {
        let map = RwLock::new(HashMap::with_capacity(capacity));
        Cache { map, capacity }
    }

    /// Create a fresh `Cache` with the same configuration.
    pub(crate) fn fresh(&self) -> Self {
        Self::new(self.capacity)
    }

    /// Clear the cache.
    pub(crate) fn clear(&self) {
        self.map.write().unwrap().clear();
    }

    #[allow(dead_code)]
    pub(crate) fn get_values<'a, I, Q>(&self, keys_iter: I) -> Option<Vec<Option<V>>>
    where
        I: Iterator<Item = &'a Q>,
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized + 'a,
    {
        if let Ok(ref mut cache) = self.map.try_read() {
            Some(keys_iter.map(|k| cache.get(k).cloned()).collect())
        } else {
            None
        }
    }

    pub(crate) fn get<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if let Ok(ref mut cache) = self.map.try_read() {
            cache.get(key).cloned()
        } else {
            None
        }
    }

    pub(crate) fn set_values<I>(&self, entries: I)
    where
        I: IntoIterator<Item = (K, V)>,
    {
        // Before trying to acquire a write lock, we check if we are already at
        // capacity with a read handler.
        if let Ok(cache) = self.map.try_read() {
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
        if let Ok(mut cache) = self.map.try_write() {
            let free = self.capacity - cache.len();
            cache.extend(entries.into_iter().take(free));
        }
    }

    pub(crate) fn set(&self, key: K, value: V) {
        self.set_values(std::iter::once((key, value)))
    }
}
