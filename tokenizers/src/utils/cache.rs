use rustc_hash::FxHashMap;
use std::borrow::Borrow;
use std::hash::{Hash, Hasher};
use std::sync::RwLock;

/// The default capacity for a `BPE`'s internal cache.
pub static DEFAULT_CACHE_CAPACITY: usize = 10_000;
/// The maximum length we should cache in a model
/// Strings that are too long have minimal chances to cache hit anyway
pub static MAX_LENGTH: usize = 256;

/// Number of shards in the shared cache.
const SHARED_CACHE_SHARDS: usize = 64;

#[inline]
fn fx_hash<K: Hash + ?Sized>(key: &K) -> u64 {
    let mut h = rustc_hash::FxHasher::default();
    key.hash(&mut h);
    h.finish()
}

struct ShardedMap<K, V> {
    shards: Vec<RwLock<FxHashMap<K, V>>>,
    per_shard_capacity: usize,
}

impl<K: Eq + Hash + Clone, V: Clone> ShardedMap<K, V> {
    fn new(total_capacity: usize) -> Self {
        let per_shard = total_capacity.div_ceil(SHARED_CACHE_SHARDS).max(1);
        let shards = (0..SHARED_CACHE_SHARDS)
            .map(|_| {
                RwLock::new(FxHashMap::with_capacity_and_hasher(
                    per_shard,
                    Default::default(),
                ))
            })
            .collect();
        ShardedMap {
            shards,
            per_shard_capacity: per_shard,
        }
    }

    #[inline]
    fn shard_for<Q: Hash + ?Sized>(key: &Q) -> usize {
        let h = fx_hash(key);
        (h >> 48) as usize % SHARED_CACHE_SHARDS
    }

    fn get<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let idx = Self::shard_for(key);
        let shard = &self.shards[idx];
        if let Ok(guard) = shard.try_read() {
            guard.get(key).cloned()
        } else {
            None
        }
    }

    fn set(&self, key: K, value: V) {
        let idx = Self::shard_for(&key);
        let shard = &self.shards[idx];
        if let Ok(guard) = shard.try_read() {
            if guard.len() >= self.per_shard_capacity {
                return;
            }
        } else {
            return;
        }
        if let Ok(mut guard) = shard.try_write() {
            if guard.len() < self.per_shard_capacity {
                guard.insert(key, value);
            }
        }
    }

    fn clear(&self) {
        for shard in &self.shards {
            if let Ok(mut guard) = shard.write() {
                guard.clear();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public Cache
// ---------------------------------------------------------------------------

/// Sharded cache for fast concurrent tokenization lookups.
///
/// Uses 64 shards with per-shard `RwLock<FxHashMap>` to minimize lock
/// contention across threads. FxHash provides fast, non-cryptographic hashing
/// suited to the small keys used in tokenization caches.
pub(crate) struct Cache<K, V>
where
    K: Eq + Hash + Clone + 'static,
    V: Clone + 'static,
{
    map: ShardedMap<K, V>,
    pub capacity: usize,
}

impl<K, V> std::fmt::Debug for Cache<K, V>
where
    K: Eq + Hash + Clone + 'static,
    V: Clone + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cache")
            .field("capacity", &self.capacity)
            .finish()
    }
}

impl<K, V> PartialEq for Cache<K, V>
where
    K: Eq + Hash + Clone + 'static,
    V: Clone + 'static,
{
    fn eq(&self, _other: &Cache<K, V>) -> bool {
        true
    }
}

impl<K, V> Default for Cache<K, V>
where
    K: Eq + Hash + Clone + 'static,
    V: Clone + 'static,
{
    fn default() -> Self {
        Self::new(DEFAULT_CACHE_CAPACITY)
    }
}

impl<K, V> Cache<K, V>
where
    K: Eq + Hash + Clone + 'static,
    V: Clone + 'static,
{
    /// Create new `Cache` with the given capacity.
    pub(crate) fn new(capacity: usize) -> Self {
        Cache {
            map: ShardedMap::new(capacity),
            capacity,
        }
    }

    /// Create a fresh `Cache` with the same configuration.
    pub(crate) fn fresh(&self) -> Self {
        Self::new(self.capacity)
    }

    /// Clear the cache.
    pub(crate) fn clear(&self) {
        self.map.clear();
    }

    #[allow(dead_code)]
    pub(crate) fn get_values<'a, I, Q>(&self, keys_iter: I) -> Option<Vec<Option<V>>>
    where
        I: Iterator<Item = &'a Q>,
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized + 'a,
    {
        Some(keys_iter.map(|k| self.get(k)).collect())
    }

    pub(crate) fn get<Q>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.get(key)
    }

    #[allow(dead_code)]
    pub(crate) fn set_values<I>(&self, entries: I)
    where
        I: IntoIterator<Item = (K, V)>,
    {
        for (k, v) in entries {
            self.map.set(k, v);
        }
    }

    pub(crate) fn set(&self, key: K, value: V) {
        self.map.set(key, value);
    }

    pub(crate) fn resize(&mut self, capacity: usize) {
        self.capacity = capacity;
        self.map = ShardedMap::new(capacity);
    }
}
