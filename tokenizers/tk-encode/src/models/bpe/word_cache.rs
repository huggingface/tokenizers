use std::ops::Range;

use ahash::RandomState;

use crate::utils::cache::MAX_LENGTH;

const WAYS: usize = 4;

#[derive(Clone, Copy, Default)]
struct CacheSlot {
    tag: u32,
    key_off: u32,
    ids_off: u32,
    key_len: u16,
    ids_len: u16,
}

#[derive(Clone, Copy, Default)]
#[repr(align(64))]
struct Bucket([CacheSlot; WAYS]);

impl CacheSlot {
    fn id_range(&self) -> Range<usize> {
        self.ids_off as usize..(self.ids_off as usize + self.ids_len as usize)
    }

    fn key_range(&self) -> Range<usize> {
        self.key_off as usize..(self.key_off as usize + self.key_len as usize)
    }
}

pub struct WordCache {
    hasher: RandomState,
    buckets: Box<[Bucket]>,
    key_bytes: Vec<u8>,
    ids: Vec<u32>,
    bucket_mask: u64,
}

impl WordCache {
    pub fn new(capacity: usize) -> Self {
        let n_buckets = (capacity.next_power_of_two() / WAYS).max(1);
        Self {
            hasher: RandomState::new(),
            buckets: vec![Bucket::default(); n_buckets].into_boxed_slice(),
            ids: Vec::with_capacity(256),
            key_bytes: Vec::with_capacity(1024),
            bucket_mask: (n_buckets as u64) - 1,
        }
    }

    // The low hash bits pick the bucket index, the high bits form the occupancy tag
	// 0x0 tag is reserved for empty spots (hence the `| 1`)
    fn locate(&self, key: &[u8]) -> (usize, u32) {
        let hash = self.hasher.hash_one(key);
        ((hash & self.bucket_mask) as usize, (hash >> 32) as u32 | 1)
    }

    pub fn get(&self, key: &[u8]) -> Option<&[u32]> {
        if key.len() > MAX_LENGTH {
            return None;
        }
        let (bucket_idx, tag) = self.locate(key);
        for slot in &self.buckets[bucket_idx].0 {
            if slot.tag == tag && key == &self.key_bytes[slot.key_range()] {
                return Some(&self.ids[slot.id_range()]);
            }
        }
        None
    }

    pub fn insert(&mut self, key: &[u8], ids: impl ExactSizeIterator<Item = u32>) {
        if key.len() > MAX_LENGTH {
            return;
        }
        let (bucket_idx, tag) = self.locate(key);
        let Some(slot) = self.buckets[bucket_idx]
            .0
            .iter()
            .position(|slot| slot.tag == 0)
        else {
            // bucket full: skip insert
            return;
        };
        self.buckets[bucket_idx].0[slot] = CacheSlot {
            tag,
            key_off: self.key_bytes.len() as u32,
            key_len: key.len() as u16,
            ids_off: self.ids.len() as u32,
            ids_len: ids.len() as u16,
        };
        self.key_bytes.extend_from_slice(key);
        self.ids.extend(ids);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let mut cache = WordCache::new(1 << 8);
        assert_eq!(cache.get(b"hello"), None);
        cache.insert(b"hello", [1u32, 2, 3].into_iter());
        cache.insert(b"world", [4u32].into_iter());
        assert_eq!(cache.get(b"hello"), Some(&[1u32, 2, 3][..]));
        assert_eq!(cache.get(b"world"), Some(&[4u32][..]));
        assert_eq!(cache.get(b"hell"), None);
    }

    #[test]
    fn single_bucket_holds_ways_entries_then_freezes() {
        // capacity <= WAYS collapses to one bucket, making conflicts deterministic
        let mut cache = WordCache::new(1);
        let keys: Vec<Vec<u8>> = (0..WAYS as u8 + 2).map(|i| vec![i; 3]).collect();
        for (i, key) in keys.iter().enumerate() {
            cache.insert(key, [i as u32].into_iter());
        }
        let cached = keys.iter().filter(|k| cache.get(k).is_some()).count();
        assert_eq!(cached, WAYS);
        for (i, key) in keys.iter().enumerate().take(WAYS) {
            assert_eq!(cache.get(key), Some(&[i as u32][..]));
        }
    }

    #[test]
    fn oversized_keys_are_ignored() {
        let mut cache = WordCache::new(1 << 8);
        let big = vec![7u8; MAX_LENGTH + 1];
        cache.insert(&big, [1u32].into_iter());
        assert_eq!(cache.get(&big), None);
    }
}
