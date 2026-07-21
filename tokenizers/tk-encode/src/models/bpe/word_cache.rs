use std::ops::Range;

use ahash::RandomState;

use crate::utils::cache::MAX_LENGTH;

#[derive(Clone, Copy, Default)]
struct CacheSlot {
    hash: u64,
    key_offsets: (u32, u16),
    ids_offsets: (u32, u16),
}

impl CacheSlot {
    fn id_range(&self) -> Range<usize> {
        let (start, len) = self.ids_offsets;
        start as usize..(start as usize + len as usize)
    }

    fn key_range(&self) -> Range<usize> {
        let (start, len) = self.key_offsets;
        start as usize..(start as usize + len as usize)
    }
}

pub struct WordCache {
    hasher: RandomState,
    slots: Box<[CacheSlot]>,
    key_bytes: Vec<u8>,
    ids: Vec<u32>,
    slot_mask: u64,
}

impl WordCache {
    pub fn new() -> Self {
        // todo: make capacity configurable
        const CAPACITY: usize = 1 << 16;
        const MASK: u64 = (CAPACITY as u64) - 1;
        Self {
            hasher: RandomState::new(),
            slots: vec![CacheSlot::default(); CAPACITY].into_boxed_slice(),
            ids: Vec::with_capacity(256),
            key_bytes: Vec::with_capacity(1024),
            slot_mask: MASK,
        }
    }

    pub fn get(&self, key: &[u8]) -> Option<&[u32]> {
        let hash = self.hasher.hash_one(key) | 1;
        let slot = self.slots[(hash & self.slot_mask) as usize];
        if hash != slot.hash {
            return None;
        }
        if key != &self.key_bytes[slot.key_range()] {
            return None;
        }
        Some(&self.ids[slot.id_range()])
    }

    pub fn insert(&mut self, key: &[u8], ids: &[u32]) {
        if key.len() > MAX_LENGTH {
            return;
        }
        let hash = self.hasher.hash_one(key) | 1;
        let slot_idx = (hash & self.slot_mask) as usize;

        if self.slots[slot_idx].hash != 0 {
            // slot already taken: skip insert
            return;
        }
        let key_offsets = (self.key_bytes.len() as u32, key.len() as u16);
        self.key_bytes.extend_from_slice(key);
        let ids_offsets = (self.ids.len() as u32, ids.len() as u16);
        self.ids.extend_from_slice(ids);

        self.slots[slot_idx] = CacheSlot {
            hash,
            key_offsets,
            ids_offsets,
        };
    }
}
