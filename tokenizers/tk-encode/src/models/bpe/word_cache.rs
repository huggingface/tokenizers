use ahash::RandomState;

const MAX_LENGTH: usize = 1024;

/// Slots searched from a key's home position. Linear probing normally settles in
/// one or two, but the walk has to stop somewhere: nothing is ever removed, so a
/// saturated table would otherwise scan forever looking for an empty slot.
const WINDOW: usize = 16;
/// Ids carried in the slot itself. Most pre-tokens encode to one or two tokens,
/// so this keeps the arena for the tail rather than the common case.
const INLINE_IDS: usize = 3;
/// `count` marker for an entry whose key bytes and/or ids live in the arenas.
const SPILLED: u8 = u8::MAX;
/// Set in `key` when it holds a hash of a > 15-byte word rather than the word
/// itself. Packed keys carry their length (1..=15) in the top byte, so the two
/// can never be confused, and neither can be 0 — the empty-slot sentinel.
const LONG_TAG: u128 = 1 << 127;

/// A key of at most 15 bytes packed into a u128: bytes in the low lanes, length
/// in the top byte. Different lengths therefore never collide, and the whole
/// comparison is one register-width equality instead of a `memcmp`.
fn pack_key(key: &[u8]) -> Option<u128> {
    if key.is_empty() || key.len() > 15 {
        return None;
    }
    let mut lanes = [0u8; 16];
    lanes[..key.len()].copy_from_slice(key);
    Some(u128::from_le_bytes(lanes) | ((key.len() as u128) << 120))
}

/// 32 bytes, two to a cache line. `w` holds the ids inline while
/// `count <= INLINE_IDS`, and otherwise the arena coordinates
/// `[key_off, ids_off, key_len << 16 | ids_len]` — where `key_len == 0` means the
/// key is packed into `key` and no bytes were stored for it.
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct CacheSlot {
    key: u128,
    w: [u32; INLINE_IDS],
    count: u8,
    freq: u8,
    _pad: u16,
}

const _: () = assert!(std::mem::size_of::<CacheSlot>() == 32);

/// An arena that reuses the space of evicted entries instead of compacting.
///
/// Runs are freed to a list per exact length, which costs nothing here because
/// `MAX_LENGTH` bounds every run: a freed run always has a list to go to, the
/// next word of that shape takes it, and no length is ever rounded up. Without
/// this the arena could only grow, and reclaiming it would mean relocating every
/// live entry into a second buffer.
struct Slab<T> {
    data: Vec<T>,
    free: Box<[Vec<u32>]>,
    budget: usize,
}

impl<T: Copy + Default> Slab<T> {
    fn new(budget: usize) -> Self {
        Self {
            data: Vec::new(),
            free: (0..=MAX_LENGTH).map(|_| Vec::new()).collect(),
            budget,
        }
    }

    fn alloc(&mut self, len: usize) -> Option<u32> {
        if let Some(off) = self.free[len].pop() {
            return Some(off);
        }
        if self.data.len() + len > self.budget {
            return None;
        }
        let off = self.data.len() as u32;
        self.data.resize(self.data.len() + len, T::default());
        Some(off)
    }

    fn release(&mut self, off: u32, len: usize) {
        self.free[len].push(off);
    }

    fn get(&self, off: u32, len: usize) -> &[T] {
        &self.data[off as usize..off as usize + len]
    }

    fn fill(&mut self, off: u32, len: usize, values: impl Iterator<Item = T>) {
        for (dst, value) in self.data[off as usize..off as usize + len]
            .iter_mut()
            .zip(values)
        {
            *dst = value;
        }
    }
}

/// Pre-token bytes to their token ids, one instance per scratch and therefore
/// per thread — no sharing, no locking.
///
/// Open addressing rather than fixed buckets, because a word that collides can
/// take the next free slot instead of being locked out. Making room does not
/// need a second copy of the table: a slot is overwritten in place, which is
/// safe because slots go empty to occupied and never back, so no probe walk is
/// ever cut short by a hole. What that costs is a bounded [`WINDOW`], and what
/// it buys is that the table and its arenas only ever hold entries that are
/// live.
///
/// Eviction picks the coldest slot of the window and halves the frequencies it
/// passed, so a word has to keep being used to keep its place, and a burst of
/// once-hot words cannot hold a window forever.
pub struct WordCache {
    hasher: RandomState,
    slots: Box<[CacheSlot]>,
    key_bytes: Slab<u8>,
    ids: Slab<u32>,
    mask: usize,
    /// Written by `get` so inline ids can be handed back as a slice.
    unpacked: [u32; INLINE_IDS],
}

impl WordCache {
    pub fn new(capacity: usize) -> Self {
        let n_slots = capacity.next_power_of_two().max(WINDOW);
        Self {
            hasher: RandomState::new(),
            slots: vec![CacheSlot::default(); n_slots].into_boxed_slice(),
            // Ceilings, not reservations — the arenas below allocate only what is
            // live. They are set high enough that the slot table runs out first:
            // an arena that refuses inserts while slots sit empty is just a
            // smaller cache.
            key_bytes: Slab::new(n_slots * 48),
            ids: Slab::new(n_slots * 16),
            mask: n_slots - 1,
            unpacked: [0; INLINE_IDS],
        }
    }

    /// The stored key, and the hash that places it. Short words key on their own
    /// bytes and need no confirmation; longer ones key on a tagged hash and are
    /// confirmed against `key_bytes`.
    fn locate(&self, key: &[u8]) -> (u128, u64) {
        let hash = self.hasher.hash_one(key);
        match pack_key(key) {
            Some(packed) => (packed, hash),
            None => ((hash as u128) | LONG_TAG, hash),
        }
    }

    fn confirmed(&self, slot: &CacheSlot, key: &[u8]) -> bool {
        if slot.key & LONG_TAG == 0 {
            return true; // the packed key is the word
        }
        let key_len = (slot.w[2] >> 16) as usize;
        key_len == key.len() && self.key_bytes.get(slot.w[0], key_len) == key
    }

    /// Return an overwritten entry's arena runs so the next entry can use them.
    fn reclaim(&mut self, index: usize) {
        let slot = self.slots[index];
        if slot.count != SPILLED {
            return;
        }
        let (key_len, ids_len) = ((slot.w[2] >> 16) as usize, (slot.w[2] & 0xFFFF) as usize);
        if key_len > 0 {
            self.key_bytes.release(slot.w[0], key_len);
        }
        if ids_len > 0 {
            self.ids.release(slot.w[1], ids_len);
        }
    }

    pub fn get(&mut self, key: &[u8]) -> Option<&[u32]> {
        if key.len() > MAX_LENGTH {
            return None;
        }
        let (stored, hash) = self.locate(key);
        let home = hash as usize;
        for step in 0..WINDOW {
            let index = (home + step) & self.mask;
            let slot = self.slots[index];
            if slot.key == 0 {
                return None;
            }
            if slot.key == stored && self.confirmed(&slot, key) {
                self.slots[index].freq = slot.freq.saturating_add(1);
                if slot.count == SPILLED {
                    return Some(self.ids.get(slot.w[1], (slot.w[2] & 0xFFFF) as usize));
                }
                self.unpacked = slot.w;
                return Some(&self.unpacked[..slot.count as usize]);
            }
        }
        None
    }

    pub fn insert(&mut self, key: &[u8], ids: impl ExactSizeIterator<Item = u32>) {
        if key.len() > MAX_LENGTH {
            return;
        }
        let (stored, hash) = self.locate(key);
        let long = stored & LONG_TAG != 0;
        let ids_len = ids.len();

        // Reserve arena space before touching the table, so a full arena costs
        // nothing but the attempt.
        let (w, count) = if long || ids_len > INLINE_IDS {
            let Some(key_off) = (if long {
                self.key_bytes.alloc(key.len())
            } else {
                Some(0)
            }) else {
                return;
            };
            let Some(ids_off) = (if ids_len > 0 {
                self.ids.alloc(ids_len)
            } else {
                Some(0)
            }) else {
                if long {
                    self.key_bytes.release(key_off, key.len());
                }
                return;
            };
            if long {
                self.key_bytes.fill(key_off, key.len(), key.iter().copied());
            }
            self.ids.fill(ids_off, ids_len, ids);
            let key_len = if long { key.len() } else { 0 };
            (
                [key_off, ids_off, ((key_len as u32) << 16) | ids_len as u32],
                SPILLED,
            )
        } else {
            let mut w = [0u32; INLINE_IDS];
            for (dst, id) in w.iter_mut().zip(ids) {
                *dst = id;
            }
            (w, ids_len as u8)
        };

        let home = hash as usize;
        let mut coldest = (u8::MAX, home & self.mask);
        let mut empty = None;
        for step in 0..WINDOW {
            let index = (home + step) & self.mask;
            if self.slots[index].key == 0 {
                empty = Some(index);
                break;
            }
            if self.slots[index].freq < coldest.0 {
                coldest = (self.slots[index].freq, index);
            }
        }
        let index = match empty {
            Some(index) => index,
            None => {
                // Age the window on the way past: every entry it holds has to be
                // used again before the next conflict or it becomes the victim.
                for step in 0..WINDOW {
                    self.slots[(home + step) & self.mask].freq >>= 1;
                }
                self.reclaim(coldest.1);
                coldest.1
            }
        };
        self.slots[index] = CacheSlot {
            key: stored,
            w,
            count,
            freq: 0, // on probation until it is used again
            _pad: 0,
        };
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
    fn oversized_keys_are_ignored() {
        let mut cache = WordCache::new(1 << 8);
        let big = vec![7u8; MAX_LENGTH + 1];
        cache.insert(&big, [1u32].into_iter());
        assert_eq!(cache.get(&big), None);
    }

    /// Both sides of the 15-byte packing boundary and both sides of the inline/
    /// arena boundary have to survive a round trip, including the long key whose
    /// stored `key` is only a hash and needs the byte comparison to confirm it.
    #[test]
    fn every_slot_shape_round_trips() {
        let mut cache = WordCache::new(1 << 8);
        let long = vec![b'x'; 200];
        let cases: [(&[u8], Vec<u32>); 4] = [
            (b"short", vec![1]),
            (b"short-wide", (0..40).collect()),
            (&long, vec![9]),
            (&long[..64], (0..64).collect()),
        ];
        for (key, ids) in &cases {
            cache.insert(key, ids.clone().into_iter());
        }
        for (key, ids) in &cases {
            assert_eq!(cache.get(key), Some(&ids[..]), "{key:?}");
        }
    }

    /// A word that hashes into a full window takes the coldest slot rather than
    /// being turned away, and the words that were actually used keep their place.
    #[test]
    fn a_full_window_evicts_its_coldest_entry() {
        let mut cache = WordCache::new(WINDOW);
        let keys: Vec<Vec<u8>> = (0..WINDOW as u8).map(|i| vec![i; 4]).collect();
        for (i, key) in keys.iter().enumerate() {
            cache.insert(key, [i as u32].into_iter());
        }
        // Every key but the first earns a hit, leaving one clear coldest slot.
        for key in &keys[1..] {
            assert!(cache.get(key).is_some());
        }
        cache.insert(b"newcomer", [999u32].into_iter());
        assert_eq!(cache.get(b"newcomer"), Some(&[999u32][..]));
        assert_eq!(
            cache.get(&keys[0]),
            None,
            "the unused entry should have gone"
        );
        for (i, key) in keys.iter().enumerate().skip(1) {
            assert_eq!(
                cache.get(key),
                Some(&[i as u32][..]),
                "used entry {i} was dropped"
            );
        }
    }

    /// Reusing an evicted entry's arena run is where this design can go wrong:
    /// hand one run to two live entries and a hit starts returning another word's
    /// ids. Churn a table far too small for the input and demand the invariant
    /// that matters — an entry may be evicted, but a hit is never wrong.
    #[test]
    fn a_hit_never_returns_another_words_ids() {
        let mut cache = WordCache::new(64);
        let mut expected: Vec<(Vec<u8>, Vec<u32>)> = Vec::new();
        for i in 0..2000usize {
            let key = match i % 4 {
                0 => format!("w{i}"),
                1 => format!("a-long-word-past-fifteen-bytes-{i}"),
                2 => format!("k{i}xxxxxxxxxxxx"),
                _ => format!("{}-{i}", "z".repeat(i % 40)),
            }
            .into_bytes();
            let ids: Vec<u32> = (0..=(i % 9) as u32).map(|k| i as u32 * 16 + k).collect();
            cache.insert(&key, ids.clone().into_iter());
            expected.push((key, ids));
        }
        let mut live = 0;
        for (key, ids) in &expected {
            if let Some(hit) = cache.get(key) {
                assert_eq!(hit, &ids[..], "{key:?}");
                live += 1;
            }
        }
        assert!(live > 0, "everything was evicted — the test proves nothing");
    }

    /// Freed runs have to come back. Without reuse the arenas would grow with
    /// every insert and the cache would be leaking rather than evicting.
    #[test]
    fn arenas_hold_the_live_set_not_every_insert() {
        let mut cache = WordCache::new(64);
        for i in 0..5000usize {
            let key = format!("a-long-word-past-fifteen-bytes-{i}");
            cache.insert(key.as_bytes(), [i as u32; 8].into_iter());
        }
        // 64 slots can hold at most 64 keys of ~34 bytes and 64 runs of 8 ids.
        assert!(cache.key_bytes.data.len() < 8 * 1024);
        assert!(cache.ids.data.len() < 8 * 1024);
    }
}
