
use std::fmt::Debug;

use std::iter::Iterator;
use wide::i8x16;

use crate::vocab::bucket_vocab_store::key_and_hash;
#[cfg(test)]
use crate::vocab::bucket_vocab_store::INLINE_KEY_BYTES;


/// How many ids a [`WordCacheSlot`] holds inline before it has to spill. A probe writes this
pub const MAX_INLINE_IDS: usize = 3;

/// A table mapping words (`[u8]`) to the token ids they encode to (`[u32]`)
pub struct WordCache {
    /// The cache slots as a contiguous table. See [`WordCacheSlot`] for more details.
    cached_words: Box<[WordCacheSlot]>,
    /// A quick lookup table to find candidates in [`Self::cached_words`] quickly
    quick_lookup: Box<[u8]>,

    placement_mask: u64,

    /// Controls eviction mechanism for spilled_buffer
    spilled_generation: u32,
    /// The maximum capacity, in number of elements, allowed for spilled_buffer
    spilled_budget: usize,
    /// An additional buffer to store ids that don't fit in a [`WordCacheSlot`]
    spilled_buffer: Vec<u32>,
}

impl<'a> WordCache {
    const EMPTY: u8 = 0;

    /// How many slots a word's window spans, starting at its home slot.
    const WINDOW_SIZE: usize = 16;

    pub fn new(num_slots: usize) -> Self {
        let next_pow2 = num_slots.next_power_of_two();
        if next_pow2 != num_slots {
        }
        let n: usize = next_pow2 + Self::WINDOW_SIZE;
        let spilled_budget = 16 * n;
        Self {
            cached_words: vec![WordCacheSlot::default(); n].into_boxed_slice(),
            quick_lookup: vec![0; n].into_boxed_slice(),
            placement_mask: (next_pow2 as u64) - 1,
            spilled_generation: 0,
            spilled_budget,
            spilled_buffer: Vec::new(),
        }
    }

    /// Looks up a word in the cache.
    #[inline]
    pub fn lookup(&'a self, word: &[u8]) -> Lookup<'a> {
        self.lookup_placed(make_lookup_key(word, self.placement_mask))
    }

    /// [`Self::lookup`] for a caller that already has the word's key and hash from
    #[inline]
    pub fn lookup_keyed(&'a self, key: u64, hash: u64) -> Lookup<'a> {
        self.lookup_placed(placement_from(LookupKey(key), hash, self.placement_mask))
    }

    /// Probe and emit in one step: on an inline hit in the home slot the ids are written straight
    /// to `dst` and the count returned, so nothing goes back to the slot and nothing becomes a
    /// slice.
    /// This is the shape the hot path wants. [`Self::lookup`] hands back a `&[u32]`, which means
    /// the caller re-reads the slot to build a fat pointer and then copies a run whose length it
    /// only learns at run time -- three trips over one 32-byte line that a single load already
    /// brought in. Here that line is read once, all [`MAX_INLINE_IDS`] lanes are stored
    /// unconditionally, and the caller advances its cursor by the count: no branch on the length,
    /// no second load, no slice.
    /// Falls back to the full window walk for anything else. The table is sized well above its
    /// load, so a word's home slot is usually the one it was placed in and the walk is a few
    /// percent of words.
    /// # Safety
    /// `dst` must have room for [`MAX_INLINE_IDS`] `u32` writes. `word` must not be empty --
    /// an empty word keys to zero, which is also what an untouched slot holds.
    #[inline]
    pub unsafe fn probe_emit(&'a self, word: &[u8], dst: *mut u32) -> ProbeEmit<'a> {
        debug_assert!(!word.is_empty(), "probe_emit needs a non-empty word");
        let (key, hash) = key_and_hash(word);
        unsafe { self.probe_emit_keyed(key, hash, dst) }
    }

    /// [`Self::probe_emit`] for a caller that already has the word's key and hash.
    /// # Safety
    /// As [`Self::probe_emit`]: `dst` must have room for [`MAX_INLINE_IDS`] `u32` writes.
    #[inline]
    pub unsafe fn probe_emit_keyed(&'a self, key: u64, hash: u64, dst: *mut u32) -> ProbeEmit<'a> {
        let placement = placement_from(LookupKey(key), hash, self.placement_mask);
        // SAFETY: `index` is masked with `placement_mask` (`next_pow2 - 1`), and the table is
        // `next_pow2 + WINDOW_SIZE` long, so the home slot is always in bounds.
        let slot = unsafe { *self.cached_words.as_ptr().add(placement.index) };
        if slot.key == placement.key && !slot.is_spilled() {
            // SAFETY: the caller guarantees room for `MAX_INLINE_IDS`. Lanes past `ids_len` are
            // dead: the caller advances its cursor by `ids_len` only, so the next word overwrites
            // them or the final `set_len` cuts them off.
            unsafe {
                for lane in 0..MAX_INLINE_IDS {
                    dst.add(lane).write(slot.payload[lane]);
                }
            }
            return ProbeEmit::Wrote(slot.ids_len as usize);
        }
        match self.lookup_placed(placement) {
            Lookup::Hit(ids) => ProbeEmit::Hit(ids),
            Lookup::Miss(at) => ProbeEmit::Miss(at),
        }
    }

    /// The window walk, once a word has been keyed and placed. Split out of [`Self::lookup`] so
    #[inline]
    fn lookup_placed(&'a self, placement: InsertPlacement) -> Lookup<'a> {
        let InsertPlacement {
            key,
            index: home,
            tag,
        } = placement;

        let tag_window = self.tag_window(home);
        let (candidates, first_empty) = tag_window.find_matches_and_first_empty(tag);

        for candidate in candidates {
            let slot = &self.cached_words[candidate];
            if slot.key == key {
                if slot.is_stale(self.spilled_generation) {
                    return Lookup::Miss(InsertPlacement {
                        index: candidate,
                        key,
                        tag,
                    });
                } else {
                    return Lookup::Hit(self.get_cached_ids(slot));
                }
            }
        }

        Lookup::Miss(InsertPlacement {
            index: first_empty.unwrap_or(home),
            key,
            tag,
        })
    }

    /// Insert a new (word, ids) pair in the cache
    pub fn insert(&mut self, placement: InsertPlacement, ids: impl ExactSizeIterator<Item = u32>) {
        let len = ids.len();
        let InsertPlacement { index, key, tag } = placement;

        let word = if len <= MAX_INLINE_IDS {
            WordCacheSlot::new_self_contained(key, ids)
        } else {
            if self.spilled_buffer.len() + len > self.spilled_budget {
                self.spilled_buffer.clear();
                self.spilled_generation = self.spilled_generation.wrapping_add(1);
                if self.spilled_generation == 0 {
                    self.reset();
                }
            }
            let start = self.spilled_buffer.len();
            self.spilled_buffer.extend(ids);
            WordCacheSlot::new_spilled(key, (start, start + len), self.spilled_generation)
        };

        self.cached_words[index] = word;
        self.quick_lookup[index] = tag;
    }
}

impl<'a> WordCache {
    /// Helper to fetch the ids from the word
    fn get_cached_ids(&'a self, word: &'a WordCacheSlot) -> &'a [u32] {
        match word.variant() {
            CacheSlotType::SelfContained(word) => word.ids(),
            CacheSlotType::Spilled(word) => {
                let (start, end) = word.ids_offsets();
                &self.spilled_buffer[start..end]
            }
        }
    }

    fn reset(&mut self) {
        self.spilled_buffer.clear();
        self.quick_lookup = vec![0; self.quick_lookup.len()].into_boxed_slice();
        self.cached_words =
            vec![WordCacheSlot::default(); self.cached_words.len()].into_boxed_slice();
    }

    fn tag_window(&self, home_index: usize) -> Window {
        Window::new(&self.quick_lookup, home_index)
    }
}

/// A cache slot, packed in 32 bytes.
/// There are two variants, depending on the number of token ids the word encodes to:
///
/// ## inline: 3 or fewer ids to cache
///
/// The token ids are encoded inline in the cache slot:
/// ```text
/// ┌────────────────────────┬────────────┬────────────┬────────────┬─────────┬─────────┐
/// │          key           │   id[0]    │   id[1]    │   id[2]    │ ids_len │   _pad  │
/// │       LookupKey        │    u32     │    u32     │    u32     │    u8   │ [u8; 3] │
/// └────────────────────────┴────────────┴────────────┴────────────┴─────────┴─────────┘
/// 0                        16           20           24           28        29        32 bytes
/// ```
///
/// ## spilled: more than 3 ids to cache
///
/// The token ids are stored in [`WordCache::spilled_buffer`].
/// The slot holds offsets in that buffer.
/// ```text
/// ┌────────────────────────┬────────────┬────────────┬────────────┬─────────┬─────────┐
/// │          key           │   start    │    end     │ generation │ SPILLED │   _pad  │
/// │       LookupKey        │    u32     │    u32     │    u32     │    u8   │ [u8; 3] │
/// └────────────────────────┴────────────┴────────────┴────────────┴─────────┴─────────┘
/// 0                        16           20           24           28        29        32 bytes
/// ```
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct WordCacheSlot {
    key: LookupKey,
    payload: [u32; 3],
    ids_len: u8,
    _pad: [u8; 3],
}

const _: () = assert!(size_of::<WordCacheSlot>() == 32);
const _: () = assert!(align_of::<WordCacheSlot>() == 32);

impl WordCacheSlot {
    /// A sentinel value that discriminates an inline slot (ids are stored in the slot) from a spilled slot
    const SPILLED: u8 = 0xFF;

    pub fn new_spilled(key: LookupKey, ids_offsets: (usize, usize), generation: u32) -> Self {
        let (start, end) = ids_offsets;
        Self {
            key,
            ids_len: Self::SPILLED,
            payload: [start as u32, end as u32, generation],
            ..Default::default()
        }
    }

    pub fn new_self_contained(key: LookupKey, ids: impl ExactSizeIterator<Item = u32>) -> Self {
        assert!(ids.len() <= MAX_INLINE_IDS);
        let ids_len = ids.len() as u8;
        let mut payload = [0u32; 3];
        for (slot, id) in payload.iter_mut().zip(ids) {
            *slot = id;
        }
        Self {
            key,
            ids_len,
            payload,
            ..Default::default()
        }
    }

    pub fn is_stale(&self, current_generation: u32) -> bool {
        match self.variant() {
            CacheSlotType::SelfContained(_) => false,
            CacheSlotType::Spilled(spilled) => spilled.generation() != current_generation,
        }
    }
}

impl WordCacheSlot {
    fn is_spilled(self) -> bool {
        self.ids_len == Self::SPILLED
    }

    fn variant<'a>(&'a self) -> CacheSlotType<'a> {
        if self.is_spilled() {
            CacheSlotType::Spilled(Spilled(self))
        } else {
            CacheSlotType::SelfContained(SelfContained(self))
        }
    }
}

/// Convenience wrapper around [`WordCacheSlot`] to discriminate
pub enum CacheSlotType<'a> {
    SelfContained(SelfContained<'a>),
    Spilled(Spilled<'a>),
}

#[repr(transparent)]
pub struct Spilled<'a>(&'a WordCacheSlot);

impl<'a> Spilled<'a> {
    fn ids_offsets(self) -> (usize, usize) {
        let [start, end, _] = self.0.payload;
        (start as usize, end as usize)
    }

    fn generation(self) -> u32 {
        self.0.payload[2]
    }
}

#[repr(C, align(32))]
pub struct SelfContained<'a>(&'a WordCacheSlot);

impl<'a> SelfContained<'a> {
    fn ids(self) -> &'a [u32] {
        let len = self.0.ids_len;
        &self.0.payload[0..len as usize]
    }
}

/// The lookup key, packed in a u128
///
/// There are two variants depending on the word's length in bytes:
///
/// ## inline: the word is 15 bytes or shorter
///
/// The key holds the word's bytes as is, plus the word's length in the top byte.
///
/// bit 127 = 0, inline: the word is its own key (len <= 15)
/// ```text
/// ┌───┬──────────────┬───────────────────────────────────────────────────────────────────────┐
/// │ 0 │    length    │                 word bytes, little-endian, zero padded                │
/// │   │    7 bits    │                                120 bits                               │
/// └───┴──────────────┴───────────────────────────────────────────────────────────────────────┘
///  127 126        120 119                                                                   0
/// ```
///
/// ## hashed: the word is longer than 15 bytes
///
/// the key holds 127 bits of hash of the word's bytes
///
/// bit 127 = 1, hashed: longer words compare by hash
/// ```text
/// ┌───┬───────────────────────────────────────────┬────────────────────────────────────────────┐
/// │ 1 │             discriminant hash             │               placement hash               │
/// │   │                  63 bits                  │                  64 bits                   │
/// └───┴───────────────────────────────────────────┴────────────────────────────────────────────┘
///  127 126                                      64 63                                         0
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
#[repr(transparent)]
pub struct LookupKey(u64);

/// The key, home slot and tag of a word.
#[inline]
fn make_lookup_key(word: &[u8], placement_mask: u64) -> InsertPlacement {
    let (key, hash) = key_and_hash(word);
    placement_from(LookupKey(key), hash, placement_mask)
}

#[inline]
fn placement_from(key: LookupKey, hash: u64, placement_mask: u64) -> InsertPlacement {
    InsertPlacement {
        key,
        index: (hash & placement_mask) as usize,
        tag: ((hash >> (64 - 8)) as u8).max(WordCache::EMPTY + 1),
    }
}

impl std::fmt::Debug for LookupKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let len = (self.0 >> 56) as usize;
        if len <= 7 {
            let bytes = self.0.to_le_bytes();
            f.debug_tuple("LookupKey")
                .field(&String::from_utf8_lossy(&bytes[..len]).into_owned())
                .finish()
        } else {
            write!(f, "LookupKey(hash {:#018x})", self.0)
        }
    }
}

pub struct InsertPlacement {
    index: usize,
    key: LookupKey,
    tag: u8,
}

pub enum Lookup<'a> {
    Hit(&'a [u32]),
    Miss(InsertPlacement),
}

/// What [`WordCache::probe_emit`] found. `Wrote` is the fast path: the ids are already at the
pub enum ProbeEmit<'a> {
    /// An inline hit in the home slot. [`MAX_INLINE_IDS`] lanes were written at `dst`; this many
    Wrote(usize),
    /// A hit the fast path could not serve -- a spilled entry, or one placed off its home slot.
    Hit(&'a [u32]),
    Miss(InsertPlacement),
}

struct Window {
    window: [u8; WordCache::WINDOW_SIZE],
    offset: usize,
}

impl Window {
    fn new(table: &[u8], start: usize) -> Self {
        Window {
            window: table[start..start + WordCache::WINDOW_SIZE]
                .try_into()
                .unwrap(),
            offset: start,
        }
    }

    /// Finds all candidates in the window that match the needle, and the leftmost 0x00 (empty slot)
    fn find_matches_and_first_empty(&self, needle: u8) -> (SlotSet, Option<usize>) {
        let window = i8x16::from(self.window.map(|byte| byte as i8));
        let matches_bitmask = window.simd_eq(i8x16::from([needle as i8; 16])).to_bitmask() as u16;
        let empty_bitmask = window
            .simd_eq(i8x16::from([WordCache::EMPTY as i8; 16]))
            .to_bitmask() as u16;

        let before_first_empty = !empty_bitmask & empty_bitmask.wrapping_sub(1);

        let candidates = SlotSet {
            mask: matches_bitmask & before_first_empty,
            offset: self.offset,
        };
        let first_empty =
            (empty_bitmask != 0).then(|| self.offset + empty_bitmask.trailing_zeros() as usize);
        (candidates, first_empty)
    }
}

#[derive(Clone, Copy)]
struct SlotSet {
    mask: u16,
    offset: usize,
}

impl Iterator for SlotSet {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.mask == 0 {
            return None;
        }
        let slot = self.mask.trailing_zeros() as usize;
        self.mask &= self.mask - 1;
        Some(slot + self.offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPILLED_BUDGET: usize = 1 << 16;

    impl<'a> Lookup<'a> {
        pub fn hit(self) -> Option<&'a [u32]> {
            match self {
                Lookup::Hit(ids) => Some(ids),
                Lookup::Miss(_) => None,
            }
        }
    }

    fn store(cache: &mut WordCache, word: &[u8], ids: &[u32]) {
        if let Lookup::Miss(at) = cache.lookup(word) {
            cache.insert(at, ids.iter().copied());
        }
    }

    /// The first `n` probe words whose home slot falls in `range`, for tests that
    fn words_homed_in(cache: &WordCache, range: std::ops::Range<usize>, n: usize) -> Vec<Vec<u8>> {
        (0u32..)
            .map(|i| format!("w{i}").into_bytes())
            .filter(|word| {
                let InsertPlacement { index: home, .. } =
                    make_lookup_key(word, cache.placement_mask);
                range.contains(&home)
            })
            .take(n)
            .collect()
    }

    /// Three ids per word, since fewer has its own test, and homes clear of the
    #[test]
    fn a_stored_word_is_found_again() {
        let mut cache = WordCache::new(1 << 8);
        let n_slots = cache.cached_words.len();
        let safe = WordCache::WINDOW_SIZE..n_slots - WordCache::WINDOW_SIZE;
        let words = words_homed_in(&cache, safe, 3);
        for (i, word) in words.iter().enumerate() {
            let base = 3 * i as u32;
            store(&mut cache, word, &[base, base + 1, base + 2]);
        }
        for (i, word) in words.iter().enumerate() {
            let base = 3 * i as u32;
            assert_eq!(
                cache.lookup(word).hit(),
                Some(&[base, base + 1, base + 2][..]),
                "{word:?}"
            );
        }
    }

    /// Home slots stop at `placement_mask`; the table holds [`WordCache::WINDOW_SIZE`]
    #[test]
    fn words_homed_on_the_last_slot_round_trip() {
        let mut cache = WordCache::new(1 << 2);
        let last = cache.placement_mask as usize;
        let words = words_homed_in(&cache, last..last + 1, 2);
        for (i, word) in words.iter().enumerate() {
            store(&mut cache, word, &[i as u32; 3]);
        }
        for (i, word) in words.iter().enumerate() {
            assert_eq!(
                cache.lookup(word).hit(),
                Some(&[i as u32; 3][..]),
                "{word:?}"
            );
        }
    }

    /// Both sides of the fifteen-byte key boundary, crossed with both sides of the
    #[test]
    fn every_key_and_slot_shape_round_trips() {
        let mut cache = WordCache::new(1 << 8);
        let cases: [(&[u8], Vec<u32>); 4] = [
            (b"short", vec![1, 2]),
            (b"short-spilled", (0..10).collect()),
            (b"a-word-past-fifteen-bytes", vec![3]),
            (b"another-word-past-fifteen-bytes", (10..30).collect()),
        ];
        for (word, ids) in &cases {
            store(&mut cache, word, ids);
        }
        for (word, ids) in &cases {
            assert_eq!(cache.lookup(word).hit(), Some(&ids[..]), "{word:?}");
        }
    }

    /// A long word is keyed on its hash, so no length is too much for the cache.
    #[test]
    fn a_very_long_word_round_trips() {
        let mut cache = WordCache::new(1 << 8);
        let word = vec![b'q'; 8192];
        let ids: Vec<u32> = (0..2000).collect();
        store(&mut cache, &word, &ids);
        assert_eq!(cache.lookup(&word).hit(), Some(&ids[..]));
    }

    /// Two distinct words must never share a key, or a hit returns the other word's
    #[test]
    fn packed_keys_are_unique_per_word() {
        let cache = WordCache::new(1 << 8);
        let key = |word: &[u8]| make_lookup_key(word, cache.placement_mask).key;
        assert_ne!(key(b"aaaaaaaaaaaaaa\x7f"), key(b"aaaaaaaaaaaaaa\xff"));
        assert_ne!(key(b"abcd"), key(b"abcd\0"));
        let mut seen = std::collections::HashSet::new();
        for len in 1..=INLINE_KEY_BYTES {
            for b in 0..=255u8 {
                let word: Vec<u8> = (0..len).map(|i| b.wrapping_add(i as u8)).collect();
                assert!(seen.insert(key(&word).0), "collision at len={len} b={b}");
            }
        }
    }

    /// One window shape per row: the needle in various lanes, an empty slot in
    #[test]
    fn the_scan_reports_matches_before_the_first_empty_and_the_empty_itself() {
        let offset = 3;
        let cases: &[(&[u8], u16, Option<usize>)] = &[
            (
                &[
                    0xA7, 0x31, 0xA7, 0x00, 0x5F, 0xA7, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                    0x18, 0x19, 0x1A,
                ],
                0b101,
                Some(3),
            ),
            (
                &[
                    0xA7, 0x31, 0xA7, 0x22, 0x5F, 0xA7, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                    0x18, 0x19, 0xA7,
                ],
                0b1000000000100101,
                None,
            ),
            (
                &[
                    0x00, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7,
                    0xA7, 0xA7, 0xA7,
                ],
                0,
                Some(0),
            ),
            (
                &[
                    0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7,
                    0xA7, 0xA7, 0x00,
                ],
                0b0111_1111_1111_1111,
                Some(15),
            ),
        ];
        for (lanes, want_mask, want_empty) in cases {
            let mut table = vec![0xEEu8; offset];
            table.extend_from_slice(lanes);
            let window = Window::new(&table, offset);
            let (candidates, first_empty) = window.find_matches_and_first_empty(0xA7);
            let found: Vec<usize> = candidates.collect();
            let want: Vec<usize> = (0..16)
                .filter(|i| want_mask >> i & 1 == 1)
                .map(|i| i + offset)
                .collect();
            assert_eq!(found, want, "lanes {lanes:?}");
            assert_eq!(
                first_empty,
                want_empty.map(|i| i + offset),
                "lanes {lanes:?}"
            );
        }
    }

    /// The tag row marks a free slot with 0x00, so no live entry may carry that
    #[test]
    fn a_live_tag_is_never_the_empty_marker() {
        for i in 0..100_000u32 {
            let placement = make_lookup_key(&i.to_le_bytes(), u64::MAX);
            assert_ne!(placement.tag, WordCache::EMPTY, "word {i}");
        }
    }

    /// Every inline length, with a different value in every byte position, so a
    #[test]
    fn an_inline_key_is_the_words_bytes_with_the_length_on_top() {
        for len in 0..=INLINE_KEY_BYTES {
            let word: Vec<u8> = (1..=len as u8).collect();
            let mut padded = [0u8; 8];
            padded[..len].copy_from_slice(&word);
            padded[7] = len as u8;
            assert_eq!(
                key_and_hash(&word).0,
                u64::from_le_bytes(padded),
                "len={len}"
            );
        }
    }

    /// A nonzero start, since every spill after the first has one and offsets that
    #[test]
    fn a_spilled_words_offsets_round_trip() {
        let cached = WordCacheSlot::new_spilled(LookupKey::default(), (5, 9), 0);
        match cached.variant() {
            CacheSlotType::Spilled(word) => assert_eq!(word.ids_offsets(), (5, 9)),
            CacheSlotType::SelfContained(_) => {
                panic!("built as spilled, read back as self-contained")
            }
        }
    }

    #[test]
    fn a_word_with_one_id_is_stored_self_contained() {
        let cached = WordCacheSlot::new_self_contained(LookupKey::default(), [42].into_iter());
        match cached.variant() {
            CacheSlotType::SelfContained(word) => assert_eq!(word.ids(), &[42]),
            CacheSlotType::Spilled(_) => panic!("one id fits without spilling"),
        }
    }

    /// A tag is one byte of hash, so about one occupied slot in 256 carries the tag
    #[test]
    fn a_tag_collision_is_confirmed_against_the_key() {
        let mut cache = WordCache::new(1 << 8);
        let InsertPlacement { index, tag, .. } = make_lookup_key(b"beta", cache.placement_mask);
        assert_ne!(tag, WordCache::EMPTY, "pick a word with a nonzero tag");
        cache.quick_lookup[index] = tag;
        cache.cached_words[index] =
            WordCacheSlot::new_self_contained(LookupKey(key_and_hash(b"decoy").0), [7].into_iter());

        assert_eq!(cache.lookup(b"beta").hit(), None);
        store(&mut cache, b"beta", &[2]);
        assert_eq!(cache.lookup(b"beta").hit(), Some(&[2][..]));
    }

    /// A word whose whole window is taken is still cached: it evicts its home slot.
    #[test]
    fn a_full_window_evicts_only_the_home_slot() {
        let mut cache = WordCache::new(1);
        let words = words_homed_in(&cache, 0..1, WordCache::WINDOW_SIZE);
        for (i, word) in words.iter().enumerate() {
            store(&mut cache, word, &[i as u32]);
        }

        store(&mut cache, b"newcomer", &[999]);
        assert_eq!(cache.lookup(b"newcomer").hit(), Some(&[999][..]));
        assert_eq!(cache.lookup(&words[0]).hit(), None);
        for (i, word) in words.iter().enumerate().skip(1) {
            assert_eq!(cache.lookup(word).hit(), Some(&[i as u32][..]), "{word:?}");
        }
    }

    /// The filler word brings a whole budget of ids on its own, so caching it
    #[test]
    fn a_spilled_word_misses_after_an_evict() {
        let mut cache = WordCache::new(1 << 6);
        store(&mut cache, b"a-spilled-word", &[1, 2, 3, 4]);
        store(&mut cache, b"the-filler-word", &vec![9; SPILLED_BUDGET]);
        assert_eq!(cache.lookup(b"a-spilled-word").hit(), None);
    }

    /// An evict drops a word's ids but leaves its slot behind. The miss the
    #[test]
    fn an_evicted_word_round_trips_once_re_inserted() {
        let mut cache = WordCache::new(1 << 6);
        store(&mut cache, b"a-spilled-word", &[1, 2, 3, 4]);
        store(&mut cache, b"the-filler-word", &vec![9; SPILLED_BUDGET]);
        store(&mut cache, b"a-spilled-word", &[5, 6, 7, 8]);
        assert_eq!(
            cache.lookup(b"a-spilled-word").hit(),
            Some(&[5, 6, 7, 8][..])
        );
    }

    /// An evict leaves the word's slot behind, key and tag intact. The miss the
    #[test]
    fn a_miss_reuses_the_slot_of_its_stale_copy() {
        let mut cache = WordCache::new(1 << 6);
        let slot = match cache.lookup(b"a-spilled-word") {
            Lookup::Miss(at) => at.index,
            Lookup::Hit(_) => panic!("the cache is empty, this must be a miss"),
        };
        store(&mut cache, b"a-spilled-word", &[1, 2, 3, 4]);
        store(&mut cache, b"the-filler-word", &vec![9; SPILLED_BUDGET]);
        match cache.lookup(b"a-spilled-word") {
            Lookup::Miss(at) => assert_eq!(at.index, slot),
            Lookup::Hit(_) => panic!("the ids were evicted, this must be a miss"),
        }
    }

    /// Five evict cycles on the same word: it must end up holding exactly one
    #[test]
    fn an_evicted_word_never_occupies_two_slots() {
        let mut cache = WordCache::new(1 << 6);
        for round in 0..5u32 {
            store(&mut cache, b"a-spilled-word", &[round; 4]);
            store(&mut cache, b"the-filler-word", &vec![9; SPILLED_BUDGET]);
        }
        let key = make_lookup_key(b"a-spilled-word", cache.placement_mask).key;
        let copies = cache
            .cached_words
            .iter()
            .filter(|slot| slot.key == key)
            .count();
        assert_eq!(copies, 1);
    }

    /// A self-contained word keeps its ids in the slot, not in the buffer, so
    #[test]
    fn a_self_contained_word_still_hits_after_an_evict() {
        let mut cache = WordCache::new(1 << 6);
        store(&mut cache, b"short", &[1, 2]);
        store(&mut cache, b"the-filler-word", &vec![9; SPILLED_BUDGET]);
        store(&mut cache, b"one-more-word", &[3, 4, 5, 6]);
        assert_eq!(cache.lookup(b"short").hit(), Some(&[1, 2][..]));
    }

    /// The evict happens inside the insert that caches this word: the slot
    #[test]
    fn the_evicting_insert_caches_its_own_word() {
        let mut cache = WordCache::new(1 << 6);
        store(&mut cache, b"the-filler-word", &vec![9; SPILLED_BUDGET]);
        store(&mut cache, b"a-spilled-word", &[1, 2, 3, 4]);
        assert_eq!(
            cache.lookup(b"a-spilled-word").hit(),
            Some(&[1, 2, 3, 4][..])
        );
    }

    /// The generation counter wraps back to zero after 2^32 evicts, where a
    #[test]
    fn the_evict_that_wraps_the_generation_clears_the_table() {
        let mut cache = WordCache::new(1 << 6);
        cache.spilled_generation = u32::MAX;
        store(&mut cache, b"short", &[1, 2]);
        store(&mut cache, b"a-spilled-word", &[1, 2, 3, 4]);
        store(&mut cache, b"the-filler-word", &vec![9; SPILLED_BUDGET]);
        assert_eq!(cache.lookup(b"short").hit(), None);
        assert_eq!(cache.lookup(b"a-spilled-word").hit(), None);
        assert_eq!(
            cache.lookup(b"the-filler-word").hit(),
            Some(&vec![9; SPILLED_BUDGET][..])
        );
    }

    /// Placements are built by hand so nothing but `insert` is on trial.
    #[test]
    fn the_spilled_buffer_never_outgrows_its_budget() {
        let mut cache = WordCache::new(1 << 6);
        let InsertPlacement { key, tag, .. } =
            make_lookup_key(b"a-word-past-fifteen-bytes", cache.placement_mask);
        for i in 0..3 * (SPILLED_BUDGET / 8) {
            let at = InsertPlacement { index: 0, key, tag };
            cache.insert(at, [i as u32; 8].into_iter());
        }
        assert!(
            cache.spilled_buffer.len() <= SPILLED_BUDGET,
            "the buffer reached {} of a {} budget",
            cache.spilled_buffer.len(),
            SPILLED_BUDGET
        );
    }

    /// Churn a table far too small for its input, mixing every key and slot shape,
    #[test]
    fn a_hit_never_returns_another_words_ids() {
        let mut cache = WordCache::new(64);
        let mut expected: Vec<(Vec<u8>, Vec<u32>)> = Vec::new();
        for i in 0..2000usize {
            let word = match i % 4 {
                0 => format!("w{i}"),
                1 => format!("a-long-word-past-fifteen-bytes-{i}"),
                2 => format!("k{i}xxxxxxxxxxxx"),
                _ => format!("{}-{i}", "z".repeat(i % 40)),
            }
            .into_bytes();
            let ids: Vec<u32> = (0..=(i % 9) as u32).map(|k| i as u32 * 16 + k).collect();
            store(&mut cache, &word, &ids);
            expected.push((word, ids));
        }

        let mut live = 0;
        for (word, ids) in &expected {
            if let Some(hit) = cache.lookup(word).hit() {
                assert_eq!(hit, &ids[..], "{word:?}");
                live += 1;
            }
        }
        assert!(live > 0, "everything was evicted, the test proves nothing");
    }
}
