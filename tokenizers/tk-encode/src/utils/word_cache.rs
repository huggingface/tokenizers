//! A table that remembers which token ids a word encodes to, so the tokenization
//! model (the expensive step) only encodes each word once.
//!
//! We determine the placement of a word in the cache table with a **hash** of the word:
//! - the bottom bits pick its **home slot**. A word can be cached in a 16-slot window around it
//! - the top byte is a **tag**, stored in a separate table ([`WordCache::quick_lookup`])
//!
//! A lookup walks a 16 byte window in [`WordCache::quick_lookup`] to find a matching tag, if the tag
//! matches the slot's 128 bit key ([`LookupKey`]) confirms whether it's a match or not.
//!
//! On miss, we return where the cache should insert ([`WordCache::insert`]) the ids;
//! Either the first empty (0x00) slot in the window or the home slot if the window is full.
//!
//! ```text
//!    lookup "hat":  tag A7, home slot 5
//!
//!    slot:         4     5     6     7
//!               ┌─────┬─────┬─────┬─────┬────
//!    tags       │ C4  │ A7  │ 31  │ A7  │ ...     one hash byte per slot
//!               └─────┴─────┴─────┴─────┴────
//!               ┌─────┬─────┬─────┬─────┬────
//!    slots      │"cat"│"the"│"sat"│"hat"│ ...     the key and the ids, 32 bytes
//!               └─────┴─────┴─────┴─────┴────
//!                        ▲           ▲
//!                        │           └ tag and key match: a hit, return the ids
//!                        └ same tag, wrong key: keep walking
//! ```
//!
//! A slot ([`WordCacheSlot`]) keeps up to three ids inline.
//! Longer encodings go to one shared buffer ([`WordCache::spilled_buffer`]) and the slot holds offsets in that buffer.
//!
//! # Note
//!
//! A cache hit for a word of 15 bytes or shorter is guaranteed to return correct ids.
//! For longer words, the cache hit relies on equality of 127 bits of hash of the word's bytes.
//! Two long words can in principle share the same 127 bit hash (a collision) which could make the
//! cache return incorrect ids for one of them, even though the collision is extremely unlikely.
//!
//! # Where the ideas come from
//!
//! - [Swiss Tables] is where the tag row comes from: one byte of hash per slot,
//!   checked before the slot itself is touched.
//! - [gigatoken] is a BPE tokenizer with a pre-token cache built from the same
//!   parts: `u128` packed keys, 32-byte self-contained slots, ids inline.
//! - [huggingface/tokenizers#2234] is an open-addressed cache for this same encode
//!   pipeline, arrived at in parallel.
//!
//! [Swiss Tables]: https://abseil.io/about/design/swisstables
//! [gigatoken]: https://github.com/marcelroed/gigatoken
//! [huggingface/tokenizers#2234]: https://github.com/huggingface/tokenizers/pull/2234

use std::{fmt::Debug, ops::Range};

use ahash::RandomState;
use std::iter::Iterator;
use wide::i8x16;

/// Hashes a word to the 64 bits its home slot and tag are taken from, and to the
/// bottom half of a long word's key ([`LookupKey::new_hash`]).
static PLACEMENT_HASHER: RandomState = RandomState::with_seeds(
    0x243f_6a88_85a3_08d3,
    0x1319_8a2e_0370_7344,
    0xa409_3822_299f_31d0,
    0x082e_fa98_ec4e_6c89,
);

/// Hashes a long word a second time, to fill the half of its key that
/// [`PLACEMENT_HASHER`] does not reach. The two hashes must be independent, or the
/// key would carry 64 bits of information instead of 127.
static DISCRIMINANT_HASHER: RandomState = RandomState::with_seeds(
    0x4528_21e6_38d0_1377,
    0xbe54_66cf_34e9_0c6c,
    0xc0ac_29b7_c97c_50dd,
    0x3f84_d5b5_b547_0917,
);

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
    /// 16 tags are 16 bytes: a whole window fits in one vector register (SIMD)
    /// and in one cache line, so scanning it for a hit can be a couple of instructions
    /// and a single memory read.
    const WINDOW_SIZE: usize = 16;

    pub fn new(num_slots: usize) -> Self {
        let next_pow2 = num_slots.next_power_of_two();
        if next_pow2 != num_slots {
            // todo: warn the user the capacity has been rounded up
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
    /// On [Lookup::Hit], returns the ids it encodes to.
    /// On [Lookup::Miss], returns the location in [Self::cached_words] where it should be inserted
    pub fn lookup(&'a self, word: &[u8]) -> Lookup<'a> {
        let InsertPlacement {
            key,
            index: home,
            tag,
        } = make_lookup_key(word, self.placement_mask);

        let tag_window = self.tag_window(home);
        let (candidates, first_empty) = tag_window.find_matches_and_first_empty(tag);

        for candidate in candidates {
            // Must validate that a candidate is indeed a match
            let slot = &self.cached_words[candidate];
            if slot.key == key {
                if slot.is_stale(self.spilled_generation) {
                    // The entry is stale: replace it with fresh ids
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

        // No match: it's a miss. The ids go in the window's first empty slot or to the home slot
        Lookup::Miss(InsertPlacement {
            index: first_empty.unwrap_or(home),
            key,
            tag,
        })
    }

    /// Insert a new (word, ids) pair in the cache
    ///
    /// The [InsertPlacement] comes from [`Lookup::Miss`]
    pub fn insert(&mut self, placement: InsertPlacement, ids: impl ExactSizeIterator<Item = u32>) {
        let len = ids.len();
        let InsertPlacement { index, key, tag } = placement;

        let word = if len <= 3 {
            WordCacheSlot::new_self_contained(key, ids)
        } else {
            if self.spilled_buffer.len() + len > self.spilled_budget {
                // Spilled buffer budget passed: we clear it
                self.spilled_buffer.clear();
                // Bump the generation to invalidate previous spilled slots
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
        // todo: log the cache clear
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

// 32-byte size and alignment, so a read is always contained in a cache line
const _: () = assert!(size_of::<WordCacheSlot>() == 32);
const _: () = assert!(align_of::<WordCacheSlot>() == 32);

impl WordCacheSlot {
    /// A sentinel value that discriminates an inline slot (ids are stored in the slot) from a spilled slot
    /// (values are stored in a buffer)
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
        assert!(ids.len() <= 3);
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
/// whether it's a spilled or self-contained slot
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
pub struct LookupKey(u128);

/// The key, home slot and tag of a word.
fn make_lookup_key(word: &[u8], placement_mask: u64) -> InsertPlacement {
    let placement_hash = PLACEMENT_HASHER.hash_one(word);
    let key = if word.len() <= 15 {
        LookupKey::new_inline(word)
    } else {
        LookupKey::new_hash(DISCRIMINANT_HASHER.hash_one(word), placement_hash)
    };
    InsertPlacement {
        key,
        index: (placement_hash & placement_mask) as usize,
        tag: ((placement_hash >> (64 - 8)) as u8).max(WordCache::EMPTY + 1),
        // ^ must be at least 0x01, otherwise can be mistaken for an EMPTY slot
    }
}

impl LookupKey {
    pub const TAG_MASK: u128 = 1 << 127;

    /// The key of a word of fifteen bytes or fewer: the word is its own key.
    pub fn new_inline(word: &[u8]) -> Self {
        let len = word.len();
        assert!(len <= 15);
        // yes, this is a bit weird :)
        //
        // We used to do this:
        // ```rust
        //  payload[..word.len()].copy_from_slice(word);
        //  payload[15] = word.len() as u8;
        //  Self(u128::from_le_bytes(payload))
        // ```
        // But that would compile into a memcpy call, probably because the len is only known at runtime.
        // memcpy turned out to be quite slow and inefficient.
        //
        // The head / tail with fixed size compiles into plain register loads which are way faster
        let raw = if len >= 8 {
            let head = u64::from_le_bytes(word[..8].try_into().unwrap()) as u128;
            let tail = u64::from_le_bytes(word[len - 8..].try_into().unwrap()) as u128;
            head | tail << (8 * (len - 8))
        } else if len >= 4 {
            let head = u32::from_le_bytes(word[..4].try_into().unwrap()) as u128;
            let tail = u32::from_le_bytes(word[len - 4..].try_into().unwrap()) as u128;
            head | tail << (8 * (len - 4))
        } else if len >= 1 {
            let first = word[0] as u128;
            let middle = (word[len / 2] as u128) << (8 * (len / 2));
            let last = (word[len - 1] as u128) << (8 * (len - 1));
            first | middle | last
        } else {
            0
        };
        Self(raw | (len as u128) << 120)
    }

    /// The key of a longer word: 127 bits of hash stand in for the word's bytes.
    pub fn new_hash(discriminant: u64, placement: u64) -> Self {
        Self(Self::TAG_MASK | (discriminant as u128) << 64 | placement as u128)
    }
}

impl std::fmt::Debug for LookupKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("LookupKey");
        if self.0 & Self::TAG_MASK == 0 {
            let bytes = self.0.to_le_bytes();
            let len = bytes[15] as usize;
            debug
                .field("type", &"inline")
                .field("word_len", &len)
                .field("word", &bytes[..len].escape_ascii().to_string());
        } else {
            debug
                .field("type", &"hashed")
                .field(
                    "discriminant",
                    &format!("{:#x}", ((self.0 & !Self::TAG_MASK) >> 64) as u64),
                )
                .field(
                    "placement",
                    &format!("{:#x}", (self.0 & u64::MAX as u128) as u64),
                );
        }
        debug.finish()
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

        // a trick to truncate matches to before the first empty
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
    /// need to pick where in the table their words land.
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
    /// table's end, which has its own test too.
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
    /// extra slots so a window starting on the last home does not run out of entries.
    /// Two words homed there force the second one into the extra slots; both have to
    /// round trip anyway.
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
    /// three-id inline boundary. The words past fifteen bytes are the ones whose
    /// stored key is a hash ([`LookupKey::new_hash`]); no shorter word exercises
    /// that path.
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
    /// ids. The first pair differs only in the top bit of the last byte, which any
    /// word ending in a multi-byte UTF-8 character has set; the second pair differs
    /// only in a trailing zero byte, so only the length tells them apart. And no
    /// inline key may carry the bit that marks a hashed one, whatever its bytes.
    #[test]
    fn packed_keys_are_unique_per_word() {
        let cache = WordCache::new(1 << 8);
        let key = |word: &[u8]| make_lookup_key(word, cache.placement_mask).key;
        assert_ne!(key(b"aaaaaaaaaaaaaa\x7f"), key(b"aaaaaaaaaaaaaa\xff"));
        assert_ne!(key(b"abcd"), key(b"abcd\0"));
        assert_eq!(key(b"aaaaaaaaaaaaaa\xff").0 & LookupKey::TAG_MASK, 0);
    }

    /// One window shape per row: the needle in various lanes, an empty slot in
    /// the middle, at the edges, or absent. Candidates past the first empty
    /// lane must not be reported (no entry can live there, see
    /// [`Window::tag_and_empty`]) and the empty lane itself is the placement.
    #[test]
    fn the_scan_reports_matches_before_the_first_empty_and_the_empty_itself() {
        let offset = 3;
        let cases: &[(&[u8], u16, Option<usize>)] = &[
            // needle at lanes 0 and 2, empty at 3: lane 5's match is out of reach
            (
                &[
                    0xA7, 0x31, 0xA7, 0x00, 0x5F, 0xA7, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                    0x18, 0x19, 0x1A,
                ],
                0b101,
                Some(3),
            ),
            // no empty slot: every match is reachable, no placement
            (
                &[
                    0xA7, 0x31, 0xA7, 0x22, 0x5F, 0xA7, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                    0x18, 0x19, 0xA7,
                ],
                0b1000000000100101,
                None,
            ),
            // empty in lane 0: nothing is reachable
            (
                &[
                    0x00, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7, 0xA7,
                    0xA7, 0xA7, 0xA7,
                ],
                0,
                Some(0),
            ),
            // every lane matches, empty in the last lane
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
    /// tag: the scan stops looking at the first 0x00 it sees, and an insert
    /// treats the slot as free space. Enough words that a tag built from a raw
    /// hash byte would land on 0x00 hundreds of times.
    #[test]
    fn a_live_tag_is_never_the_empty_marker() {
        for i in 0..100_000u32 {
            let placement = make_lookup_key(&i.to_le_bytes(), u64::MAX);
            assert_ne!(placement.tag, WordCache::EMPTY, "word {i}");
        }
    }

    /// Every inline length, with a different value in every byte position, so a
    /// packing that drops, duplicates or misplaces a byte fails. The reference is
    /// the construction the packing must be equivalent to: the bytes copied into
    /// a zeroed array, the length written in the top byte.
    #[test]
    fn an_inline_key_is_the_words_bytes_with_the_length_on_top() {
        for len in 0..=15usize {
            let word: Vec<u8> = (1..=len as u8).collect();
            let mut padded = [0u8; 16];
            padded[..len].copy_from_slice(&word);
            padded[15] = len as u8;
            assert_eq!(
                LookupKey::new_inline(&word),
                LookupKey(u128::from_le_bytes(padded)),
                "len={len}"
            );
        }
    }

    /// A nonzero start, since every spill after the first has one and offsets that
    /// only round trip from zero would still pass.
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
    /// of a word it does not hold. Too rare to hit by chance here, so the collision
    /// is forged. The lookup must confirm the key and keep walking, not trust the
    /// tag.
    #[test]
    fn a_tag_collision_is_confirmed_against_the_key() {
        let mut cache = WordCache::new(1 << 8);
        let InsertPlacement { index, tag, .. } = make_lookup_key(b"beta", cache.placement_mask);
        assert_ne!(tag, WordCache::EMPTY, "pick a word with a nonzero tag");
        cache.quick_lookup[index] = tag;
        cache.cached_words[index] =
            WordCacheSlot::new_self_contained(LookupKey::new_inline(b"decoy"), [7].into_iter());

        assert_eq!(cache.lookup(b"beta").hit(), None);
        store(&mut cache, b"beta", &[2]);
        assert_eq!(cache.lookup(b"beta").hit(), Some(&[2][..]));
    }

    /// A word whose whole window is taken is still cached: it evicts its home slot.
    /// The other fifteen words in the window keep their ids.
    #[test]
    fn a_full_window_evicts_only_the_home_slot() {
        // A one-home table: placement_mask is 0, so every word homes at slot 0 and
        // the sixteen slots from there are one shared window.
        let mut cache = WordCache::new(1);
        let words = words_homed_in(&cache, 0..1, WordCache::WINDOW_SIZE);
        for (i, word) in words.iter().enumerate() {
            store(&mut cache, word, &[i as u32]);
        }

        store(&mut cache, b"newcomer", &[999]);
        assert_eq!(cache.lookup(b"newcomer").hit(), Some(&[999][..]));
        // Inserts fill the window in order, so the home slot holds the first word.
        assert_eq!(cache.lookup(&words[0]).hit(), None);
        for (i, word) in words.iter().enumerate().skip(1) {
            assert_eq!(cache.lookup(word).hit(), Some(&[i as u32][..]), "{word:?}");
        }
    }

    /// The filler word brings a whole budget of ids on its own, so caching it
    /// evicts whatever the buffer holds. The evicted word's slot keeps its tag
    /// and key, but its ids are gone from the buffer: looking the word up
    /// again has to be a miss, not a hit on the filler's ids.
    #[test]
    fn a_spilled_word_misses_after_an_evict() {
        let mut cache = WordCache::new(1 << 6);
        store(&mut cache, b"a-spilled-word", &[1, 2, 3, 4]);
        store(&mut cache, b"the-filler-word", &vec![9; SPILLED_BUDGET]);
        assert_eq!(cache.lookup(b"a-spilled-word").hit(), None);
    }

    /// An evict drops a word's ids but leaves its slot behind. The miss the
    /// word's lookup then returns must be a placement `insert` accepts, and
    /// the word must round trip through it. New ids for the second insertion,
    /// so a hit on the first ones cannot pass.
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

    /// A self-contained word keeps its ids in the slot, not in the buffer, so
    /// an evict must not cost it its hit. Two filler words, because the first
    /// fills an empty buffer exactly to the budget; the second overflows it.
    #[test]
    fn a_self_contained_word_still_hits_after_an_evict() {
        let mut cache = WordCache::new(1 << 6);
        store(&mut cache, b"short", &[1, 2]);
        store(&mut cache, b"the-filler-word", &vec![9; SPILLED_BUDGET]);
        store(&mut cache, b"one-more-word", &[3, 4, 5, 6]);
        assert_eq!(cache.lookup(b"short").hit(), Some(&[1, 2][..]));
    }

    /// The evict happens inside the insert that caches this word: the slot
    /// must carry the generation the evict moved to, not the one the insert
    /// started with, or the word would be stale the moment it is cached.
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
    /// slot stamped long ago would look fresh again and hit on another word's
    /// ids. The wrap therefore clears the whole table, self-contained slots
    /// included; only the word the wrapping insert caches survives it.
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
    /// and check the one hard promise: the cache may forget a word, it must never
    /// answer with another word's ids.
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
            // No two words share an id (k stays below 16), so a wrong hit cannot
            // return the right ids by luck.
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
