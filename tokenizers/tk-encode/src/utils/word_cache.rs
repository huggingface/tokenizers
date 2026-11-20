//! A table that remembers which token ids a word encodes to, so the tokenization
//! model (the expensive step) only encodes each word once.
//!
//! A word is looked up by its [`Signature`]: its length, first eight bytes and last eight bytes.
//! A hash of the signature places the word in the table:
//! - the bottom bits pick its **home slot**. A word can be cached in the 16-slot window that starts there
//! - the top byte is a **tag**, stored in a separate table ([`WordCache::quick_lookup`])
//!
//! A lookup walks a 16 byte window in [`WordCache::quick_lookup`] to find a matching tag.
//! When a tag matches, the slot's signature confirms whether it's a hit or not.
//!
//! On miss, we return where the cache should insert ([`WordCache::insert`]) the ids;
//! Either the slot already holding a stale copy of the word ([`WordCacheSlot::is_stale`]),
//! the first empty (0x00) slot in the window, or the home slot if the window is full.
//!
//! ```text
//!    lookup "hat":  tag A7, home slot 5
//!
//!    slot:         4     5     6     7
//!               ┌─────┬─────┬─────┬─────┬────
//!    tags       │ C4  │ A7  │ 31  │ A7  │ ...     one hash byte per slot
//!               └─────┴─────┴─────┴─────┴────
//!               ┌─────┬─────┬─────┬─────┬────
//!    slots      │"cat"│"the"│"sat"│"hat"│ ...     the signature and the ids, 32 bytes
//!               └─────┴─────┴─────┴─────┴────
//!                        ▲           ▲
//!                        │           └ tag and signature match: a hit, return the ids
//!                        └ same tag, wrong signature: keep walking
//! ```
//!
//! A slot ([`WordCacheSlot`]) keeps up to three ids inline.
//! Longer encodings go to one shared buffer ([`WordCache::spilled_buffer`]) and the slot holds offsets in that buffer.
//!
//! # A hit is always the word that was stored
//!
//! A signature holds a word of [`Signature::BYTES`] bytes or less whole, so a matching slot is the same word.
//! A longer word also stores the bytes its signature leaves out, in [`WordCache::spilled_buffer`] after its ids.
//! A lookup compares them before it reports a hit.
//!
//! # Where the ideas come from
//!
//! - [Swiss Tables] is where the tag row comes from: one byte of hash per slot,
//!   checked before the slot itself is touched.
//! - [gigatoken] is a BPE tokenizer with a pre-token cache built from the same
//!   parts: packed keys, 32-byte self-contained slots, ids inline.
//! - [huggingface/tokenizers#2234] is an open-addressed cache for this same encode
//!   pipeline, arrived at in parallel.
//!
//! [Swiss Tables]: https://abseil.io/about/design/swisstables
//! [gigatoken]: https://github.com/marcelroed/gigatoken
//! [huggingface/tokenizers#2234]: https://github.com/huggingface/tokenizers/pull/2234

use std::iter::Iterator;

use prefetch_index::prefetch_index;
use wide::i8x16;

use crate::utils::signature::Signature;

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
    /// Holds what does not fit in a [`WordCacheSlot`]: more than three ids, and the bytes of a
    /// long word that its signature leaves out, packed four to a `u32` right after its ids.
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
            warn!(
                "WordCache: rounding cache_capacity up from {num_slots} to {next_pow2} slots \
                 (a power of two is required for slot masking); \
                 behavior is unchanged, the table just uses slightly more memory"
            );
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
    #[inline]
    // Only `unigram` and `wordpiece` call this. `bpe` holds the signature already.
    #[cfg_attr(not(any(feature = "unigram", feature = "wordpiece")), allow(dead_code))]
    pub fn lookup(&'a self, word: &[u8]) -> Lookup<'a> {
        self.lookup_signed(word, Signature::new(word))
    }

    /// Starts loading the home slot of the word with this signature, so that a lookup a few
    /// words later finds it in the CPU cache.
    #[inline(always)]
    pub fn prefetch(&self, signature: Signature) {
        prefetch_index(&*self.cached_words, self.home(signature));
    }

    /// Answers the leading words of a batch straight from their home slots, into the caller's
    /// buffer. Returns how many words it answered and how many ids it wrote.
    ///
    /// It stops at the first word whose home slot does not hold it with inline ids. That word
    /// goes through [`Self::lookup_signed`]. On text the cache has seen, nearly every word is
    /// answered here, so this loop is kept to a few loads, compares and stores per word.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writes of [`MAX_INLINE_IDS`] `u32`s per signature. A hit writes
    /// every inline lane, whatever its number of ids.
    // Out of line: inside the encode loop, the table's address and mask were reloaded per word.
    #[inline(never)]
    pub unsafe fn emit_inline_hits(
        &self,
        signatures: &[Signature],
        dst: *mut u32,
    ) -> (usize, usize) {
        let slots = self.cached_words.as_ptr();
        let mut next = dst;
        let mut answered = 0;
        for &signature in signatures {
            // SAFETY: a home is masked with `placement_mask` (`next_pow2 - 1`), and the table
            // holds `next_pow2 + WINDOW_SIZE` slots.
            let slot = unsafe { *slots.add(self.home(signature)) };
            if !slot.holds_inline(signature) {
                break;
            }
            // SAFETY: `next` is at most `MAX_INLINE_IDS * answered` past `dst`, and the caller
            // left room for `MAX_INLINE_IDS` more.
            unsafe {
                next.copy_from_nonoverlapping(slot.payload.as_ptr(), MAX_INLINE_IDS);
                next = next.add(slot.ids_len());
            }
            answered += 1;
        }
        // SAFETY: `next` was derived from `dst` and only moved forward.
        (answered, unsafe { next.offset_from_unsigned(dst) })
    }

    /// [`Self::lookup`] for a caller that already holds the word's signature.
    pub fn lookup_signed(&'a self, word: &[u8], signature: Signature) -> Lookup<'a> {
        let home = self.home(signature);
        let tag = tag_of(signature);
        let tag_window = self.tag_window(home);
        let (candidates, first_empty) = tag_window.find_matches_and_first_empty(tag);

        for candidate in candidates {
            let slot = &self.cached_words[candidate];
            if !slot.has_signature(signature) {
                continue;
            }
            if slot.is_stale(self.spilled_generation) {
                // The entry is stale: replace it with fresh ids
                return Lookup::Miss(InsertPlacement {
                    index: candidate,
                    signature,
                    tag,
                });
            }
            if self.holds_uncovered(slot, word) {
                return Lookup::Hit(self.get_cached_ids(slot));
            }
        }

        // No match: it's a miss. The ids go in the window's first empty slot or to the home slot
        Lookup::Miss(InsertPlacement {
            index: first_empty.unwrap_or(home),
            signature,
            tag,
        })
    }

    /// Insert a new (word, ids) pair in the cache
    ///
    /// The [InsertPlacement] comes from a [`Lookup::Miss`] of the same `word`.
    pub fn insert(
        &mut self,
        placement: InsertPlacement,
        word: &[u8],
        ids: impl ExactSizeIterator<Item = u32>,
    ) {
        let InsertPlacement {
            mut index,
            signature,
            tag,
        } = placement;
        // A slot has 24 bits for the word's length, so a longer word is not cached.
        if signature.len > WordCacheSlot::MAX_WORD_BYTES {
            return;
        }
        let uncovered = Signature::uncovered(word);
        let n_ids = ids.len();

        let slot = if n_ids <= MAX_INLINE_IDS && uncovered.is_empty() {
            WordCacheSlot::new_self_contained(signature, ids)
        } else {
            let needed = n_ids + uncovered.len().div_ceil(4);
            if self.spilled_buffer.len() + needed > self.spilled_budget {
                debug!(
                    "WordCache: spilled-ids budget ({} ids) exhausted; evicting {} spilled ids",
                    self.spilled_budget,
                    self.spilled_buffer.len()
                );
                self.spilled_buffer.clear();
                // Bump the generation to invalidate previous spilled slots
                self.spilled_generation = self.spilled_generation.wrapping_add(1);
                if self.spilled_generation == 0 {
                    self.reset();
                    // The placement was chosen in the table as it was. In the empty one, a slot
                    // past the home slot sits behind an empty slot, where no lookup reaches it.
                    index = self.home(signature);
                }
            }
            let start = self.spilled_buffer.len();
            self.spilled_buffer.extend(ids);
            self.spilled_buffer.extend(pack_bytes(uncovered));
            WordCacheSlot::new_spilled(signature, (start, start + n_ids), self.spilled_generation)
        };

        self.cached_words[index] = slot;
        self.quick_lookup[index] = tag;
    }
}

impl<'a> WordCache {
    #[inline(always)]
    fn home(&self, signature: Signature) -> usize {
        (signature.hash & self.placement_mask) as usize
    }

    /// Whether `slot`, whose signature is `word`'s, also holds the bytes of `word` that the
    /// signature leaves out.
    fn holds_uncovered(&self, slot: &WordCacheSlot, word: &[u8]) -> bool {
        let uncovered = Signature::uncovered(word);
        if uncovered.is_empty() {
            return true;
        }
        // `insert` spills every word longer than its signature, bytes right after the ids.
        let (_, ids_end) = Spilled(slot).ids_offsets();
        let stored: &[u8] = bytemuck::cast_slice(&self.spilled_buffer[ids_end..]);
        stored[..uncovered.len()] == *uncovered
    }

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
        info!(
            "WordCache: eviction generation wrapped past u32::MAX; \
             cleared all {} cached words to avoid stale hits \
             (lookups will miss until the cache refills)",
            self.cached_words.len()
        );
        self.spilled_buffer.clear();
        self.quick_lookup = vec![0; self.quick_lookup.len()].into_boxed_slice();
        self.cached_words =
            vec![WordCacheSlot::default(); self.cached_words.len()].into_boxed_slice();
    }

    fn tag_window(&self, home_index: usize) -> Window {
        Window::new(&self.quick_lookup, home_index)
    }
}

#[inline(always)]
fn tag_of(signature: Signature) -> u8 {
    // At least 0x01, otherwise it can be mistaken for an EMPTY slot
    ((signature.hash >> (64 - 8)) as u8).max(WordCache::EMPTY + 1)
}

/// Packs bytes four to a `u32`, zero padding the last one. The `u32`s read back as the same
/// bytes in memory order.
fn pack_bytes(bytes: &[u8]) -> impl Iterator<Item = u32> + '_ {
    let (quads, rest) = bytes.as_chunks::<4>();
    let last = (!rest.is_empty()).then(|| {
        let mut padded = [0u8; 4];
        padded[..rest.len()].copy_from_slice(rest);
        u32::from_ne_bytes(padded)
    });
    quads
        .iter()
        .map(|&quad| u32::from_ne_bytes(quad))
        .chain(last)
}

/// A cache slot, packed in 32 bytes.
/// There are two variants:
///
/// ## inline: 3 or fewer ids to cache, for a word of [`Signature::BYTES`] bytes or less
///
/// The token ids are encoded inline in the cache slot:
/// ```text
/// ┌───────────┬───────────┬──────────┬──────────┬──────────┬──────────────────────────┐
/// │   head    │   tail    │  id[0]   │  id[1]   │  id[2]   │           meta           │
/// │    u64    │    u64    │   u32    │   u32    │   u32    │ word_len << 8 | ids_len  │
/// └───────────┴───────────┴──────────┴──────────┴──────────┴──────────────────────────┘
/// 0           8           16         20         24         28                       32 bytes
/// ```
///
/// ## spilled: every other word
///
/// The token ids are stored in [`WordCache::spilled_buffer`], followed by the bytes of the word
/// that its signature leaves out. The slot holds the offsets of the ids in that buffer.
/// ```text
/// ┌───────────┬───────────┬──────────┬──────────┬────────────┬──────────────────────────┐
/// │   head    │   tail    │  start   │   end    │ generation │           meta           │
/// │    u64    │    u64    │   u32    │   u32    │    u32     │ word_len << 8 | SPILLED  │
/// └───────────┴───────────┴──────────┴──────────┴────────────┴──────────────────────────┘
/// 0           8           16         20         24           28                       32 bytes
/// ```
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WordCacheSlot {
    head: u64,
    tail: u64,
    payload: [u32; 3],
    /// The word's length in the top 24 bits. The number of inline ids in the low 8, or
    /// [`Self::SPILLED`].
    meta: u32,
}

const _: () = assert!(
    (MAX_INLINE_IDS + 1).is_power_of_two(),
    "WordCacheSlot::INLINE_IDS is a bit mask: it needs MAX_INLINE_IDS to be all ones"
);
// 32-byte size and alignment, so a read is always contained in a cache line
const _: () = assert!(size_of::<WordCacheSlot>() == 32);
const _: () = assert!(align_of::<WordCacheSlot>() == 32);

impl WordCacheSlot {
    /// A sentinel value that discriminates an inline slot (ids are stored in the slot) from a spilled slot
    /// (values are stored in a buffer)
    const SPILLED: u32 = 0xFF;

    /// The longest word whose length fits in `meta`.
    const MAX_WORD_BYTES: usize = (1 << 24) - 1;

    /// The bits of `meta` that hold up to [`MAX_INLINE_IDS`]: an inline hit ignores them.
    const INLINE_IDS: u64 = MAX_INLINE_IDS as u64;

    fn new(signature: Signature, payload: [u32; 3], ids_len: u32) -> Self {
        Self {
            head: signature.head,
            tail: signature.tail,
            payload,
            meta: (signature.len as u32) << 8 | ids_len,
        }
    }

    pub fn new_spilled(signature: Signature, ids_offsets: (usize, usize), generation: u32) -> Self {
        let (start, end) = ids_offsets;
        Self::new(
            signature,
            [start as u32, end as u32, generation],
            Self::SPILLED,
        )
    }

    pub fn new_self_contained(
        signature: Signature,
        ids: impl ExactSizeIterator<Item = u32>,
    ) -> Self {
        assert!(ids.len() <= MAX_INLINE_IDS);
        let ids_len = ids.len() as u32;
        let mut payload = [0u32; 3];
        for (slot, id) in payload.iter_mut().zip(ids) {
            *slot = id;
        }
        Self::new(signature, payload, ids_len)
    }

    pub fn is_stale(&self, current_generation: u32) -> bool {
        match self.variant() {
            CacheSlotType::SelfContained(_) => false,
            CacheSlotType::Spilled(spilled) => spilled.generation() != current_generation,
        }
    }
}

impl WordCacheSlot {
    // Both compare lengths as `u64`: a word too long for `meta` then matches no slot.
    #[inline(always)]
    fn has_signature(&self, signature: Signature) -> bool {
        self.head == signature.head
            && self.tail == signature.tail
            && u64::from(self.meta >> 8) == signature.len as u64
    }

    /// [`Self::has_signature`] with the ids inline, in one compare of `meta`.
    /// Such a slot never holds a word longer than its signature: [`WordCache::insert`] spills it.
    #[inline(always)]
    fn holds_inline(&self, signature: Signature) -> bool {
        self.head == signature.head
            && self.tail == signature.tail
            && u64::from(self.meta) & !Self::INLINE_IDS == (signature.len as u64) << 8
    }

    #[inline(always)]
    fn ids_len(&self) -> usize {
        (self.meta & 0xFF) as usize
    }

    fn is_spilled(self) -> bool {
        self.meta & 0xFF == Self::SPILLED
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
        &self.0.payload[..self.0.ids_len()]
    }
}

pub struct InsertPlacement {
    index: usize,
    signature: Signature,
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
            cache.insert(at, word, ids.iter().copied());
        }
    }

    /// The first `n` probe words whose home slot falls in `range`, for tests that
    /// need to pick where in the table their words land.
    fn words_homed_in(cache: &WordCache, range: std::ops::Range<usize>, n: usize) -> Vec<Vec<u8>> {
        (0u32..)
            .map(|i| format!("w{i}").into_bytes())
            .filter(|word| range.contains(&cache.home(Signature::new(word))))
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

    /// Both sides of the [`Signature::BYTES`] boundary, crossed with both sides of the
    /// three-id inline boundary. The words past sixteen bytes are the ones that store
    /// bytes in the buffer; no shorter word exercises that path.
    #[test]
    fn every_key_and_slot_shape_round_trips() {
        let mut cache = WordCache::new(1 << 8);
        let cases: [(&[u8], Vec<u32>); 4] = [
            (b"short", vec![1, 2]),
            (b"short-spilled", (0..10).collect()),
            (b"a-word-past-sixteen-bytes", vec![3]),
            (b"another-word-past-sixteen-bytes", (10..30).collect()),
        ];
        for (word, ids) in &cases {
            store(&mut cache, word, ids);
        }
        for (word, ids) in &cases {
            assert_eq!(cache.lookup(word).hit(), Some(&ids[..]), "{word:?}");
        }
    }

    #[test]
    fn a_very_long_word_round_trips() {
        let mut cache = WordCache::new(1 << 8);
        let word = vec![b'q'; 8192];
        let ids: Vec<u32> = (0..2000).collect();
        store(&mut cache, &word, &ids);
        assert_eq!(cache.lookup(&word).hit(), Some(&ids[..]));
    }

    /// A slot has 24 bits for the word's length. Two words past that limit would look alike
    /// to a slot that truncated their lengths, so neither is cached.
    #[test]
    fn a_word_too_long_for_a_slot_is_never_cached() {
        let mut cache = WordCache::new(1 << 8);
        for len in [1 << 24, (1 << 24) + 5] {
            let word = vec![b'q'; len];
            store(&mut cache, &word, &[1, 2, 3, 4]);
            assert_eq!(cache.lookup(&word).hit(), None, "len {len}");
        }
    }

    /// [`WordCache::emit_inline_hits`] is a shortcut for [`WordCache::lookup`]. It must write the
    /// ids a lookup returns for each word it answers, and stop at the first word whose home slot
    /// does not hold inline ids: a word with four ids ends the first run, a word that was never
    /// stored ends the second.
    #[test]
    fn a_run_of_inline_hits_writes_what_lookup_returns() {
        let mut cache = WordCache::new(1 << 8);
        let stored: [(&[u8], &[u32]); 5] = [
            (b"the", &[1]),
            (b"sixteen-byte-wd?", &[2, 3, 4]),
            (b"a", &[5, 6]),
            (b"spilled", &[7, 8, 9, 10]),
            (b"after", &[11]),
        ];
        for (word, ids) in stored {
            store(&mut cache, word, ids);
        }
        let words: Vec<&[u8]> = stored
            .iter()
            .map(|&(word, _)| word)
            .chain([&b"never-stored"[..]])
            .collect();
        let signatures: Vec<Signature> = words.iter().map(|word| Signature::new(word)).collect();
        let homes: std::collections::HashSet<usize> = signatures
            .iter()
            .map(|&signature| cache.home(signature))
            .collect();
        assert_eq!(
            homes.len(),
            words.len(),
            "pick words with distinct home slots"
        );

        let mut next = 0;
        let mut answers = Vec::new();
        while next < words.len() {
            let run = &signatures[next..];
            let mut out = vec![0u32; MAX_INLINE_IDS * run.len()];
            // SAFETY: `out` holds `MAX_INLINE_IDS` ids per signature of the run.
            let (answered, written) = unsafe { cache.emit_inline_hits(run, out.as_mut_ptr()) };
            let mut ids = out[..written].iter();
            for word in &words[next..next + answered] {
                let expected = cache
                    .lookup(word)
                    .hit()
                    .expect("the run answered this word");
                let got: Vec<u32> = ids.by_ref().take(expected.len()).copied().collect();
                assert_eq!(got, expected, "{word:?}");
            }
            answers.push(answered);
            next += answered + 1;
        }
        assert_eq!(answers, [3, 1]);
    }

    /// A one-home table sends every word to slot 0. The run must not answer a word that has only
    /// part of the stored word's signature: ten `a`s against nine (another length), against a
    /// last byte changed (another tail), and against a first byte changed (another head).
    #[test]
    fn a_run_does_not_answer_a_word_with_part_of_the_signature() {
        let mut cache = WordCache::new(1);
        store(&mut cache, b"aaaaaaaaaa", &[1]);
        let answered = |word: &[u8]| {
            let mut out = [0u32; MAX_INLINE_IDS];
            // SAFETY: `out` holds `MAX_INLINE_IDS` ids for the one signature.
            unsafe {
                cache
                    .emit_inline_hits(&[Signature::new(word)], out.as_mut_ptr())
                    .0
            }
        };
        assert_eq!(answered(b"aaaaaaaaaa"), 1);
        for near in [&b"aaaaaaaaa"[..], b"aaaaaaaaab", b"baaaaaaaaa"] {
            assert_eq!(answered(near), 0, "{near:?}");
        }
    }

    /// Two words past [`Signature::BYTES`] with the same length, start and end have the same
    /// signature, home slot and tag. A lookup that stopped at the signature would hand one
    /// word the other's ids. One id each, so the words are spilled only for their bytes.
    #[test]
    fn words_with_one_signature_keep_their_own_ids() {
        let mut cache = WordCache::new(1 << 8);
        let (first, second) = (b"01234567-first--89abcdef", b"01234567-second-89abcdef");
        assert_eq!(Signature::new(first), Signature::new(second));

        store(&mut cache, first, &[1]);
        assert_eq!(cache.lookup(second).hit(), None);
        store(&mut cache, second, &[2]);
        assert_eq!(cache.lookup(first).hit(), Some(&[1][..]));
        assert_eq!(cache.lookup(second).hit(), Some(&[2][..]));
    }

    /// One window shape per row: the needle in various lanes, an empty slot in
    /// the middle, at the edges, or absent. Candidates past the first empty
    /// lane must not be reported (no entry can live there, since inserts always
    /// fill the first empty lane) and the empty lane itself is the placement.
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
            let tag = tag_of(Signature::new(&i.to_le_bytes()));
            assert_ne!(tag, WordCache::EMPTY, "word {i}");
        }
    }

    /// A nonzero start, since every spill after the first has one and offsets that
    /// only round trip from zero would still pass.
    #[test]
    fn a_spilled_words_offsets_round_trip() {
        let cached = WordCacheSlot::new_spilled(Signature::default(), (5, 9), 0);
        match cached.variant() {
            CacheSlotType::Spilled(word) => assert_eq!(word.ids_offsets(), (5, 9)),
            CacheSlotType::SelfContained(_) => {
                panic!("built as spilled, read back as self-contained")
            }
        }
    }

    #[test]
    fn a_word_with_one_id_is_stored_self_contained() {
        let cached = WordCacheSlot::new_self_contained(Signature::default(), [42].into_iter());
        match cached.variant() {
            CacheSlotType::SelfContained(word) => assert_eq!(word.ids(), &[42]),
            CacheSlotType::Spilled(_) => panic!("one id fits without spilling"),
        }
    }

    /// A tag is one byte of hash, so about one occupied slot in 256 carries the tag
    /// of a word it does not hold. Too rare to hit by chance here, so the collision
    /// is forged. The lookup must confirm the signature and keep walking, not trust
    /// the tag.
    #[test]
    fn a_tag_collision_is_confirmed_against_the_signature() {
        let mut cache = WordCache::new(1 << 8);
        let signature = Signature::new(b"beta");
        let (index, tag) = (cache.home(signature), tag_of(signature));
        cache.quick_lookup[index] = tag;
        cache.cached_words[index] =
            WordCacheSlot::new_self_contained(Signature::new(b"decoy"), [7].into_iter());

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
    /// and signature, but its ids are gone from the buffer: looking the word up
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

    /// An evict leaves the word's slot behind, signature and tag intact. The miss the
    /// word's lookup then returns must place the fresh ids back into that slot,
    /// not into an empty one: the window stays clean of stale copies, which is
    /// what lets the walk stop at the first match.
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
    /// slot. A placement that preferred an empty slot would leave one more
    /// stale copy behind per cycle.
    #[test]
    fn an_evicted_word_never_occupies_two_slots() {
        let mut cache = WordCache::new(1 << 6);
        for round in 0..5u32 {
            store(&mut cache, b"a-spilled-word", &[round; 4]);
            store(&mut cache, b"the-filler-word", &vec![9; SPILLED_BUDGET]);
        }
        let signature = Signature::new(b"a-spilled-word");
        let copies = cache
            .cached_words
            .iter()
            .filter(|slot| slot.has_signature(signature))
            .count();
        assert_eq!(copies, 1);
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
        let word = b"a-word-past-sixteen-bytes";
        let signature = Signature::new(word);
        let tag = tag_of(signature);
        for i in 0..3 * (SPILLED_BUDGET / 8) {
            let at = InsertPlacement {
                index: 0,
                signature,
                tag,
            };
            cache.insert(at, word, [i as u32; 8].into_iter());
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
                1 => format!("a-long-word-past-sixteen-bytes-{i}"),
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
