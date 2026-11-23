use prefetch_index::prefetch_index;

use super::ids::{CachedIds, MAX_RUN_IDS};
use super::slot::{CacheSlotType, MAX_INLINE_IDS, Spilled, WordCacheSlot};
use super::tags::{Window, tag_of};
use crate::utils::signature::Signature;
use crate::utils::uncovered::Uncovered;

/// A table mapping words (`[u8]`) to the token ids they encode to (`[u32]`)
pub struct WordCache {
    /// The cache slots as a contiguous table. See [`WordCacheSlot`] for more details.
    pub(super) cached_words: Box<[WordCacheSlot]>,
    /// A quick lookup table to find candidates in [`Self::cached_words`] quickly
    pub(super) quick_lookup: Box<[u8]>,

    pub(super) placement_mask: u64,

    /// Controls eviction mechanism for spilled_buffer
    pub(super) spilled_generation: u32,
    /// The maximum capacity, in number of elements, allowed for spilled_buffer
    pub(super) spilled_budget: usize,
    /// Holds what does not fit in a [`WordCacheSlot`]: more than three ids, and the
    /// [`Uncovered`] bytes of a long word, packed four to a `u32` right after its ids.
    ///
    /// It ends with [`MAX_RUN_IDS`] zeros, so that many `u32` can be read from the start of
    /// any word's ids.
    pub(super) spilled_buffer: Vec<u32>,
}

impl<'a> WordCache {
    pub(super) const EMPTY: u8 = 0;

    /// How many slots a word's window spans, starting at its home slot.
    /// 16 tags are 16 bytes: a whole window fits in one vector register (SIMD)
    /// and in one cache line, so scanning it for a hit can be a couple of instructions
    /// and a single memory read.
    pub(super) const WINDOW_SIZE: usize = 16;

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
        // 16 per slot held the ids alone. A word longer than its signature now keeps its other
        // bytes here too, about half as much again on Chinese and Japanese text.
        let spilled_budget = 24 * n;
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
        self.lookup_with_signature(word, Signature::new(word))
    }

    /// Starts loading the home slot of the word with this signature, so that a lookup a few
    /// words later finds it in the CPU cache.
    #[inline(always)]
    pub fn prefetch(&self, signature: Signature) {
        prefetch_index(&*self.cached_words, self.home(signature));
    }

    /// [`Self::lookup`] for a caller that already holds the word's signature.
    // Forced: out of line, each call returned its 56-byte `Lookup` through memory.
    #[inline(always)]
    pub fn lookup_with_signature(&'a self, word: &[u8], signature: Signature) -> Lookup<'a> {
        let home = self.home(signature);
        let tag = tag_of(signature);
        // The home slot answers most lookups on its own, and a caller that prefetched it or
        // just read it has it at hand. The scan below costs a load from the tags and a walk.
        if self.cached_words[home].is_vacant() {
            // The scan stops at the first empty slot, so this is where it would end.
            return Lookup::Miss(InsertPlacement {
                index: home,
                signature,
                tag,
            });
        }
        if let Some(found) = self.examine(home, word, signature, tag) {
            return found;
        }

        let tag_window = self.tag_window(home);
        let (candidates, first_empty) = tag_window.find_matches_and_first_empty(tag);
        for candidate in candidates {
            if let Some(found) = self.examine(candidate, word, signature, tag) {
                return found;
            }
        }

        // No match: it's a miss. The ids go in the window's first empty slot or to the home slot
        Lookup::Miss(InsertPlacement {
            index: first_empty.unwrap_or(home),
            signature,
            tag,
        })
    }

    /// What the slot at `index` says about the word: a hit, a stale copy of it to replace, or
    /// nothing when the slot holds another word.
    #[inline(always)]
    fn examine(
        &'a self,
        index: usize,
        word: &[u8],
        signature: Signature,
        tag: u8,
    ) -> Option<Lookup<'a>> {
        let slot = &self.cached_words[index];
        if !slot.has_signature(signature) {
            return None;
        }
        if slot.is_stale(self.spilled_generation) {
            return Some(Lookup::Miss(InsertPlacement {
                index,
                signature,
                tag,
            }));
        }
        self.holds_uncovered(slot, word)
            .then(|| Lookup::Hit(self.get_cached_ids(slot)))
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
        // A slot has 24 bits for the word's length, so a longer word is not cached. Nor is the
        // empty word: with no ids, its slot would be all zeros, which is a vacant slot.
        if signature.len == 0 || signature.len > WordCacheSlot::MAX_WORD_BYTES {
            return;
        }
        let uncovered = Uncovered::of(word).map_or(&[][..], Uncovered::as_bytes);
        let n_ids = ids.len();

        let slot = if n_ids <= MAX_INLINE_IDS && uncovered.is_empty() {
            WordCacheSlot::new_self_contained(signature, ids)
        } else {
            // The zeros at the end come off, and go back on after this word.
            let used = self.spilled_buffer.len().saturating_sub(MAX_RUN_IDS);
            self.spilled_buffer.truncate(used);
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
            // The bytes go in as bytes, zero padded to whole `u32`s: one copy, where packing
            // them four at a time was a loop per word.
            let bytes_start = self.spilled_buffer.len();
            self.spilled_buffer
                .resize(bytes_start + uncovered.len().div_ceil(4), 0);
            let stored: &mut [u8] =
                bytemuck::cast_slice_mut(&mut self.spilled_buffer[bytes_start..]);
            stored[..uncovered.len()].copy_from_slice(uncovered);
            self.spilled_buffer
                .resize(self.spilled_buffer.len() + MAX_RUN_IDS, 0);
            WordCacheSlot::new_spilled(signature, (start, start + n_ids), self.spilled_generation)
        };

        self.cached_words[index] = slot;
        self.quick_lookup[index] = tag;
    }
}

impl<'a> WordCache {
    #[inline(always)]
    pub(super) fn home(&self, signature: Signature) -> usize {
        (signature.hash & self.placement_mask) as usize
    }

    /// Whether `slot`, whose signature is `word`'s, also holds its [`Uncovered`] bytes.
    #[inline(always)]
    pub(super) fn holds_uncovered(&self, slot: &WordCacheSlot, word: &[u8]) -> bool {
        let Some(uncovered) = Uncovered::of(word) else {
            return true;
        };
        // `insert` spills every word longer than its signature, bytes right after the ids.
        let (_, ids_end) = Spilled(slot).ids_offsets();
        uncovered.is_prefix_of(bytemuck::cast_slice(&self.spilled_buffer[ids_end..]))
    }

    /// Helper to fetch the ids from the word
    #[inline(always)]
    pub(super) fn get_cached_ids(&'a self, word: &'a WordCacheSlot) -> CachedIds<'a> {
        match word.variant() {
            CacheSlotType::SelfContained(word) => word.ids(),
            CacheSlotType::Spilled(word) => {
                let (start, end) = word.ids_offsets();
                CachedIds {
                    padded: &self.spilled_buffer[start..],
                    len: end - start,
                }
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

pub struct InsertPlacement {
    pub(super) index: usize,
    pub(super) signature: Signature,
    pub(super) tag: u8,
}

pub enum Lookup<'a> {
    Hit(CachedIds<'a>),
    Miss(InsertPlacement),
}
