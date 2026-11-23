//! The cache itself: where a word is looked up, and where it is stored.

use prefetch_index::prefetch_index;

use super::ids::CachedIds;
use super::slot::{MAX_INLINE_IDS, WordCacheSlot};
use super::spill::SpillBuffer;
use super::tags::{EMPTY, WINDOW_SIZE, Window, tag_of};
use crate::utils::signature::Signature;
use crate::utils::uncovered::Uncovered;

/// A table mapping words (`[u8]`) to the token ids they encode to (`[u32]`)
///
/// A word is stored in the first free slot of its window, and a slot never becomes free again
/// short of a reset. So every slot from a word's home slot to the one that holds it is taken:
/// a lookup that meets a vacant slot has seen every slot the word can be in.
pub struct WordCache {
    /// One slot per cached word. See [`WordCacheSlot`] for more details.
    pub(super) slots: Box<[WordCacheSlot]>,
    /// One byte of hash per slot, scanned to find the slots of a window that may hold a word.
    tags: Box<[u8]>,
    pub(super) placement_mask: u64,
    /// What does not fit in a slot.
    pub(super) spill: SpillBuffer,
}

impl WordCache {
    pub fn new(num_slots: usize) -> Self {
        let next_pow2 = num_slots.next_power_of_two();
        if next_pow2 != num_slots {
            warn!(
                "WordCache: rounding cache_capacity up from {num_slots} to {next_pow2} slots \
                 (a power of two is required for slot masking); \
                 behavior is unchanged, the table just uses slightly more memory"
            );
        }
        let n: usize = next_pow2 + WINDOW_SIZE;
        // 16 per slot held the ids alone. A word longer than its signature now keeps its other
        // bytes here too, about half as much again on Chinese and Japanese text.
        let spill_budget = 24 * n;
        Self {
            slots: vec![WordCacheSlot::default(); n].into_boxed_slice(),
            tags: vec![EMPTY; n].into_boxed_slice(),
            placement_mask: (next_pow2 as u64) - 1,
            spill: SpillBuffer::new(spill_budget),
        }
    }

    /// Looks up a word in the cache.
    /// On [Lookup::Hit], returns the ids it encodes to.
    /// On [Lookup::Miss], returns the location in [Self::slots] where it should be inserted
    #[inline]
    // Only `unigram` and `wordpiece` call this. `bpe` holds the signature already.
    #[cfg_attr(not(any(feature = "unigram", feature = "wordpiece")), allow(dead_code))]
    pub fn lookup(&self, word: &[u8]) -> Lookup<'_> {
        self.lookup_with_signature(word, Signature::new(word))
    }

    /// Starts loading the home slot of the word with this signature, so that a lookup a few
    /// words later finds it in the CPU cache.
    #[inline(always)]
    pub fn prefetch(&self, signature: Signature) {
        prefetch_index(&*self.slots, self.home(signature));
    }

    /// [`Self::lookup`] for a caller that already holds the word's signature.
    // Forced: out of line, each call returned its 56-byte `Lookup` through memory.
    #[inline(always)]
    pub fn lookup_with_signature(&self, word: &[u8], signature: Signature) -> Lookup<'_> {
        let home = self.home(signature);
        let tag = tag_of(signature);
        // The home slot answers most lookups on its own, and a caller that prefetched it or
        // just read it has it at hand. The scan below costs a load from the tags and a walk.
        if self.slots[home].is_vacant() {
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

        let window = Window::new(&self.tags, home);
        let (candidates, first_empty) = window.find_matches_and_first_empty(tag);
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
        &self,
        index: usize,
        word: &[u8],
        signature: Signature,
        tag: u8,
    ) -> Option<Lookup<'_>> {
        let slot = &self.slots[index];
        if !slot.has_signature(signature) {
            return None;
        }
        let Some(record) = slot.record() else {
            // `insert` spills every word longer than its signature, so this one has no more
            // bytes to compare.
            return Some(Lookup::Hit(slot.inline_ids()));
        };
        if record.generation != self.spill.generation() {
            return Some(Lookup::Miss(InsertPlacement {
                index,
                signature,
                tag,
            }));
        }
        Uncovered::of(word)
            .is_none_or(|uncovered| self.spill.holds(record, uncovered))
            .then(|| Lookup::Hit(self.spill.ids(record)))
    }

    /// Where to insert a word whose home slot is vacant: in that slot.
    #[inline(always)]
    pub fn vacant_home(&self, signature: Signature) -> InsertPlacement {
        debug_assert!(self.slots[self.home(signature)].is_vacant());
        InsertPlacement {
            index: self.home(signature),
            signature,
            tag: tag_of(signature),
        }
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
        let uncovered = Uncovered::of(word);

        let slot = if ids.len() <= MAX_INLINE_IDS && uncovered.is_none() {
            WordCacheSlot::inline(signature, ids)
        } else {
            let (record, wrapped) = self.spill.store(ids, uncovered);
            if wrapped {
                self.reset();
                // The placement was chosen in the table as it was. In the empty one, a slot
                // past the home slot sits behind an empty slot, where no lookup reaches it.
                index = self.home(signature);
            }
            WordCacheSlot::spilled(signature, record)
        };

        self.slots[index] = slot;
        self.tags[index] = tag;
    }

    #[inline(always)]
    pub(super) fn home(&self, signature: Signature) -> usize {
        (signature.hash & self.placement_mask) as usize
    }

    /// Forgets every word, after the spill buffer's generation wrapped.
    fn reset(&mut self) {
        info!(
            "WordCache: eviction generation wrapped past u32::MAX; \
             cleared all {} cached words to avoid stale hits \
             (lookups will miss until the cache refills)",
            self.slots.len()
        );
        self.tags.fill(EMPTY);
        self.slots.fill(WordCacheSlot::default());
    }
}

pub struct InsertPlacement {
    pub(super) index: usize,
    signature: Signature,
    tag: u8,
}

pub enum Lookup<'a> {
    Hit(CachedIds<'a>),
    Miss(InsertPlacement),
}
