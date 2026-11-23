//! The loop that answers a run of words straight from their slots.

use super::ids::MAX_RUN_IDS;
use super::slot::WordCacheSlot;
use super::table::WordCache;
use crate::pipeline::Token;
use crate::utils::signature::Signature;
use crate::utils::uncovered::Uncovered;

/// What a run saw in the home slot of the word it stopped on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HomeSlot {
    /// Vacant: the cache does not hold the word.
    Vacant,
    /// Taken, or not read. Only a lookup tells whether the cache holds the word.
    Unknown,
}

impl WordCache {
    /// Answers the leading words of a batch straight from their slots, and appends their ids to
    /// `output`. Returns how many words it answered, and what it saw of the next one.
    ///
    /// A word is answered from its home slot, or from the slot after it, where most words that
    /// lost their home slot are. The run stops at the first word neither slot holds, or holds
    /// with more than [`MAX_RUN_IDS`] ids. That word goes through
    /// [`Self::lookup_with_signature`]. On text the cache has seen, nearly every word is answered
    /// here, so this loop is kept to a few loads, compares and stores per word.
    ///
    /// `word` gives the bytes of the word at an index of `signatures`. It is only called for a
    /// word longer than its signature.
    // Forced into its one caller, a small function of its own: inside the whole encode loop, the
    // table's address and mask were reloaded per word.
    #[inline(always)]
    pub fn emit_run<'w>(
        &self,
        signatures: &[Signature],
        word: impl Fn(usize) -> &'w [u8],
        output: &mut Vec<Token>,
    ) -> (usize, HomeSlot) {
        // A hit copies `MAX_RUN_IDS` ids or fewer at one size, whatever the word's own number.
        output.reserve(MAX_RUN_IDS * signatures.len());
        let slots = self.slots.as_ptr();
        // A local. Read from `self` in the loop, the mask was loaded again for every word.
        let placement_mask = self.placement_mask;
        // SAFETY: the end of a `Vec`'s contents is inside its allocation.
        let dst = unsafe { output.as_mut_ptr().add(output.len()).cast::<u32>() };
        let mut next = dst;
        let mut answered = 0;
        let mut stopped_on = HomeSlot::Unknown;
        for (i, &signature) in signatures.iter().enumerate() {
            let word = || word(i);
            // SAFETY: a home is masked with `placement_mask` (`next_pow2 - 1`), and the table
            // holds `next_pow2 + WINDOW_SIZE` slots.
            // A reference, not a copy: a copy whose address is taken lives on the stack, where
            // each compare of this loop would reload it.
            let home = unsafe { &*slots.add((signature.hash & placement_mask) as usize) };
            // SAFETY, for the three calls: `next` is at most `MAX_RUN_IDS * answered` ids past
            // `dst`, and the room reserved above holds `MAX_RUN_IDS` more. `home` holds its
            // ids in the first, and is a home slot in the third.
            if home.holds_inline(signature) {
                next = unsafe { home.copy_inline_ids(next) };
            } else if let Some(after) = unsafe { self.emit_spilled(home, signature, word, next) } {
                next = after;
            } else if home.is_vacant() {
                // No word is stored past a vacant slot of its window: the cache does not hold
                // this one.
                stopped_on = HomeSlot::Vacant;
                break;
            } else if let Some(after) =
                unsafe { self.emit_after_home(home, &signature, word, next) }
            {
                next = after;
            } else {
                break;
            }
            answered += 1;
        }
        // SAFETY: `next` started at `dst` and moved past the ids of each word the loop answered.
        unsafe { output.set_len(output.len() + next.offset_from_unsigned(dst)) };
        (answered, stopped_on)
    }

    /// Copies the ids of the word with this signature to `next` when `slot` holds them in the
    /// spill buffer and they are [`MAX_RUN_IDS`] or fewer. Returns `next` moved past them.
    ///
    /// # Safety
    ///
    /// `next` must be valid for writes of [`MAX_RUN_IDS`] `u32`s.
    #[inline(always)]
    unsafe fn emit_spilled<'w>(
        &self,
        slot: &WordCacheSlot,
        signature: Signature,
        word: impl FnOnce() -> &'w [u8],
        next: *mut u32,
    ) -> Option<*mut u32> {
        let record = slot.run_record(signature, self.spill.generation())?;
        if signature.len > Signature::BYTES
            && !Uncovered::of(word()).is_some_and(|bytes| self.spill.holds(record, bytes))
        {
            return None;
        }
        let padded = self.spill.padded_ids(record);
        // SAFETY: the caller left room for `MAX_RUN_IDS` ids, and the record has no more.
        unsafe {
            next.cast::<[u32; MAX_RUN_IDS]>().write(*padded);
            Some(next.add(record.ids_len()))
        }
    }

    /// Copies the ids of the word with this signature to `next` when the slot after `home`
    /// holds them, inline or in the spill buffer. Returns `next` moved past them.
    ///
    /// # Safety
    ///
    /// `home` must be a home slot of the table, and `next` must be valid for writes of
    /// [`MAX_RUN_IDS`] `u32`s.
    // Out of line: inline, these checks took registers from the run loop, which then ran four
    // more instructions for every word it answered.
    #[inline(never)]
    unsafe fn emit_after_home<'w>(
        &self,
        home: &WordCacheSlot,
        signature: &Signature,
        word: impl FnOnce() -> &'w [u8],
        next: *mut u32,
    ) -> Option<*mut u32> {
        // SAFETY: the table holds `WINDOW_SIZE` slots past its last home slot.
        let slot = unsafe { &*std::ptr::from_ref(home).add(1) };
        if !slot.has_signature(*signature) {
            return None;
        }
        // SAFETY: the caller left room for `MAX_RUN_IDS` ids.
        unsafe { self.emit_slot(slot, signature, word, next) }
    }

    /// Copies the ids of a word to `next` from a slot that has its signature: inline ids, or
    /// ids in the spill buffer when they are [`MAX_RUN_IDS`] or fewer and still there.
    ///
    /// # Safety
    ///
    /// `next` must be valid for writes of [`MAX_RUN_IDS`] `u32`s.
    // Out of line: most calls of `emit_after_home` end on its first test, and with this code
    // inline each one saved and restored seven registers around it.
    #[inline(never)]
    unsafe fn emit_slot<'w>(
        &self,
        slot: &WordCacheSlot,
        signature: &Signature,
        word: impl FnOnce() -> &'w [u8],
        next: *mut u32,
    ) -> Option<*mut u32> {
        if slot.record().is_none() {
            // SAFETY: the slot holds its ids, and the caller left room for them.
            return Some(unsafe { slot.copy_inline_ids(next) });
        }
        // SAFETY: the caller left room for `MAX_RUN_IDS` ids.
        unsafe { self.emit_spilled(slot, *signature, word, next) }
    }
}
