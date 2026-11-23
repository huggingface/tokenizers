use super::cache::WordCache;
#[cfg(doc)]
use super::ids::MAX_RUN_IDS;
use super::slot::MAX_INLINE_IDS;
use crate::utils::signature::Signature;

impl WordCache {
    /// Answers the leading words of a batch straight from their home slots, into the caller's
    /// buffer. Returns how many words it answered and how many ids it wrote.
    ///
    /// It stops at the first word its home slot does not hold, or holds with more than
    /// [`MAX_RUN_IDS`] ids. That word goes through [`Self::lookup_with_signature`]. On text the
    /// cache has seen, nearly every word is answered here, so this loop is kept to a few loads,
    /// compares and stores per word.
    ///
    /// `word` gives the bytes of the word at an index of `signatures`. It is only called for a
    /// word longer than its signature.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writes of [`MAX_RUN_IDS`] `u32`s per signature. A hit writes a
    /// fixed number of ids, whatever the word's own.
    // Forced into its one caller, a small function of its own: inside the whole encode loop, the
    // table's address and mask were reloaded per word.
    #[inline(always)]
    pub unsafe fn emit_home_hits<'w>(
        &self,
        signatures: &[Signature],
        word: impl Fn(usize) -> &'w [u8],
        dst: *mut u32,
    ) -> (usize, usize) {
        let slots = self.cached_words.as_ptr();
        let mut next = dst;
        let mut answered = 0;
        for (i, &signature) in signatures.iter().enumerate() {
            // SAFETY: a home is masked with `placement_mask` (`next_pow2 - 1`), and the table
            // holds `next_pow2 + WINDOW_SIZE` slots.
            // A reference, not a copy: the calls below take the slot's address, and a copy
            // would then live on the stack, where each compare of this loop would reload it.
            let slot = unsafe { &*slots.add(self.home(signature)) };
            if slot.holds_inline(signature) {
                // SAFETY: `next` is at most `MAX_RUN_IDS * answered` past `dst`, and the caller
                // left room for `MAX_RUN_IDS` more.
                unsafe {
                    next.copy_from_nonoverlapping(slot.payload.as_ptr(), MAX_INLINE_IDS);
                    next = next.add(slot.ids_len());
                }
            } else if slot.holds_spilled(signature, self.spilled_generation)
                && (signature.len <= Signature::BYTES || self.holds_uncovered(slot, word(i)))
            {
                // SAFETY: as above, and `holds_spilled` saw no more than `MAX_RUN_IDS` ids.
                unsafe { next = next.add(self.get_cached_ids(slot).write(next)) };
            } else {
                break;
            }
            answered += 1;
        }
        // SAFETY: `next` was derived from `dst` and only moved forward.
        (answered, unsafe { next.offset_from_unsigned(dst) })
    }
}
