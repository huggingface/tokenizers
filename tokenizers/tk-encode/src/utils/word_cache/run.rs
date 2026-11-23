//! The loop that answers a run of words straight from their home slots.

use super::ids::MAX_RUN_IDS;
use super::table::WordCache;
use crate::pipeline::Token;
use crate::utils::signature::Signature;
use crate::utils::uncovered::Uncovered;

impl WordCache {
    /// Answers the leading words of a batch straight from their home slots, and appends their
    /// ids to `output`. Returns how many words it answered.
    ///
    /// It stops at the first word its home slot does not hold, or holds with more than
    /// [`MAX_RUN_IDS`] ids. That word goes through [`Self::lookup_with_signature`]. On text the
    /// cache has seen, nearly every word is answered here, so this loop is kept to a few loads,
    /// compares and stores per word.
    ///
    /// `word` gives the bytes of the word at an index of `signatures`. It is only called for a
    /// word longer than its signature.
    // Forced into its one caller, a small function of its own: inside the whole encode loop, the
    // table's address and mask were reloaded per word.
    #[inline(always)]
    pub fn emit_home_hits<'w>(
        &self,
        signatures: &[Signature],
        word: impl Fn(usize) -> &'w [u8],
        output: &mut Vec<Token>,
    ) -> usize {
        // A hit copies `MAX_RUN_IDS` ids or fewer at one size, whatever the word's own number.
        output.reserve(MAX_RUN_IDS * signatures.len());
        let slots = self.slots.as_ptr();
        // SAFETY: the end of a `Vec`'s contents is inside its allocation.
        let dst = unsafe { output.as_mut_ptr().add(output.len()).cast::<u32>() };
        let mut next = dst;
        let mut answered = 0;
        for (i, &signature) in signatures.iter().enumerate() {
            // SAFETY: a home is masked with `placement_mask` (`next_pow2 - 1`), and the table
            // holds `next_pow2 + WINDOW_SIZE` slots.
            // A reference, not a copy: the calls below take the slot's address, and a copy
            // would then live on the stack, where each compare of this loop would reload it.
            let slot = unsafe { &*slots.add(self.home(signature)) };
            if slot.holds_inline(signature) {
                // SAFETY: `next` is at most `MAX_RUN_IDS * answered` ids past `dst`, and the
                // room reserved above holds `MAX_RUN_IDS` more.
                next = unsafe { slot.copy_inline_ids(next) };
            } else if let Some(record) = slot.run_record(signature, self.spill.generation())
                && (signature.len <= Signature::BYTES
                    || Uncovered::of(word(i)).is_some_and(|bytes| self.spill.holds(record, bytes)))
            {
                let padded = self.spill.padded_ids(record);
                // SAFETY: as above, and the record is of no more than `MAX_RUN_IDS` ids.
                unsafe {
                    next.cast::<[u32; MAX_RUN_IDS]>().write(*padded);
                    next = next.add(record.ids_len());
                }
            } else {
                break;
            }
            answered += 1;
        }
        // SAFETY: `next` started at `dst` and moved past the ids of each word the loop answered.
        unsafe { output.set_len(output.len() + next.offset_from_unsigned(dst)) };
        answered
    }
}
