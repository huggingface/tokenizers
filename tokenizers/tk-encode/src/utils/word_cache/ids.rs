//! The ids a lookup hands back, and how they are copied out.

#[cfg(doc)]
use super::spill::SpillBuffer;
#[cfg(doc)]
use super::table::WordCache;
use crate::pipeline::Token;

// Ids are copied straight into the spare room of a `Vec<Token>`. That is only sound while a
// token is laid out as its id.
const _: () = assert!(size_of::<Token>() == size_of::<u32>());
const _: () = assert!(align_of::<Token>() == align_of::<u32>());

/// The most ids of a word that [`WordCache::emit_home_hits`] answers.
///
/// Up to this number, a hit copies ids at one size whatever the word's own: see
/// [`CachedIds::append_to`]. A word with more ids is cached too, and takes the slower lookup.
pub const MAX_RUN_IDS: usize = 32;

/// The ids a cached word encodes to.
#[derive(Clone, Copy)]
pub struct CachedIds<'a> {
    /// The ids, then whatever is stored after them.
    pub(super) padded: &'a [u32],
    pub(super) len: usize,
}

impl<'a> CachedIds<'a> {
    // Only `unigram` and `wordpiece` call this. `bpe` copies with `append_to`.
    #[cfg_attr(not(any(feature = "unigram", feature = "wordpiece")), allow(dead_code))]
    pub fn as_slice(self) -> &'a [u32] {
        &self.padded[..self.len]
    }

    /// Appends the ids to `output`.
    ///
    /// Up to [`MAX_RUN_IDS`] ids from a [`SpillBuffer`] are copied as `MAX_RUN_IDS`: the word's
    /// own, then what follows them. A copy of one size is a few moves, with no branch on the
    /// number of ids.
    #[inline(always)]
    pub fn append_to(self, output: &mut Vec<Token>) {
        output.reserve(self.len.max(MAX_RUN_IDS));
        // SAFETY: the spare room holds `MAX_RUN_IDS` ids, and `len` when that is more. Either
        // copy stays inside it, and the first `len` ids it writes are the word's.
        unsafe {
            let dst = output.as_mut_ptr().add(output.len()).cast::<u32>();
            match self.padded.first_chunk::<MAX_RUN_IDS>() {
                Some(padded) if self.len <= MAX_RUN_IDS => {
                    dst.cast::<[u32; MAX_RUN_IDS]>().write(*padded)
                }
                // More ids than one copy holds, or ids inline in a slot past the word's home.
                _ => dst.copy_from_nonoverlapping(self.padded.as_ptr(), self.len),
            }
            output.set_len(output.len() + self.len);
        }
    }
}
