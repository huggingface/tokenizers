#[cfg(doc)]
use super::cache::WordCache;

/// The most ids of a word that [`WordCache::emit_home_hits`] answers.
///
/// Up to this number, a hit copies ids at one size whatever the word's own: see
/// [`CachedIds::write`]. A word with more ids is cached too, and takes the slower lookup.
pub const MAX_RUN_IDS: usize = 32;

/// The ids a cached word encodes to.
#[derive(Clone, Copy)]
pub struct CachedIds<'a> {
    /// The ids, then whatever is stored after them.
    pub(super) padded: &'a [u32],
    pub(super) len: usize,
}

impl<'a> CachedIds<'a> {
    // Only `unigram` and `wordpiece` call this. `bpe` copies with `write`.
    #[cfg_attr(not(any(feature = "unigram", feature = "wordpiece")), allow(dead_code))]
    pub fn as_slice(self) -> &'a [u32] {
        &self.padded[..self.len]
    }

    /// How many `u32`s [`Self::write`] needs room for.
    pub fn room(self) -> usize {
        self.len.max(MAX_RUN_IDS)
    }

    /// Copies the ids to `dst` and returns how many they are.
    ///
    /// Up to [`MAX_RUN_IDS`] ids from [`WordCache::spilled_buffer`] are copied as `MAX_RUN_IDS`:
    /// the word's own, then what follows them. A copy of one size is a few moves, with no
    /// branch on the number of ids.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writes of [`Self::room`] `u32`s.
    #[inline(always)]
    pub unsafe fn write(self, dst: *mut u32) -> usize {
        match self.padded.first_chunk::<MAX_RUN_IDS>() {
            // SAFETY: the caller left room for `MAX_RUN_IDS` ids.
            Some(padded) if self.len <= MAX_RUN_IDS => unsafe {
                dst.cast::<[u32; MAX_RUN_IDS]>().write(*padded)
            },
            // More ids than one copy holds, or ids inline in a slot past the word's home.
            // SAFETY: the caller left room for `len` ids.
            _ => unsafe { dst.copy_from_nonoverlapping(self.padded.as_ptr(), self.len) },
        }
        self.len
    }
}
