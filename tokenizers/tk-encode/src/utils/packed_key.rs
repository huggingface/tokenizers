//! A short byte string as a `u128`, so that comparing two of them is one
//! register-wide equality instead of a `memcmp` against bytes somewhere else in
//! memory.
//!
//! Used by anything that keys on token or word bytes — the BPE word cache and the
//! vocabulary store both do. They share this rather than each holding a copy,
//! because the two would have to agree byte for byte and nothing would catch them
//! drifting apart: a key built one way and compared the other is not a crash, it is
//! a silently wrong token id.
//!
//! ```text
//!   b"tion"                 │ t  i  o  n  00 … 00 │ 4 │   bytes, then the length
//!   b"unbelievabilities"    │        hash         │▲│     too long: only a hash
//!                                                  └ LONG_TAG
//! ```
//!
//! The length in the top byte does two jobs: it keeps `"a"` and `"a\0"` apart, and
//! it means a packed key can never be `0` or collide with a [`LONG_TAG`] one.

/// Longest byte string that fits in a key. One byte of the sixteen holds the
/// length.
pub(crate) const MAX_PACKED: usize = 15;

/// Set when the key holds a hash instead of the bytes themselves. A holder of such
/// a key has to confirm a match against the real bytes, since two long strings can
/// hash alike.
pub(crate) const LONG_TAG: u128 = 1 << 127;

/// `word` as a key, or `None` when it is empty or longer than [`MAX_PACKED`] and
/// the caller has to fall back to a hash.
///
/// `head` is 16 readable bytes starting where `word` does, which a caller that
/// knows what surrounds it can hand over — a pre-tokenized word has the rest of its
/// chunk after it, so `Split::head` supplies one. Reading the whole window and
/// masking the surplus off costs one load, where copying just `word`'s own bytes
/// costs a call into `memcpy`: the length is only known at run time, and LLVM folds
/// a copy into a load only when the length is a constant. Passing `None` is correct
/// and produces the same key, it just pays for that call.
#[inline]
pub(crate) fn pack(word: &[u8], head: Option<&[u8; 16]>) -> Option<u128> {
    let len = word.len();
    if len == 0 || len > MAX_PACKED {
        return None;
    }
    debug_assert!(
        head.is_none_or(|head| head[..len] == *word),
        "head has to start where the word does"
    );
    let lanes = match head {
        Some(head) => u128::from_le_bytes(*head),
        None => {
            let mut lanes = [0u8; 16];
            lanes[..len].copy_from_slice(word);
            u128::from_le_bytes(lanes)
        }
    };
    Some((lanes & (u128::MAX >> (8 * (16 - len)))) | ((len as u128) << 120))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The properties every holder of a key relies on: it is unique per byte
    /// string, is never `0` (which callers use as "nothing here"), and never looks
    /// like a hashed key.
    #[test]
    fn a_packed_key_is_unique_and_never_looks_long() {
        assert_ne!(pack(b"a", None), pack(b"a\0", None));
        assert_ne!(pack(&[0u8; 15], None), Some(0));
        assert_eq!(pack(&[0u8; 15], None).unwrap() & LONG_TAG, 0);
        assert_eq!(pack(b"", None), None);
        assert_eq!(pack(&[b'x'; 16], None), None);
    }

    /// The window is a cheaper way to read the same bytes, not part of the key. If
    /// any byte past the word survived, the same word would key differently
    /// depending on what happened to follow it.
    #[test]
    fn the_window_past_the_word_is_masked_off() {
        let chunk = b"the quick brown fox jumps over the lazy dog";
        for (start, len) in [(0usize, 3usize), (4, 5), (10, 5), (16, 3), (26, 4)] {
            let word = &chunk[start..start + len];
            let head: &[u8; 16] = chunk[start..start + 16].try_into().unwrap();
            assert_eq!(pack(word, None), pack(word, Some(head)), "{word:?}");
        }
    }
}
