//! The bytes a table compares for a word longer than its [`Signature`].

use crate::utils::signature::Signature;

/// How many bytes a compare of [`Uncovered`] bytes reads at a time.
const BLOCK: usize = 16;

/// The bytes of a word longer than [`Signature::BYTES`] that a table compares on top of its
/// signature.
///
/// They hold every byte the signature leaves out. A word that leaves out fewer than [`BLOCK`]
/// adds bytes its signature holds, so there is always a whole block to compare.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Uncovered<'a>(&'a [u8]);

impl<'a> Uncovered<'a> {
    /// `None` for a word its signature holds whole.
    #[inline(always)]
    pub(crate) fn of(word: &'a [u8]) -> Option<Self> {
        let len = word.len();
        if len <= Signature::BYTES {
            return None;
        }
        let start = (len - BLOCK).min(8);
        let end = (start + BLOCK).max(len - 8);
        Some(Self(&word[start..end]))
    }

    pub(crate) fn as_bytes(self) -> &'a [u8] {
        self.0
    }

    /// Whether `stored` starts with these bytes. `stored` is where a table kept them, followed
    /// by whatever it kept next.
    #[inline(always)]
    pub(crate) fn is_prefix_of(self, stored: &[u8]) -> bool {
        stored
            .get(..self.0.len())
            .is_some_and(|stored| same_blocks(stored, self.0))
    }
}

impl PartialEq for Uncovered<'_> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len() && same_blocks(self.0, other.0)
    }
}

/// Whether two byte strings of one length, [`BLOCK`] or more, are equal.
///
/// It compares one [`BLOCK`] at a time and tests the outcome once, so no branch depends on the
/// bytes. The last block is read from the end, and may overlap the block before it.
#[inline(always)]
fn same_blocks(a: &[u8], b: &[u8]) -> bool {
    debug_assert_eq!(a.len(), b.len());
    let (Some(last_a), Some(last_b)) = (a.last_chunk::<BLOCK>(), b.last_chunk::<BLOCK>()) else {
        return false;
    };
    let mut diff = u128::from_ne_bytes(*last_a) ^ u128::from_ne_bytes(*last_b);
    // Stops one byte short. The block at the end is read above, whole or not.
    let whole = a.len() - 1;
    let (blocks_a, blocks_b) = (a[..whole].as_chunks().0, b[..whole].as_chunks().0);
    for (a, b) in blocks_a.iter().zip(blocks_b) {
        diff |= u128::from_ne_bytes(*a) ^ u128::from_ne_bytes(*b);
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lengths on both sides of each edge: 17 and 23 take in bytes of the tail, 24 to 32 are one
    /// block, 33 is the first to need two, 48 and 49 are whole blocks and one byte past.
    const LONG_LENGTHS: [usize; 7] = [17, 23, 24, 32, 33, 48, 49];

    /// Each byte is its own offset in the word, so the first byte returned says where they start.
    #[test]
    fn the_uncovered_bytes_hold_what_the_signature_leaves_out_in_a_block_or_more() {
        assert_eq!(Uncovered::of(b""), None);
        assert_eq!(Uncovered::of(b"0123456789abcdef"), None);
        for len in LONG_LENGTHS {
            let word: Vec<u8> = (0..len as u8).collect();
            let uncovered = Uncovered::of(&word)
                .expect("longer than a signature")
                .as_bytes();
            let start = uncovered[0] as usize;
            let end = start + uncovered.len();
            assert_eq!(uncovered, &word[start..end], "len {len}");
            assert!(start <= 8 && end >= len - 8, "len {len}: {start}..{end}");
            assert!(uncovered.len() >= BLOCK, "len {len}: {start}..{end}");
        }
    }

    /// A compare that skipped any one of the bytes would call two words equal.
    #[test]
    fn uncovered_bytes_that_differ_in_one_byte_are_not_the_same() {
        for len in LONG_LENGTHS {
            let word: Vec<u8> = (0..len as u8).collect();
            let uncovered = Uncovered::of(&word).expect("longer than a signature");
            assert!(uncovered.is_prefix_of(uncovered.as_bytes()), "len {len}");
            for at in 0..uncovered.as_bytes().len() {
                let mut other = uncovered.as_bytes().to_vec();
                other[at] ^= 0x80;
                assert!(!uncovered.is_prefix_of(&other), "len {len} byte {at}");
                assert_ne!(uncovered, Uncovered(&other), "len {len} byte {at}");
            }
        }
    }

    /// Past 254 bytes the vocabulary knows a token's length only through this compare.
    #[test]
    fn uncovered_bytes_are_not_the_same_as_longer_ones_that_start_with_them() {
        let (short, long) = ([b'm'; 300], [b'm'; 301]);
        assert_ne!(Uncovered::of(&short), Uncovered::of(&long));
    }
}
