//! The key the word cache and the vocabulary look a word up by.
//!
//! A [`Signature`] is a word's length, its first eight bytes and its last eight bytes.
//! The two halves overlap in a word shorter than 16 bytes, and together they hold every byte of it.
//! Two words of [`Signature::BYTES`] bytes or less are equal exactly when their signatures are.
//!
//! A longer word shares its signature with every word of the same length, start and end.
//! A table compares [`Signature::uncovered`] as well before it accepts such a word.
//! No lookup accepts a word on its hash alone.
//!
//! Reading a signature costs two 8-byte loads whatever the word's length.
//! There is no branch on the length, so the encode loop reads the signatures of many words ahead
//! of looking any of them up.

/// The length, first eight bytes and last eight bytes of a word.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Signature {
    /// The first eight bytes, little-endian, zero padded when the word is shorter.
    pub(crate) head: u64,
    /// The last eight bytes, little-endian. Equal to `head` when the word is 8 bytes or shorter.
    pub(crate) tail: u64,
    pub(crate) len: usize,
    /// 64 bits computed from the three fields above, which spread signatures over a table.
    /// Computed once here, for every table that looks the word up.
    pub(crate) hash: u64,
}

const SEEDS: [u64; 2] = [0x243F_6A88_85A3_08D3, 0x1319_8A2E_0370_7344];

/// `LOW_BYTES[n]` keeps the low `n` bytes of a `u64`.
const LOW_BYTES: [u64; 9] = [
    0,
    0xFF,
    0xFFFF,
    0xFF_FFFF,
    0xFFFF_FFFF,
    0xFF_FFFF_FFFF,
    0xFFFF_FFFF_FFFF,
    0xFF_FFFF_FFFF_FFFF,
    u64::MAX,
];

impl Signature {
    /// The longest word a signature holds whole.
    pub(crate) const BYTES: usize = 16;

    pub(crate) fn new(word: &[u8]) -> Self {
        let len = word.len();
        if let (Some(head), Some(tail)) = (word.first_chunk(), word.last_chunk()) {
            return Self::from_parts(u64::from_le_bytes(*head), u64::from_le_bytes(*tail), len);
        }
        // Overlapping fixed-size reads: a copy into a padded buffer is a `memcpy` call per word.
        let head = if let (Some(low), Some(high)) = (word.first_chunk(), word.last_chunk()) {
            let (low, high) = (u32::from_le_bytes(*low), u32::from_le_bytes(*high));
            u64::from(low) | u64::from(high) << (8 * (len - 4))
        } else if let (Some(&first), Some(&last)) = (word.first(), word.last()) {
            let middle = u64::from(word[len / 2]) << (8 * (len / 2));
            u64::from(first) | middle | u64::from(last) << (8 * (len - 1))
        } else {
            0
        };
        Self::from_parts(head, head, len)
    }

    /// [`Self::new`] for a word inside a larger buffer, with `readable` bytes from the word's
    /// start to the buffer's end. It reads 8 bytes at the word's start even when the word is
    /// shorter, which is what spares the branch on the length.
    #[inline(always)]
    pub(crate) fn new_readable(word: &[u8], readable: usize) -> Self {
        debug_assert!(readable >= word.len());
        if readable < 8 {
            return Self::new(word);
        }
        let len = word.len();
        let short = len.min(8);
        let mask = LOW_BYTES[short];
        // SAFETY: `readable >= 8` bytes exist from the word's start. The second read starts
        // `len - short` bytes in: at the start when `len < 8`, 8 bytes before the end otherwise.
        let (head, tail) = unsafe {
            let start = word.as_ptr();
            (
                start.cast::<[u8; 8]>().read_unaligned(),
                start.add(len - short).cast::<[u8; 8]>().read_unaligned(),
            )
        };
        Self::from_parts(
            u64::from_le_bytes(head) & mask,
            u64::from_le_bytes(tail) & mask,
            len,
        )
    }

    /// The bytes of `word` its signature does not hold: empty unless the word is longer than
    /// [`Self::BYTES`].
    #[inline]
    pub(crate) fn uncovered(word: &[u8]) -> &[u8] {
        word.get(8..word.len().saturating_sub(8))
            .unwrap_or_default()
    }

    /// The hash is one 64x64 to 128 bit multiply with its two halves xored, the shape foldhash
    /// uses for an input of 16 bytes or less.
    #[inline(always)]
    fn from_parts(head: u64, tail: u64, len: usize) -> Self {
        let product = u128::from(head ^ SEEDS[0]) * u128::from(tail ^ SEEDS[1] ^ len as u64);
        Self {
            head,
            tail,
            len,
            hash: product as u64 ^ (product >> 64) as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The expected value is built byte by byte, so it is the same on a big-endian target.
    fn expected(word: &[u8]) -> Signature {
        let pack = |bytes: &[u8]| {
            bytes.iter().enumerate().fold(0u64, |packed, (i, &byte)| {
                packed | u64::from(byte) << (8 * i)
            })
        };
        let short = word.len().min(8);
        Signature::from_parts(
            pack(&word[..short]),
            pack(&word[word.len() - short..]),
            word.len(),
        )
    }

    /// Words are slices of a larger buffer, which is what makes 8 bytes readable past a short one.
    /// Passing the word's own length as `readable` covers a word at the very end of a buffer.
    #[test]
    fn both_constructors_read_the_same_signature() {
        let buffer: &[u8] = b"\x00\xffhello \xc0\xc1world\x80\x7f\x01\xfe\x00 the quick brown\xff";
        for start in 0..8 {
            for len in 0..=20 {
                let word = &buffer[start..start + len];
                let want = expected(word);
                assert_eq!(Signature::new(word), want, "start {start} len {len}");
                let readable = buffer.len() - start;
                assert_eq!(
                    Signature::new_readable(word, readable),
                    want,
                    "start {start} len {len}"
                );
                assert_eq!(
                    Signature::new_readable(word, len),
                    want,
                    "start {start} len {len}"
                );
            }
        }
    }

    /// Runs of one byte differ only in length up to 8 bytes, and `"ab"` against `"ab\0"` differ
    /// only in a zero byte the padding hides.
    #[test]
    fn words_of_sixteen_bytes_or_less_have_distinct_signatures() {
        let mut words: Vec<Vec<u8>> = Vec::new();
        for len in 0..=Signature::BYTES {
            words.push(vec![b'a'; len]);
            words.push(vec![0; len]);
            words.push((0..len as u8).collect());
            words.push((0..len as u8).rev().collect());
        }
        words.push(b"ab\0".to_vec());
        words.push(b"xaaaaaaay".to_vec());
        words.push(b"xaaaaaaaay".to_vec());
        words.sort();
        words.dedup();
        for (i, a) in words.iter().enumerate() {
            for b in &words[i + 1..] {
                assert_ne!(Signature::new(a), Signature::new(b), "{a:?} / {b:?}");
            }
        }
    }

    #[test]
    fn a_word_past_sixteen_bytes_leaves_its_middle_uncovered() {
        assert_eq!(Signature::uncovered(b""), b"");
        assert_eq!(Signature::uncovered(b"0123456789abcdef"), b"");
        assert_eq!(Signature::uncovered(b"0123456789abcdefg"), b"8");
        assert_eq!(
            Signature::uncovered(b"01234567-middle-89abcdef"),
            b"-middle-"
        );
        assert_eq!(
            Signature::new(b"01234567-middle-89abcdef"),
            Signature::new(b"01234567+MIDDLE+89abcdef")
        );
    }
}
