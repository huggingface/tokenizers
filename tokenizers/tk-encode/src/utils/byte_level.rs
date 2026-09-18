use crate::vocab::bucket_vocab_store::BucketVocabStore;
use std::sync::LazyLock;

// The GPT-2 pre-tokenize regex is the canonical spec in bitsplit (single source of truth); re-export
// under the historical name so call sites are unchanged. `pub` because the `ByteLevel` pre-tokenizer
// lowering -- which rewrites a `use_regex` ByteLevel into a `Split` on exactly this pattern -- lives
// in `tk-convert`.
pub use bitsplit::regexes::GPT2 as GPT2_REGEX_STR;

/// Maps each byte to its GPT-2 byte-level unicode character, indexed by the byte value.
///
/// This is the reversible bytes-to-unicode scheme from GPT-2's BPE encoder
/// ([reference](https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9)): each of the
/// 256 byte values is assigned a distinct, printable unicode codepoint so that arbitrary
/// (including non-UTF-8) byte sequences can be represented as text and survive a round-trip.
/// Bytes that are already printable map to themselves (codepoint `== byte`); the rest are
/// shifted into the `256..` range so they never collide with the first group.
///
/// The mapping is a bijection — see [`CHAR_BYTES_LOOKUP`] for the inverse.
pub static BYTES_CHAR_LOOKUP: LazyLock<[char; 256]> = LazyLock::new(make_byte_char_lookup);
/// Inverse of [`BYTES_CHAR_LOOKUP`], indexed by codepoint: `CHAR_BYTES_LOOKUP[c as usize]` is
/// the byte that `c` stands for, or `None` if `c` is not a byte-level character.
///
/// [`make_byte_char_lookup`] only ever emits codepoints below 324, so a 512-entry table holds
/// the whole alphabet and the inverse needs no hashing. Read it through [`char_to_byte`], which
/// handles the out-of-range case.
pub static CHAR_BYTES_LOOKUP: LazyLock<[Option<u8>; 512]> = LazyLock::new(|| {
    let mut table = [None; 512];
    for byte in 0..=255u8 {
        table[BYTES_CHAR_LOOKUP[byte as usize] as usize] = Some(byte);
    }
    table
});

/// The byte `c` stands for in the byte-level alphabet, or `None` if it is not one of its
/// characters. One array index; the reverse mapping used to be a hash probe.
#[inline]
pub fn char_to_byte(c: char) -> Option<u8> {
    *CHAR_BYTES_LOOKUP.get(c as usize)?
}

fn make_byte_char_lookup() -> [char; 256] {
    let mut lookup: [char; 256] = ['\0'; 256];

    let mut counter = 0;
    for byte in 0..=255u8 {
        let is_printable_utf32 = matches!(byte,
            // Printable ASCII
            b'!'..=b'~'
            // printable latin1
            | b'\xA1'..=b'\xAC'
            | b'\xAE'..=b'\xFF'
        );
        if is_printable_utf32 {
            // SAFETY: a printable byte (< 0x100) is always a valid Unicode scalar value.
            lookup[byte as usize] = unsafe { char::from_u32_unchecked(byte as u32) };
        } else {
            // Byte isn't printable on its own: remap it to 256 + n which is a printable codepoint.
            // SAFETY: there are 68 non-printable bytes, so counter is in 0..=67 and the
            // argument is in 256..=323 which are valid unicode scalar value.
            lookup[byte as usize] = unsafe { char::from_u32_unchecked(u32::pow(2, 8) + counter) };
            counter += 1;
        }
    }

    lookup
}

fn reverse_lookup(c: char) -> Vec<u8> {
    char_to_byte(c).map_or_else(|| c.to_string().into_bytes(), |b| vec![b])
}

pub(crate) fn transform_vocab(vocab: BucketVocabStore) -> BucketVocabStore {
    BucketVocabStore::build(
        vocab
            .content()
            .into_iter()
            .map(|(string, token_id)| (string.chars().flat_map(reverse_lookup).collect(), token_id))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `CHAR_BYTES_LOOKUP` indexes by codepoint without a guard, so it would panic on first use
    /// if [`make_byte_char_lookup`] ever moved the alphabet past the end of the table.
    #[test]
    fn alphabet_fits_the_table() {
        for c in BYTES_CHAR_LOOKUP.iter() {
            assert!(
                (*c as usize) < CHAR_BYTES_LOOKUP.len(),
                "{c:?} (U+{:04X}) is past the table",
                *c as u32
            );
        }
    }

    #[test]
    fn every_byte_round_trips() {
        for byte in 0..=255u8 {
            assert_eq!(char_to_byte(BYTES_CHAR_LOOKUP[byte as usize]), Some(byte));
        }
    }

    /// Codepoints the alphabet does not claim, on both sides of the table's end: the
    /// non-printable ASCII and latin1 holes that `make_byte_char_lookup` remaps away from, and
    /// a character far beyond 512.
    #[test]
    fn chars_outside_the_alphabet_have_no_byte() {
        for c in [
            '\0',
            '\u{7f}',
            '\u{a0}',
            '\u{ad}',
            '\u{1ff}',
            '漢',
            '\u{1f600}',
        ] {
            assert_eq!(char_to_byte(c), None, "{c:?} should not map to a byte");
        }
    }
}
