use crate::vocab::bucket_vocab_store::BucketVocabStore;
use ahash::AHashMap;
use std::sync::LazyLock;

// The GPT-2 pre-tokenize regex is the canonical spec in atomsplit (single source of truth); re-export
// under the historical name so call sites are unchanged. `pub` because the `ByteLevel` pre-tokenizer
// lowering — which rewrites a `use_regex` ByteLevel into a `Split` on exactly this pattern — lives
// in `tk-convert`.
pub use atomsplit::regexes::GPT2 as GPT2_REGEX_STR;

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
/// Inverse of [`BYTES_CHAR_LOOKUP`]: maps each byte-level unicode character back to its byte.
///
/// Used when decoding byte-level tokens back into raw bytes. Because [`BYTES_CHAR_LOOKUP`] is
/// a bijection over all 256 bytes, this map has exactly 256 entries with no collisions.
pub static CHAR_BYTES_LOOKUP: LazyLock<AHashMap<char, u8>> = LazyLock::new(|| {
    (0..=255u8)
        .map(|byte| (BYTES_CHAR_LOOKUP[byte as usize], byte))
        .collect()
});

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
    CHAR_BYTES_LOOKUP
        .get(&c)
        .map_or_else(|| c.to_string().into_bytes(), |b| vec![*b])
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
