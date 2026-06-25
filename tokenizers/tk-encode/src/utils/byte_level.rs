use std::sync::LazyLock;

use ahash::AHashMap;

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

/// Expands a UTF-8 string into its GPT-2 byte-level character sequence, paired with the
/// alignment deltas expected by [`NormalizedString::transform`].
///
/// Each byte of `input` becomes one [`BYTES_CHAR_LOOKUP`] character, so a multi-byte
/// codepoint explodes into several byte-level chars. The accompanying `isize` is the
/// alignment `change`: `0` for the first byte of a source char (it replaces that char) and
/// `1` for every following byte (each is a new char mapped back onto the same source char),
/// keeping offset tracking correct through the expansion.
///
/// The result feeds straight into `normalized.transform(byte_level_transform(s), 0)`.
///
/// [`NormalizedString::transform`]: crate::tokenizer::NormalizedString::transform
pub(crate) fn byte_level_transform(input: &str) -> Vec<(char, isize)> {
    let mut transformations: Vec<(char, isize)> = vec![];
    for (index, character) in input.char_indices() {
        let char_size = character.len_utf8();
        transformations.extend(
            input.as_bytes()[index..index + char_size]
                .iter()
                .enumerate()
                .map(|(i, b)| (BYTES_CHAR_LOOKUP[*b as usize], isize::from(i > 0))),
        );
    }
    transformations
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::NormalizedString;

    #[test]
    fn test_byte_level_transform() {
        let original = "Hello 我今天能为你做什么";
        let normalized = "HelloĠæĪĳä»Ĭå¤©èĥ½ä¸ºä½łåģļä»Ģä¹Ī";
        assert_ne!(original, normalized);
        let mut n = NormalizedString::from(original);
        n.transform(byte_level_transform(n.get()), 0);
        assert_eq!(&n.get(), &normalized);
        assert_eq!(
            n,
            NormalizedString::new(
                original.to_string(),
                normalized.to_string(),
                vec![
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (4, 5),
                    (5, 6),
                    (5, 6),
                    (6, 9),
                    (6, 9),
                    (6, 9),
                    (6, 9),
                    (6, 9),
                    (6, 9),
                    (9, 12),
                    (9, 12),
                    (9, 12),
                    (9, 12),
                    (9, 12),
                    (9, 12),
                    (12, 15),
                    (12, 15),
                    (12, 15),
                    (12, 15),
                    (12, 15),
                    (12, 15),
                    (15, 18),
                    (15, 18),
                    (15, 18),
                    (15, 18),
                    (15, 18),
                    (15, 18),
                    (18, 21),
                    (18, 21),
                    (18, 21),
                    (18, 21),
                    (18, 21),
                    (18, 21),
                    (21, 24),
                    (21, 24),
                    (21, 24),
                    (21, 24),
                    (21, 24),
                    (21, 24),
                    (24, 27),
                    (24, 27),
                    (24, 27),
                    (24, 27),
                    (24, 27),
                    (24, 27),
                    (27, 30),
                    (27, 30),
                    (27, 30),
                    (27, 30),
                    (27, 30),
                    (27, 30),
                    (30, 33),
                    (30, 33),
                    (30, 33),
                    (30, 33),
                    (30, 33),
                    (30, 33)
                ],
                0
            )
        );
        assert_eq!(
            n.alignments_original(),
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 7),
                (7, 13),
                (7, 13),
                (7, 13),
                (13, 19),
                (13, 19),
                (13, 19),
                (19, 25),
                (19, 25),
                (19, 25),
                (25, 31),
                (25, 31),
                (25, 31),
                (31, 37),
                (31, 37),
                (31, 37),
                (37, 43),
                (37, 43),
                (37, 43),
                (43, 49),
                (43, 49),
                (43, 49),
                (49, 55),
                (49, 55),
                (49, 55),
                (55, 61),
                (55, 61),
                (55, 61)
            ]
        );
    }
}
