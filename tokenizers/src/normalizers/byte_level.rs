use crate::processors::byte_level::bytes_char;
use crate::tokenizer::{NormalizedString, Normalizer, Result};
use crate::utils::macro_rules_attribute;
use ahash::{AHashMap, AHashSet};
use std::sync::LazyLock;

#[derive(Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct ByteLevel;

static BYTES_CHAR: LazyLock<AHashMap<u8, char>> = LazyLock::new(bytes_char);

/// Pre-encoded UTF-8 lookup: for each input byte, the UTF-8 encoding of its
/// byte-level char (1 or 2 bytes) + length.  Avoids HashMap lookup + per-char
/// UTF-8 encoding in the hot `normalize_str` path.
struct Utf8Entry {
    bytes: [u8; 2],
    len: u8,
}

static BYTES_CHAR_UTF8: LazyLock<[Utf8Entry; 256]> = LazyLock::new(|| {
    let map = bytes_char();
    std::array::from_fn(|i| {
        let c = map[&(i as u8)];
        let mut buf = [0u8; 2];
        let s = c.encode_utf8(&mut buf);
        let len = s.len() as u8;
        Utf8Entry { bytes: buf, len }
    })
});

impl Default for ByteLevel {
    fn default() -> Self {
        Self::new()
    }
}

impl ByteLevel {
    pub fn new() -> Self {
        Self {}
    }

    pub fn alphabet() -> AHashSet<char> {
        BYTES_CHAR.values().copied().collect()
    }
}

impl Normalizer for ByteLevel {
    /// Strip the normalized string inplace
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        if !normalized.is_empty() {
            let s = normalized.get();
            let mut transformations: Vec<(char, isize)> = Vec::with_capacity(s.len());
            for (i, cur_char) in s.char_indices() {
                let size = cur_char.len_utf8();
                transformations.extend(
                    s.as_bytes()[i..i + size]
                        .iter()
                        .enumerate()
                        .map(|(i, b)| (BYTES_CHAR[b], isize::from(i > 0))),
                );
            }
            normalized.transform(transformations, 0);
        }
        Ok(())
    }

    /// Fast path: map each byte to its byte-level char without alignment tracking.
    /// Uses a pre-encoded UTF-8 lookup table — no HashMap, no per-char encoding.
    fn normalize_str(&self, s: &str) -> Result<String> {
        let table = &*BYTES_CHAR_UTF8;
        let mut out = Vec::with_capacity(s.len() * 2);
        for &b in s.as_bytes() {
            let entry = &table[b as usize];
            out.extend_from_slice(&entry.bytes[..entry.len as usize]);
        }
        // SAFETY: every entry in the table is valid UTF-8 (encoded from a char).
        Ok(unsafe { String::from_utf8_unchecked(out) })
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_byte_level_normalize() {
        let original = "Hello 我今天能为你做什么";
        let normalized = "HelloĠæĪĳä»Ĭå¤©èĥ½ä¸ºä½łåģļä»Ģä¹Ī";
        assert_ne!(original, normalized);
        let mut n = NormalizedString::from(original);
        let byte_level = ByteLevel::new();
        byte_level.normalize(&mut n).unwrap();
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

    #[test]
    fn normalize_str_matches_normalize() {
        let bl = ByteLevel::new();
        for input in &["Hello", "Hello 我今天能为你做什么", "", "abc\x00\x01\x7f"] {
            let mut ns = NormalizedString::from(*input);
            bl.normalize(&mut ns).unwrap();
            let fast = bl.normalize_str(input).unwrap();
            assert_eq!(ns.get(), fast, "mismatch for input: {input:?}");
        }
    }
}
