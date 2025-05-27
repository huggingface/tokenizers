use crate::processors::byte_level::bytes_char;
use crate::tokenizer::{NormalizedString, Normalizer, Result};
use crate::utils::macro_rules_attribute;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

#[derive(Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct ByteLevel;

static BYTES_CHAR: LazyLock<HashMap<u8, char>> = LazyLock::new(bytes_char);

impl Default for ByteLevel {
    fn default() -> Self {
        Self::new()
    }
}

impl ByteLevel {
    pub fn new() -> Self {
        Self {}
    }

    pub fn alphabet() -> HashSet<char> {
        BYTES_CHAR.values().copied().collect()
    }
}

impl Normalizer for ByteLevel {
    /// Strip the normalized string inplace
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        if !normalized.is_empty() {
            let s = normalized.get();
            let s_bytes = s.as_bytes();
            let mut transformations: Vec<(char, isize)> = Vec::with_capacity(s.len());
            let mut i = 0;
            for cur_char in s.chars() {
                let size = cur_char.len_utf8();
                let bytes = &s_bytes[i..i + size];
                i += size;
                for (j, b) in bytes.iter().enumerate() {
                    transformations.push((BYTES_CHAR[b], if j > 0 { 1 } else { 0 }));
                }
            }
            normalized.transform(transformations, 0);
        }
    Ok(())
    }

    /// Fast normalization: byte-to-char mapping without tracking positions
    fn normalize_fast(&self, input: &str) -> String {
        let mut out = String::with_capacity(input.len());
            for b in input.as_bytes() {
                out.push(BYTES_CHAR[b]);
            }
        out
    }
}

unsafe impl Sync for ByteLevel {}

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
}
