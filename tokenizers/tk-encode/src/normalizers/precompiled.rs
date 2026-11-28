use std::borrow::Cow;

use crate::pipeline;
use crate::tokenizer::Result;
pub use spm_precompiled::Precompiled;
use unicode_segmentation::UnicodeSegmentation;

/// A [`Precompiled`] together with the `precompiled_charsmap` bytes it was parsed from.
///
/// The bytes are kept because `spm_precompiled` holds its `precompiled_charsmap` in a private field
/// and publishes it only through its `Serialize` impl. A serde-free writer therefore has no way to
/// ask the value what it was built from, and this is the one normalizer whose configuration *is*
/// that blob — so either the pipeline remembers it or the normalizer cannot be written back out.
///
/// Remembering it costs a second copy of the map: 237 KB for the SentencePiece charsmap that t5,
/// albert and xlm-roberta all ship. Only a config that has a `Precompiled` pays it, and the copy
/// goes away the day upstream grows a three-line getter.
///
/// [`charsmap`](Self::charsmap) is an `Option` because not every producer has the bytes to hand:
/// one that only has an already-parsed [`Precompiled`] records `None`, and a writer then reports
/// that rather than inventing a blob. `from_charsmap` is currently the only constructor, so today
/// it is always `Some`.
#[derive(Debug, Clone)]
pub struct PrecompiledNormalizer {
    parsed: Precompiled,
    charsmap: Option<Box<[u8]>>,
}

impl PrecompiledNormalizer {
    /// Parse a charsmap and keep it, which is what a reader does.
    pub fn from_charsmap(charsmap: &[u8]) -> Result<Self> {
        let parsed =
            Precompiled::from(charsmap).map_err(|e| -> crate::Error { e.to_string().into() })?;
        check_charsmap(charsmap)?;
        Ok(Self {
            parsed,
            charsmap: Some(charsmap.into()),
        })
    }

    /// The bytes this was parsed from, when they are known.
    pub fn charsmap(&self) -> Option<&[u8]> {
        self.charsmap.as_deref()
    }
}

/// Refuses a charsmap that `Precompiled::transform` would index out of bounds with.
///
/// `spm_precompiled` only checks that the blob splits into a trie and a UTF-8 string. Its trie
/// walk (`common_prefix_search`) then indexes the array with whatever offsets the trie holds, and
/// `transform` slices the string at whatever a leaf points to, so a crafted `precompiled_charsmap`
/// panics on every encode.
///
/// The walk steps from a node to `node ^ byte`, for any byte of the input, and on a label match
/// moves on to that unit's child, reading the child itself when the unit has a leaf. So it is
/// enough that the root's node and every child a unit can lead to sit in a 256-unit block that is
/// wholly inside the array, and that every leaf points at a character boundary of the string. A
/// unit whose label is no byte never matches and is skipped, which includes every value unit:
/// darts-clone sets their bit 31. This is one pass over the units rather than a walk of the trie,
/// and SentencePiece's maps pass it, since darts-clone lays units out in whole 256-unit blocks.
fn check_charsmap(charsmap: &[u8]) -> Result<()> {
    let invalid =
        || -> crate::Error { "invalid `precompiled_charsmap`: trie out of bounds".into() };
    // The layout `Precompiled::from` just accepted: a u32 byte size, the trie units, the string.
    let trie_size = u32::from_le_bytes(charsmap[..4].try_into().unwrap()) as usize / 4;
    let units: Vec<usize> = charsmap[4..4 + 4 * trie_size]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|unit| u32::from_le_bytes(*unit) as usize)
        .collect();
    let normalized = std::str::from_utf8(&charsmap[4 + 4 * trie_size..]).map_err(|_| invalid())?;

    // The unit layout, as `spm_precompiled` reads it.
    let offset = |unit: usize| (unit >> 10) << ((unit & (1 << 9)) >> 6);
    let label = |unit: usize| unit & ((1 << 31) | 0xFF);
    let has_leaf = |unit: usize| (unit >> 8) & 1 == 1;
    let value = |unit: usize| unit & ((1 << 31) - 1);
    let block_in_bounds = |node: usize| (node | 0xFF) < units.len();

    let root = *units.first().ok_or_else(invalid)?;
    if !block_in_bounds(offset(root)) {
        return Err(invalid());
    }
    for (index, &unit) in units.iter().enumerate() {
        if !(1..=0xFF).contains(&label(unit)) {
            continue;
        }
        let child = index ^ offset(unit);
        if !block_in_bounds(child) {
            return Err(invalid());
        }
        if has_leaf(unit) && !normalized.is_char_boundary(value(units[child])) {
            return Err(invalid());
        }
    }
    Ok(())
}

impl pipeline::Normalize for PrecompiledNormalizer {
    fn normalize<'a>(&self, input: &'a str, offset: usize) -> Result<Cow<'a, str>> {
        pipeline::Normalize::normalize(&self.parsed, input, offset)
    }
}

impl pipeline::Normalize for Precompiled {
    fn normalize<'a>(&self, input: &'a str, _offset: usize) -> Result<Cow<'a, str>> {
        let mut transformed: Option<String> = None;
        for (g_idx, grapheme) in input.grapheme_indices(true) {
            if grapheme.len() < 6
                && let Some(replacement) = self.transform(grapheme)
            {
                let string = transformed.get_or_insert_with(|| {
                    let mut s = String::with_capacity(input.len());
                    s.push_str(&input[..g_idx]);
                    s
                });
                string.push_str(replacement);
                continue;
            }
            for (c_idx, character) in grapheme.char_indices() {
                if let Some(replacement) =
                    self.transform(&grapheme[c_idx..c_idx + character.len_utf8()])
                {
                    let string = transformed.get_or_insert_with(|| {
                        let mut s = String::with_capacity(input.len());
                        s.push_str(&input[..g_idx + c_idx]);
                        s
                    });
                    string.push_str(replacement);
                } else if let Some(transformed) = transformed.as_mut() {
                    transformed.push(character);
                }
            }
        }
        if let Some(string) = transformed {
            Ok(Cow::Owned(string))
        } else {
            Ok(Cow::Borrowed(input))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::Normalize;

    /// A charsmap blob: the trie's byte size, its units, then the normalized strings.
    fn charsmap(units: &[u32], normalized: &str) -> Vec<u8> {
        let mut blob = ((units.len() * 4) as u32).to_le_bytes().to_vec();
        for unit in units {
            blob.extend_from_slice(&unit.to_le_bytes());
        }
        blob.extend_from_slice(normalized.as_bytes());
        blob
    }

    /// One 256-unit block mapping `a` to `leaf`'s value: the root's offset is 0, so `a` lands on
    /// unit 0x61, whose offset 0x61 ^ 200 leads to the value in unit 200.
    fn map_a(leaf: u32) -> Vec<u32> {
        let mut units = vec![0u32; 256];
        units[0x61] = ((0x61 ^ 200) << 10) | (1 << 8) | 0x61;
        units[200] = leaf;
        units
    }

    #[test]
    fn a_well_formed_charsmap_normalizes() {
        let normalizer = PrecompiledNormalizer::from_charsmap(&charsmap(&map_a(0), "b\0")).unwrap();
        assert_eq!(normalizer.normalize("cab", 0).unwrap(), "cbb");
    }

    #[test]
    fn a_trie_out_of_bounds_is_refused() {
        let bad = [
            // no trie at all: the walk reads unit 0
            charsmap(&[], ""),
            // one unit: the first byte of any input steps past it
            charsmap(&[0], ""),
            // a leaf value past the end of the normalized strings
            charsmap(&map_a(50), "b\0"),
            // a leaf value inside a multi-byte character
            charsmap(&map_a(1), "é\0"),
        ];
        for blob in bad {
            assert!(
                PrecompiledNormalizer::from_charsmap(&blob).is_err(),
                "{blob:?}"
            );
        }
    }
}
