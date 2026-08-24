use std::borrow::Cow;

use crate::pipeline;
use crate::tokenizer::{NormalizedString, Normalizer, Result};
pub use spm_precompiled::Precompiled;
use std::cmp::Ordering;
use unicode_segmentation::UnicodeSegmentation;

/// A [`Precompiled`] together with the `precompiled_charsmap` bytes it was parsed from.
/// TODO: I think this has an extra copy, but once we move this to bitmapgen we can get rid of the
/// upstream library entirely.
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
        Ok(Self {
            parsed,
            charsmap: Some(charsmap.into()),
        })
    }

    /// Adopt an already-parsed value, whose bytes are gone.
    pub fn from_parsed(parsed: Precompiled) -> Self {
        Self {
            parsed,
            charsmap: None,
        }
    }

    /// The bytes this was parsed from, when they are known.
    pub fn charsmap(&self) -> Option<&[u8]> {
        self.charsmap.as_deref()
    }

    /// The parsed value, for anything that wants to normalize with it directly.
    pub fn parsed(&self) -> &Precompiled {
        &self.parsed
    }
}

impl Normalizer for PrecompiledNormalizer {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        self.parsed.normalize(normalized)
    }
}

impl pipeline::Normalizer for PrecompiledNormalizer {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        pipeline::Normalizer::normalize(&self.parsed, input)
    }
}

fn replace(transformations: &mut Vec<(char, isize)>, old_part: &str, new_part: &str) {
    let old_count = old_part.chars().count() as isize;
    let new_count = new_part.chars().count() as isize;
    let diff = new_count - old_count;

    // If we are just replacing characters, all changes should be == 0
    transformations.extend(new_part.chars().map(|c| (c, 0)));

    match diff.cmp(&0) {
        // If we are adding some characters, the last DIFF characters should be == 1
        Ordering::Greater => {
            transformations
                .iter_mut()
                .rev()
                .take(diff as usize)
                .for_each(|(_, cs)| *cs = 1);
        }
        // If we are removing some characters, the last one should include the diff
        Ordering::Less => {
            if let Some((_, cs)) = transformations.last_mut() {
                *cs += diff;
            }
        }
        _ => {}
    }
}

impl Normalizer for Precompiled {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        let mut transformations = Vec::with_capacity(normalized.get().len());
        // Future reader. From @Narsil.
        // Yes, this is weird,
        // Yes, this seems broken
        // No, I don't know why Google did this.
        // If you question this code, check this normalizer against
        // XNLI database (all languages) with Unigram model against
        // Mbart, XLMRoberta *AND* Marian. If you don't get 100% or
        // break a single test.
        // You don't pass.
        let mut modified = false;
        normalized.get().graphemes(true).for_each(|grapheme| {
            if grapheme.len() < 6
                && let Some(norm) = self.transform(grapheme)
            {
                modified = true;
                replace(&mut transformations, grapheme, norm);
                return;
            }
            for (char_index, c) in grapheme.char_indices() {
                let part = &grapheme[char_index..char_index + c.len_utf8()];
                if let Some(norm) = self.transform(part) {
                    modified = true;
                    replace(&mut transformations, part, norm);
                } else {
                    transformations.push((c, 0));
                }
            }
        });
        if modified {
            normalized.transform(transformations, 0);
        }
        Ok(())
    }
}

impl pipeline::Normalizer for Precompiled {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
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

    #[test]
    fn expansion_followed_by_removal() {
        // Simulate transformations from "™\x1eg" to "TMg"
        let mut transformations = vec![];

        let mut n = NormalizedString::from("™\x1eg");
        replace(&mut transformations, "™", "TM");
        replace(&mut transformations, "\x1e", "");
        transformations.push(('g', 0));

        n.transform(transformations, 0);

        assert_eq!(n.get(), "TMg");
    }
}
