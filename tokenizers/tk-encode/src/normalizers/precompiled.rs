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
        Ok(Self {
            parsed,
            charsmap: Some(charsmap.into()),
        })
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

impl pipeline::Normalizer for PrecompiledNormalizer {
    fn normalize<'a>(&self, input: &'a str, is_sequence_start: bool) -> Result<Cow<'a, str>> {
        pipeline::Normalizer::normalize(&self.parsed, input, is_sequence_start)
    }
}

impl pipeline::Normalizer for Precompiled {
    fn normalize<'a>(&self, input: &'a str, _is_sequence_start: bool) -> Result<Cow<'a, str>> {
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
