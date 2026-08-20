use std::borrow::Cow;

use crate::pipeline;
use crate::tokenizer::{NormalizedString, Normalizer, Result};
pub use spm_precompiled::Precompiled;
use std::cmp::Ordering;
use unicode_segmentation::UnicodeSegmentation;

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

    /// The only charsmap to test against is one read out of a `tokenizer.json`, so this needs a JSON
    /// parser and therefore the `serde` feature.
    #[cfg(feature = "serde")]
    fn albert_precompiled() -> Precompiled {
        let json = std::fs::read_to_string("../data/albert-base-v1-tokenizer.json").unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let precompiled = value["normalizer"]["normalizers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n["type"] == "Precompiled")
            .unwrap();
        // Precompiled can't deserialize through serde_json::Value (the base64 charsmap only decodes
        // via the string deserializer) -- same dance as NormalizerWrapper's Deserialize impl.
        serde_json::from_str(&serde_json::to_string(precompiled).unwrap()).unwrap()
    }

    #[test]
    #[cfg(feature = "serde")]
    fn pipeline_precompiled_matches_legacy() {
        let n = albert_precompiled();
        let mut any_modified = false;
        for input in &[
            "\u{2122}\x1eg",
            "\u{ff2b}\u{ff21}\u{ff24}\u{ff2f}\u{ff2b}\u{ff21}\u{ff37}\u{ff21}",
            "\u{ff11}\u{ff12}\u{ff13}",
            "\u{2026}",
            "\u{fb01}",
            "e\u{0301}",
            "\u{337f}",
            "abc def",
            "",
        ] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            any_modified |= ns.get() != *input;
            assert_eq!(
                ns.get(),
                &*pipeline::Normalizer::normalize(&n, input).unwrap(),
                "pipeline output diverges from legacy for {input:?}"
            );
        }
        // Guard against the oracle silently becoming a no-op on these inputs
        assert!(any_modified);
    }

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
