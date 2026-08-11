use std::borrow::Cow;

use crate::pipeline;
use crate::tokenizer::Result;
pub use spm_precompiled::Precompiled;
use unicode_segmentation::UnicodeSegmentation;

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
    use crate::normalizers::assert_normalizes;

    fn albert_precompiled() -> Precompiled {
        let json = std::fs::read_to_string("../data/albert-base-v1-tokenizer.json").unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let precompiled = value["normalizer"]["normalizers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n["type"] == "Precompiled")
            .unwrap();
        // Precompiled can't deserialize through serde_json::Value (the base64
        // charsmap only decodes via the string deserializer) — same dance as
        // NormalizerWrapper's Deserialize impl
        serde_json::from_str(&serde_json::to_string(precompiled).unwrap()).unwrap()
    }

    #[test]
    fn albert_charsmap_replacements() {
        assert_normalizes(
            &albert_precompiled(),
            &[
                // "™" expands to two chars and "\x1e" is dropped, in one grapheme run
                ("™\x1eg", "TMg"),
                ("ＫＡＤＯＫＡＷＡ", "KADOKAWA"),
                ("１２３", "123"),
                ("…", "..."),
                ("\u{fb01}", "fi"),
                ("e\u{0301}", "é"),
                // One grapheme standing for four characters
                ("㍿", "株式会社"),
                ("abc def", "abc def"),
                ("", ""),
            ],
        );
    }
}
