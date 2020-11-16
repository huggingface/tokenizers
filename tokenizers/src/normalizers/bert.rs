use crate::tokenizer::{NormalizedString, Normalizer, Result};

use serde::{Deserialize, Serialize};
use unicode_categories::UnicodeCategories;

/// Checks whether a character is whitespace
fn is_whitespace(c: char) -> bool {
    // These are technically control characters but we count them as whitespace
    if c == '\t' || c == '\n' || c == '\r' {
        true
    } else {
        c.is_whitespace()
    }
}

/// Checks whether a character is a control character
fn is_control(c: char) -> bool {
    // These are technically control characters but we count them as whitespace
    if c == '\t' || c == '\n' || c == '\r' {
        false
    } else {
        // The definition of `is_control` here is quite large and contains also
        // Cc, Cf, Cn or Co
        // cf. https://unicode.org/reports/tr44/ (Table 12)
        c.is_other()
    }
}

/// Checks whether a character is chinese
/// This defines a "chinese character" as anything in the CJK Unicode block:
///   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
///
/// Note that the CJK Unicode block is NOT all Japanese and Korean characters,
/// despite its name. The modern Korean Hangul alphabet is a different block,
/// as is Japanese Hiragana and Katakana. Those alphabets are used to write
/// space-separated words, so they are not treated specially and handled
/// like for all of the other languages.
fn is_chinese_char(c: char) -> bool {
    match c as usize {
        0x4E00..=0x9FFF => true,
        0x3400..=0x4DBF => true,
        0x20000..=0x2A6DF => true,
        0x2A700..=0x2B73F => true,
        0x2B740..=0x2B81F => true,
        0x2B920..=0x2CEAF => true,
        0xF900..=0xFAFF => true,
        0x2F800..=0x2FA1F => true,
        _ => false,
    }
}

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
#[non_exhaustive]
pub struct BertNormalizer {
    /// Whether to do the bert basic cleaning:
    ///   1. Remove any control characters
    ///   2. Replace all sorts of whitespace by the classic one ` `
    pub clean_text: bool,
    /// Whether to put spaces around chinese characters so they get split
    pub handle_chinese_chars: bool,
    /// Whether to strip accents
    pub strip_accents: Option<bool>,
    /// Whether to lowercase the input
    pub lowercase: bool,
}

impl Default for BertNormalizer {
    fn default() -> Self {
        Self {
            clean_text: true,
            handle_chinese_chars: true,
            strip_accents: None,
            lowercase: true,
        }
    }
}

impl BertNormalizer {
    pub fn new(
        clean_text: bool,
        handle_chinese_chars: bool,
        strip_accents: Option<bool>,
        lowercase: bool,
    ) -> Self {
        BertNormalizer {
            clean_text,
            handle_chinese_chars,
            strip_accents,
            lowercase,
        }
    }

    fn do_clean_text(&self, normalized: &mut NormalizedString) {
        normalized
            .filter(|c| !(c as usize == 0 || c as usize == 0xfffd || is_control(c)))
            .map(|c| if is_whitespace(c) { ' ' } else { c });
    }

    fn do_handle_chinese_chars(&self, normalized: &mut NormalizedString) {
        let mut new_chars: Vec<(char, isize)> = vec![];
        normalized.for_each(|c| {
            if is_chinese_char(c) {
                new_chars.extend(&[(' ', 0), (c, 1), (' ', 1)]);
            } else {
                new_chars.push((c, 0));
            }
        });
        normalized.transform(new_chars.into_iter(), 0);
    }

    fn do_strip_accents(&self, normalized: &mut NormalizedString) {
        normalized.nfd().filter(|c| !c.is_mark_nonspacing());
    }

    fn do_lowercase(&self, normalized: &mut NormalizedString) {
        normalized.lowercase();
    }
}

impl Normalizer for BertNormalizer {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        if self.clean_text {
            self.do_clean_text(normalized);
        }
        if self.handle_chinese_chars {
            self.do_handle_chinese_chars(normalized);
        }
        let strip_accents = self.strip_accents.unwrap_or(self.lowercase);
        if strip_accents {
            self.do_strip_accents(normalized);
        }
        if self.lowercase {
            self.do_lowercase(normalized);
        }

        Ok(())
    }
}
