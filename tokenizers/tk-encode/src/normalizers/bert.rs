use std::borrow::Cow;

use crate::{
    pipeline,
    tokenizer::{NormalizedString, Normalizer, Result},
};

use serde::{Deserialize, Serialize};
use unicode_categories::UnicodeCategories;
use unicode_normalization::{is_nfd_quick, IsNormalized, UnicodeNormalization};

/// Checks whether a character is whitespace
fn is_whitespace(c: char) -> bool {
    // These are technically control characters but we count them as whitespace
    match c {
        '\t' | '\n' | '\r' => true,
        _ => c.is_whitespace(),
    }
}

/// Checks whether a character is a control character
fn is_control(c: char) -> bool {
    // These are technically control characters but we count them as whitespace
    match c {
        '\t' | '\n' | '\r' => false,
        // The definition of `is_control` here is quite large and contains also
        // Cc, Cf, Cn or Co
        // cf. https://unicode.org/reports/tr44/ (Table 12)
        _ => c.is_other(),
    }
}

/// Whether lowercasing `c` leaves it unchanged (a single, identical char)
fn lowercases_to_self(c: char) -> bool {
    let mut it = c.to_lowercase();
    matches!((it.next(), it.next()), (Some(first), None) if first == c)
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
    matches!(
        c as usize,
        0x4E00..=0x9FFF |
        0x3400..=0x4DBF |
        0x20000..=0x2A6DF |
        0x2A700..=0x2B73F |
        0x2B740..=0x2B81F |
        0x2B920..=0x2CEAF |
        0xF900..=0xFAFF |
        0x2F800..=0x2FA1F
    )
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
        Self {
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
                new_chars.extend([(' ', 0), (c, 1), (' ', 1)]);
            } else {
                new_chars.push((c, 0));
            }
        });
        normalized.transform(new_chars, 0);
    }

    fn do_strip_accents(&self, normalized: &mut NormalizedString) {
        normalized.nfd().filter(|c| !c.is_mark_nonspacing());
    }

    fn do_lowercase(&self, normalized: &mut NormalizedString) {
        normalized.lowercase();
    }

    fn is_noop(&self, input: &str, strip_accents: bool) -> bool {
        if strip_accents && !matches!(is_nfd_quick(input.chars()), IsNormalized::Yes) {
            return false;
        }
        let changes = |c: char| {
            (self.clean_text
                && (c as usize == 0
                    || c as usize == 0xfffd
                    || is_control(c)
                    || (is_whitespace(c) && c != ' ')))
                || (self.handle_chinese_chars && is_chinese_char(c))
                || (strip_accents && c.is_mark_nonspacing())
                || (self.lowercase && !lowercases_to_self(c))
        };
        !input.chars().any(changes)
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

impl pipeline::Normalizer for BertNormalizer {
    fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str> {
        let strip_accents = self.strip_accents.unwrap_or(self.lowercase);

        if self.is_noop(input, strip_accents) {
           return input.into()
        }

        let cleaned: String = input
            .chars()
            .filter(|&c| {
                !(self.clean_text && (c as usize == 0 || c as usize == 0xfffd || is_control(c)))
            })
            .flat_map(|c| {
                let c = if self.clean_text && is_whitespace(c) {
                    ' '
                } else {
                    c
                };
                if self.handle_chinese_chars && is_chinese_char(c) {
                    [Some(' '), Some(c), Some(' ')]
                } else {
                    [Some(c), None, None]
                }
            })
            .flatten()
            .collect();

        let stripped = if strip_accents {
            cleaned.nfd().filter(|c| !c.is_mark_nonspacing()).collect()
        } else {
            cleaned
        };

        let lowered = if self.lowercase {
            stripped.chars().flat_map(char::to_lowercase).collect()
        } else {
            stripped
        };

        Cow::Owned(lowered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INPUTS: &[&str] = &[
        "Héllo World",
        "中文字",
        "a中b文c",
        "  spaced  ",
        "abc",
        "",
        "\tTab\n\r",
        "MiXeD Café",
        "e\u{0301}",           // already-NFD combining acute
        "\u{fb01}ligature",    // NFKC ligature (unchanged by NFD)
        "null\0here",
        "repl\u{fffd}char",
        "ctrl\u{0007}char",
        "ǅ",                   // titlecase, lowercases to multi-mapping
        "İstanbul",            // dotted capital I: lowercases to 2 chars
        "straße",
    ];

    #[test]
    fn pipeline_bert_matches_legacy() {
        for &clean_text in &[true, false] {
            for &handle_chinese_chars in &[true, false] {
                for &strip_accents in &[None, Some(true), Some(false)] {
                    for &lowercase in &[true, false] {
                        let n = BertNormalizer::new(
                            clean_text,
                            handle_chinese_chars,
                            strip_accents,
                            lowercase,
                        );
                        for input in INPUTS {
                            let mut ns = NormalizedString::from(*input);
                            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
                            assert_eq!(
                                ns.get(),
                                &*pipeline::Normalizer::normalize(&n, input),
                                "config={n:?} input={input:?}",
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn pipeline_bert_borrows_when_noop() {
        let n = BertNormalizer::default();
        for input in &["hello world", "already lowercase ascii", ""] {
            assert!(matches!(
                pipeline::Normalizer::normalize(&n, input),
                Cow::Borrowed(_)
            ));
        }
        // Anything that must change is owned.
        for input in &["Héllo", "中", "\tx", "ABC"] {
            assert!(matches!(
                pipeline::Normalizer::normalize(&n, input),
                Cow::Owned(_)
            ));
        }
    }
}
