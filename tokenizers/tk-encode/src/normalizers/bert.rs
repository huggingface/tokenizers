use std::borrow::Cow;

use crate::{pipeline, tokenizer::Result};

use super::utils::lowercases_to_self;

use unicode_categories::UnicodeCategories;
use unicode_normalization::{IsNormalized, UnicodeNormalization, is_nfd_quick};

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

/// Whether BERT text cleaning removes `c` entirely
fn clean_text_removes(c: char) -> bool {
    c == '\0' || c == '\u{fffd}' || is_control(c)
}

/// The whitespace folding BERT text cleaning applies to kept characters
fn clean_text_map(c: char) -> char {
    if is_whitespace(c) { ' ' } else { c }
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

/// All four fields are required -- including `strip_accents`, which is an `Option<bool>` but has no
/// `#[serde(default)]`, so a config has to spell it, if only as `null`. That is not an oversight to
/// tidy: `null` and *absent* would mean the same thing to the type (`None`, meaning "follow
/// `lowercase`"), but they do not mean the same thing to serde, and `tk-serialize`'s slim reader
/// rejects the absent case with a message of its own.
///
/// `#[serde(tag = "type")]` on a struct of this name writes `"type":"BertNormalizer"`, which is what
/// every real config on disk says -- while `NormalizerWrapper`'s `EnumType` spells the variant
/// `Bert`. Both spellings load, and only because a bare `tag` attribute *ignores* the tag's value on
/// the way in. `bert_loads_under_both_tag_spellings` is the test that stops a tidy-up from giving
/// this a required tag and quietly breaking one of them.
#[derive(Copy, Clone, Debug)]
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

    fn is_noop(&self, input: &str, strip_accents: bool) -> bool {
        if strip_accents && !matches!(is_nfd_quick(input.chars()), IsNormalized::Yes) {
            return false;
        }
        let changes = |c: char| {
            (self.clean_text && (clean_text_removes(c) || clean_text_map(c) != c))
                || (self.handle_chinese_chars && is_chinese_char(c))
                || (strip_accents && c.is_mark_nonspacing())
                || (self.lowercase && !lowercases_to_self(c))
        };
        !input.chars().any(changes)
    }
}

impl pipeline::Normalizer for BertNormalizer {
    fn normalize<'a>(&self, input: &'a str, _is_sequence_start: bool) -> Result<Cow<'a, str>> {
        let strip_accents = self.strip_accents.unwrap_or(self.lowercase);

        if self.is_noop(input, strip_accents) {
            return Ok(input.into());
        }

        let cleaned = input
            .chars()
            .filter(|&c| !(self.clean_text && clean_text_removes(c)))
            .flat_map(|c| {
                let c = if self.clean_text {
                    clean_text_map(c)
                } else {
                    c
                };
                if self.handle_chinese_chars && is_chinese_char(c) {
                    [Some(' '), Some(c), Some(' ')]
                } else {
                    [Some(c), None, None]
                }
            })
            .flatten();

        // `.nfd()` changes the iterator's type, so the stage toggles can't be
        // plain `if`s mid-chain: each config combination gets its own
        // statically-typed chain, all funneled into one pre-sized String.
        let mut normalized = String::with_capacity(input.len());
        match (strip_accents, self.lowercase) {
            (true, true) => normalized.extend(
                cleaned
                    .nfd()
                    .filter(|c| !c.is_mark_nonspacing())
                    .flat_map(char::to_lowercase),
            ),
            (true, false) => normalized.extend(cleaned.nfd().filter(|c| !c.is_mark_nonspacing())),
            (false, true) => normalized.extend(cleaned.flat_map(char::to_lowercase)),
            (false, false) => normalized.extend(cleaned),
        }

        Ok(Cow::Owned(normalized))
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
        "e\u{0301}",        // already-NFD combining acute
        "\u{fb01}ligature", // NFKC ligature (unchanged by NFD)
        "null\0here",
        "repl\u{fffd}char",
        "ctrl\u{0007}char",
        "ǅ",        // titlecase, lowercases to multi-mapping
        "İstanbul", // dotted capital I: lowercases to 2 chars
        "straße",
    ];

    /// The 24-config x 16-input sweep this used to run against the legacy `NormalizedString`
    /// normalizer, pinned as one FNV-1a digest over every `(config, output)` pair instead of 384
    /// literals. The value was captured from the legacy oracle's own output on the commit that
    /// removed it, so it still encodes exactly what the two implementations agreed on.
    ///
    /// If this fires, print the pairs in the loop below and diff them to find the cell that moved.
    #[test]
    fn pipeline_bert_sweep_digest() {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        let mut feed = |bytes: &[u8]| {
            for b in bytes {
                hash ^= u64::from(*b);
                hash = hash.wrapping_mul(0x100_0000_01b3);
            }
        };
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
                            let out = pipeline::Normalizer::normalize(&n, input, true).unwrap();
                            feed(format!("{n:?}").as_bytes());
                            feed(&[0x01]);
                            feed(out.as_bytes());
                            feed(&[0x02]);
                        }
                    }
                }
            }
        }
        assert_eq!(
            hash, 0x9cde_0fd5_9e74_731d,
            "sweep digest moved: {hash:#018x}"
        );
    }
}
