use std::{borrow::Cow, sync::OnceLock};

use crate::{
    normalizers::utils::{CharacterCache, Rewriter, unstable_spans_iter},
    pipeline,
    tokenizer::Result,
};

use super::lowercase::lowercases_to_self;

use unicode_categories::UnicodeCategories;
use unicode_normalization::{
    IsNormalized, UnicodeNormalization, char::canonical_combining_class, is_nfd_quick,
};

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

    fn unstable_cache(&self, strip_accents: bool) -> &'static CharacterCache {
        static CACHES: [OnceLock<CharacterCache>; 16] = [const { OnceLock::new() }; 16];
        let (clean_text, handle_chinese, lowercase) =
            (self.clean_text, self.handle_chinese_chars, self.lowercase);
        let index = clean_text as usize
            | (handle_chinese as usize) << 1
            | (strip_accents as usize) << 2
            | (lowercase as usize) << 3;
        CACHES[index].get_or_init(|| {
            CharacterCache::new(move |c| {
                (clean_text && (clean_text_removes(c) || clean_text_map(c) != c))
                    || (handle_chinese && is_chinese_char(c))
                    || (strip_accents
                        && (c.is_mark_nonspacing()
                            || canonical_combining_class(c) != 0
                            || !matches!(is_nfd_quick(std::iter::once(c)), IsNormalized::Yes)))
                    || (lowercase && !lowercases_to_self(c))
            })
        })
    }

    fn normalize_into(&self, input: &str, strip_accents: bool, out: &mut String) {
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
        // statically-typed chain
        match (strip_accents, self.lowercase) {
            (true, true) => out.extend(
                cleaned
                    .nfd()
                    .filter(|c| !c.is_mark_nonspacing())
                    .flat_map(char::to_lowercase),
            ),
            (true, false) => out.extend(cleaned.nfd().filter(|c| !c.is_mark_nonspacing())),
            (false, true) => out.extend(cleaned.flat_map(char::to_lowercase)),
            (false, false) => out.extend(cleaned),
        }
    }

    fn normalize_ascii<'a>(&self, input: Cow<'a, str>) -> Cow<'a, str> {
        let needs_change = |byte: u8| {
            // Control chars and spaces
            self.clean_text && (byte < 0x20 || byte == 0x7F)
            // Uppercase ascii: A-Z
                || (self.lowercase && byte.is_ascii_uppercase())
        };
        // Check for no-op
        if !input.bytes().any(needs_change) {
            return input;
        }
        // At least one byte needs to change: copy
        let mut out = String::with_capacity(input.len());
        for c in input.chars() {
            // lowercase ascii chars
            let c = if self.lowercase {
                c.to_ascii_lowercase()
            } else {
                c
            };
            // Clean controls and spaces, if needed
            match c {
                '\t' | '\n' | '\r' if self.clean_text => out.push(' '),
                '\0'..='\x1f' | '\x7f' if self.clean_text => {},
                _ => out.push(c),
            };
        }
        Cow::Owned(out)
    }
}

impl pipeline::Normalizer for BertNormalizer {
    fn normalize<'a>(&self, input: &'a str, _offset: usize) -> Result<Cow<'a, str>> {
        let strip_accents = self.strip_accents.unwrap_or(self.lowercase);

        // Pass 1: normalize non-ascii segments
        let mut rewriter = Rewriter::new(input);
        for span in unstable_spans_iter(input, self.unstable_cache(strip_accents), false) {
            let text = &input[span.clone()];
            self.normalize_into(text, strip_accents, rewriter.rewrite(span));
        }
        // Pass 2: Normalize ascii text
        // 1. clean_text: transform \t, \r and \n to ' ', drop other control bytes (eg 0x00-0x01, 0x7F)
        // 2. lowercase ASCII chars
        // chinese, nfd, strip accents: nothing to do as those were handled by the previous pass
        Ok(self.normalize_ascii(rewriter.finish()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalizers::utils::random_texts;

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
                            let out = pipeline::Normalizer::normalize(&n, input, 0).unwrap();
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

    fn configs() -> Vec<BertNormalizer> {
        let mut configs = Vec::new();
        for clean_text in [true, false] {
            for chinese in [true, false] {
                for strip_accents in [None, Some(true), Some(false)] {
                    for lowercase in [true, false] {
                        configs.push(BertNormalizer::new(
                            clean_text,
                            chinese,
                            strip_accents,
                            lowercase,
                        ));
                    }
                }
            }
        }
        configs
    }

    /// The per-character chain run on the whole string: what `normalize` must reproduce.
    fn chain(n: &BertNormalizer, text: &str) -> String {
        let mut out = String::new();
        n.normalize_into(text, n.strip_accents.unwrap_or(n.lowercase), &mut out);
        out
    }

    #[test]
    fn normalize_matches_the_char_chain_on_the_whole_string() {
        let pieces = [
            "the quick brown fox jumps over the lazy dog, THE QUICK BROWN FOX",
            "a",
            "Z",
            " ",
            "\0",
            "\u{7}",
            "\u{b}",
            "\u{1f}",
            "\u{7f}",
            "\t",
            "\n",
            "\r",
            "\u{e9}",
            "\u{c9}",
            "\u{df}",
            "\u{ff}",
            "e\u{301}",
            "E\u{301}",
            "\u{301}",
            "\u{323}\u{301}",
            "\u{301}\u{323}",
            "\u{1d165}\u{1d16e}",
            "\u{212a}",
            "\u{130}",
            "\u{1c5}",
            "\u{3a3}",
            "\u{3ac}",
            "\u{390}",
            "\u{416}",
            "\u{439}",
            "\u{5d0}\u{5b0}\u{5b4}",
            "\u{4e2d}\u{6587}",
            "\u{3000}",
            "\u{a0}",
            "\u{85}",
            "\u{ad}",
            "\u{fffd}",
            "\u{fb01}",
            "\u{1f600}",
        ];
        let texts = random_texts(7, &pieces, 1500, 160);
        for n in configs() {
            for text in &texts {
                let got = pipeline::Normalizer::normalize(&n, text, 0).unwrap();
                assert_eq!(got, chain(&n, text), "{n:?} on {text:?}");
            }
        }
    }

    #[test]
    fn ascii_stages_match_the_chain_at_every_block_offset() {
        let ascii = [
            "a", "Q", " ", "\t", "\n", "\r", "\0", "\u{1b}", "\u{7f}", "~",
        ];
        for n in configs() {
            for align in 0..16 {
                for text in random_texts(align as u64, &ascii, 20, 48) {
                    let text = "x".repeat(align) + &text + "\u{e9}";
                    let got = pipeline::Normalizer::normalize(&n, &text, 0).unwrap();
                    assert_eq!(got, chain(&n, &text), "{n:?} on {text:?}");
                }
            }
        }
    }

    #[test]
    fn unchanged_text_comes_back_borrowed() {
        let candidates = [
            "",
            "the quick brown fox jumps over the lazy dog",
            "The Quick Brown Fox",
            "caf\u{e9} au lait",
            "cafe\u{301}",
            "\u{43f}\u{440}\u{438}\u{432}\u{435}\u{442} \u{43c}\u{438}\u{440}",
            "\u{4e2d}\u{6587}",
            "\u{5d0}\u{5b0}",
        ];
        let mut checked = 0;
        for n in configs() {
            for text in candidates {
                if chain(&n, text) == text {
                    let got = pipeline::Normalizer::normalize(&n, text, 0).unwrap();
                    assert!(matches!(got, Cow::Borrowed(_)), "{n:?} on {text:?}");
                    checked += 1;
                }
            }
        }
        assert!(checked > 60, "only {checked} unchanged cases");
    }
}
