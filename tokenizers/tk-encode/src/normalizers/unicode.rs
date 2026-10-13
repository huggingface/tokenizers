use std::borrow::Cow;
use std::sync::LazyLock;

use crate::normalizers::utils::{Rewriter, unstable_spans_iter};
use crate::tokenizer::Result;
use crate::{normalizers::utils::CharacterCache, pipeline};

use unicode_normalization::char::canonical_combining_class;
use unicode_normalization::{
    IsNormalized, UnicodeNormalization, is_nfc, is_nfc_quick, is_nfd, is_nfd_quick, is_nfkc,
    is_nfkc_quick, is_nfkd, is_nfkd_quick,
};

// The quick check, not `is_normalized`: alone, a Hangul trailing consonant is in NFC, yet it
// composes with the syllable before it, and only its `Maybe` answer says so.
fn is_unstable_char(form: NormalizedForm, c: char) -> bool {
    let quick = match form {
        NormalizedForm::NFC => is_nfc_quick(std::iter::once(c)),
        NormalizedForm::NFD => is_nfd_quick(std::iter::once(c)),
        NormalizedForm::NFKC => is_nfkc_quick(std::iter::once(c)),
        NormalizedForm::NFKD => is_nfkd_quick(std::iter::once(c)),
    };
    canonical_combining_class(c) != 0 || !matches!(quick, IsNormalized::Yes)
}

static NFC_CACHE: LazyLock<CharacterCache> =
    LazyLock::new(|| CharacterCache::new(|c| is_unstable_char(NormalizedForm::NFC, c)));

static NFD_CACHE: LazyLock<CharacterCache> =
    LazyLock::new(|| CharacterCache::new(|c| is_unstable_char(NormalizedForm::NFD, c)));

static NFKC_CACHE: LazyLock<CharacterCache> =
    LazyLock::new(|| CharacterCache::new(|c| is_unstable_char(NormalizedForm::NFKC, c)));

static NFKD_CACHE: LazyLock<CharacterCache> =
    LazyLock::new(|| CharacterCache::new(|c| is_unstable_char(NormalizedForm::NFKD, c)));

#[derive(Clone, Copy)]
enum NormalizedForm {
    NFD,
    NFC,
    NFKD,
    NFKC,
}

impl NormalizedForm {
    fn is_normalized(self, text: &str) -> bool {
        match self {
            Self::NFC => is_nfc(text),
            Self::NFD => is_nfd(text),
            Self::NFKC => is_nfkc(text),
            Self::NFKD => is_nfkd(text),
        }
    }

    fn normalize_into(self, input: &str, output: &mut String) {
        match self {
            Self::NFC => output.extend(input.nfc()),
            Self::NFD => output.extend(input.nfd()),
            Self::NFKC => output.extend(input.nfkc()),
            Self::NFKD => output.extend(input.nfkd()),
        };
    }

    fn unstable_cache(self) -> &'static CharacterCache {
        match self {
            NormalizedForm::NFC => &NFC_CACHE,
            NormalizedForm::NFD => &NFD_CACHE,
            NormalizedForm::NFKC => &NFKC_CACHE,
            NormalizedForm::NFKD => &NFKD_CACHE,
        }
    }

    fn needs_previous_stable(self) -> bool {
        matches!(self, Self::NFC | Self::NFKC)
    }
}

fn normalize_form<'a>(form: NormalizedForm, input: &'a str) -> Cow<'a, str> {
    let mut rewriter = Rewriter::new(input);
    for unstable_span in
        unstable_spans_iter(input, form.unstable_cache(), form.needs_previous_stable())
    {
        let unstable_text = &input[unstable_span.clone()];
        if !form.is_normalized(unstable_text) {
            form.normalize_into(unstable_text, rewriter.rewrite(unstable_span));
        }
    }
    rewriter.finish()
}

#[derive(Default, Copy, Clone, Debug)]
pub struct NFD;
impl pipeline::Normalizer for NFD {
    fn normalize<'a>(&self, input: &'a str, _offset: usize) -> Result<Cow<'a, str>> {
        Ok(normalize_form(NormalizedForm::NFD, input))
    }
}

#[derive(Default, Copy, Clone, Debug)]
pub struct NFKD;
impl pipeline::Normalizer for NFKD {
    fn normalize<'a>(&self, input: &'a str, _offset: usize) -> Result<Cow<'a, str>> {
        Ok(normalize_form(NormalizedForm::NFKD, input))
    }
}

#[derive(Default, Copy, Clone, Debug)]
pub struct NFC;
impl pipeline::Normalizer for NFC {
    fn normalize<'a>(&self, input: &'a str, _offset: usize) -> Result<Cow<'a, str>> {
        Ok(normalize_form(NormalizedForm::NFC, input))
    }
}

#[derive(Default, Copy, Clone, Debug)]
pub struct NFKC;
impl pipeline::Normalizer for NFKC {
    fn normalize<'a>(&self, input: &'a str, _offset: usize) -> Result<Cow<'a, str>> {
        Ok(normalize_form(NormalizedForm::NFKC, input))
    }
}

/// Control characters the NMT normalizer removes
fn nmt_removes(c: char) -> bool {
    matches!(
        c as u32,
        0x0001..=0x0008 |
        0x000B |
        0x000E..=0x001F |
        0x007F |
        0x008F |
        0x009F
    )
}

/// Code points the NMT normalizer considers whitespace, folded to ' '
fn nmt_to_space(c: char) -> char {
    match c as u32 {
        0x0009
        | 0x000A
        | 0x000C
        | 0x000D
        | 0x1680
        | 0x200B..=0x200F
        | 0x2028
        | 0x2029
        | 0x2581
        | 0xFEFF
        | 0xFFFD => ' ',
        _ => c,
    }
}

#[derive(Default, Copy, Clone, Debug)]
pub struct Nmt;

impl pipeline::Normalizer for Nmt {
    fn normalize<'a>(&self, input: &'a str, _offset: usize) -> Result<Cow<'a, str>> {
        let mut rewriter = Rewriter::new(input);
        // XXX: could leverage SIMD to avoid scanning every single char
        for (i, c) in input.char_indices() {
            let range = i..i + c.len_utf8();
            if nmt_removes(c) {
                rewriter.rewrite(range);
            } else if nmt_to_space(c) != c {
                rewriter.rewrite(range).push(' ');
            }
        }
        Ok(rewriter.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalizers::utils::random_texts;

    /// Expectations come from Python's `unicodedata.normalize` -- an implementation independent
    /// of both this crate and the legacy normalizer this test used to compare against, so it
    /// pins NFD instead of just agreeing with another of our own code paths.
    #[test]
    fn pipeline_nfd() {
        for (input, expected) in [
            ("\u{e9}", "e\u{301}"),
            ("caf\u{e9}", "cafe\u{301}"),
            ("abc", "abc"),
            ("", ""),
            ("\u{c5}", "A\u{30a}"),
        ] {
            assert_eq!(
                &*pipeline::Normalizer::normalize(&NFD, input, 0).unwrap(),
                expected,
                "input={input:?}"
            );
        }
    }

    /// Expectations come from Python's `unicodedata.normalize` -- an implementation independent
    /// of both this crate and the legacy normalizer this test used to compare against, so it
    /// pins NFKD instead of just agreeing with another of our own code paths.
    #[test]
    fn pipeline_nfkd() {
        for (input, expected) in [
            ("\u{fb01}", "fi"),
            ("\u{b2}", "2"),
            ("caf\u{e9}", "cafe\u{301}"),
            ("abc", "abc"),
            ("", ""),
        ] {
            assert_eq!(
                &*pipeline::Normalizer::normalize(&NFKD, input, 0).unwrap(),
                expected,
                "input={input:?}"
            );
        }
    }

    /// Expectations come from Python's `unicodedata.normalize` -- an implementation independent
    /// of both this crate and the legacy normalizer this test used to compare against, so it
    /// pins NFC instead of just agreeing with another of our own code paths.
    #[test]
    fn pipeline_nfc() {
        for (input, expected) in [
            ("e\u{301}", "\u{e9}"),
            ("abc", "abc"),
            ("", ""),
            ("cafe\u{301}", "caf\u{e9}"),
        ] {
            assert_eq!(
                &*pipeline::Normalizer::normalize(&NFC, input, 0).unwrap(),
                expected,
                "input={input:?}"
            );
        }
    }

    /// Expectations come from Python's `unicodedata.normalize` -- an implementation independent
    /// of both this crate and the legacy normalizer this test used to compare against, so it
    /// pins NFKC instead of just agreeing with another of our own code paths.
    #[test]
    fn pipeline_nfkc() {
        for (input, expected) in [
            ("\u{fb01}", "fi"),
            ("\u{b2}", "2"),
            ("e\u{301}", "\u{e9}"),
            ("abc", "abc"),
            ("", ""),
        ] {
            assert_eq!(
                &*pipeline::Normalizer::normalize(&NFKC, input, 0).unwrap(),
                expected,
                "input={input:?}"
            );
        }
    }

    /// Expected values were captured from the legacy `NormalizedString` normalizer this test used
    /// to compare against, on the commit that removed it -- so they still pin exactly what the two
    /// implementations agreed on, without keeping the legacy code alive to ask.
    #[test]
    fn pipeline_nmt() {
        let n = Nmt;
        for (input, expected) in [
            ("a\tb", "a b"),
            ("x\u{200b}y", "x y"),
            ("abc", "abc"),
            ("", ""),
            ("\u{feff}hi", " hi"),
            ("c\u{7}d", "cd"),
        ] {
            assert_eq!(
                &*pipeline::Normalizer::normalize(&n, input, 0).unwrap(),
                expected,
                "input={input:?}"
            );
        }
    }

    fn forms(text: &str) -> [(&'static str, Cow<'_, str>, String); 4] {
        let run = |n: &dyn pipeline::Normalizer| n.normalize(text, 0).unwrap();
        [
            ("NFD", run(&NFD), text.nfd().collect()),
            ("NFKD", run(&NFKD), text.nfkd().collect()),
            ("NFC", run(&NFC), text.nfc().collect()),
            ("NFKC", run(&NFKC), text.nfkc().collect()),
        ]
    }

    #[test]
    fn every_character_normalizes_like_the_whole_string_form() {
        let long_ascii = "abcdefghijklmnopqrstuvwxyz".repeat(2);
        let code_points = (0..0x10000u32).chain((0x10000..0x110000).step_by(7));
        for c in code_points.filter_map(char::from_u32) {
            for text in [
                c.to_string(),
                format!("e{c}\u{301}"),
                format!("{long_ascii}{c}\u{323}{long_ascii}"),
            ] {
                for (name, got, expected) in forms(&text) {
                    assert_eq!(got, expected, "{name} on {text:?}");
                }
            }
        }
    }

    #[test]
    fn nfc_composes_with_the_ascii_starter_before_a_segment() {
        let text = "a".repeat(40) + "e\u{301}";
        for (name, got, expected) in forms(&text) {
            assert_eq!(got, expected, "{name} on {text:?}");
        }
    }

    #[test]
    fn mixed_scripts_normalize_like_the_whole_string_form() {
        let pieces = [
            "abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz",
            "a",
            " ",
            "e",
            "q",
            "\u{301}",
            "\u{323}",
            "\u{323}\u{301}",
            "\u{e9}",
            "\u{419}",
            "\u{438}\u{306}",
            "\u{1100}\u{1161}\u{11a8}",
            "\u{1161}",
            "\u{11a8}",
            "\u{ac00}",
            "\u{fb01}",
            "\u{212b}",
            "\u{a0}",
            "\u{2460}",
            "\u{1e9b}\u{323}",
            "\u{5d0}\u{5b0}\u{5b4}",
            "\u{915}\u{93c}\u{94d}",
            "\u{9c7}\u{9be}",
            "\u{4e2d}",
            "\u{1d15f}\u{1d16e}",
            "\u{1f600}",
            "\u{2f800}",
        ];
        for text in random_texts(9, &pieces, 4000, 120) {
            for (name, got, expected) in forms(&text) {
                assert_eq!(got, expected, "{name} on {text:?}");
            }
        }
    }

    #[test]
    fn normalized_text_comes_back_borrowed() {
        let candidates = [
            "",
            "abc",
            "caf\u{e9}",
            "cafe\u{301}",
            "q\u{301}",
            "\u{915}\u{93c}",
            "\u{41f}\u{440}\u{438}\u{432}\u{435}\u{442} \u{43c}\u{438}\u{440}",
            "\u{ac00}",
            "\u{1100}\u{1161}",
            "\u{4e2d}\u{6587}",
        ];
        for text in candidates {
            for (name, got, expected) in forms(text) {
                if expected == text {
                    assert!(matches!(got, Cow::Borrowed(_)), "{name} on {text:?}");
                }
            }
        }
    }
}
