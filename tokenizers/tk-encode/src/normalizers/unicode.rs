use std::borrow::Cow;

use crate::pipeline;
use crate::tokenizer::Result;

use unicode_general_category::{GeneralCategory, get_general_category};
use unicode_normalization::{
    IsNormalized, UnicodeNormalization, is_nfc_quick, is_nfd_quick, is_nfkc_quick, is_nfkd_quick,
};

#[derive(Default, Copy, Clone, Debug)]
pub struct NFD;
impl pipeline::Normalizer for NFD {
    fn normalize<'a>(&self, input: &'a str, _offset: usize) -> Result<Cow<'a, str>> {
        if let IsNormalized::Yes = is_nfd_quick(input.chars()) {
            Ok(input.into())
        } else {
            Ok(Cow::Owned(input.nfd().collect()))
        }
    }
}

#[derive(Default, Copy, Clone, Debug)]
pub struct NFKD;
impl pipeline::Normalizer for NFKD {
    fn normalize<'a>(&self, input: &'a str, _offset: usize) -> Result<Cow<'a, str>> {
        if let IsNormalized::Yes = is_nfkd_quick(input.chars()) {
            Ok(input.into())
        } else {
            Ok(Cow::Owned(input.nfkd().collect()))
        }
    }
}

#[derive(Default, Copy, Clone, Debug)]
pub struct NFC;
impl pipeline::Normalizer for NFC {
    fn normalize<'a>(&self, input: &'a str, _offset: usize) -> Result<Cow<'a, str>> {
        if let IsNormalized::Yes = is_nfc_quick(input.chars()) {
            Ok(input.into())
        } else {
            Ok(Cow::Owned(input.nfc().collect()))
        }
    }
}

#[derive(Default, Copy, Clone, Debug)]
pub struct NFKC;
impl pipeline::Normalizer for NFKC {
    fn normalize<'a>(&self, input: &'a str, _offset: usize) -> Result<Cow<'a, str>> {
        if let IsNormalized::Yes = is_nfkc_quick(input.chars()) {
            Ok(input.into())
        } else {
            Ok(Cow::Owned(input.nfkc().collect()))
        }
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
        if input
            .chars()
            .any(|c| nmt_removes(c) || nmt_to_space(c) != c)
        {
            let normalized: String = input
                .chars()
                .filter(|&c| !nmt_removes(c))
                .map(nmt_to_space)
                .collect();
            return Ok(Cow::Owned(normalized));
        }

        Ok(input.into())
    }
}

/// Drops code points by general category, so text carrying characters no vocabulary can have a
/// meaning for (unassigned code points, private use areas) never reaches the model.
///
/// Categories come from the Unicode 16.0 tables of the `unicode-general-category` crate. With
/// `filter_unassigned`, a character first assigned in a later Unicode version (such as U+20C1,
/// added in 17.0) counts as unassigned and is removed.
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub struct UnicodeFilter {
    pub filter_unassigned: bool,
    pub filter_private_use: bool,
}

impl UnicodeFilter {
    pub fn new(filter_unassigned: bool, filter_private_use: bool) -> Self {
        Self {
            filter_unassigned,
            filter_private_use,
        }
    }

    fn removes(&self, c: char) -> bool {
        match get_general_category(c) {
            GeneralCategory::Unassigned => self.filter_unassigned,
            GeneralCategory::PrivateUse => self.filter_private_use,
            _ => false,
        }
    }
}

impl Default for UnicodeFilter {
    fn default() -> Self {
        Self::new(true, true)
    }
}

impl pipeline::Normalizer for UnicodeFilter {
    fn normalize<'a>(&self, input: &'a str, _offset: usize) -> Result<Cow<'a, str>> {
        if input.chars().any(|c| self.removes(c)) {
            return Ok(Cow::Owned(
                input.chars().filter(|&c| !self.removes(c)).collect(),
            ));
        }
        Ok(input.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn pipeline_unicode_filter() {
        // Euro sign (assigned), U+10FFFF (unassigned), U+E000 (private use, BMP) and U+F0000
        // (private use, plane 15).
        let input = "A\u{20AC}\u{10FFFF}\u{E000}\u{F0000}";
        for (filter, expected) in [
            (UnicodeFilter::default(), "A\u{20AC}"),
            (
                UnicodeFilter::new(true, false),
                "A\u{20AC}\u{E000}\u{F0000}",
            ),
            (UnicodeFilter::new(false, true), "A\u{20AC}\u{10FFFF}"),
            (UnicodeFilter::new(false, false), input),
        ] {
            assert_eq!(
                &*pipeline::Normalizer::normalize(&filter, input, 0).unwrap(),
                expected,
                "filter={filter:?}"
            );
        }
        let clean = UnicodeFilter::default();
        assert!(matches!(
            pipeline::Normalizer::normalize(&clean, "abc", 0).unwrap(),
            Cow::Borrowed("abc")
        ));
    }

    /// U+1FAE9 FACE WITH BAGS UNDER EYES was assigned in Unicode 16.0, after the 15.0 tables of
    /// `unicode-general-category` 0.6, and must survive.
    #[test]
    fn unicode_filter_keeps_recent_assignments() {
        // U+0378 is unassigned in every Unicode version.
        let input = "\u{1FAE9}\u{378}\u{E000}";
        assert_eq!(
            &*pipeline::Normalizer::normalize(&UnicodeFilter::default(), input, 0).unwrap(),
            "\u{1FAE9}"
        );
    }
}
