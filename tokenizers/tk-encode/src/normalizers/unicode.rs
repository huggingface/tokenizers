use std::borrow::Cow;

use crate::pipeline;
use crate::tokenizer::Result;

use unicode_normalization::{
    IsNormalized, UnicodeNormalization, is_nfc_quick, is_nfd_quick, is_nfkc_quick, is_nfkd_quick,
};

#[derive(Default, Copy, Clone, Debug)]
pub struct NFD;
impl pipeline::Normalizer for NFD {
    fn normalize<'a>(&self, input: &'a str, _is_sequence_start: bool) -> Result<Cow<'a, str>> {
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
    fn normalize<'a>(&self, input: &'a str, _is_sequence_start: bool) -> Result<Cow<'a, str>> {
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
    fn normalize<'a>(&self, input: &'a str, _is_sequence_start: bool) -> Result<Cow<'a, str>> {
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
    fn normalize<'a>(&self, input: &'a str, _is_sequence_start: bool) -> Result<Cow<'a, str>> {
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
    fn normalize<'a>(&self, input: &'a str, _is_sequence_start: bool) -> Result<Cow<'a, str>> {
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
                &*pipeline::Normalizer::normalize(&NFD, input, true).unwrap(),
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
                &*pipeline::Normalizer::normalize(&NFKD, input, true).unwrap(),
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
                &*pipeline::Normalizer::normalize(&NFC, input, true).unwrap(),
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
                &*pipeline::Normalizer::normalize(&NFKC, input, true).unwrap(),
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
                &*pipeline::Normalizer::normalize(&n, input, true).unwrap(),
                expected,
                "input={input:?}"
            );
        }
    }
}
