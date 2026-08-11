use std::borrow::Cow;

use crate::pipeline;
use crate::tokenizer::{Normalizer, Result};
use crate::utils::macro_rules_attribute;

use unicode_normalization::{
    IsNormalized, UnicodeNormalization, is_nfc_quick, is_nfd_quick, is_nfkc_quick, is_nfkd_quick,
};

#[derive(Default, Copy, Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct NFD;
impl Normalizer for NFD {}
impl pipeline::Normalizer for NFD {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        if let IsNormalized::Yes = is_nfd_quick(input.chars()) {
            Ok(input.into())
        } else {
            Ok(Cow::Owned(input.nfd().collect()))
        }
    }
}

#[derive(Default, Copy, Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct NFKD;
impl Normalizer for NFKD {}
impl pipeline::Normalizer for NFKD {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        if let IsNormalized::Yes = is_nfkd_quick(input.chars()) {
            Ok(input.into())
        } else {
            Ok(Cow::Owned(input.nfkd().collect()))
        }
    }
}

#[derive(Default, Copy, Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct NFC;
impl Normalizer for NFC {}
impl pipeline::Normalizer for NFC {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        if let IsNormalized::Yes = is_nfc_quick(input.chars()) {
            Ok(input.into())
        } else {
            Ok(Cow::Owned(input.nfc().collect()))
        }
    }
}

#[derive(Default, Copy, Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct NFKC;
impl Normalizer for NFKC {}
impl pipeline::Normalizer for NFKC {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
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
#[macro_rules_attribute(impl_serde_type!)]
pub struct Nmt;
impl Normalizer for Nmt {}

impl pipeline::Normalizer for Nmt {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
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
    use crate::normalizers::assert_normalizes;

    #[test]
    fn nfd_decomposes() {
        assert_normalizes(
            &NFD,
            &[
                ("é", "e\u{301}"),
                ("café", "cafe\u{301}"),
                ("Å", "A\u{30a}"),
                ("abc", "abc"),
                ("", ""),
            ],
        );
    }

    #[test]
    fn nfkd_decomposes_compatibility_forms() {
        assert_normalizes(
            &NFKD,
            &[
                ("\u{fb01}", "fi"),
                ("²", "2"),
                ("café", "cafe\u{301}"),
                ("abc", "abc"),
                ("", ""),
            ],
        );
    }

    #[test]
    fn nfc_composes() {
        assert_normalizes(
            &NFC,
            &[
                ("e\u{301}", "é"),
                ("cafe\u{301}", "café"),
                ("abc", "abc"),
                ("", ""),
            ],
        );
    }

    #[test]
    fn nfkc_composes_compatibility_forms() {
        assert_normalizes(
            &NFKC,
            &[
                ("\u{fb01}", "fi"),
                ("²", "2"),
                ("e\u{301}", "é"),
                ("abc", "abc"),
                ("", ""),
            ],
        );
    }

    #[test]
    fn nmt_folds_whitespace_and_drops_controls() {
        assert_normalizes(
            &Nmt,
            &[
                ("a\tb", "a b"),
                ("x\u{200b}y", "x y"),
                ("\u{feff}hi", " hi"),
                ("c\u{0007}d", "cd"),
                ("abc", "abc"),
                ("", ""),
            ],
        );
    }
}
