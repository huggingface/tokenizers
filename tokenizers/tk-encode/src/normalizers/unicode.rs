use std::borrow::Cow;

use crate::pipeline;
use crate::tokenizer::{NormalizedString, Normalizer, Result};
use crate::utils::macro_rules_attribute;

use unicode_normalization::{
    is_nfc_quick, is_nfd_quick, is_nfkc_quick, is_nfkd_quick, IsNormalized, UnicodeNormalization,
};

#[derive(Default, Copy, Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct NFD;
impl Normalizer for NFD {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.nfd();
        Ok(())
    }
}
impl pipeline::Normalizer for NFD {
    fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str> {
        if let IsNormalized::Yes = is_nfd_quick(input.chars()) {
            input.into()
        } else {
            Cow::Owned(input.nfd().collect())
        }
    }
}

#[derive(Default, Copy, Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct NFKD;
impl Normalizer for NFKD {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.nfkd();
        Ok(())
    }
}
impl pipeline::Normalizer for NFKD {
    fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str> {
        if let IsNormalized::Yes = is_nfkd_quick(input.chars()) {
            input.into()
        } else {
            Cow::Owned(input.nfkd().collect())
        }
    }
}

#[derive(Default, Copy, Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct NFC;
impl Normalizer for NFC {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.nfc();
        Ok(())
    }
}
impl pipeline::Normalizer for NFC {
    fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str> {
        if let IsNormalized::Yes = is_nfc_quick(input.chars()) {
            input.into()
        } else {
            Cow::Owned(input.nfc().collect())
        }
    }
}

#[derive(Default, Copy, Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct NFKC;
impl Normalizer for NFKC {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.nfkc();
        Ok(())
    }
}
impl pipeline::Normalizer for NFKC {
    fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str> {
        if let IsNormalized::Yes = is_nfkc_quick(input.chars()) {
            input.into()
        } else {
            Cow::Owned(input.nfkc().collect())
        }
    }
}

fn do_nmt(normalized: &mut NormalizedString) {
    // Ascii Control characters
    normalized
        .filter(|c| {
            !matches!(
                c as u32,
                0x0001..=0x0008 |
                0x000B |
                0x000E..=0x001F |
                0x007F |
                0x008F |
                0x009F
            )
        })
        // Other code points considered as whitespace.
        .map(|c| match c as u32 {
            0x0009 => ' ',
            0x000A => ' ',
            0x000C => ' ',
            0x000D => ' ',
            0x1680 => ' ',
            0x200B..=0x200F => ' ',
            0x2028 => ' ',
            0x2029 => ' ',
            0x2581 => ' ',
            0xFEFF => ' ',
            0xFFFD => ' ',
            _ => c,
        });
}

#[derive(Default, Copy, Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct Nmt;
impl Normalizer for Nmt {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        do_nmt(normalized);
        Ok(())
    }
}
impl pipeline::Normalizer for Nmt {
    fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str> {
        let normalized: String = input
            .chars()
            .filter(|&c| {
                !matches!(
                    c as u32,
                    0x0001..=0x0008 |
                    0x000B |
                    0x000E..=0x001F |
                    0x007F |
                    0x008F |
                    0x009F
                )
            })
            // Other code points considered as whitespace.
            .map(|c| match c as u32 {
                0x0009 => ' ',
                0x000A => ' ',
                0x000C => ' ',
                0x000D => ' ',
                0x1680 => ' ',
                0x200B..=0x200F => ' ',
                0x2028 => ' ',
                0x2029 => ' ',
                0x2581 => ' ',
                0xFEFF => ' ',
                0xFFFD => ' ',
                _ => c,
            })
            .collect();
        Cow::Owned(normalized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nfkc() {
        let original = "\u{fb01}".to_string();
        let normalized = "fi".to_string();
        let mut n = NormalizedString::from(original.clone());
        NFKC.normalize(&mut n).unwrap();

        assert_eq!(
            n,
            NormalizedString::new(original, normalized, vec![(0, 3), (0, 3)], 0)
        );

        assert_eq!(n.alignments_original(), vec![(0, 2), (0, 2), (0, 2)]);
    }

    #[test]
    fn pipeline_nfd_matches_legacy() {
        let n = NFD;
        for input in &["é", "café", "abc", "", "Å"] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            assert_eq!(ns.get(), &*pipeline::Normalizer::normalize(&n, input));
        }
    }

    #[test]
    fn pipeline_nfkd_matches_legacy() {
        let n = NFKD;
        for input in &["\u{fb01}", "²", "café", "abc", ""] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            assert_eq!(ns.get(), &*pipeline::Normalizer::normalize(&n, input));
        }
    }

    #[test]
    fn pipeline_nfc_matches_legacy() {
        let n = NFC;
        for input in &["e\u{0301}", "abc", "", "cafe\u{0301}"] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            assert_eq!(ns.get(), &*pipeline::Normalizer::normalize(&n, input));
        }
    }

    #[test]
    fn pipeline_nfkc_matches_legacy() {
        let n = NFKC;
        for input in &["\u{fb01}", "²", "e\u{0301}", "abc", ""] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            assert_eq!(ns.get(), &*pipeline::Normalizer::normalize(&n, input));
        }
    }

    #[test]
    fn pipeline_nmt_matches_legacy() {
        let n = Nmt;
        for input in &["a\tb", "x\u{200b}y", "abc", "", "\u{feff}hi", "c\u{0007}d"] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            assert_eq!(ns.get(), &*pipeline::Normalizer::normalize(&n, input));
        }
    }
}
