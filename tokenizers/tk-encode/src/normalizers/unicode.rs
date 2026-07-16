use std::borrow::Cow;

use crate::pipeline;
use crate::tokenizer::{NormalizedString, Normalizer, Result};
use crate::utils::macro_rules_attribute;

// All four forms (NFD/NFKD/NFC/NFKC) are pure-Rust via `atomnorm`; the legacy `Normalizer` impls
// keep `NormalizedString`'s own `nfd()`/`nfc()`/… (alignment-tracking) for the non-pipeline path.

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
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        // Pure-Rust SIMD NFD: no classify pass, no per-byte tag buffer. Bulk-skips NFD-stable runs
        // (bitset probe over vld2/vld3 deinterleave) and decomposes the rest from a baked table.
        Ok(atomnorm::nfd(input))
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
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        // Pure-Rust SIMD compatibility decomposition — same skip-based kernel as NFD, NFKD tables.
        Ok(atomnorm::nfkd(input))
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
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        // Pure-Rust NFC: a quick-check relevance bitset borrows already-composed input untouched;
        // otherwise decompose (NFD tables) then canonically recompose from the baked COMPOSE table.
        Ok(atomnorm::nfc(input))
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
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        // Pure-Rust NFKC: NFKC relevance bitset borrow gate, else NFKD decompose + canonical recompose.
        Ok(atomnorm::nfkc(input))
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

fn do_nmt(normalized: &mut NormalizedString) {
    normalized.filter(|c| !nmt_removes(c)).map(nmt_to_space);
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
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        // atomnorm scan over the baked NMT remove/fold sets; borrows when clean
        Ok(atomnorm::nmt(input))
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
        for input in &[
            "é",
            "café",
            "abc",
            "",
            "Å",
            "한국어 테스트",          // Hangul → arithmetic decompose
            "a\u{0323}\u{0301}",      // combining marks needing canonical reorder (ccc 220 vs 230)
            "Ἀρχαία ἑλληνικά",        // polytonic Greek (3-byte decomposers)
            "这是中文",               // CJK, all NFD-stable → borrowed unchanged
            "mixed 世界 café Москва", // multi-script (ASCII + CJK + Latin-accent + Cyrillic)
            "\u{2F800}",              // astral CJK-compat ideograph
        ] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            assert_eq!(
                ns.get(),
                &*pipeline::Normalizer::normalize(&n, input).unwrap()
            );
        }
    }

    #[test]
    fn pipeline_nfkd_matches_legacy() {
        let n = NFKD;
        for input in &[
            "\u{fb01}",
            "²",
            "café",
            "abc",
            "",
            "한국어",
            "½ ﷺ ㍿",
            "Ấ ṩ",
            "mixed 世界 ﬁ Москва",
        ] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            assert_eq!(
                ns.get(),
                &*pipeline::Normalizer::normalize(&n, input).unwrap()
            );
        }
    }

    #[test]
    fn pipeline_nfc_matches_legacy() {
        let n = NFC;
        for input in &[
            "e\u{0301}",
            "abc",
            "",
            "cafe\u{0301}",
            "A\u{0302}\u{0301}",        // nested compose → Ấ
            "\u{1100}\u{1161}\u{11A8}", // jamo → Hangul syllable
            "a\u{0323}\u{0301}",
            "café",
            "这是中文",
            "мир",
        ] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            assert_eq!(
                ns.get(),
                &*pipeline::Normalizer::normalize(&n, input).unwrap()
            );
        }
    }

    #[test]
    fn pipeline_nfkc_matches_legacy() {
        let n = NFKC;
        for input in &[
            "\u{fb01}",
            "²",
            "e\u{0301}",
            "abc",
            "",
            "Ấ ṩ",
            "½ ﷺ",
            "한국어",
            "A\u{0302}\u{0301}",
            "mixed 世界 ﬁ",
        ] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            assert_eq!(
                ns.get(),
                &*pipeline::Normalizer::normalize(&n, input).unwrap()
            );
        }
    }

    #[test]
    fn pipeline_nmt_matches_legacy() {
        let n = Nmt;
        for input in &["a\tb", "x\u{200b}y", "abc", "", "\u{feff}hi", "c\u{0007}d"] {
            let mut ns = NormalizedString::from(*input);
            Normalizer::normalize(&n, &mut ns).unwrap(); // legacy oracle
            assert_eq!(
                ns.get(),
                &*pipeline::Normalizer::normalize(&n, input).unwrap()
            );
        }
    }
}
