use crate::tokenizer::{NormalizedString, Normalizer, Result};
use crate::utils::macro_rules_attribute;

#[derive(Default, Copy, Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct NFD;
impl Normalizer for NFD {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.nfd();
        Ok(())
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

#[derive(Default, Copy, Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct NFC;
impl Normalizer for NFC {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        // ASCII strings are NFC by definition (U+0000..=U+007F have no
        // decomposition or composition mappings), so we can skip the
        // per-`char` Unicode pass and the alignments rebuild it triggers.
        // Any non-ASCII byte falls through to the original path unchanged.
        if normalized.get().is_ascii() {
            return Ok(());
        }
        normalized.nfc();
        Ok(())
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
    fn nfc_ascii_fast_path_is_no_op() {
        // After an NFKD step expands a ligature, `normalized` is all-ASCII but
        // `alignments` is non-trivial (each output byte still maps back to the
        // 3-byte ligature). NFC over ASCII must leave every field untouched.
        let mut n = NormalizedString::from("\u{fb00}");
        n.nfkd();
        assert!(n.get().is_ascii());

        let before = n.clone();
        NFC.normalize(&mut n).unwrap();
        assert_eq!(n, before);
    }

    #[test]
    fn nfc_non_ascii_still_runs_unicode_path() {
        // A combining-mark sequence ("e" + COMBINING ACUTE) must still be
        // composed to "é" by the original NFC path; the gate must not skip it.
        let mut n = NormalizedString::from("e\u{0301}");
        NFC.normalize(&mut n).unwrap();
        assert_eq!(n.get(), "\u{00e9}");
    }
}
