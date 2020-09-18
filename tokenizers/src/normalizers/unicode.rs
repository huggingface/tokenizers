use crate::tokenizer::{NormalizedString, Normalizer, Result};

#[derive(Default, Copy, Clone, Debug)]
pub struct NFD;
impl Normalizer for NFD {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.nfd();
        Ok(())
    }
}

#[derive(Default, Copy, Clone, Debug)]
pub struct NFKD;
impl Normalizer for NFKD {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.nfkd();
        Ok(())
    }
}

#[derive(Default, Copy, Clone, Debug)]
pub struct NFC;
impl Normalizer for NFC {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.nfc();
        Ok(())
    }
}

#[derive(Default, Copy, Clone, Debug)]
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
        .filter(|c| match c as u32 {
            0x0001..=0x0008 => false,
            0x000B => false,
            0x000E..=0x001F => false,
            0x007F => false,
            0x008F => false,
            0x009F => false,
            _ => true,
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
pub struct Nmt;
impl Normalizer for Nmt {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        do_nmt(normalized);
        Ok(())
    }
}

impl_serde_unit_struct!(NFCVisitor, NFC);
impl_serde_unit_struct!(NFCKVisitor, NFKC);
impl_serde_unit_struct!(NFKDVisitor, NFKD);
impl_serde_unit_struct!(NFDVisitor, NFD);
impl_serde_unit_struct!(NMTVisitor, Nmt);

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
}
