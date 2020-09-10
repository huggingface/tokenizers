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
            0x0001 => false,
            0x0002 => false,
            0x0003 => false,
            0x0004 => false,
            0x0005 => false,
            0x0006 => false,
            0x0007 => false,
            0x0008 => false,
            0x000B => false,
            0x000E => false,
            0x000F => false,
            0x0010 => false,
            0x0011 => false,
            0x0012 => false,
            0x0013 => false,
            0x0014 => false,
            0x0015 => false,
            0x0016 => false,
            0x0017 => false,
            0x0018 => false,
            0x0019 => false,
            0x001A => false,
            0x001B => false,
            0x001C => false,
            0x001D => false,
            0x001E => false,
            0x001F => false,
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
            0x200B => ' ',
            0x200E => ' ',
            0x200F => ' ',
            0x2028 => ' ',
            0x2029 => ' ',
            0x2581 => ' ',
            0xFEFF => ' ',
            0xFFFD => ' ',
            0x200C => ' ',
            0x200D => ' ',
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
