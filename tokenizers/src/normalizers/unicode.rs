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

impl_serde_unit_struct!(NFCVisitor, NFC);
impl_serde_unit_struct!(NFCKVisitor, NFKC);
impl_serde_unit_struct!(NFKDVisitor, NFKD);
impl_serde_unit_struct!(NFDVisitor, NFD);
