use crate::tokenizer::{NormalizedString, Normalizer, Result};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct NFD;
#[typetag::serde]
impl Normalizer for NFD {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.nfd();
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
pub struct NFKD;
#[typetag::serde]
impl Normalizer for NFKD {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.nfkd();
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
pub struct NFC;
#[typetag::serde]
impl Normalizer for NFC {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.nfc();
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
pub struct NFKC;
#[typetag::serde]
impl Normalizer for NFKC {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        normalized.nfkc();
        Ok(())
    }
}
