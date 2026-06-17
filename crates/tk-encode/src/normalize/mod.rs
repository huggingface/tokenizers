mod error;
mod unicode;

use std::borrow::Cow;

use error::Result;

use crate::normalize::unicode::{NFCNormalizer, NFDNormalizer, NFKCNormalizer, NFKDNormalizer};
pub trait Normalizer: Send + Sync {
    fn normalize<'a>(&mut self, string: Cow<'a, str>) -> error::Result<Cow<'a, str>>;
}

pub enum NormalizePlan {
    Custom(Box<dyn Normalizer>),
    Lowercase(LowercaseNormalizer),
    NFD(NFDNormalizer),
    NFC(NFCNormalizer),
    NFKC(NFKCNormalizer),
    NFKD(NFKDNormalizer),
    Sequence(Box<[NormalizePlan]>),
    None,
}

impl Normalizer for NormalizePlan {
    fn normalize<'a>(&mut self, string: Cow<'a, str>) -> Result<Cow<'a, str>> {
        match self {
            NormalizePlan::Custom(normalizer) => normalizer.normalize(string),
            NormalizePlan::NFC(normalizer) => normalizer.normalize(string),
            NormalizePlan::NFD(normalizer) => normalizer.normalize(string),
            NormalizePlan::NFKC(normalizer) => normalizer.normalize(string),
            NormalizePlan::NFKD(normalizer) => normalizer.normalize(string),
            NormalizePlan::Lowercase(normalizer) => normalizer.normalize(string),
            NormalizePlan::Sequence(seq) => {
                let mut string = string;
                for member in seq {
                    string = member.normalize(string)?;
                }
                Ok(string)
            },
            // No normalization: pass the string as-is
            NormalizePlan::None => Ok(string),

        }
    }
}



#[derive(Debug)]
pub struct LowercaseNormalizer {}

impl LowercaseNormalizer {
    pub fn new() -> Self {
        Self {}
    }
}

impl Normalizer for LowercaseNormalizer {
    fn normalize<'a>(&mut self, s: Cow<'a, str>) -> Result<Cow<'a, str>> {
        todo!("Not implemented")
    }
}
