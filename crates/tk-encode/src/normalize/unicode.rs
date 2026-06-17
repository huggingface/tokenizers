use std::borrow::Cow;

use crate::normalize::Normalizer;
use crate::normalize::error::Result;

use unicode_normalization::{UnicodeNormalization, is_nfc, is_nfd, is_nfkc, is_nfkd};

pub struct NFCNormalizer;
pub struct NFDNormalizer;
pub struct NFKCNormalizer;
pub struct NFKDNormalizer;

impl Normalizer for NFDNormalizer {
    fn normalize<'a>(&mut self, mut string: Cow<'a, str>) -> Result<Cow<'a, str>> {
        Ok(if !is_nfd(&string) {
            string.to_mut().nfd().collect()
        } else {
            string
        })
    }
}

impl Normalizer for NFCNormalizer {
    fn normalize<'a>(&mut self, mut string: Cow<'a, str>) -> Result<Cow<'a, str>> {
        Ok(if !is_nfc(&string) {
            string.to_mut().nfc().collect()
        } else {
            string
        })
    }
}

impl Normalizer for NFKCNormalizer {
    fn normalize<'a>(&mut self, mut string: Cow<'a, str>) -> Result<Cow<'a, str>> {
        Ok(if !is_nfkc(&string) {
            string.to_mut().nfkc().collect()
        } else {
            string
        })
    }
}

impl Normalizer for NFKDNormalizer {
    fn normalize<'a>(&mut self, mut string: Cow<'a, str>) -> Result<Cow<'a, str>> {
        Ok(if !is_nfkd(&string) {
            string.to_mut().nfkd().collect()
        } else {
            string
        })
    }
}
