use std::borrow::Cow;

use tk_core::NormalizerImpl;

pub enum Normalizer {
    Lowercase(LowercaseNormalizer),
    None(IdentityNormalizer),
}

impl NormalizerImpl for Normalizer {
    fn normalize<'a>(&mut self, s: Cow<'a, str>) -> Cow<'a, str> {
        match self {
            Normalizer::Lowercase(normalizer) => normalizer.normalize(s),
            Normalizer::None(normalizer) => normalizer.normalize(s),
        }
    }
}

#[derive(Debug)]
pub struct IdentityNormalizer {}

impl IdentityNormalizer {
    pub fn new() -> Self {
        Self {}
    }
}

impl NormalizerImpl for IdentityNormalizer {
    fn normalize<'a>(&mut self, mut _s: Cow<'a, str>) -> Cow<'a, str> {
        _s
    }
}

#[derive(Debug)]
pub struct LowercaseNormalizer {}

impl LowercaseNormalizer {
    pub fn new() -> Self {
        Self {}
    }
}

impl NormalizerImpl for LowercaseNormalizer {
    fn normalize<'a>(&mut self, mut s: Cow<'a, str>) -> Cow<'a, str> {
        if s.chars().any(|c| c.is_uppercase()) {
            s.to_mut().make_ascii_lowercase();
        }
        s
    }
}
