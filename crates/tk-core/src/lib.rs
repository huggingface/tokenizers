use std::borrow::Cow;

mod encoding;
mod error;
mod token;
mod vocab;

pub type AddedTokens = Vec<(usize, usize)>;
pub type Offsets = Vec<(usize, usize)>;

pub trait NormalizerImpl: Send + Sync {
    fn normalize<'a>(&mut self, s: Cow<'a, str>) -> Cow<'a, str>;
}
