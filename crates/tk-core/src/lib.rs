mod encoding;
mod error;
mod token;
mod vocab;

pub type AddedTokens = Vec<(usize, usize)>;
pub type Offsets = Vec<(usize, usize)>;

pub trait Normalizer: Send + Sync {
    fn normalize(&self, mut s: Cow<str>) -> Cow<str>;
}
