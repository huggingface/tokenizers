mod encoding;
mod error;
mod token;
mod vocab;

pub type AddedTokens = Vec<(usize, usize)>;
pub type Offsets = Vec<(usize, usize)>;
