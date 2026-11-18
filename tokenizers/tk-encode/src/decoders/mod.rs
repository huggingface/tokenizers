pub mod bpe;
pub mod byte_fallback;
pub mod byte_level;
pub mod ctc;
pub mod fuse;
pub mod metaspace;
pub mod replace;
pub mod runtime;
pub mod strip;
pub mod wordpiece;

pub use runtime::Decoder;

use crate::tokenizer::Result;

/// Turns tokens back into text. Every decoder implements it.
///
/// This lives here, with its implementors, rather than in `tokenizer` alongside the encode-side
/// traits: those were the legacy `NormalizedString` pipeline and are gone, while this one is what
/// `Tokenizer::decode` actually dispatches through.
pub trait Decode {
    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        let results = self.decode_chain(tokens)?;
        Ok(results.join(""))
    }
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>>;
}
