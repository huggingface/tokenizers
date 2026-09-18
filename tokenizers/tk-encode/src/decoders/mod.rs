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

pub use runtime::DecoderRuntime;

use crate::tokenizer::Result;

/// A `Decoder` changes the raw tokens into its more readable form.
///
/// This lives here, with its implementors, rather than in `tokenizer` alongside the encode-side
/// traits: those were the legacy `NormalizedString` pipeline and are gone, while this one is what
/// `PipelineTokenizer::decode` actually dispatches through.
pub trait Decoder {
    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        let results = self.decode_chain(tokens)?;
        Ok(results.join(""))
    }
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>>;
}
