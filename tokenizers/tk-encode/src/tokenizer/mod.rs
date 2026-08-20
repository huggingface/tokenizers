//! Represents a tokenization pipeline.
//!
//! A [`Tokenizer`](struct.Tokenizer.html) is composed of some of the following parts.
//!   - [`Normalizer`](trait.Normalizer.html): Takes care of the text normalization (like unicode normalization).
//!   - [`PreTokenizer`](trait.PreTokenizer.html): Takes care of the pre tokenization (ie. How to split tokens and pre-process
//!     them.
//!   - [`Model`](trait.Model.html): A model encapsulates the tokenization algorithm (like BPE, Word base, character
//!     based, ...).
//!   - [`PostProcessor`](trait.PostProcessor.html): Takes care of the processing after tokenization (like truncating, padding,
//!     ...).

// The `Path` types are reachable only from `Model::save`.
use std::path::{Path, PathBuf};

mod encoding;
pub mod normalizer;
pub mod pattern;
pub mod pipeline;
pub mod pre_tokenizer;

// `Tokenizer`, `TokenizerImpl`, `TokenizerBuilder`, `DecodeStream`, the legacy `AddedVocabulary`
// (and its `AddedToken`) and the whole serde load/save surface were deleted with the config layer.
// What stays here is the vocabulary of traits every component implements, plus the input/output
// types they are written against.
pub use crate::decoders::DecoderRuntime;
pub use crate::utils::iter::LinesWithEnding;
pub use crate::utils::padding::{PaddingDirection, PaddingParams, PaddingStrategy, pad_encodings};
pub use crate::utils::truncation::{
    TruncationDirection, TruncationParams, TruncationStrategy, truncate_encodings,
};
pub use encoding::*;
pub use normalizer::{NormalizedString, OffsetReferential, SplitDelimiterBehavior};
pub use pre_tokenizer::*;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;
pub type Offsets = (usize, usize);

/// Takes care of pre-processing strings.
pub trait Normalizer: Sync {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()>;
}

/// The `PreTokenizer` is in charge of doing the pre-segmentation step. It splits the given string
/// in multiple substrings, keeping track of the offsets of said substrings from the
/// `NormalizedString`. In some occasions, the `PreTokenizer` might need to modify the given
/// `NormalizedString` to ensure we can entirely keep track of the offsets and the mapping with
/// the original string.
pub trait PreTokenizer {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()>;
}

/// Represents a model used during Tokenization (like BPE or Word or Unigram).
pub trait Model {
    /// Tokenize the given sequence into multiple underlying `Token`. The `offsets` on the `Token`
    /// are expected to be relative to the given sequence.
    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>>;
    /// Find the ID associated to a string token
    fn token_to_id(&self, token: &str) -> Option<u32>;
    /// Find the string token associated to an ID
    fn id_to_token(&self, id: u32) -> Option<String>;
    /// Retrieve the entire vocabulary mapping (token -> ID)
    fn get_vocab(&self) -> HashMap<String, u32>;
    /// Retrieve the size of the vocabulary
    fn get_vocab_size(&self) -> usize;
    /// Save the current `Model` in the given folder, using the given `prefix` for the various
    /// files that need to be saved.
    ///
    /// Writing files is config-layer work and would rather not be on this trait at all, but it
    /// cannot leave: this is a *trait method*, and both bindings write their own `save` inside
    /// `impl Model for PyModel` / `impl tk::Model for Model`. Removing it from the trait breaks
    /// them, and an extension trait would only work where it is imported — which would silently
    /// route those `save` calls to a different method. So it stays.
    ///
    /// It used to sit behind `config`, which was really a proxy for "serde". It needs no serde at
    /// all -- only `std::path` -- so with that feature gone it is simply unconditional.
    fn save(&self, folder: &Path, prefix: Option<&str>) -> Result<Vec<PathBuf>>;

    /// Tokenize every pre-token within a `PreTokenizedString` in one call.
    ///
    /// When `truncation` is `Some((max_tokens, direction))`, the
    /// `tokenize_with_limit` early-exit path is taken; otherwise every
    /// pre-token is tokenized.
    ///
    /// The default calls `self.tokenize()` per pre-token, which is correct
    /// for self-contained `Model` implementations.  Implementations that
    /// wrap their inner model behind a lock (e.g. the `PyModel` and Node
    /// `Model` bindings, both `Arc<RwLock<_>>`) can override this to
    /// acquire the lock once for the whole sequence of pre-tokens; that
    /// removes ~one atomic load/store pair per pre-token from the hot path
    /// (~1 500 pre-tokens per ~10 KB document).  Both the truncated and
    /// non-truncated paths benefit from the override because they share
    /// this entry point.
    fn tokenize_in_pretokenized(
        &self,
        pretokenized: &mut PreTokenizedString,
        truncation: Option<(usize, TruncationDirection)>,
    ) -> Result<()> {
        match truncation {
            Some((max_tokens, direction)) => pretokenized.tokenize_with_limit(
                |normalized| self.tokenize(normalized.get()),
                max_tokens,
                direction,
            ),
            None => pretokenized.tokenize(|normalized| self.tokenize(normalized.get())),
        }
    }
}

/// A `PostProcessor` has the responsibility to post process an encoded output of the `Tokenizer`.
/// It adds any special tokens that a language model would require.
pub trait PostProcessor {
    /// Returns the number of tokens that will be added during the processing step
    fn added_tokens(&self, is_pair: bool) -> usize;
    /// Process both encodings and returns a new merged one
    fn process(
        &self,
        encoding: Encoding,
        pair_encoding: Option<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        let mut encodings = if let Some(pair_encoding) = pair_encoding {
            vec![encoding, pair_encoding]
        } else {
            vec![encoding]
        };
        encodings.iter_mut().enumerate().for_each(|(i, encoding)| {
            encoding.set_sequence_id(i);
            encoding
                .get_overflowing_mut()
                .iter_mut()
                .for_each(|encoding| encoding.set_sequence_id(i));
            encoding.set_type_ids(vec![i as u32; encoding.len()]);
        });

        let encodings = self.process_encodings(encodings, add_special_tokens)?;
        Ok(Encoding::merge(encodings, false))
    }

    /// Process any amount of encodings and returns a series of encoding (might merge them)
    fn process_encodings(
        &self,
        encodings: Vec<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>>;
}
impl dyn PostProcessor {
    pub fn default_process(
        encodings: Vec<Encoding>,
        _add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        match encodings.len() {
            1 => Ok(encodings),
            _ => {
                let mut final_encoding = Encoding::default();
                for (i, mut encoding) in encodings.into_iter().enumerate() {
                    encoding.set_sequence_id(i);
                    final_encoding.merge_with(encoding, false);
                }
                Ok(vec![final_encoding])
            }
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ProcessorError {
    #[error("encodings vector length must be either 1 or 2")]
    InvalidEncodingsVecLength,
}

/// A `Decoder` changes the raw tokens into its more readable form.
pub trait Decoder {
    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        let results = self.decode_chain(tokens)?;
        Ok(results.join(""))
    }
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub id: u32,
    pub value: String,
    pub offsets: (usize, usize),
}
impl Token {
    pub fn new(id: u32, value: String, offsets: (usize, usize)) -> Self {
        Self { id, value, offsets }
    }
}

use std::borrow::Cow;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum InputSequence<'s> {
    Raw(Cow<'s, str>),
    PreTokenized(Cow<'s, [&'s str]>),
    PreTokenizedOwned(Cow<'s, [String]>),
    PreTokenizedCow(Cow<'s, [Cow<'s, str>]>),
}

impl<'s> From<Cow<'s, str>> for InputSequence<'s> {
    fn from(input: Cow<'s, str>) -> Self {
        Self::Raw(input)
    }
}

impl<'s> From<&'s str> for InputSequence<'s> {
    fn from(input: &'s str) -> Self {
        Self::Raw(Cow::Borrowed(input))
    }
}

impl From<String> for InputSequence<'_> {
    fn from(input: String) -> Self {
        Self::Raw(Cow::Owned(input))
    }
}

impl<'s> From<&'s [&'s str]> for InputSequence<'s> {
    fn from(input: &'s [&'s str]) -> Self {
        Self::PreTokenized(Cow::Borrowed(input))
    }
}

impl<'s> From<Vec<&'s str>> for InputSequence<'s> {
    fn from(input: Vec<&'s str>) -> Self {
        Self::PreTokenized(Cow::Owned(input))
    }
}

impl<'s> From<&'s [String]> for InputSequence<'s> {
    fn from(input: &'s [String]) -> Self {
        Self::PreTokenizedOwned(Cow::Borrowed(input))
    }
}

impl From<Vec<String>> for InputSequence<'_> {
    fn from(input: Vec<String>) -> Self {
        Self::PreTokenizedOwned(Cow::Owned(input))
    }
}

impl<'s> From<Vec<Cow<'s, str>>> for InputSequence<'s> {
    fn from(input: Vec<Cow<'s, str>>) -> Self {
        Self::PreTokenizedCow(Cow::Owned(input))
    }
}

impl<'s> From<&'s [Cow<'s, str>]> for InputSequence<'s> {
    fn from(input: &'s [Cow<'s, str>]) -> Self {
        Self::PreTokenizedCow(Cow::Borrowed(input))
    }
}

#[derive(Debug, Clone)]
pub enum EncodeInput<'s> {
    Single(InputSequence<'s>),
    Dual(InputSequence<'s>, InputSequence<'s>),
}

impl<'s, I: Into<InputSequence<'s>>> From<I> for EncodeInput<'s> {
    fn from(input: I) -> Self {
        Self::Single(input.into())
    }
}

impl<'s, I1, I2> From<(I1, I2)> for EncodeInput<'s>
where
    I1: Into<InputSequence<'s>>,
    I2: Into<InputSequence<'s>>,
{
    fn from(input: (I1, I2)) -> Self {
        Self::Dual(input.0.into(), input.1.into())
    }
}
