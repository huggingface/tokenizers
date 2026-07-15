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

use ahash::AHashMap;
use std::{
    fs::{read_to_string, File},
    io::prelude::*,
    ops::{Deref, DerefMut},
    path::{Path, PathBuf},
};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};


mod added_vocabulary;
pub mod normalizer;
pub mod pattern;
pub mod pipeline;
pub mod pre_tokenizer;
mod serialization;

// Re-export wrappers
pub use crate::models::ModelWrapper;
pub use crate::normalizers::NormalizerWrapper;
pub use crate::pre_tokenizers::PreTokenizerWrapper;
// And some other types
pub use crate::tokenizer::added_vocabulary::{AddedToken, AddedVocabulary};
pub use crate::utils::iter::LinesWithEnding;
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
    fn save(&self, folder: &Path, prefix: Option<&str>) -> Result<Vec<PathBuf>>;
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

use std::collections::HashMap;

#[derive(thiserror::Error, Debug)]
#[error("{0}")]
pub struct BuilderError(String);

/// Builder for Tokenizer structs.
///
/// `build()` fails if the `model` is missing.
pub struct TokenizerBuilder<M, N, PT> {
    model: Option<M>,
    normalizer: Option<N>,
    pre_tokenizer: Option<PT>,

    added_vocabulary: AddedVocabulary,
}

impl<M, N, PT> Default for TokenizerBuilder<M, N, PT>
where
    M: Model,
    N: Normalizer,
    PT: PreTokenizer,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<M, N, PT> TokenizerBuilder<M, N, PT>
where
    M: Model,
    N: Normalizer,
    PT: PreTokenizer,
{
    /// Get an empty TokenizerBuilder.
    pub fn new() -> Self {
        Self {
            model: None,
            normalizer: None,
            pre_tokenizer: None,
            added_vocabulary: AddedVocabulary::new(),
        }
    }

    /// Convert the TokenizerBuilder to a Tokenizer.
    ///
    /// Conversion fails if the `model` is missing.
    pub fn build(self) -> Result<TokenizerImpl<M, N, PT>> {
        let model = self
            .model
            .ok_or_else(|| Box::new(BuilderError("Model missing.".into())))?;
        Ok(TokenizerImpl {
            normalizer: self.normalizer,
            pre_tokenizer: self.pre_tokenizer,
            model,
            added_vocabulary: self.added_vocabulary,
        })
    }

    /// Set the model.
    #[must_use]
    pub fn with_model(mut self, model: M) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the normalizer.
    #[must_use]
    pub fn with_normalizer(mut self, normalizer: Option<N>) -> Self {
        self.normalizer = normalizer;
        self
    }

    /// Set the pre-tokenizer.
    #[must_use]
    pub fn with_pre_tokenizer(mut self, pretokenizer: Option<PT>) -> Self {
        self.pre_tokenizer = pretokenizer;
        self
    }

    /// Set the added vocabulary.
    pub fn with_added_vocabulary(mut self, added_vocabulary: AddedVocabulary) -> Self {
        self.added_vocabulary = added_vocabulary;
        self
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Tokenizer(TokenizerImpl<ModelWrapper, NormalizerWrapper, PreTokenizerWrapper>);

impl Tokenizer {
    /// Construct a new Tokenizer based on the model.
    pub fn new(model: impl Into<ModelWrapper>) -> Self {
        Self(TokenizerImpl::new(model.into()))
    }

    /// Unwrap the TokenizerImpl.
    pub fn into_inner(self) -> TokenizerImpl<ModelWrapper, NormalizerWrapper, PreTokenizerWrapper> {
        self.0
    }
    pub fn from_file<P: AsRef<Path>>(file: P) -> Result<Self> {
        let content = read_to_string(file)?;
        let tokenizer = serde_json::from_str(&content)?;
        Ok(tokenizer)
    }
    pub fn from_bytes<P: AsRef<[u8]>>(bytes: P) -> Result<Self> {
        let tokenizer = serde_json::from_slice(bytes.as_ref())?;
        Ok(tokenizer)
    }
    #[cfg(feature = "http")]
    #[cfg_attr(docsrs, doc(cfg(feature = "http")))]
    pub fn from_pretrained<S: AsRef<str>>(
        identifier: S,
        params: Option<crate::utils::from_pretrained::FromPretrainedParameters>,
    ) -> Result<Self> {
        let tokenizer_file = crate::utils::from_pretrained::from_pretrained(identifier, params)?;
        Tokenizer::from_file(tokenizer_file)
    }
}

impl std::str::FromStr for Tokenizer {
    type Err = Box<dyn std::error::Error + Send + Sync>;

    fn from_str(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }
}

impl<M, N, PT> From<TokenizerImpl<M, N, PT>> for Tokenizer
where
    M: Into<ModelWrapper>,
    N: Into<NormalizerWrapper>,
    PT: Into<PreTokenizerWrapper>,
{
    fn from(t: TokenizerImpl<M, N, PT>) -> Self {
        Self(TokenizerImpl {
            model: t.model.into(),
            normalizer: t.normalizer.map(Into::into),
            pre_tokenizer: t.pre_tokenizer.map(Into::into),
            added_vocabulary: t.added_vocabulary,
        })
    }
}

impl Deref for Tokenizer {
    type Target = TokenizerImpl<ModelWrapper, NormalizerWrapper, PreTokenizerWrapper>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Tokenizer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// A `Tokenizer` is capable of encoding text into token ids (via the pipeline).
#[derive(Clone, Debug)]
pub struct TokenizerImpl<M, N, PT> {
    // Tokenizer parts
    normalizer: Option<N>,
    pre_tokenizer: Option<PT>,
    model: M,

    // Added Vocabulary capabilities
    added_vocabulary: AddedVocabulary,
}

impl<M, N, PT> TokenizerImpl<M, N, PT>
where
    M: Model,
    N: Normalizer,
    PT: PreTokenizer,
{
    /// Instantiate a new Tokenizer, with the given Model
    pub fn new(model: M) -> Self {
        Self {
            normalizer: None,
            pre_tokenizer: None,
            model,
            added_vocabulary: AddedVocabulary::new(),
        }
    }

    /// Set the normalizer.
    ///
    /// # Performance note
    ///
    /// When added tokens with `normalized = true` are already present, this method
    /// re-normalizes all of them and rebuilds the matching trie. For tokenizers with
    /// many added tokens this may be slow. Prefer setting the normalizer before adding
    /// tokens when constructing a tokenizer programmatically.
    pub fn with_normalizer(&mut self, normalizer: Option<impl Into<N>>) -> Result<&mut Self> {
        self.normalizer = normalizer.map(|norm| norm.into());
        self.added_vocabulary
            .refresh_normalized_tokens(self.normalizer.as_ref())?;
        Ok(self)
    }
    /// Get the normalizer
    pub fn get_normalizer(&self) -> Option<&N> {
        self.normalizer.as_ref()
    }

    /// Set the pre tokenizer
    pub fn with_pre_tokenizer(&mut self, pre_tokenizer: Option<impl Into<PT>>) -> &mut Self {
        self.pre_tokenizer = pre_tokenizer.map(|tok| tok.into());
        self
    }

    /// Get the pre tokenizer
    pub fn get_pre_tokenizer(&self) -> Option<&PT> {
        self.pre_tokenizer.as_ref()
    }

    /// Set the model
    pub fn with_model(&mut self, model: impl Into<M>) -> &mut Self {
        self.model = model.into();
        self
    }

    /// Get the model
    pub fn get_model(&self) -> &M {
        &self.model
    }

    /// Get a mutable reference to the model. Mainly used by the training
    /// extension in the `tk-train` crate so a `Trainer` can write the trained
    /// vocabulary/merges back into the model.
    pub fn get_model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Set the added vocabulary.
    pub fn with_added_vocabulary(&mut self, added_vocabulary: AddedVocabulary) -> &mut Self {
        self.added_vocabulary = added_vocabulary;
        self
    }

    /// Get the added vocabulary
    pub fn get_added_vocabulary(&self) -> &AddedVocabulary {
        &self.added_vocabulary
    }

    // Get the vocabulary as a plain HashMap for bindings compatibility
    pub fn get_vocab(&self, with_added_tokens: bool) -> HashMap<String, u32> {
        let mut final_vocab = self.model.get_vocab();

        if with_added_tokens {
            let added_vocab = self.added_vocabulary.get_vocab();
            if !added_vocab.is_empty() {
                final_vocab.reserve(added_vocab.len());
                for (token, id) in added_vocab {
                    final_vocab.insert(token.clone(), *id);
                }
            }
        }

        final_vocab
    }

    /// Get the added tokens decoder
    pub fn get_added_tokens_decoder(&self) -> AHashMap<u32, AddedToken> {
        self.added_vocabulary.get_added_tokens_decoder().clone()
    }

    /// Get the size of the vocabulary
    pub fn get_vocab_size(&self, with_added_tokens: bool) -> usize {
        let base = self.model.get_vocab_size();
        if with_added_tokens {
            let added = self.added_vocabulary.get_vocab();
            let overlapping = added
                .keys()
                .filter(|t| self.model.token_to_id(t).is_some())
                .count();
            base + added.len() - overlapping
        } else {
            base
        }
    }

    /// Converts a token in the corresponding id.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.added_vocabulary.token_to_id(token, &self.model)
    }

    /// Converts an id to the corresponding token.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.added_vocabulary
            .simple_id_to_token(id)
            .or_else(|| self.model.id_to_token(id))
    }

    /// set the added vocab's splitting scheme
    pub fn set_encode_special_tokens(&mut self, value: bool) {
        self.added_vocabulary.set_encode_special_tokens(value);
    }

    /// Get added token value
    pub fn get_encode_special_tokens(&self) -> bool {
        self.added_vocabulary.get_encode_special_tokens()
    }
}

impl<M, N, PT> TokenizerImpl<M, N, PT>
where
    N: Normalizer,
    M: Model,
{
    /// Register the given tokens as special tokens. This is especially useful for removing
    /// these special tokens while decoding
    pub fn add_special_tokens(
        &mut self,
        tokens: impl IntoIterator<Item = AddedToken>,
    ) -> Result<usize> {
        self.added_vocabulary
            .add_special_tokens(tokens, &self.model, self.normalizer.as_ref())
    }

    /// Add the given tokens to the added vocabulary
    pub fn add_tokens(&mut self, tokens: impl IntoIterator<Item = AddedToken>) -> Result<usize> {
        self.added_vocabulary
            .add_tokens(tokens, &self.model, self.normalizer.as_ref())
    }
}

impl<M, N, PT> std::str::FromStr for TokenizerImpl<M, N, PT>
where
    M: for<'de> Deserialize<'de> + Model,
    N: for<'de> Deserialize<'de> + Normalizer,
    PT: for<'de> Deserialize<'de> + PreTokenizer,
{
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }
}

impl<M, N, PT> TokenizerImpl<M, N, PT>
where
    M: DeserializeOwned + Model,
    N: DeserializeOwned + Normalizer,
    PT: DeserializeOwned + PreTokenizer,
{
    /// Instantiate a new Tokenizer from the given file
    pub fn from_file<P: AsRef<Path>>(file: P) -> Result<Self> {
        let content = read_to_string(file)?;
        let tokenizer = serde_json::from_str(&content)?;
        Ok(tokenizer)
    }
}

impl<M, N, PT> TokenizerImpl<M, N, PT>
where
    M: DeserializeOwned + Model,
    N: DeserializeOwned + Normalizer,
    PT: DeserializeOwned + PreTokenizer,
{
    /// Instantiate a new Tokenizer from bytes
    pub fn from_bytes<P: AsRef<[u8]>>(bytes: P) -> Result<Self> {
        let tokenizer = serde_json::from_slice(bytes.as_ref())?;
        Ok(tokenizer)
    }
}

impl<M, N, PT> TokenizerImpl<M, N, PT>
where
    M: DeserializeOwned + Model,
    N: DeserializeOwned + Normalizer,
    PT: DeserializeOwned + PreTokenizer,
{
    #[deprecated(
        since = "0.14.0",
        note = "Users should download the file separately using https://github.com/huggingface/hf-hub instead, which splits concerns of accessing the web, and should use the new cache layout"
    )]
    #[cfg(feature = "http")]
    #[cfg_attr(docsrs, doc(cfg(feature = "http")))]
    /// Instantiate a new Tokenizer from a file hosted on the Hugging Face Hub.
    /// It expects the `identifier` of a model that includes a `tokenizer.json` file.
    pub fn from_pretrained<S: AsRef<str>>(
        identifier: S,
        params: Option<crate::utils::from_pretrained::FromPretrainedParameters>,
    ) -> Result<Self> {
        let tokenizer_file = crate::utils::from_pretrained::from_pretrained(identifier, params)?;
        TokenizerImpl::from_file(tokenizer_file)
    }
}

impl<M, N, PT> TokenizerImpl<M, N, PT>
where
    M: Serialize,
    N: Serialize,
    PT: Serialize,
{
    /// Serialize the current tokenizer as a String
    pub fn to_string(&self, pretty: bool) -> Result<String> {
        Ok(if pretty {
            serde_json::to_string_pretty(self)?
        } else {
            serde_json::to_string(self)?
        })
    }

    /// Save the current tokenizer at the given path
    pub fn save<P: AsRef<Path>>(&self, path: P, pretty: bool) -> Result<()> {
        let serialized = self.to_string(pretty)?;

        let mut file = File::create(path)?;
        file.write_all(serialized.as_bytes())?;

        Ok(())
    }
}

