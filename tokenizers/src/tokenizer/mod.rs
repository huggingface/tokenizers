//! Represents a tokenization pipeline.
//!
//! A [`Tokenizer`](struct.Tokenizer.html) is composed of some of the following parts.
//!   - [`Normalizer`](trait.Normalizer.html): Takes care of the text normalization (like unicode normalization).
//!   - [`PreTokenizer`](trait.PreTokenizer.html): Takes care of the pre tokenization (ie. How to split tokens and pre-process
//!   them.
//!   - [`Model`](trait.Model.html): A model encapsulates the tokenization algorithm (like BPE, Word base, character
//!   based, ...).
//!   - [`PostProcessor`](trait.PostProcessor.html): Takes care of the processing after tokenization (like truncating, padding,
//!   ...).

use std::{
    collections::HashMap,
    fmt,
    fs::{read_to_string, File},
    io::prelude::*,
    io::BufReader,
    ops::{Deref, DerefMut},
    path::{Path, PathBuf},
};

use serde::de::DeserializeOwned;
use serde::export::Formatter;
use serde::{Deserialize, Serialize};

use crate::utils::iter::ResultShunt;
use crate::utils::parallelism::*;
use crate::utils::progress::{ProgressBar, ProgressStyle};

mod added_vocabulary;
mod encoding;
pub mod normalizer;
pub mod pattern;
pub mod pre_tokenizer;
mod serialization;

// Re-export wrappers
pub use crate::decoders::DecoderWrapper;
pub use crate::models::ModelWrapper;
pub use crate::normalizers::NormalizerWrapper;
pub use crate::pre_tokenizers::PreTokenizerWrapper;
pub use crate::processors::PostProcessorWrapper;
// And some other types
pub use crate::utils::iter::LinesWithEnding;
pub use crate::utils::padding::{pad_encodings, PaddingDirection, PaddingParams, PaddingStrategy};
pub use crate::utils::truncation::{truncate_encodings, TruncationParams, TruncationStrategy};
pub use added_vocabulary::*;
pub use encoding::*;
pub use normalizer::{NormalizedString, OffsetReferential, SplitDelimiterBehavior};
pub use pre_tokenizer::*;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;
pub type Offsets = (usize, usize);

/// Takes care of pre-processing strings.
pub trait Normalizer {
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
    type Trainer: Trainer + Sync;
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
    /// Get an instance of a Trainer capable of training this Model
    fn get_trainer(&self) -> <Self as Model>::Trainer;
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
    ) -> Result<Encoding>;
}
impl dyn PostProcessor {
    pub fn default_process(
        mut encoding: Encoding,
        pair_encoding: Option<Encoding>,
        _add_special_tokens: bool,
    ) -> Result<Encoding> {
        match pair_encoding {
            None => Ok(encoding),
            Some(mut pair) => {
                encoding.set_sequence_id(0);
                pair.set_sequence_id(1);
                encoding.merge_with(pair, false);
                Ok(encoding)
            }
        }
    }
}

/// A `Decoder` has the responsibility to merge the given `Vec<String>` in a `String`.
pub trait Decoder {
    fn decode(&self, tokens: Vec<String>) -> Result<String>;
}

/// A `Trainer` has the responsibility to train a model. We feed it with lines/sentences
/// and then it can train the given `Model`.
pub trait Trainer {
    type Model: Model + Sized;
    /// Whether we should show progress during the training.
    fn should_show_progress(&self) -> bool;
    /// The actual training method. This will return a new trained Model as well as a list
    /// of `special_tokens` to be added directly to the tokenizer along with the model.
    fn train(&self, model: &mut Self::Model) -> Result<Vec<AddedToken>>;
    /// Process an iterator of sequences, calling `process` for each of them in order to
    /// pre-process the said sequence as relevant.
    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync;
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub id: u32,
    pub value: String,
    pub offsets: (usize, usize),
}
impl Token {
    pub fn new(id: u32, value: String, offsets: (usize, usize)) -> Self {
        Token { id, value, offsets }
    }
}

use std::borrow::Cow;
#[derive(Debug, Clone)]
pub enum InputSequence<'s> {
    Raw(Cow<'s, str>),
    PreTokenized(Cow<'s, [&'s str]>),
    PreTokenizedOwned(Cow<'s, [String]>),
    PreTokenizedCow(Cow<'s, [Cow<'s, str>]>),
}

impl<'s> From<Cow<'s, str>> for InputSequence<'s> {
    fn from(input: Cow<'s, str>) -> Self {
        InputSequence::Raw(input)
    }
}

impl<'s> From<&'s str> for InputSequence<'s> {
    fn from(input: &'s str) -> Self {
        InputSequence::Raw(Cow::Borrowed(input))
    }
}

impl From<String> for InputSequence<'_> {
    fn from(input: String) -> Self {
        InputSequence::Raw(Cow::Owned(input))
    }
}

impl<'s> From<&'s [&'s str]> for InputSequence<'s> {
    fn from(input: &'s [&'s str]) -> Self {
        InputSequence::PreTokenized(Cow::Borrowed(input))
    }
}

impl<'s> From<Vec<&'s str>> for InputSequence<'s> {
    fn from(input: Vec<&'s str>) -> Self {
        InputSequence::PreTokenized(Cow::Owned(input))
    }
}

impl<'s> From<&'s [String]> for InputSequence<'s> {
    fn from(input: &'s [String]) -> Self {
        InputSequence::PreTokenizedOwned(Cow::Borrowed(input))
    }
}

impl<'s> From<Vec<String>> for InputSequence<'s> {
    fn from(input: Vec<String>) -> Self {
        InputSequence::PreTokenizedOwned(Cow::Owned(input))
    }
}

impl<'s> From<Vec<Cow<'s, str>>> for InputSequence<'s> {
    fn from(input: Vec<Cow<'s, str>>) -> Self {
        InputSequence::PreTokenizedCow(Cow::Owned(input))
    }
}

impl<'s> From<&'s [Cow<'s, str>]> for InputSequence<'s> {
    fn from(input: &'s [Cow<'s, str>]) -> Self {
        InputSequence::PreTokenizedCow(Cow::Borrowed(input))
    }
}

#[derive(Debug, Clone)]
pub enum EncodeInput<'s> {
    Single(InputSequence<'s>),
    Dual(InputSequence<'s>, InputSequence<'s>),
}

impl<'s, I: Into<InputSequence<'s>>> From<I> for EncodeInput<'s> {
    fn from(input: I) -> Self {
        EncodeInput::Single(input.into())
    }
}

impl<'s, I1, I2> From<(I1, I2)> for EncodeInput<'s>
where
    I1: Into<InputSequence<'s>>,
    I2: Into<InputSequence<'s>>,
{
    fn from(input: (I1, I2)) -> Self {
        EncodeInput::Dual(input.0.into(), input.1.into())
    }
}

#[derive(Debug)]
pub struct BuilderError(String);

impl std::error::Error for BuilderError {}

impl fmt::Display for BuilderError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Builder for Tokenizer structs.
///
/// `build()` fails if the `model` is missing.
pub struct TokenizerBuilder<M, N, PT, PP, D> {
    model: Option<M>,
    normalizer: Option<N>,
    pre_tokenizer: Option<PT>,
    post_processor: Option<PP>,
    decoder: Option<D>,

    added_vocabulary: AddedVocabulary,

    truncation: Option<TruncationParams>,
    padding: Option<PaddingParams>,
}

impl<M, N, PT, PP, D> Default for TokenizerBuilder<M, N, PT, PP, D>
where
    M: Model,
    N: Normalizer,
    PT: PreTokenizer,
    PP: PostProcessor,
    D: Decoder,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<M, N, PT, PP, D> TokenizerBuilder<M, N, PT, PP, D>
where
    M: Model,
    N: Normalizer,
    PT: PreTokenizer,
    PP: PostProcessor,
    D: Decoder,
{
    /// Get an empty TokenizerBuilder.
    pub fn new() -> Self {
        TokenizerBuilder {
            model: None,
            normalizer: None,
            pre_tokenizer: None,
            post_processor: None,
            decoder: None,
            added_vocabulary: AddedVocabulary::new(),
            truncation: None,
            padding: None,
        }
    }

    /// Convert the TokenizerBuilder to a Tokenizer.
    ///
    /// Conversion fails if the `model` is missing.
    pub fn build(self) -> Result<TokenizerImpl<M, N, PT, PP, D>> {
        let model = self
            .model
            .ok_or_else(|| Box::new(BuilderError("Model missing.".into())))?;
        Ok(TokenizerImpl {
            normalizer: self.normalizer,
            pre_tokenizer: self.pre_tokenizer,
            model,

            post_processor: self.post_processor,
            decoder: self.decoder,
            added_vocabulary: self.added_vocabulary,
            truncation: self.truncation,
            padding: self.padding,
        })
    }

    /// Set the model.
    pub fn with_model(mut self, model: M) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the normalizer.
    pub fn with_normalizer(mut self, normalizer: Option<N>) -> Self {
        self.normalizer = normalizer;
        self
    }

    /// Set the pre-tokenizer.
    pub fn with_pre_tokenizer(mut self, pretokenizer: Option<PT>) -> Self {
        self.pre_tokenizer = pretokenizer;
        self
    }

    /// Set the post-processor.
    pub fn with_post_processor(mut self, post_processor: Option<PP>) -> Self {
        self.post_processor = post_processor;
        self
    }

    /// Set the decoder.
    pub fn with_decoder(mut self, decoder: Option<D>) -> Self {
        self.decoder = decoder;
        self
    }

    /// Set the trunaction parameters.
    pub fn with_truncation(mut self, trunc: Option<TruncationParams>) -> Self {
        self.truncation = trunc;
        self
    }

    /// Set the padding parameters.
    pub fn with_padding(mut self, padding: Option<PaddingParams>) -> Self {
        self.padding = padding;
        self
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Tokenizer(
    TokenizerImpl<
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >,
);

impl Tokenizer {
    /// Construct a new Tokenizer based on the model.
    pub fn new(model: impl Into<ModelWrapper>) -> Self {
        Self(TokenizerImpl::new(model.into()))
    }

    /// Unwrap the TokenizerImpl.
    pub fn into_inner(
        self,
    ) -> TokenizerImpl<
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    > {
        self.0
    }
    pub fn from_file<P: AsRef<Path>>(file: P) -> Result<Self> {
        let content = read_to_string(file)?;
        Ok(serde_json::from_str(&content)?)
    }
}

impl std::str::FromStr for Tokenizer {
    type Err = Box<dyn std::error::Error + Send + Sync>;

    fn from_str(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }
}

impl<M, N, PT, PP, D> From<TokenizerImpl<M, N, PT, PP, D>> for Tokenizer
where
    M: Into<ModelWrapper>,
    N: Into<NormalizerWrapper>,
    PT: Into<PreTokenizerWrapper>,
    PP: Into<PostProcessorWrapper>,
    D: Into<DecoderWrapper>,
{
    fn from(t: TokenizerImpl<M, N, PT, PP, D>) -> Self {
        Self(TokenizerImpl {
            model: t.model.into(),
            normalizer: t.normalizer.map(Into::into),
            pre_tokenizer: t.pre_tokenizer.map(Into::into),
            post_processor: t.post_processor.map(Into::into),
            decoder: t.decoder.map(Into::into),
            added_vocabulary: t.added_vocabulary,
            padding: t.padding,
            truncation: t.truncation,
        })
    }
}

impl Deref for Tokenizer {
    type Target = TokenizerImpl<
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Tokenizer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// A `Tokenizer` is capable of encoding/decoding any text.
#[derive(Clone, Debug)]
pub struct TokenizerImpl<M, N, PT, PP, D> {
    // Tokenizer parts
    normalizer: Option<N>,
    pre_tokenizer: Option<PT>,
    model: M,
    post_processor: Option<PP>,
    decoder: Option<D>,

    // Added Vocabulary capabilities
    added_vocabulary: AddedVocabulary,

    // General processing parameters
    truncation: Option<TruncationParams>,
    padding: Option<PaddingParams>,
}

impl<M, N, PT, PP, D> TokenizerImpl<M, N, PT, PP, D>
where
    M: Model,
    N: Normalizer,
    PT: PreTokenizer,
    PP: PostProcessor,
    D: Decoder,
{
    /// Instantiate a new Tokenizer, with the given Model
    pub fn new(model: M) -> Self {
        TokenizerImpl {
            normalizer: None,
            pre_tokenizer: None,
            model,
            post_processor: None,
            decoder: None,

            added_vocabulary: AddedVocabulary::new(),

            truncation: None,
            padding: None,
        }
    }

    /// Set the normalizer
    pub fn with_normalizer(&mut self, normalizer: impl Into<N>) -> &mut Self {
        self.normalizer = Some(normalizer.into());
        self
    }

    /// Get the normalizer
    pub fn get_normalizer(&self) -> Option<&N> {
        self.normalizer.as_ref()
    }

    /// Set the pre tokenizer
    pub fn with_pre_tokenizer(&mut self, pre_tokenizer: impl Into<PT>) -> &mut Self {
        self.pre_tokenizer = Some(pre_tokenizer.into());
        self
    }

    /// Get the pre tokenizer
    pub fn get_pre_tokenizer(&self) -> Option<&PT> {
        self.pre_tokenizer.as_ref()
    }

    /// Set the post processor
    pub fn with_post_processor(&mut self, post_processor: impl Into<PP>) -> &mut Self {
        self.post_processor = Some(post_processor.into());
        self
    }

    /// Get the post processor
    pub fn get_post_processor(&self) -> Option<&PP> {
        self.post_processor.as_ref()
    }

    /// Set the decoder
    pub fn with_decoder(&mut self, decoder: impl Into<D>) -> &mut Self {
        self.decoder = Some(decoder.into());
        self
    }

    /// Get the decoder
    pub fn get_decoder(&self) -> Option<&D> {
        self.decoder.as_ref()
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

    /// Set the truncation parameters
    pub fn with_truncation(&mut self, trunc: Option<TruncationParams>) -> &mut Self {
        self.truncation = trunc;
        self
    }

    /// Get the currently set truncation parameters
    pub fn get_truncation(&self) -> Option<&TruncationParams> {
        self.truncation.as_ref()
    }

    /// Get a mutable reference to the currently set truncation parameters
    pub fn get_truncation_mut(&mut self) -> Option<&mut TruncationParams> {
        self.truncation.as_mut()
    }

    /// Set the padding parameters
    pub fn with_padding(&mut self, padding: Option<PaddingParams>) -> &mut Self {
        self.padding = padding;
        self
    }

    /// Get the currently set padding parameters
    pub fn get_padding(&self) -> Option<&PaddingParams> {
        self.padding.as_ref()
    }

    /// Get a mutable reference to the currently set padding parameters
    pub fn get_padding_mut(&mut self) -> Option<&mut PaddingParams> {
        self.padding.as_mut()
    }

    /// Get the vocabulary
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

    /// Get the size of the vocabulary
    pub fn get_vocab_size(&self, with_added_tokens: bool) -> usize {
        self.model.get_vocab_size()
            + if with_added_tokens {
                self.added_vocabulary.len()
            } else {
                0
            }
    }

    /// Converts a token in the corresponding id.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.added_vocabulary.token_to_id(token, &self.model)
    }

    /// Converts an id to the corresponding token.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.added_vocabulary.id_to_token(id, &self.model)
    }

    /// Encode a single sequence
    fn encode_single_sequence(
        &self,
        sequence: InputSequence,
        type_id: u32,
        offsets_type: OffsetType,
    ) -> Result<Encoding> {
        let encode = |is_pre_tokenized, subseq_idx, subseq| -> Result<Encoding> {
            let normalized = self
                .added_vocabulary
                .extract_and_normalize(self.normalizer.as_ref(), subseq);
            let pre_tokenized = self.do_pre_tokenize(normalized)?;
            let subseq_encoding = self.do_tokenize(
                pre_tokenized,
                type_id,
                if is_pre_tokenized {
                    Some(subseq_idx as u32)
                } else {
                    None
                },
                offsets_type,
            )?;

            Ok(subseq_encoding)
        };

        match sequence {
            InputSequence::PreTokenized(seq) => seq
                .iter()
                .enumerate()
                .map(|(i, sequence)| encode(true, i, sequence))
                .collect(),
            InputSequence::PreTokenizedOwned(seq) => seq
                .iter()
                .enumerate()
                .map(|(i, sequence)| encode(true, i, sequence))
                .collect(),
            InputSequence::PreTokenizedCow(seq) => seq
                .iter()
                .enumerate()
                .map(|(i, sequence)| encode(true, i, sequence))
                .collect(),
            InputSequence::Raw(seq) => encode(false, 0, seq.as_ref()),
        }
    }

    /// Encode the given input. This method accepts both single sequences, as well as pair
    /// sequences. Also, a sequence can be a string, or already pre-tokenized input directly:
    ///
    /// ```
    /// # use tokenizers::Tokenizer;
    /// # use tokenizers::models::bpe::BPE;
    /// # let mut tokenizer = Tokenizer::new(BPE::default());
    /// #
    /// // Sequences:
    /// tokenizer.encode("Single sequence", false);
    /// tokenizer.encode(("Sequence A", "Sequence B"), false);
    ///
    /// // Pre-tokenized sequences:
    /// tokenizer.encode(&["Single", "sequence"][..], false);
    /// tokenizer.encode((
    ///     &["Sequence", "A"][..],
    ///     &["Sequence", "B"][..]
    /// ), false);
    ///
    /// // or even both types together:
    /// tokenizer.encode(("A complete sequence", &["And", "a", "tokenized"][..]), false);
    /// ```
    pub fn encode<'s, E>(&self, input: E, add_special_tokens: bool) -> Result<Encoding>
    where
        E: Into<EncodeInput<'s>>,
    {
        // Extract sequences from the EncodeInput
        let (sequence, pair) = match input.into() {
            EncodeInput::Single(s1) => (s1, None),
            EncodeInput::Dual(s1, s2) => (s1, Some(s2)),
        };

        // Encode each sequence
        let encoding = self.encode_single_sequence(sequence, 0, OffsetType::Byte)?;
        let pair_encoding = match pair {
            Some(sequence) => Some(self.encode_single_sequence(sequence, 1, OffsetType::Byte)?),
            None => None,
        };

        // And finally post process
        self.post_process(encoding, pair_encoding, add_special_tokens)
    }

    /// Encode the given input, using offsets relative to chars instead of bytes.
    /// This method accepts both single sequences, as well as pair sequences. Also,
    /// a sequence can be a string, or already pre-tokenized input directly:
    ///
    /// ```
    /// # use tokenizers::Tokenizer;
    /// # use tokenizers::models::bpe::BPE;
    /// # let mut tokenizer = Tokenizer::new(BPE::default());
    /// #
    /// // Sequences:
    /// tokenizer.encode("Single sequence", false);
    /// tokenizer.encode(("Sequence A", "Sequence B"), false);
    ///
    /// // Pre-tokenized sequences:
    /// tokenizer.encode(&["Single", "sequence"][..], false);
    /// tokenizer.encode((
    ///     &["Sequence", "A"][..],
    ///     &["Sequence", "B"][..]
    /// ), false);
    ///
    /// // or even both types together:
    /// tokenizer.encode(("A complete sequence", &["And", "a", "tokenized"][..]), false);
    /// ```
    pub fn encode_char_offsets<'s, E>(&self, input: E, add_special_tokens: bool) -> Result<Encoding>
    where
        E: Into<EncodeInput<'s>>,
    {
        // Extract sequences from the EncodeInput
        let (sequence, pair) = match input.into() {
            EncodeInput::Single(s1) => (s1, None),
            EncodeInput::Dual(s1, s2) => (s1, Some(s2)),
        };

        // Encode each sequence
        let encoding = self.encode_single_sequence(sequence, 0, OffsetType::Char)?;
        let pair_encoding = match pair {
            Some(sequence) => Some(self.encode_single_sequence(sequence, 1, OffsetType::Char)?),
            None => None,
        };

        // And finally post process
        self.post_process(encoding, pair_encoding, add_special_tokens)
    }

    /// Decode the given ids, back to a String
    pub fn decode(&self, ids: Vec<u32>, skip_special_tokens: bool) -> Result<String> {
        let tokens = ids
            .into_iter()
            .filter_map(|id| {
                self.added_vocabulary
                    .id_to_token(id, &self.model)
                    .filter(|token| {
                        !skip_special_tokens || !self.added_vocabulary.is_special_token(token)
                    })
            })
            .collect::<Vec<_>>();

        if let Some(decoder) = &self.decoder {
            decoder.decode(tokens)
        } else {
            Ok(tokens.join(" "))
        }
    }
}

impl<M, N, PT, PP, D> TokenizerImpl<M, N, PT, PP, D>
where
    M: Model,
{
    /// Tokenization logic, makes the bridge between the pre-tokenization phase and the real
    /// tokenization phase, and converting offsets back to the original referential.
    fn do_tokenize<P: Into<PreTokenizedString>>(
        &self,
        pretokenized: P,
        type_id: u32,
        word_idx: Option<u32>,
        offsets_type: OffsetType,
    ) -> Result<Encoding> {
        let mut pretokenized: PreTokenizedString = pretokenized.into();
        pretokenized.tokenize(|normalized| self.model.tokenize(normalized.get()))?;
        pretokenized.into_encoding(word_idx, type_id, offsets_type)
    }
}

impl<M, N, PT, PP, D> TokenizerImpl<M, N, PT, PP, D>
where
    N: Normalizer,
{
    /// Normalization logic, go through all normalizers
    fn do_normalize<V: Into<NormalizedString>>(&self, normalized: V) -> Result<NormalizedString> {
        let mut normalized: NormalizedString = normalized.into();

        if let Some(ref normalizer) = self.normalizer {
            normalizer.normalize(&mut normalized)?;
        }

        Ok(normalized)
    }
}

impl<M, N, PT, PP, D> TokenizerImpl<M, N, PT, PP, D>
where
    N: Normalizer,
    M: Model,
{
    /// Register the given tokens as special tokens. This is especially useful for removing
    /// these special tokens while decoding
    pub fn add_special_tokens(&mut self, tokens: &[AddedToken]) -> usize {
        self.added_vocabulary
            .add_special_tokens(tokens, &self.model, self.normalizer.as_ref())
    }

    /// Add the given tokens to the added vocabulary
    pub fn add_tokens(&mut self, tokens: &[AddedToken]) -> usize {
        self.added_vocabulary
            .add_tokens(tokens, &self.model, self.normalizer.as_ref())
    }
}

impl<M, N, PT, PP, D> TokenizerImpl<M, N, PT, PP, D>
where
    PT: PreTokenizer,
{
    /// PreTokenization logic, handling the case where there is no PreTokenizer set
    fn do_pre_tokenize<P: Into<PreTokenizedString>>(
        &self,
        pretokenized: P,
    ) -> Result<PreTokenizedString> {
        let mut pretokenized: PreTokenizedString = pretokenized.into();

        if let Some(ref pretok) = self.pre_tokenizer {
            pretok.pre_tokenize(&mut pretokenized)?;
        }

        Ok(pretokenized)
    }
}

impl<M, N, PT, PP, D> TokenizerImpl<M, N, PT, PP, D>
where
    PP: PostProcessor,
{
    /// Post processing logic, handling the case where there is no PostProcessor set
    pub fn post_process(
        &self,
        encoding: Encoding,
        pair_encoding: Option<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        // 1. First we truncate if needed
        let (encoding, pair_encoding) = {
            if let Some(trunc) = &self.truncation {
                let n_added_tokens = if let Some(processor) = &self.post_processor {
                    processor.added_tokens(pair_encoding.is_some())
                } else {
                    0
                };

                if add_special_tokens && n_added_tokens > 0 {
                    let params = TruncationParams {
                        max_length: trunc.max_length - n_added_tokens,
                        ..*trunc
                    };
                    truncate_encodings(encoding, pair_encoding, &params)?
                } else {
                    truncate_encodings(encoding, pair_encoding, &trunc)?
                }
            } else {
                (encoding, pair_encoding)
            }
        };

        // 2. Then We post process
        let final_encoding = if let Some(processor) = &self.post_processor {
            processor.process(encoding, pair_encoding, add_special_tokens)?
        } else {
            PostProcessor::default_process(encoding, pair_encoding, add_special_tokens)?
        };

        // 3. Then we pad if needed
        let [final_encoding] = if let Some(params) = &self.padding {
            let mut arr = [final_encoding];
            pad_encodings(&mut arr, params)?;
            arr
        } else {
            [final_encoding]
        };

        Ok(final_encoding)
    }
}

impl<M, N, PT, PP, D> TokenizerImpl<M, N, PT, PP, D>
where
    M: Model + Send + Sync,
    N: Normalizer + Send + Sync,
    PT: PreTokenizer + Send + Sync,
    PP: PostProcessor + Send + Sync,
    D: Decoder + Send + Sync,
{
    /// Encode all the sentences in parallel, using multiple threads
    pub fn encode_batch<'s, E>(
        &self,
        inputs: Vec<E>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>>
    where
        E: Into<EncodeInput<'s>> + Send,
    {
        let mut encodings = inputs
            .into_maybe_par_iter()
            .map(|input| self.encode(input, add_special_tokens))
            .collect::<Result<Vec<Encoding>>>()?;

        if let Some(params) = &self.padding {
            // We do the padding here to make sure we handle the batch padding
            pad_encodings(&mut encodings, &params)?;
        }

        Ok(encodings)
    }

    /// Encode all the sentences in parallel, using multiple threads.
    /// The offsets on each `Encoding` will be relative to chars instead of bytes.
    pub fn encode_batch_char_offsets<'s, E>(
        &self,
        inputs: Vec<E>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>>
    where
        E: Into<EncodeInput<'s>> + Send,
    {
        let mut encodings = inputs
            .into_maybe_par_iter()
            .map(|input| self.encode_char_offsets(input, add_special_tokens))
            .collect::<Result<Vec<Encoding>>>()?;

        if let Some(params) = &self.padding {
            // We do the padding here to make sure we handle the batch padding
            pad_encodings(&mut encodings, &params)?;
        }

        Ok(encodings)
    }

    /// Decode all sentences in parallel
    pub fn decode_batch(
        &self,
        sentences: Vec<Vec<u32>>,
        skip_special_tokens: bool,
    ) -> Result<Vec<String>>
    where
        M: Send + Sync,
    {
        sentences
            .into_maybe_par_iter()
            .map(|sentence| self.decode(sentence, skip_special_tokens))
            .collect()
    }

    /// Train our Model from files
    pub fn train_from_files<T>(&mut self, trainer: &mut T, files: Vec<String>) -> Result<&mut Self>
    where
        T: Trainer<Model = M> + Sync,
    {
        let mut len = 0;
        for file in files.iter() {
            len += File::open(file)
                .and_then(|f| f.metadata())
                .map(|m| m.len())?;
        }

        let max_read = 1_000_000;

        ResultShunt::process(
            files.into_iter().flat_map(|filename| {
                match File::open(filename) {
                    Ok(file) => {
                        let file = BufReader::with_capacity(max_read, file);
                        // We read new lines using this API instead of the Lines Iterator
                        // on purpose. We want to keep the `\n` and potential `\r` between each lines
                        // We use an iterator to be able to chain with par_bridge.
                        itertools::Either::Left(file.lines_with_ending())
                    }
                    Err(e) => itertools::Either::Right(std::iter::once(Err(e))),
                }
            }),
            |sequences| -> Result<()> {
                let progress = if trainer.should_show_progress() {
                    let progress = ProgressBar::new(len);
                    progress.set_style(
                        ProgressStyle::default_bar()
                            .template("[{elapsed_precise}] {msg:<40!} {wide_bar} {percent:>18!}%"),
                    );
                    progress
                        .set_message(&format!("Pre-processing files ({:.2} Mo)", len / 1_000_000));
                    progress.set_draw_delta(len / 100); // Redraw only every 2%
                    Some(progress)
                } else {
                    None
                };

                trainer.feed(
                    sequences.map(|s| {
                        if let Some(progress) = &progress {
                            progress.inc(s.len() as u64)
                        }
                        s
                    }),
                    |seq| {
                        let normalized = self.do_normalize(seq.as_ref())?;
                        let pre_tokenized = self.do_pre_tokenize(normalized)?;
                        Ok(pre_tokenized
                            .get_splits(OffsetReferential::Original, OffsetType::Byte)
                            .into_iter()
                            .map(|(s, _, _)| s.to_owned())
                            .collect())
                    },
                )?;

                if let Some(pbar) = progress {
                    pbar.finish();
                }
                let special_tokens = trainer.train(&mut self.model)?;
                self.add_special_tokens(&special_tokens);

                Ok(())
            },
        )??;
        Ok(self)
    }

    /// Train our Model, using the given Trainer and iterator
    pub fn train<T, I, S>(&mut self, trainer: &mut T, sequences: I) -> Result<&mut Self>
    where
        T: Trainer<Model = M> + Sync,
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
    {
        let (lower, upper) = sequences.size_hint();
        let len = upper.unwrap_or(lower) as u64;
        let progress = if trainer.should_show_progress() {
            let progress = ProgressBar::new(len);
            progress.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {msg:<40!} {wide_bar} {pos:<9!}/{len:>9!}"),
            );
            progress.set_message("Pre-processing sequences");
            if len > 0 {
                progress.set_draw_delta(len / 100); // Redraw only every 2%
            } else {
                // Trying to have a good default to avoid progress tracking being the bottleneck
                progress.set_draw_delta(1000);
            }
            Some(progress)
        } else {
            None
        };

        trainer.feed(
            sequences.map(|s| {
                if let Some(progress) = &progress {
                    progress.inc(1)
                }
                s
            }),
            |seq| {
                let normalized = self.do_normalize(seq.as_ref())?;
                let pre_tokenized = self.do_pre_tokenize(normalized)?;
                Ok(pre_tokenized
                    .get_splits(OffsetReferential::Original, OffsetType::Byte)
                    .into_iter()
                    .map(|(s, _, _)| s.to_owned())
                    .collect())
            },
        )?;
        if let Some(pbar) = progress {
            pbar.finish();
        }

        let special_tokens = trainer.train(&mut self.model)?;
        self.add_special_tokens(&special_tokens);

        Ok(self)
    }
}

impl<M, N, PT, PP, D> std::str::FromStr for TokenizerImpl<M, N, PT, PP, D>
where
    M: for<'de> Deserialize<'de> + Model,
    N: for<'de> Deserialize<'de> + Normalizer,
    PT: for<'de> Deserialize<'de> + PreTokenizer,
    PP: for<'de> Deserialize<'de> + PostProcessor,
    D: for<'de> Deserialize<'de> + Decoder,
{
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }
}

impl<M, N, PT, PP, D> TokenizerImpl<M, N, PT, PP, D>
where
    M: DeserializeOwned + Model,
    N: DeserializeOwned + Normalizer,
    PT: DeserializeOwned + PreTokenizer,
    PP: DeserializeOwned + PostProcessor,
    D: DeserializeOwned + Decoder,
{
    /// Instantiate a new Tokenizer from the given file
    pub fn from_file<P: AsRef<Path>>(file: P) -> Result<Self> {
        let content = read_to_string(file)?;
        Ok(serde_json::from_str(&content)?)
    }
}

impl<M, N, PT, PP, D> TokenizerImpl<M, N, PT, PP, D>
where
    M: Serialize,
    N: Serialize,
    PT: Serialize,
    PP: Serialize,
    D: Serialize,
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
        file.write_all(&serialized.as_bytes())?;

        Ok(())
    }
}
