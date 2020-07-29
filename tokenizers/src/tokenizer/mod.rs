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

pub use crate::utils::iter::LinesWithEnding;
pub use crate::utils::padding::{pad_encodings, PaddingDirection, PaddingParams, PaddingStrategy};
pub use crate::utils::truncation::{truncate_encodings, TruncationParams, TruncationStrategy};
use indicatif::{ProgressBar, ProgressStyle};
use normalizer::Range;
use std::{
    collections::HashMap,
    fs::File,
    io::prelude::*,
    io::BufReader,
    path::{Path, PathBuf},
};

mod added_vocabulary;
mod encoding;
pub mod normalizer;
pub mod pattern;
pub mod pre_tokenizer;
mod serialization;

pub use added_vocabulary::*;
pub use encoding::*;
pub use normalizer::{NormalizedString, SplitDelimiterBehavior};
pub use pre_tokenizer::*;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;
pub type Offsets = (usize, usize);

use crate::utils::parallelism::*;

#[typetag::serde(tag = "type")]
/// Takes care of pre-processing strings.
pub trait Normalizer: Send + Sync {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()>;
}

#[typetag::serde(tag = "type")]
/// The `PreTokenizer` is in charge of doing the pre-segmentation step. It splits the given string
/// in multiple substrings, keeping track of the offsets of said substrings from the
/// `NormalizedString`. In some occasions, the `PreTokenizer` might need to modify the given
/// `NormalizedString` to ensure we can entirely keep track of the offsets and the mapping with
/// the original string.
pub trait PreTokenizer: Send + Sync {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()>;
}

#[typetag::serde(tag = "type")]
/// Represents a model used during Tokenization (like BPE or Word or Unigram).
pub trait Model: Send + Sync {
    /// Tokenize the given sequence into multiple underlying `Token`. The `offsets` on the `Token`
    /// are expected to be relative to the given sequence.
    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>>;
    /// Find the ID associated to a string token
    fn token_to_id(&self, token: &str) -> Option<u32>;
    /// Find the string token associated to an ID
    fn id_to_token(&self, id: u32) -> Option<&str>;
    /// Retrieve the entire vocabulary mapping (token -> ID)
    fn get_vocab(&self) -> &HashMap<String, u32>;
    /// Retrieve the size of the vocabulary
    fn get_vocab_size(&self) -> usize;
    /// Save the current `Model` in the given folder, using the given `prefix` for the various
    /// files that need to be saved.
    fn save(&self, folder: &Path, prefix: Option<&str>) -> Result<Vec<PathBuf>>;
}

#[typetag::serde(tag = "type")]
/// A `PostProcessor` has the responsibility to post process an encoded output of the `Tokenizer`.
/// It adds any special tokens that a language model would require.
pub trait PostProcessor: Send + Sync {
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
            Some(pair) => {
                encoding.merge_with(pair, false);
                Ok(encoding)
            }
        }
    }
}

#[typetag::serde(tag = "type")]
/// A `Decoder` has the responsibility to merge the given `Vec<String>` in a `String`.
pub trait Decoder: Send + Sync {
    fn decode(&self, tokens: Vec<String>) -> Result<String>;
}

/// A `Trainer` has the responsibility to train a model. We feed it with lines/sentences
/// and it returns a `Model` when done.
pub trait Trainer: Sync {
    /// Whether we should show progress during the training.
    fn should_show_progress(&self) -> bool;
    /// The actual training method. This will return a new trained Model as well as a list
    /// of `special_tokens` to be added directly to the tokenizer along with the model.
    fn train(&self, words: HashMap<String, u32>) -> Result<(Box<dyn Model>, Vec<AddedToken>)>;
    /// Process a bunch of token, counting them as relevant.
    fn process_tokens(&self, words: &mut HashMap<String, u32>, tokens: Vec<String>);
}

#[derive(Debug, PartialEq)]
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

#[derive(Debug, Clone)]
pub enum InputSequence {
    Raw(String),
    PreTokenized(Vec<String>),
}

impl From<String> for InputSequence {
    fn from(input: String) -> Self {
        InputSequence::Raw(input)
    }
}

impl From<&str> for InputSequence {
    fn from(input: &str) -> Self {
        InputSequence::Raw(input.to_owned())
    }
}

impl From<Vec<String>> for InputSequence {
    fn from(input: Vec<String>) -> Self {
        InputSequence::PreTokenized(input)
    }
}

impl From<&[String]> for InputSequence {
    fn from(input: &[String]) -> Self {
        InputSequence::PreTokenized(input.to_vec())
    }
}

impl From<&[&str]> for InputSequence {
    fn from(input: &[&str]) -> Self {
        InputSequence::PreTokenized(input.iter().map(|&i| i.to_string()).collect())
    }
}

#[derive(Debug, Clone)]
pub enum EncodeInput {
    Single(InputSequence),
    Dual(InputSequence, InputSequence),
}

impl<I: Into<InputSequence>> From<I> for EncodeInput {
    fn from(input: I) -> Self {
        EncodeInput::Single(input.into())
    }
}

impl<I1: Into<InputSequence>, I2: Into<InputSequence>> From<(I1, I2)> for EncodeInput {
    fn from(input: (I1, I2)) -> Self {
        EncodeInput::Dual(input.0.into(), input.1.into())
    }
}

/// A `Tokenizer` is capable of encoding/decoding any text.
pub struct Tokenizer {
    // Tokenizer parts
    normalizer: Option<Box<dyn Normalizer>>,
    pre_tokenizer: Option<Box<dyn PreTokenizer>>,
    model: Box<dyn Model>,
    post_processor: Option<Box<dyn PostProcessor>>,
    decoder: Option<Box<dyn Decoder>>,

    // Added Vocabulary capabilities
    added_vocabulary: AddedVocabulary,

    // General processing parameters
    truncation: Option<TruncationParams>,
    padding: Option<PaddingParams>,
}

impl std::str::FromStr for Tokenizer {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }
}

impl Tokenizer {
    /// Instantiate a new Tokenizer, with the given Model
    pub fn new(model: Box<dyn Model>) -> Self {
        Tokenizer {
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

    /// Instantiate a new Tokenizer from the given file
    pub fn from_file<P: AsRef<Path>>(file: P) -> Result<Self> {
        let file = File::open(file)?;
        let buf = BufReader::new(file);
        Ok(serde_json::from_reader(buf)?)
    }

    /// Serialize the current tokenizer as a String
    pub fn to_string(&self, pretty: bool) -> Result<String> {
        Ok(if pretty {
            serde_json::to_string_pretty(self)?
        } else {
            serde_json::to_string(self)?
        })
    }

    /// Save the current tokenizer at the given path
    pub fn save(&self, path: &str, pretty: bool) -> Result<()> {
        let serialized = self.to_string(pretty)?;

        let mut file = File::create(path)?;
        file.write_all(&serialized.as_bytes())?;

        Ok(())
    }

    /// Set the normalizer
    pub fn with_normalizer(&mut self, normalizer: Box<dyn Normalizer>) -> &Self {
        self.normalizer = Some(normalizer);
        self
    }

    /// Get the normalizer
    #[allow(clippy::borrowed_box)]
    pub fn get_normalizer(&self) -> Option<&Box<dyn Normalizer>> {
        self.normalizer.as_ref()
    }

    /// Set the pre tokenizer
    pub fn with_pre_tokenizer(&mut self, pre_tokenizer: Box<dyn PreTokenizer>) -> &Self {
        self.pre_tokenizer = Some(pre_tokenizer);
        self
    }

    /// Get the pre tokenizer
    #[allow(clippy::borrowed_box)]
    pub fn get_pre_tokenizer(&self) -> Option<&Box<dyn PreTokenizer>> {
        self.pre_tokenizer.as_ref()
    }

    /// Set the post processor
    pub fn with_post_processor(&mut self, post_processor: Box<dyn PostProcessor>) -> &Self {
        self.post_processor = Some(post_processor);
        self
    }

    /// Get the post processor
    #[allow(clippy::borrowed_box)]
    pub fn get_post_processor(&self) -> Option<&Box<dyn PostProcessor>> {
        self.post_processor.as_ref()
    }

    /// Set the decoder
    pub fn with_decoder(&mut self, decoder: Box<dyn Decoder>) -> &Self {
        self.decoder = Some(decoder);
        self
    }

    /// Get the decoder
    #[allow(clippy::borrowed_box)]
    pub fn get_decoder(&self) -> Option<&Box<dyn Decoder>> {
        self.decoder.as_ref()
    }

    /// Set the model
    pub fn with_model(&mut self, model: Box<dyn Model>) -> &Self {
        self.model = model;
        self
    }

    /// Get the model
    #[allow(clippy::borrowed_box)]
    pub fn get_model(&self) -> &Box<dyn Model> {
        &self.model
    }

    /// Set the truncation parameters
    pub fn with_truncation(&mut self, trunc: Option<TruncationParams>) -> &Self {
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
    pub fn with_padding(&mut self, padding: Option<PaddingParams>) -> &Self {
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
        let mut final_vocab = self.model.get_vocab().clone();

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
        self.added_vocabulary
            .token_to_id(token, self.model.as_ref())
    }

    /// Converts an id to the corresponding token.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.added_vocabulary.id_to_token(id, self.model.as_ref())
    }

    /// Normalize the given sentence and return the corresponding normalized string
    pub fn normalize(&self, sentence: &str) -> Result<NormalizedString> {
        self.added_vocabulary
            .extract_and_normalize(self.normalizer.as_deref(), sentence)
            .flat_map(|(sentence, _, id)| {
                if id.is_some() {
                    itertools::Either::Left(std::iter::once(Ok(sentence)))
                } else {
                    // The PreTokenizers can still manipulate the normalized strings, so we do
                    // this anyway, and will merge it back to a NormalizedString
                    match self.do_pre_tokenize(sentence) {
                        Ok(pretok) => itertools::Either::Right(
                            pretok.into_iter().map(|substring| Ok(substring.normalized)),
                        ),
                        Err(e) => itertools::Either::Left(std::iter::once(Err(e))),
                    }
                }
            })
            .collect::<Result<NormalizedString>>()
    }

    /// Encode a single sequence
    fn encode_single_sequence(&self, sequence: InputSequence, type_id: u32) -> Result<Encoding> {
        let (sequence, pre_tokenized) = match sequence {
            InputSequence::PreTokenized(seq) => (seq, true),
            InputSequence::Raw(seq) => (vec![seq], false),
        };

        sequence
            .into_iter()
            .enumerate()
            .map(|(subseq_idx, subseq)| {
                let mut subseq_encoding = self
                    .added_vocabulary
                    .extract_and_normalize(self.normalizer.as_deref(), &subseq)
                    .map(|(normalized, original_offsets, id)| match id {
                        Some(id) => Ok(Encoding::from_tokens(
                            vec![Token::new(
                                id,
                                normalized.get().to_owned(),
                                original_offsets,
                            )],
                            type_id,
                        )),
                        None => self.do_tokenize(
                            self.do_pre_tokenize(normalized)?,
                            original_offsets,
                            type_id,
                        ),
                    })
                    .collect::<Result<Encoding>>()?;

                // If we are handling already pre-tokenized input, each word should have the
                // relevant index from the given input, not determined by the pre-tokenization step
                if pre_tokenized {
                    subseq_encoding.get_words_mut().iter_mut().for_each(|word| {
                        if let Some(ref mut word) = word {
                            *word = subseq_idx as u32;
                        }
                    });
                }

                Ok(subseq_encoding)
            })
            .collect::<Result<Encoding>>()
    }

    /// Encode the given input. This method accepts both single sequences, as well as pair
    /// sequences. Also, a sequence can be a string, or already pre-tokenized input directly:
    ///
    /// ```
    /// # use tokenizers::Tokenizer;
    /// # use tokenizers::models::bpe::BPE;
    /// # let tokenizer = Tokenizer::new(Box::new(BPE::default()));
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
    pub fn encode<E: Into<EncodeInput>>(
        &self,
        input: E,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        // Extract sequences from the EncodeInput
        let (sequence, pair) = match input.into() {
            EncodeInput::Single(s1) => (s1, None),
            EncodeInput::Dual(s1, s2) => (s1, Some(s2)),
        };

        // Encode each sequence
        let encoding = self.encode_single_sequence(sequence, 0)?;
        let pair_encoding = match pair {
            Some(sequence) => Some(self.encode_single_sequence(sequence, 1)?),
            None => None,
        };

        // And finally post process
        self.post_process(encoding, pair_encoding, add_special_tokens)
    }

    /// Encode all the sentences in parallel, using multiple threads
    pub fn encode_batch<E: Into<EncodeInput> + Send>(
        &self,
        inputs: Vec<E>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
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

    /// Decode the given ids, back to a String
    pub fn decode(&self, ids: Vec<u32>, skip_special_tokens: bool) -> Result<String> {
        let tokens = ids
            .into_iter()
            .filter_map(|id| {
                self.added_vocabulary
                    .id_to_token(id, self.model.as_ref())
                    .filter(|token| {
                        !skip_special_tokens || !self.added_vocabulary.is_special_token(token)
                    })
                    .map(|t| t.to_owned())
            })
            .collect::<Vec<_>>();

        if let Some(decoder) = &self.decoder {
            decoder.decode(tokens)
        } else {
            Ok(tokens.join(" "))
        }
    }

    /// Decode all sentences in parallel
    pub fn decode_batch(
        &self,
        sentences: Vec<Vec<u32>>,
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        sentences
            .into_maybe_par_iter()
            .map(|sentence| self.decode(sentence, skip_special_tokens))
            .collect()
    }

    /// Train a model and replace our current Model, using the given Trainer
    #[allow(clippy::borrowed_box)]
    fn word_count(
        &mut self,
        trainer: &Box<dyn Trainer>,
        files: Vec<String>,
    ) -> Result<HashMap<String, u32>> {
        let max_read = 1_000_000;
        let mut len = 0;
        for file in files.iter() {
            len += File::open(file)
                .and_then(|f| f.metadata())
                .map(|m| m.len())?;
        }

        let progress = if trainer.should_show_progress() {
            let progress = ProgressBar::new(len);
            progress.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {msg:<40!} {wide_bar} {percent:>19!}"),
            );
            progress.set_message(&format!("Reading files ({:.2} Mo)", len / 1_000_000));
            progress.set_draw_delta(len / 100); // Redraw only every 2%
            Some(progress)
        } else {
            None
        };
        let words = files
            .into_iter()
            .map(|filename| -> Result<HashMap<String, u32>> {
                let file = File::open(filename)?;
                let file = BufReader::with_capacity(max_read, file);
                // We read new lines using this API instead of the Lines Iterator
                // on purpose. We want to keep the `\n` and potential `\r` between each lines
                // We use an iterator to be able to chain with par_bridge.
                file.lines_with_ending()
                    .maybe_par_bridge()
                    .map_with(
                        &progress,
                        |progress, line| -> Result<HashMap<String, u32>> {
                            let newline = line?;
                            let b = newline.len();
                            let mut words = HashMap::new();
                            let normalized = self.do_normalize(newline)?;
                            let pre_tokenized = self.do_pre_tokenize(normalized)?;
                            trainer.process_tokens(
                                &mut words,
                                pre_tokenized
                                    .into_iter()
                                    .map(|sub| sub.normalized.get().to_owned())
                                    .collect(),
                            );

                            if let Some(pbar) = progress {
                                pbar.inc(b as u64);
                            }
                            Ok(words)
                        },
                    )
                    .reduce(
                        || Ok(HashMap::new()),
                        |acc, ws| {
                            let mut acc = acc?;
                            for (k, v) in ws? {
                                acc.entry(k).and_modify(|c| *c += v).or_insert(v);
                            }
                            Ok(acc)
                        },
                    )
            })
            .try_fold(
                HashMap::new(),
                |mut acc, ws| -> Result<HashMap<String, u32>> {
                    for (k, v) in ws? {
                        acc.entry(k).and_modify(|c| *c += v).or_insert(v);
                    }
                    Ok(acc)
                },
            )?;
        if let Some(pbar) = progress {
            pbar.finish();
        }
        Ok(words)
    }

    /// Train a model and replace our current Model, using the given Trainer
    #[allow(clippy::borrowed_box)]
    pub fn train(&mut self, trainer: &Box<dyn Trainer>, files: Vec<String>) -> Result<()> {
        let words = self.word_count(trainer, files)?;

        let (model, special_tokens) = trainer.train(words)?;
        self.model = model;
        self.add_special_tokens(&special_tokens);

        Ok(())
    }

    /// Tokenization logic, makes the bridge between the pre-tokenization phase and the real
    /// tokenization phase, and converting offsets back to the original referential.
    fn do_tokenize<P: Into<PreTokenizedString>>(
        &self,
        pretokenized: P,
        original_offsets: Offsets,
        type_id: u32,
    ) -> Result<Encoding> {
        let pretokenized: PreTokenizedString = pretokenized.into();

        pretokenized
            .into_iter()
            .enumerate()
            .map(|(word_idx, substr)| {
                let mut tokens = self.model.tokenize(substr.normalized.get())?;

                // Update the offsets to match the original input
                tokens.iter_mut().for_each(|token| {
                    // We convert the normalized offsets back to the original
                    let converted_offsets = substr
                        .normalized
                        .convert_offsets(Range::Normalized(token.offsets.0..token.offsets.1))
                        .map_or(token.offsets, |range| {
                            (
                                original_offsets.0 + substr.original_offsets.0 + range.start,
                                original_offsets.0 + substr.original_offsets.0 + range.end,
                            )
                        });

                    // And we update the token to these original offsets, applying the original offset
                    // of the sequence we just tokenized.
                    token.offsets = converted_offsets;
                });

                // Then build the encoding from these tokens, setting the `words` as relevant
                let mut encoding = Encoding::from_tokens(tokens, type_id);
                encoding.get_words_mut().iter_mut().for_each(|word| {
                    *word = Some(word_idx as u32);
                });

                Ok(encoding)
            })
            .collect()
    }

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

    /// Normalization logic, go through all normalizers
    fn do_normalize<N: Into<NormalizedString>>(&self, normalized: N) -> Result<NormalizedString> {
        let mut normalized: NormalizedString = normalized.into();

        if let Some(ref normalizer) = self.normalizer {
            normalizer.normalize(&mut normalized)?;
        }

        Ok(normalized)
    }

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

    /// Register the given tokens as special tokens. This is especially useful for removing
    /// these special tokens while decoding
    pub fn add_special_tokens(&mut self, tokens: &[AddedToken]) -> usize {
        self.added_vocabulary.add_special_tokens(
            tokens,
            self.model.as_ref(),
            self.normalizer.as_deref(),
        )
    }

    /// Add the given tokens to the added vocabulary
    pub fn add_tokens(&mut self, tokens: &[AddedToken]) -> usize {
        self.added_vocabulary
            .add_tokens(tokens, self.model.as_ref(), self.normalizer.as_deref())
    }
}
