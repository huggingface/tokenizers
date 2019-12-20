//!
//! # Tokenizer module
//!
//! Represents a tokenization pipeline.
//!
//! A Tokenizer is composed of some of the following parts.
//!   - Normalizer: Takes care of the text normalization (like unicode normalization).
//!   - PreTokenizer: Takes care of the pre tokenization (ie. How to split tokens and pre-process
//!   them.
//!   - Model: A model encapsulates the tokenization algorithm. (Like BPE, Word base, character
//!   based, ...)
//!   - PostProcessor: Takes care of the processing after tokenization. (Like truncating, padding,
//!   ...)
//!
pub use crate::utils::{
    pad_encodings, truncate_encodings, PaddingParams, PaddingStrategy, TruncationParams,
    TruncationStrategy,
};
use rayon::prelude::*;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
};

mod encoding;
pub use encoding::*;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// A Normalizer takes care of pre-processing strings
pub trait Normalizer {
    fn normalize(&self, s: String) -> Result<String>;
}

/// A PreTokenizer takes care of pre-tokenizing strings before this goes to the model
pub trait PreTokenizer {
    // TODO: Should return offsets with each substring
    fn pre_tokenize(&self, s: &str) -> Result<Vec<String>>;
}

/// Represents a `Model` used during Tokenization (Like BPE or Word or Unigram)
pub trait Model {
    fn tokenize(&self, tokens: Vec<String>) -> Result<Vec<Token>>;
    fn token_to_id(&self, token: &str) -> Option<u32>;
    fn id_to_token(&self, id: u32) -> Option<String>;
    fn get_vocab_size(&self) -> usize;
}

/// A PostProcessor has the responsibility to post process an encoded output of the Tokenizer.
/// It adds any special tokens that a language model would require
pub trait PostProcessor {
    /// Returns the number of tokens that will be added during the processing step
    fn added_tokens(&self, encoding: &Encoding, pair_encoding: &Option<Encoding>) -> Result<usize>;
    /// Process both encodings and returns a new merged one
    fn process(&self, encoding: Encoding, pair_encoding: Option<Encoding>) -> Result<Encoding>;
}

/// A Decoder has the responsibility to merge the given Vec<String> in a String
pub trait Decoder {
    fn decode(&self, tokens: Vec<String>) -> Result<String>;
}

/// A Trainer has the responsibility to train a Model. We feed it with lines/sentences
/// and it returns a Model when done.
pub trait Trainer: Sync {
    fn train(&self, words: HashMap<String, u32>) -> Result<Box<dyn Model + Sync>>;
    fn process_tokens(&self, words: &mut HashMap<String, u32>, tokens: Vec<String>);
}

/// A Token
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

pub enum EncodeInput {
    Single(String),
    Dual(String, String),
}

#[derive(Debug, Clone)]
pub struct AddedToken {
    /// The content of the added token
    pub content: String,
    /// Whether this token must be a single word or can break words
    pub single_word: bool,
}
impl AddedToken {
    fn from(content: String) -> Self {
        AddedToken {
            content,
            ..Default::default()
        }
    }
}
impl Default for AddedToken {
    fn default() -> Self {
        AddedToken {
            content: String::new(),
            single_word: false,
        }
    }
}
// We only want to hash on the content. AddedToken cannot be added multiple times with different
// options
impl std::hash::Hash for AddedToken {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.content.hash(state);
    }
}
impl std::cmp::PartialEq for AddedToken {
    fn eq(&self, other: &Self) -> bool {
        self.content == other.content
    }
}
impl std::cmp::Eq for AddedToken {}

///
/// ## Tokenizer
///
/// A Tokenizer is capable of encoding/decoding any text
///
pub struct Tokenizer {
    // Tokenizer parts
    normalizer: Option<Box<dyn Normalizer + Sync>>,
    pre_tokenizer: Option<Box<dyn PreTokenizer + Sync>>,
    model: Box<dyn Model + Sync>,
    post_processor: Option<Box<dyn PostProcessor + Sync>>,
    decoder: Option<Box<dyn Decoder + Sync>>,

    // Added Vocabulary capabilities
    added_tokens: HashMap<AddedToken, u32>,
    added_tokens_r: HashMap<u32, AddedToken>,
    split_re: Option<regex::Regex>,
    special_tokens: HashMap<String, u32>,

    // General processing parameters
    trunc: Option<TruncationParams>,
    padding: Option<PaddingParams>,
}

impl Tokenizer {
    /// Instanciate a new Tokenizer, with the given Model
    pub fn new(model: Box<dyn Model + Sync>) -> Self {
        Tokenizer {
            normalizer: None,
            pre_tokenizer: None,
            model,
            post_processor: None,
            decoder: None,

            added_tokens: HashMap::new(),
            added_tokens_r: HashMap::new(),
            split_re: None,
            special_tokens: HashMap::new(),

            trunc: None,
            padding: None,
        }
    }

    /// Set the normalizer
    pub fn with_normalizer(&mut self, normalizer: Box<dyn Normalizer + Sync>) -> &Self {
        self.normalizer = Some(normalizer);
        self
    }

    /// Set the pre tokenizer
    pub fn with_pre_tokenizer(&mut self, pre_tokenizer: Box<dyn PreTokenizer + Sync>) -> &Self {
        self.pre_tokenizer = Some(pre_tokenizer);
        self
    }

    /// Set the post processor
    pub fn with_post_processor(&mut self, post_processor: Box<dyn PostProcessor + Sync>) -> &Self {
        self.post_processor = Some(post_processor);
        self
    }

    /// Set the decoder
    pub fn with_decoder(&mut self, decoder: Box<dyn Decoder + Sync>) -> &Self {
        self.decoder = Some(decoder);
        self
    }

    /// Set the model
    pub fn with_model(&mut self, model: Box<dyn Model + Sync>) -> &Self {
        self.model = model;
        self
    }

    /// Set the truncation parameters
    pub fn with_truncation(&mut self, trunc: Option<TruncationParams>) -> &Self {
        self.trunc = trunc;
        self
    }

    /// Set the padding strategy
    pub fn with_padding(&mut self, padding: Option<PaddingParams>) -> &Self {
        self.padding = padding;
        self
    }

    /// Get the size of the vocabulary
    pub fn get_vocab_size(&self, with_added_tokens: bool) -> usize {
        self.model.get_vocab_size()
            + if with_added_tokens {
                self.added_tokens.len()
            } else {
                0
            }
    }

    /// Converts a token in the corresponding id.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        if let Some(id) = self.added_tokens.get(&AddedToken::from(token.to_owned())) {
            Some(*id)
        } else {
            self.model.token_to_id(token)
        }
    }

    /// Converts an id to the corresponding token.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        if let Some(token) = self.added_tokens_r.get(&id) {
            Some(token.content.clone())
        } else {
            self.model.id_to_token(id)
        }
    }

    /// Encode the given sentence
    pub fn encode(&self, input: EncodeInput) -> Result<Encoding> {
        let generate_output = move |sentence: String, type_id: u32| -> Result<Encoding> {
            // First we need to split into as many sequences as needed to avoid splitting
            // on our added tokens
            let mut encodings = self
                .split_on_added_tokens(&sentence)
                .into_iter()
                .map(|(sentence, id)| -> Result<Encoding> {
                    // If this is one of our added tokens, lets return an encoding directly
                    if let Some(id) = id {
                        return Ok(Encoding::new(
                            sentence.clone(),
                            sentence.clone(),
                            vec![id],
                            vec![type_id],
                            vec![sentence.to_owned()],
                            vec![(0, sentence.len())],
                            vec![0],
                            vec![1],
                            None,
                        ));
                    }

                    // 1. Normalization
                    // TODO: Make sure we have the offsets update necessary to go from the original text to
                    // the normalized one
                    let original = sentence.clone();
                    let normalized = self.normalize(sentence)?;

                    // 2. Pre tokenization
                    let pre_tokenized = self.pre_tokenize(&normalized)?;

                    // 3. Model
                    let output = self.model.tokenize(pre_tokenized)?;
                    let length = output.len();

                    let (ids, tokens, offsets) = output.into_iter().fold(
                        (
                            Vec::with_capacity(length),
                            Vec::with_capacity(length),
                            Vec::with_capacity(length),
                        ),
                        |(mut ids, mut tokens, mut offsets), t| {
                            ids.push(t.id);
                            tokens.push(t.value);
                            offsets.push(t.offsets);
                            (ids, tokens, offsets)
                        },
                    );

                    Ok(Encoding::new(
                        original,
                        normalized,
                        ids,
                        vec![type_id; length],
                        tokens,
                        offsets,
                        vec![0; length],
                        vec![1; length],
                        None,
                    ))
                })
                .collect::<Result<Vec<Encoding>>>()?;

            if encodings.is_empty() {
                return Ok(Encoding::default());
            }

            let others = encodings.split_off(1);
            let mut first: Encoding = encodings.into_iter().nth(0).unwrap();

            for encoding in others {
                first.merge_with(encoding);
            }

            Ok(first)
        };

        let (sentence, pair) = match input {
            EncodeInput::Single(s1) => (s1, None),
            EncodeInput::Dual(s1, s2) => (s1, Some(s2)),
        };

        let encoding = generate_output(sentence, 0)?;
        let pair_encoding = match pair {
            Some(pair) => Some(generate_output(pair, 1)?),
            None => None,
        };

        // 4. Post processing
        self.post_process(encoding, pair_encoding)
    }

    /// Encode all the sentences in parallel, using multiple threads
    pub fn encode_batch(&self, inputs: Vec<EncodeInput>) -> Result<Vec<Encoding>> {
        let encodings = inputs
            .into_par_iter()
            .map(|input| self.encode(input))
            .collect::<Result<Vec<Encoding>>>()?;

        if let Some(params) = &self.padding {
            // We do the padding here to make sure we handle the batch padding
            pad_encodings(encodings, &params)
        } else {
            Ok(encodings)
        }
    }

    /// Decode the given ids, back to a String
    pub fn decode(&self, ids: Vec<u32>, skip_special_tokens: bool) -> Result<String> {
        let tokens = ids
            .into_iter()
            .map(|id| {
                let token = if let Some(token) = self.added_tokens_r.get(&id) {
                    Some(token.content.to_owned())
                } else {
                    self.model.id_to_token(id)
                };

                token.filter(|token| {
                    !skip_special_tokens || !self.special_tokens.contains_key(token)
                })
            })
            .filter(|token| token.is_some())
            .map(|id| id.unwrap())
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
            .into_par_iter()
            .map(|sentence| self.decode(sentence, skip_special_tokens))
            .collect()
    }

    /// Train a model and replace our current Model, using the given Trainer
    #[allow(clippy::borrowed_box)]
    pub fn train(&mut self, trainer: &Box<dyn Trainer>, files: Vec<String>) -> Result<()> {
        let results = files
            .par_iter()
            .map(|file| -> Result<HashMap<String, u32>> {
                let mut words = HashMap::new();

                let file: std::fs::File = File::open(file)?;
                let file = BufReader::new(file);

                for line in file.lines() {
                    let line = line?;
                    let normalized = self.normalize(line)?;
                    let pre_tokenized = self.pre_tokenize(&normalized)?;
                    trainer.process_tokens(&mut words, pre_tokenized);
                }

                Ok(words)
            })
            .collect::<Vec<_>>();

        let mut words = HashMap::new();
        for result in results {
            for (word, count) in result? {
                words
                    .entry(word)
                    .and_modify(|c| *c += count)
                    .or_insert(count);
            }
        }

        self.model = trainer.train(words)?;

        Ok(())
    }

    /// PreTokenization logic, handling the case where there is no PreTokenizer set
    fn pre_tokenize(&self, sentence: &str) -> Result<Vec<String>> {
        match &self.pre_tokenizer {
            None => Ok(vec![sentence.to_owned()]),
            Some(pre_tokenizer) => pre_tokenizer.pre_tokenize(sentence),
        }
    }

    /// Normalization logic, go through all normalizers
    fn normalize(&self, sentence: String) -> Result<String> {
        if let Some(normalizer) = &self.normalizer {
            normalizer.normalize(sentence)
        } else {
            Ok(sentence)
        }
    }

    /// Post processing logic, handling the case where there is no PostProcessor set
    fn post_process(
        &self,
        encoding: Encoding,
        pair_encoding: Option<Encoding>,
    ) -> Result<Encoding> {
        // 1. First we truncate if needed
        let (mut encoding, pair_encoding) = {
            if let Some(trunc) = &self.trunc {
                let n_added_tokens = if let Some(processor) = &self.post_processor {
                    processor.added_tokens(&encoding, &pair_encoding)?
                } else {
                    0
                };

                if n_added_tokens > 0 {
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
        let mut final_encoding = if let Some(processor) = &self.post_processor {
            processor.process(encoding, pair_encoding)?
        } else {
            match pair_encoding {
                None => encoding,
                Some(pair) => {
                    encoding.merge_with(pair);
                    encoding
                }
            }
        };

        // 3. Then we pad if needed
        if let Some(params) = &self.padding {
            // We can only pad for a given size. If the Strategy is BatchLongest, it will be done
            // when we handle a batch
            if let PaddingStrategy::Fixed(size) = params.strategy {
                final_encoding.pad(
                    size,
                    params.pad_id,
                    params.pad_type_id,
                    &params.pad_token,
                    &params.direction,
                );
            }
        }

        Ok(final_encoding)
    }

    /// Register the given tokens as special tokens. This is especially useful for removing
    /// these special tokens while decoding
    pub fn add_special_tokens(&mut self, tokens: &[&str]) -> usize {
        let added_tokens = tokens
            .iter()
            .map(|t| AddedToken::from((*t).to_owned()))
            .collect::<Vec<_>>();

        let added = self.add_tokens(&added_tokens);
        for token in tokens {
            if let Some(id) = self.token_to_id(token) {
                self.special_tokens.entry((*token).to_owned()).or_insert(id);
            }
        }

        added
    }

    /// Add the given tokens to the added vocabulary
    pub fn add_tokens(&mut self, tokens: &[AddedToken]) -> usize {
        let mut ignored = 0;
        for token in tokens {
            if token.content.is_empty() || self.token_to_id(&token.content).is_some() {
                ignored += 1;
                continue;
            }

            let new_id = (self.model.get_vocab_size() - 1 + self.added_tokens.len()) as u32;
            let id = self
                .added_tokens
                .entry(token.clone())
                .and_modify(|_| ignored += 1)
                .or_insert(new_id);

            // Update the current revert operation
            self.added_tokens_r
                .entry(*id)
                .and_modify(|t| *t = token.clone())
                .or_insert_with(|| token.clone());
        }

        // We rebuild the regex here everytime on purpose, because the added tokens may
        // have changed
        let added_tokens = self
            .added_tokens
            .keys()
            .map(|token| {
                if token.single_word {
                    let first_b = token
                        .content
                        .chars()
                        .nth(0)
                        .map(|c| {
                            if regex_syntax::is_word_character(c) {
                                r"\b"
                            } else {
                                ""
                            }
                        })
                        .unwrap();
                    let last_b = token
                        .content
                        .chars()
                        .last()
                        .map(|c| {
                            if regex_syntax::is_word_character(c) {
                                r"\b"
                            } else {
                                ""
                            }
                        })
                        .unwrap();
                    format!(r"{}{}{}", first_b, regex::escape(&token.content), last_b)
                } else {
                    regex::escape(&token.content)
                }
            })
            .collect::<Vec<_>>();

        self.split_re = Some(regex::Regex::new(&format!(r"({})", added_tokens.join("|"))).unwrap());

        // Return the number of added tokens
        tokens.len() - ignored
    }

    /// Split the given sentence on multiple parts, finding the added tokens and their id in the process
    fn split_on_added_tokens(&self, sentence: &str) -> Vec<(String, Option<u32>)> {
        if let Some(split_re) = &self.split_re {
            let splits = split_re
                .find_iter(&sentence)
                .map(|m| (m.start(), m.end()))
                .collect::<Vec<_>>();

            // We also insert the splits that are inbetween the added tokens, to split the entire string
            let mut start_offset = 0;
            let mut splits = splits
                .into_iter()
                .map(|(start, end)| {
                    let mut splits = vec![];
                    if start_offset < start {
                        splits.push((start_offset, start));
                    }
                    splits.push((start, end));
                    start_offset = end;

                    splits
                })
                .flatten()
                .collect::<Vec<_>>();
            if let Some((_, end)) = splits.iter().last().copied() {
                if end < sentence.len() {
                    splits.push((end, sentence.len()));
                }
            }

            if splits.is_empty() {
                vec![(sentence.to_owned(), None)]
            } else {
                splits
                    .into_iter()
                    .map(|(start, end)| unsafe {
                        let s = sentence.get_unchecked(start..end).to_owned();
                        let id = self.added_tokens.get(&AddedToken {
                            content: s.clone(),
                            ..Default::default()
                        });
                        (s, id.copied())
                    })
                    .collect()
            }
        } else {
            vec![(sentence.to_owned(), None)]
        }
    }
}
