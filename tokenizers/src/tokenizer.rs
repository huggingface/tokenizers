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
use rayon::prelude::*;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
};

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
    fn decode(&self, ids: Vec<u32>) -> Result<Vec<String>>;
    fn token_to_id(&self, token: &str) -> Option<u32>;
    fn id_to_token(&self, id: u32) -> Option<String>;
    fn get_vocab_size(&self) -> usize;
}

/// A PostProcessor has the responsibility to post process an encoded output of the Tokenizer.
/// Truncating, Padding, etc... are PostProcessor steps
pub trait PostProcessor {
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

/// The Encoding struct represents the output of the Tokenizer
#[derive(Default)]
pub struct Encoding {
    original: String,
    normalized: String,
    ids: Vec<u32>,
    type_ids: Vec<u32>,
    tokens: Vec<String>,
    offsets: Vec<(usize, usize)>,
    special_tokens_mask: Vec<u32>,
    attention_mask: Vec<u32>,
    overflowing: Option<Box<Encoding>>,
}
impl Encoding {
    pub fn new(
        original: String,
        normalized: String,
        ids: Vec<u32>,
        type_ids: Vec<u32>,
        tokens: Vec<String>,
        offsets: Vec<(usize, usize)>,
        special_tokens_mask: Vec<u32>,
        attention_mask: Vec<u32>,
        overflowing: Option<Box<Encoding>>,
    ) -> Self {
        Encoding {
            original,
            normalized,
            ids,
            type_ids,
            tokens,
            offsets,
            special_tokens_mask,
            attention_mask,
            overflowing,
        }
    }

    pub fn get_original(&self) -> &str {
        &self.original
    }

    pub fn get_normalized(&self) -> &str {
        &self.normalized
    }

    pub fn get_tokens(&self) -> &[String] {
        &self.tokens[..]
    }

    pub fn get_ids(&self) -> &[u32] {
        &self.ids
    }

    pub fn get_type_ids(&self) -> &[u32] {
        &self.type_ids
    }

    pub fn get_offsets(&self) -> &[(usize, usize)] {
        &self.offsets
    }

    pub fn get_special_tokens_mask(&self) -> &[u32] {
        &self.special_tokens_mask
    }

    pub fn get_attention_mask(&self) -> &[u32] {
        &self.attention_mask
    }

    pub fn take_overflowing(&mut self) -> Option<Box<Encoding>> {
        self.overflowing.take()
    }

    pub fn truncate(&mut self, max_len: usize, stride: usize) {
        if max_len > self.ids.len() {
            return;
        }

        let mut o_ids = self.ids.split_off(max_len);
        let mut o_type_ids = self.type_ids.split_off(max_len);
        let mut o_tokens = self.tokens.split_off(max_len);
        let mut o_offsets = self.offsets.split_off(max_len);
        let mut o_spe_toks = self.special_tokens_mask.split_off(max_len);
        let mut o_attent = self.attention_mask.split_off(max_len);

        // Figure out offsets for original and normalized
        // TODO: We will be able to retrive the right part of original
        // only when we will have the alignment difference between both
        // For now we will use the normalized offset...
        let max = self
            .offsets
            .iter()
            .fold(0, |max, (_, end)| if *end > max { *end } else { max });
        let trunc_original = self.original.split_off(max);
        let trunc_normalized = self.normalized.split_off(max);

        if stride > 0 {
            o_ids = prepend_stride(&self.ids, o_ids, stride);
            o_type_ids = prepend_stride(&self.type_ids, o_type_ids, stride);
            o_tokens = prepend_stride(&self.tokens, o_tokens, stride);
            o_offsets = prepend_stride(&self.offsets, o_offsets, stride);
            o_spe_toks = prepend_stride(&self.special_tokens_mask, o_spe_toks, stride);
            o_attent = prepend_stride(&self.attention_mask, o_attent, stride);
        }

        self.overflowing = Some(Box::new(Encoding {
            original: trunc_original,
            normalized: trunc_normalized,
            ids: o_ids,
            type_ids: o_type_ids,
            tokens: o_tokens,
            offsets: o_offsets,
            special_tokens_mask: o_spe_toks,
            attention_mask: o_attent,
            overflowing: None,
        }));
    }
}

fn prepend_stride<T: Clone>(previous: &Vec<T>, current: Vec<T>, stride: usize) -> Vec<T> {
    let prev = previous
        .iter()
        .rev()
        .take(stride)
        .map(|v| v.clone())
        .rev()
        .collect::<Vec<_>>();

    [&prev[..], &current[..]].concat()
}

pub enum EncodeInput {
    Single(String),
    Dual(String, String),
}

///
/// ## Tokenizer
///
/// A Tokenizer is capable of encoding/decoding any text
///
pub struct Tokenizer {
    normalizer: Option<Box<dyn Normalizer + Sync>>,
    pre_tokenizer: Option<Box<dyn PreTokenizer + Sync>>,
    model: Box<dyn Model + Sync>,
    post_processor: Option<Box<dyn PostProcessor + Sync>>,
    decoder: Option<Box<dyn Decoder + Sync>>,
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
        }
    }

    /// Set the normalizers
    pub fn with_normalizers(&mut self, normalizer: Box<dyn Normalizer + Sync>) -> &Self {
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

    /// Get the size of the vocabulary
    pub fn get_vocab_size(&self) -> usize {
        self.model.get_vocab_size()
    }

    /// Converts a token in the corresponding id.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.model.token_to_id(token)
    }

    /// Converts an id to the corresponding token.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.model.id_to_token(id)
    }

    /// Encode the given sentence
    pub fn encode(&self, input: EncodeInput) -> Result<Encoding> {
        let generate_output = move |sentence: String, type_id: u32| -> Result<Encoding> {
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

            Ok(Encoding {
                original,
                normalized,
                ids,
                type_ids: vec![type_id; length],
                tokens,
                offsets,
                attention_mask: vec![1; length],
                special_tokens_mask: vec![0; length],
                overflowing: None,
            })
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
        inputs
            .into_par_iter()
            .map(|input| self.encode(input))
            .collect()
    }

    /// Decode the given ids, back to a String
    pub fn decode(&self, ids: Vec<u32>) -> Result<String> {
        let tokens = self.model.decode(ids)?;

        if let Some(decoder) = &self.decoder {
            decoder.decode(tokens)
        } else {
            Ok(tokens.join(" "))
        }
    }

    /// Decode all sentences in parallel
    pub fn decode_batch(&self, sentences: Vec<Vec<u32>>) -> Result<Vec<String>> {
        sentences
            .into_par_iter()
            .map(|sentence| self.decode(sentence))
            .collect()
    }

    /// Train a model and replace our current Model, using the given Trainer
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
            Ok(sentence.to_owned())
        }
    }

    /// Post processing logic, handling the case where there is no PostProcessor set
    fn post_process(
        &self,
        encoding: Encoding,
        pair_encoding: Option<Encoding>,
    ) -> Result<Encoding> {
        if let Some(processor) = &self.post_processor {
            processor.process(encoding, pair_encoding)
        } else {
            match pair_encoding {
                None => Ok(encoding),
                Some(pair) => Ok(Encoding {
                    original: format!("{}{}", encoding.original, pair.original),
                    normalized: format!("{}{}", encoding.normalized, pair.normalized),
                    ids: [&encoding.ids[..], &pair.ids[..]].concat(),
                    type_ids: [&encoding.type_ids[..], &pair.type_ids[..]].concat(),
                    tokens: [&encoding.tokens[..], &pair.tokens[..]].concat(),
                    offsets: [
                        &encoding.offsets[..],
                        &pair
                            .offsets
                            .into_iter()
                            .map(|(start, end)| {
                                (
                                    start + encoding.original.len(),
                                    end + encoding.original.len(),
                                )
                            })
                            .collect::<Vec<_>>(),
                    ]
                    .concat(),
                    special_tokens_mask: [
                        &encoding.special_tokens_mask[..],
                        &pair.special_tokens_mask[..],
                    ]
                    .concat(),
                    attention_mask: [&encoding.attention_mask[..], &pair.attention_mask[..]]
                        .concat(),
                    overflowing: None,
                }),
            }
        }
    }
}
