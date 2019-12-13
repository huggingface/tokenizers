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

mod encoding;
pub use encoding::Encoding;

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
        mut encoding: Encoding,
        pair_encoding: Option<Encoding>,
    ) -> Result<Encoding> {
        if let Some(processor) = &self.post_processor {
            processor.process(encoding, pair_encoding)
        } else {
            match pair_encoding {
                None => Ok(encoding),
                Some(pair) => {
                    encoding.merge_with(pair);
                    Ok(encoding)
                }
            }
        }
    }
}
