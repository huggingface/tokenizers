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
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
};

/// A Normalizer takes care of pre-processing strings
pub trait Normalizer {
    // TODO: Use Cow here to avoid useless allocation if nothing is modified
    fn normalize(&self, s: &str) -> String;
}

/// A PreTokenizer takes care of pre-tokenizing strings before this goes to the model
pub trait PreTokenizer {
    fn pre_tokenize(&self, s: &str) -> Vec<String>;
}

/// Represents a `Model` used during Tokenization (Like BPE or Word or Unigram)
pub trait Model {
    fn tokenize(&self, tokens: Vec<String>) -> Vec<Token>;
    fn decode(&self, ids: Vec<u32>) -> Vec<String>;
    fn token_to_id(&self, token: &str) -> Option<u32>;
    fn id_to_token(&self, id: u32) -> Option<String>;
}

/// A PostProcessor has the responsibility to post process an encoded output of the Tokenizer.
/// Truncating, Padding, etc... are PostProcessor steps
pub trait PostProcessor {
    fn process(&self, tokens: Vec<Token>) -> Vec<Token>;
}

/// A Decoder has the responsibility to merge the given Vec<String> in a String
pub trait Decoder {
    fn decode(&self, tokens: Vec<String>) -> String;
}

/// A Trainer has the responsibility to train a Model. We feed it with lines/sentences
/// and it returns a Model when done.
pub trait Trainer: Sync {
    fn train(&self, words: HashMap<String, u32>) -> Result<Box<dyn Model + Sync>, Box<dyn Error>>;
    fn process_tokens(&self, words: &mut HashMap<String, u32>, tokens: Vec<String>);
}

/// A Token represents the output of the Tokenizer
#[derive(Debug, PartialEq)]
pub struct Token {
    pub id: u32,
    pub value: String,
    pub offsets: (usize, usize),
    // TODO: Find out the best way to define the customizable part (For post processing steps)
}
impl Token {
    pub fn new(id: u32, value: String, offsets: (usize, usize)) -> Self {
        Token { id, value, offsets }
    }
}

///
/// ## Tokenizer
///
/// A Tokenizer is capable of encoding/decoding any text
///
pub struct Tokenizer {
    normalizers: Vec<Box<dyn Normalizer + Sync>>,
    pre_tokenizer: Option<Box<dyn PreTokenizer + Sync>>,
    model: Box<dyn Model + Sync>,
    post_processors: Vec<Box<dyn PostProcessor + Sync>>,
    decoder: Option<Box<dyn Decoder + Sync>>,
}

impl Tokenizer {
    /// Instanciate a new Tokenizer, with the given Model
    pub fn new(model: Box<dyn Model + Sync>) -> Self {
        Tokenizer {
            normalizers: vec![],
            pre_tokenizer: None,
            model,
            post_processors: vec![],
            decoder: None,
        }
    }

    /// Set the normalizers
    pub fn with_normalizers(&mut self, normalizers: Vec<Box<dyn Normalizer + Sync>>) -> &Self {
        self.normalizers = normalizers;
        self
    }

    /// Set the pre tokenizer
    pub fn with_pre_tokenizer(&mut self, pre_tokenizer: Box<dyn PreTokenizer + Sync>) -> &Self {
        self.pre_tokenizer = Some(pre_tokenizer);
        self
    }

    /// Set the post processors
    pub fn with_post_processors(
        &mut self,
        post_processors: Vec<Box<dyn PostProcessor + Sync>>,
    ) -> &Self {
        self.post_processors = post_processors;
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

    /// Converts a token in the corresponding id.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.model.token_to_id(token)
    }

    /// Converts an id to the corresponding token.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.model.id_to_token(id)
    }

    /// Encode the given sentence
    pub fn encode(&self, sentence: &str) -> Vec<Token> {
        let normalized = self.normalize(sentence);
        let pre_tokenized = self.pre_tokenize(&normalized);

        self.model.tokenize(pre_tokenized)
    }

    /// Encode all the sentences in parallel, using multiple threads
    pub fn encode_batch(&self, sentences: Vec<&str>) -> Vec<Vec<Token>> {
        sentences
            .par_iter()
            .map(|sentence| self.encode(sentence))
            .collect()
    }

    /// Decode the given ids, back to a String
    pub fn decode(&self, ids: Vec<u32>) -> String {
        let tokens = self.model.decode(ids);

        if let Some(decoder) = &self.decoder {
            decoder.decode(tokens)
        } else {
            tokens.join(" ")
        }
    }

    /// Decode all sentences in parallel
    pub fn decode_batch(&self, sentences: Vec<Vec<u32>>) -> Vec<String> {
        sentences
            .into_par_iter()
            .map(|sentence| self.decode(sentence))
            .collect()
    }

    /// Train a model and replace our current Model, using the given Trainer
    pub fn train(
        &mut self,
        trainer: &Box<dyn Trainer>,
        files: Vec<String>,
    ) -> Result<(), Box<dyn Error>> {
        let results = files
            .par_iter()
            .map(|file| -> std::io::Result<HashMap<String, u32>> {
                let mut words = HashMap::new();

                let file: std::fs::File = File::open(file)?;
                let file = BufReader::new(file);

                for line in file.lines() {
                    let line = line?;
                    let normalized = self.normalize(&line);
                    let pre_tokenized = self.pre_tokenize(&normalized);
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
    fn pre_tokenize(&self, sentence: &str) -> Vec<String> {
        match &self.pre_tokenizer {
            None => vec![sentence.to_owned()],
            Some(pre_tokenizer) => pre_tokenizer.pre_tokenize(sentence),
        }
    }

    /// Normalization logic, go through all normalizers
    fn normalize(&self, sentence: &str) -> String {
        if self.normalizers.len() == 0 {
            sentence.to_owned()
        } else {
            unimplemented!("Normalization has not been implemented yet")
        }
    }
}
