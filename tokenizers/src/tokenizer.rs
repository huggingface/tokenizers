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

/// A Normalizer takes care of pre-processing strings
pub trait Normalizer {
    fn normalize(&self, s: &str) -> String;
}

/// A PreTokenizer takes care of pre-tokenizing strings before this goes to the model
pub trait PreTokenizer {
    fn pre_tokenize(&self, s: &str) -> Vec<String>;
}

/// Represents a `Model` used during Tokenization (Like BPE or Word or Unigram)
pub trait Model {
    fn tokenize(&self, tokens: Vec<String>) -> Vec<Token>;
    fn token_to_id(&self, token: &str) -> Option<u32>;
    fn id_to_token(&self, id: u32) -> Option<String>;
}

/// A PostProcessor has the responsibility to post process an encoded output of the Tokenizer.
/// Truncating, Padding, etc... are PostProcessor steps
pub trait PostProcessor {
    fn process(&self, tokens: Vec<Token>) -> Vec<Token>;
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
}

impl Tokenizer {
    /// Instanciate a new Tokenizer, with the given Model
    pub fn new(model: Box<dyn Model + Sync>) -> Self {
        Tokenizer {
            normalizers: vec![],
            pre_tokenizer: None,
            model,
            post_processors: vec![],
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
        let pre_tokenized = match &self.pre_tokenizer {
            None => vec![sentence.to_owned()],
            Some(pre_tokenizer) => pre_tokenizer.pre_tokenize(sentence),
        };

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
    pub fn decode(&self, tokens: Vec<u32>) -> String {
        unimplemented!("Decode is not implemented yet");
    }
}
