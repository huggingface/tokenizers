use super::OrderedVocabIter;
use crate::tokenizer::{Model, Result, Token};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

mod serialization;
mod trainer;

// Re-export
pub use trainer::*;

type Vocab = HashMap<String, u32>;

#[derive(Debug)]
pub enum Error {
    MissingUnkToken,
    BadVocabulary,
}
impl std::error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::MissingUnkToken => write!(
                fmt,
                "WordLevel error: Missing [UNK] token from the vocabulary"
            ),
            Error::BadVocabulary => write!(fmt, "Bad vocabulary json file"),
        }
    }
}

struct Config {
    files: Option<String>,
    vocab: HashMap<String, u32>,
    unk_token: String,
}

/// A `WordLevelBuilder` can be used to create a `WordLevel`
/// model with a custom configuration.
pub struct WordLevelBuilder {
    config: Config,
}

impl Default for WordLevelBuilder {
    fn default() -> Self {
        Self {
            config: Config {
                files: None,
                vocab: HashMap::new(),
                unk_token: String::from("<unk>"),
            },
        }
    }
}

impl WordLevelBuilder {
    /// Construct a new `WordLevelBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the input files.
    pub fn files(mut self, vocab: String) -> Self {
        self.config.files = Some(vocab);
        self
    }

    /// Set the vocab (token -> ID) mapping.
    pub fn vocab(mut self, vocab: HashMap<String, u32>) -> Self {
        self.config.vocab = vocab;
        self
    }

    /// The the `UNK` token for the vocab.
    pub fn unk_token(mut self, unk_token: String) -> Self {
        self.config.unk_token = unk_token;
        self
    }

    /// Contructs a `WordLevel` model that uses the `WordLevelBuilder`'s configuration.
    pub fn build(mut self) -> Result<WordLevel> {
        if let Some(vocab) = self.config.files {
            self.config.vocab = WordLevel::read_file(&vocab)?;
        }

        let vocab_r = self
            .config
            .vocab
            .iter()
            .map(|(key, val)| (*val, key.to_owned()))
            .collect();

        Ok(WordLevel {
            vocab: self.config.vocab,
            vocab_r,
            unk_token: self.config.unk_token,
        })
    }
}

#[derive(PartialEq, Clone)]
pub struct WordLevel {
    vocab: HashMap<String, u32>,
    vocab_r: HashMap<u32, String>,
    pub unk_token: String,
}

impl std::fmt::Debug for WordLevel {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("WordLevel")
            .field("unk_token", &self.unk_token)
            .field("vocab", &self.vocab.len())
            .finish()
    }
}

impl WordLevel {
    pub fn builder() -> WordLevelBuilder {
        WordLevelBuilder::new()
    }

    pub fn read_file(vocab_path: &str) -> Result<Vocab> {
        let vocab_file = File::open(vocab_path)?;
        let mut vocab_file = BufReader::new(vocab_file);
        let mut buffer = String::new();
        let mut vocab = HashMap::new();

        vocab_file.read_to_string(&mut buffer)?;
        let json: Value = serde_json::from_str(&buffer)?;

        match json {
            Value::Object(m) => {
                for (token, id) in m {
                    if let Value::Number(id) = id {
                        let id = id.as_u64().ok_or(Error::BadVocabulary)? as u32;
                        vocab.insert(token, id);
                    }
                }
            }
            _ => return Err(Box::new(Error::BadVocabulary)),
        };
        Ok(vocab)
    }

    /// Initialize a WordLevel model from vocab and merges file.
    pub fn from_file(vocab_path: &str, unk_token: String) -> Result<WordLevel> {
        let vocab = WordLevel::read_file(vocab_path)?;
        Self::builder().vocab(vocab).unk_token(unk_token).build()
    }
}

impl Default for WordLevel {
    fn default() -> Self {
        Self {
            vocab: HashMap::new(),
            vocab_r: HashMap::new(),
            unk_token: String::from("<unk>"),
        }
    }
}

impl Model for WordLevel {
    type Trainer = WordLevelTrainer;

    fn tokenize(&self, token: &str) -> Result<Vec<Token>> {
        Ok(vec![Token {
            id: *self
                .vocab
                .get(&*token)
                .or_else(|| self.vocab.get(&*self.unk_token))
                .ok_or(Error::MissingUnkToken)?,
            value: token.to_owned(),
            offsets: (0, token.len()),
        }])
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab_r.get(&id).cloned()
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.clone()
    }

    fn get_vocab_size(&self) -> usize {
        self.vocab.keys().len()
    }

    fn save(&self, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>> {
        let vocab_file_name = match name {
            Some(name) => format!("{}-vocab.json", name),
            None => "vocab.json".to_string(),
        };

        // Write vocab.json
        let vocab_path: PathBuf = [folder, Path::new(vocab_file_name.as_str())]
            .iter()
            .collect();
        let mut vocab_file = File::create(&vocab_path)?;
        let order_vocab_iter = OrderedVocabIter::new(&self.vocab_r);
        let serialized = serde_json::to_string(&order_vocab_iter)?;
        vocab_file.write_all(&serialized.as_bytes())?;

        Ok(vec![vocab_path])
    }

    fn get_trainer(&self) -> Self::Trainer {
        WordLevelTrainer::default()
    }
}
