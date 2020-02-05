use crate::tokenizer::{Model, Result, Token};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

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
    pub fn build(self) -> WordLevel {
        let vocab_r = self
            .config
            .vocab
            .iter()
            .map(|(key, val)| (*val, key.to_owned()))
            .collect();
        WordLevel {
            vocab: self.config.vocab,
            vocab_r,
            unk_token: self.config.unk_token,
        }
    }
}

pub struct WordLevel {
    vocab: HashMap<String, u32>,
    vocab_r: HashMap<u32, String>,
    unk_token: String,
}

impl WordLevel {
    fn builder() -> WordLevelBuilder {
        WordLevelBuilder::new()
    }

    /// Initialize a WordLevel model from vocab and merges file.
    pub fn from_files(vocab_path: &str, unk_token: String) -> Result<WordLevel> {
        // Read vocab.json
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

        Ok(Self::builder().vocab(vocab).unk_token(unk_token).build())
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
    fn tokenize(&self, tokens: Vec<(String, (usize, usize))>) -> Result<Vec<Token>> {
        let mut output_tokens = vec![];

        for (token, initial_offsets) in tokens {
            let t = Token {
                id: *self
                    .vocab
                    .get(&*token)
                    .or_else(|| self.vocab.get(&*self.unk_token))
                    .ok_or(Error::MissingUnkToken)?,
                value: token,
                offsets: initial_offsets,
            };

            output_tokens.push(t);
        }

        Ok(output_tokens)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab_r.get(&id).cloned()
    }

    fn get_vocab_size(&self) -> usize {
        self.vocab.keys().len()
    }

    fn save(&self, folder: &Path, name: &str) -> Result<Vec<PathBuf>> {
        // Write vocab.txt
        let vocab_path: PathBuf = [folder, Path::new(&format!("{}-vocab.txt", name))]
            .iter()
            .collect();
        let mut vocab_file = File::create(&vocab_path)?;
        let mut vocab: Vec<(&String, &u32)> = self.vocab.iter().collect();
        vocab.sort_unstable_by_key(|k| *k.1);
        vocab_file.write_all(
            &vocab
                .into_iter()
                .map(|(token, _)| format!("{}\n", token).as_bytes().to_owned())
                .flatten()
                .collect::<Vec<_>>()[..],
        )?;

        Ok(vec![vocab_path])
    }
}
