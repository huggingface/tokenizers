use crate::tokenizer::{Model, Result, Token};
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum Error {
    MissingUnkToken,
}
impl std::error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::MissingUnkToken => write!(
                fmt,
                "LookupTable error: Missing [UNK] token from the vocabulary"
            ),
        }
    }
}

struct Config {
    vocab: HashMap<String, u32>,
    unk_token: String,
}

/// A `LookupTableModelBuilder` can be used to create a `LookupTableModel`
/// model with a custom configuration.
pub struct LookupTableBuilder {
    config: Config,
}

impl Default for LookupTableBuilder {
    fn default() -> Self {
        Self {
            config: Config {
                vocab: HashMap::new(),
                unk_token: String::from("<unk>"),
            },
        }
    }
}

impl LookupTableBuilder {
    /// Construct a new `WordPieceBuilder`.
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

    /// Contructs a `WordPiece` model that uses the `WordPieceBuilder`'s configuration.
    pub fn build(self) -> LookupTable {
        let vocab_r = self
            .config
            .vocab
            .iter()
            .map(|(key, val)| (*val, key.to_owned()))
            .collect();
        LookupTable {
            vocab: self.config.vocab,
            vocab_r,
            unk_token: self.config.unk_token,
        }
    }
}

pub struct LookupTable {
    vocab: HashMap<String, u32>,
    vocab_r: HashMap<u32, String>,
    unk_token: String,
}

impl Default for LookupTable {
    fn default() -> Self {
        Self {
            vocab: HashMap::new(),
            vocab_r: HashMap::new(),
            unk_token: String::from("<unk>"),
        }
    }
}

impl Model for LookupTable {
    fn tokenize(&self, tokens: Vec<(String, (usize, usize))>) -> Result<Vec<Token>> {
        let mut output_tokens = vec![];

        for (token, initial_offsets) in tokens {
            let t = Token {
                id: *self
                    .vocab
                    .get(&*token)
                    .or(self.vocab.get(&*self.unk_token))
                    .ok_or(Error::MissingUnkToken)?,
                value: token,
                offsets: initial_offsets,
            };

            output_tokens.push(t);
        }

        Ok(output_tokens)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> { self.vocab.get(token).copied() }

    fn id_to_token(&self, id: u32) -> Option<String> { self.vocab_r.get(&id).cloned() }

    fn get_vocab_size(&self) -> usize { self.vocab.keys().len() }

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
