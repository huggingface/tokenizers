//! [WordPiece](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)
//! model.

use crate::models::bpe::BPE;
use crate::tokenizer::{Model, Offsets, Result, Token};
use std::{
    collections::HashMap,
    fmt,
    fs::File,
    io::prelude::*,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

mod trainer;
pub use trainer::*;

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
                "WordPiece error: Missing [UNK] token from the vocabulary"
            ),
        }
    }
}

struct Config {
    vocab: HashMap<String, u32>,
    unk_token: String,
    continuing_subword_prefix: String,
    max_input_chars_per_word: usize,
}

/// A `WordPieceBuilder` can be used to create a `WordPiece` model with a custom configuration.
pub struct WordPieceBuilder {
    config: Config,
}

impl Default for WordPieceBuilder {
    fn default() -> Self {
        Self {
            config: Config {
                vocab: HashMap::new(),
                unk_token: String::from("[UNK]"),
                continuing_subword_prefix: String::from("##"),
                max_input_chars_per_word: 100,
            },
        }
    }
}

impl WordPieceBuilder {
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

    /// Set the prefix for continuing subwords.
    pub fn continuing_subword_prefix(mut self, continuing_subword_prefix: String) -> Self {
        self.config.continuing_subword_prefix = continuing_subword_prefix;
        self
    }

    /// Set the maximum number of input characters per word.
    pub fn max_input_chars_per_word(mut self, max_input_chars_per_word: usize) -> Self {
        self.config.max_input_chars_per_word = max_input_chars_per_word;
        self
    }

    /// Contructs a `WordPiece` model that uses the `WordPieceBuilder`'s configuration.
    pub fn build(self) -> WordPiece {
        let vocab_r = self
            .config
            .vocab
            .iter()
            .map(|(key, val)| (*val, key.to_owned()))
            .collect();
        WordPiece {
            vocab: self.config.vocab,
            vocab_r,
            unk_token: self.config.unk_token,
            continuing_subword_prefix: self.config.continuing_subword_prefix,
            max_input_chars_per_word: self.config.max_input_chars_per_word,
        }
    }
}

/// A
/// [WordPiece](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)
/// model.
pub struct WordPiece {
    vocab: HashMap<String, u32>,
    vocab_r: HashMap<u32, String>,
    unk_token: String,
    continuing_subword_prefix: String,
    max_input_chars_per_word: usize,
}

impl Default for WordPiece {
    fn default() -> Self {
        Self {
            vocab: HashMap::new(),
            vocab_r: HashMap::new(),
            unk_token: String::from("[UNK]"),
            continuing_subword_prefix: String::from("##"),
            max_input_chars_per_word: 100,
        }
    }
}

impl WordPiece {
    /// Get a `WordPieceBuilder`.
    pub fn builder() -> WordPieceBuilder {
        WordPieceBuilder::new()
    }

    /// Initialize a `WordPiece` model from a vocab mapping file.
    pub fn from_files(
        vocab: &str,
        unk_token: String,
        max_input_chars_per_word: Option<usize>,
    ) -> std::io::Result<Self> {
        let file = File::open(vocab)?;
        let file = BufReader::new(file);

        let mut vocab = HashMap::new();
        for (index, line) in file.lines().enumerate() {
            let line = line?;
            vocab.insert(line.trim_end().to_owned(), index as u32);
        }

        let mut builder = Self::builder().vocab(vocab).unk_token(unk_token);
        if let Some(max_chars) = max_input_chars_per_word {
            builder = builder.max_input_chars_per_word(max_chars);
        }
        Ok(builder.build())
    }

    /// Create a `WordPiece` model from a `BPE` model.
    pub fn from_bpe(bpe: &BPE) -> Self {
        let mut wp = Self::builder().vocab(bpe.get_vocab().clone()).build();
        if let Some(unk) = bpe.get_unk_token() {
            wp.unk_token = unk.to_owned();
        }
        if let Some(prefix) = bpe.get_continuing_subword_prefix() {
            wp.continuing_subword_prefix = prefix.to_owned();
        }
        wp
    }
}

impl Model for WordPiece {
    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn tokenize(&self, sentence: Vec<(String, Offsets)>) -> Result<Vec<Token>> {
        let mut output_tokens = vec![];

        for (token, initial_offsets) in sentence {
            let char_len = token.chars().count();
            if char_len > self.max_input_chars_per_word {
                output_tokens.push(Token {
                    value: self.unk_token.clone(),
                    id: *self
                        .vocab
                        .get(&self.unk_token)
                        .ok_or(Error::MissingUnkToken)?,
                    offsets: initial_offsets,
                });
                continue;
            }

            let mut is_bad = false;
            let mut start = 0;
            let mut sub_tokens: Vec<Token> = vec![];
            let chars = token.chars().collect::<Vec<_>>();

            while start < chars.len() {
                let mut end = chars.len();
                let mut cur_str = None;

                while start < end {
                    let mut substr = chars[start..end].iter().collect::<String>();
                    if start > 0 {
                        substr = format!("{}{}", self.continuing_subword_prefix, substr);
                    }
                    if self.vocab.contains_key(&substr) {
                        cur_str = Some(Token {
                            id: self.vocab[&substr],
                            value: substr,
                            offsets: (initial_offsets.0 + start, initial_offsets.0 + end),
                        });
                        break;
                    }
                    end -= 1;
                }

                if cur_str.is_none() {
                    is_bad = true;
                    break;
                }

                sub_tokens.push(cur_str.unwrap());
                start = end;
            }

            if is_bad {
                output_tokens.push(Token {
                    value: self.unk_token.clone(),
                    id: *self
                        .vocab
                        .get(&self.unk_token)
                        .ok_or(Error::MissingUnkToken)?,
                    offsets: initial_offsets,
                });
            } else {
                output_tokens.extend(sub_tokens);
            }
        }

        Ok(output_tokens)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab_r.get(&id).cloned()
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
                .flat_map(|(token, _)| format!("{}\n", token).as_bytes().to_owned())
                .collect::<Vec<_>>()[..],
        )?;

        Ok(vec![vocab_path])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        assert!(format!("{}", Error::MissingUnkToken).contains("Missing [UNK] token"));
    }
}
