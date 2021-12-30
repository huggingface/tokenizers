//! [WordPiece](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)
//! model.

use crate::models::bpe::BPE;
use crate::tokenizer::{Model, Result, Token};
use crate::utils::trie::Trie;
use std::{
    borrow::Cow,
    collections::HashMap,
    fmt,
    fs::File,
    io::prelude::*,
    io::{BufRead, BufReader},
    iter::FromIterator,
    path::{Path, PathBuf},
};

mod serialization;
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

type Vocab = HashMap<String, u32>;
type VocabR = HashMap<u32, String>;

struct Config {
    files: Option<String>,
    vocab: Vocab,
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
                files: None,
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

    /// Set the input files.
    pub fn files(mut self, vocab: String) -> Self {
        self.config.files = Some(vocab);
        self
    }

    /// Set the vocab (token -> ID) mapping.
    pub fn vocab(mut self, vocab: Vocab) -> Self {
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
    pub fn build(mut self) -> Result<WordPiece> {
        if let Some(vocab) = self.config.files {
            self.config.vocab = WordPiece::read_file(&vocab)?;
        }

        let vocab_r = self
            .config
            .vocab
            .iter()
            .map(|(key, val)| (*val, key.to_owned()))
            .collect();

        let mut trie = Trie::default();
        let n = self.config.continuing_subword_prefix.chars().count();
        for key in self.config.vocab.keys() {
            let chars = if key.starts_with(&self.config.continuing_subword_prefix) {
                key.chars().skip(n).collect::<Vec<_>>()
            } else {
                key.chars().collect::<Vec<_>>()
            };
            println!("Saving in trie {:?}", chars);
            trie.push(&chars);
        }

        Ok(WordPiece {
            vocab: self.config.vocab,
            vocab_r,
            trie,
            unk_token: self.config.unk_token,
            continuing_subword_prefix: self.config.continuing_subword_prefix,
            max_input_chars_per_word: self.config.max_input_chars_per_word,
        })
    }
}

/// A
/// [WordPiece](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)
/// model.
#[derive(Clone)]
pub struct WordPiece {
    vocab: Vocab,
    vocab_r: VocabR,
    trie: Trie<char>,
    pub unk_token: String,
    pub continuing_subword_prefix: String,
    pub max_input_chars_per_word: usize,
}

impl PartialEq for WordPiece {
    fn eq(&self, rhs: &Self) -> bool {
        self.vocab == rhs.vocab
            && self.vocab_r == rhs.vocab_r
            && self.unk_token == rhs.unk_token
            && self.continuing_subword_prefix == rhs.continuing_subword_prefix
            && self.max_input_chars_per_word == rhs.max_input_chars_per_word
    }
}

impl std::fmt::Debug for WordPiece {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("WordPiece")
            .field("unk_token", &self.unk_token)
            .field("continuing_subword_prefix", &self.continuing_subword_prefix)
            .field("max_input_chars_per_word", &self.max_input_chars_per_word)
            .field("vocab", &self.vocab.len())
            .finish()
    }
}

impl Default for WordPiece {
    fn default() -> Self {
        Self {
            vocab: HashMap::new(),
            vocab_r: HashMap::new(),
            trie: Trie::default(),
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

    /// Read the given files to extract the vocab
    pub fn read_file(vocab: &str) -> Result<Vocab> {
        let file = File::open(vocab)?;
        let file = BufReader::new(file);

        let mut vocab = HashMap::new();
        for (index, line) in file.lines().enumerate() {
            let line = line?;
            vocab.insert(line.trim_end().to_owned(), index as u32);
        }

        Ok(vocab)
    }

    /// Initialize a `WordPiece` model from a vocab mapping file.
    pub fn from_file(vocab: &str) -> WordPieceBuilder {
        WordPiece::builder().files(vocab.to_owned())
    }

    /// Create a `WordPiece` model from a `BPE` model.
    pub fn from_bpe(bpe: &BPE) -> Self {
        let mut wp = Self::builder().vocab(bpe.get_vocab()).build().unwrap();
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
    type Trainer = WordPieceTrainer;

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.clone()
    }

    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        println!("Sequence {:?}", sequence);
        println!("Vocab {:?}", self.vocab);
        let chars: Vec<_> = sequence.chars().collect();
        if chars.len() > self.max_input_chars_per_word {
            return Ok(vec![Token {
                value: self.unk_token.clone(),
                id: *self
                    .vocab
                    .get(&self.unk_token)
                    .ok_or(Error::MissingUnkToken)?,
                offsets: (0, sequence.len()),
            }]);
        }

        let mut is_bad = false;
        let mut start_offset = 0;
        let splits: Vec<_> = self
            .trie
            .matches(&chars)
            .flat_map(|(start, stop)| {
                let mut subsplits = vec![];
                if start_offset < start {
                    is_bad = true;
                }
                subsplits.push((start, stop));
                start_offset = stop;
                subsplits
            })
            .collect();
        println!("Splits {:?}", splits);
        println!("Is bad {:?}", is_bad);
        if start_offset != chars.len() {
            is_bad = true;
        }

        let sub_tokens: Vec<_> = splits
            .into_iter()
            .filter_map(|(start, stop)| {
                let mut substr: Cow<str> = Cow::Owned(String::from_iter(&chars[start..stop]));
                if start > 0 {
                    substr = Cow::Owned(format!("{}{}", self.continuing_subword_prefix, substr));
                }
                if self.vocab.contains_key(substr.as_ref()) {
                    Some(Token {
                        id: self.vocab[substr.as_ref()],
                        value: substr.to_string(),
                        offsets: (start, stop),
                    })
                } else {
                    is_bad = true;
                    None
                }
            })
            .collect();

        if is_bad {
            Ok(vec![Token {
                value: self.unk_token.clone(),
                id: *self
                    .vocab
                    .get(&self.unk_token)
                    .ok_or(Error::MissingUnkToken)?,
                offsets: (0, sequence.len()),
            }])
        } else {
            Ok(sub_tokens)
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab_r.get(&id).cloned()
    }

    fn save(&self, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>> {
        let vocab_file_name = match name {
            Some(name) => format!("{}-vocab.txt", name),
            None => "vocab.txt".to_string(),
        };

        // Write vocab.txt
        let vocab_path: PathBuf = [folder, Path::new(vocab_file_name.as_str())]
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

    fn get_trainer(&self) -> Self::Trainer {
        WordPieceTrainer::builder().build()
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
