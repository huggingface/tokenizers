//! [WordPiece](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)
//! model.

use crate::models::bpe::BPE;
use crate::pipeline::{self, PipelineToken};
use crate::tokenizer::{Model, Result, Token};
use ahash::AHashMap;
use std::collections::HashMap;
use std::{
    borrow::Cow,
    fs::File,
    io::prelude::*,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

mod serialization;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("WordPiece error: Missing [UNK] token from the vocabulary")]
    MissingUnkToken,
}

type Vocab = AHashMap<String, u32>;
type VocabR = AHashMap<u32, String>;

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
                vocab: AHashMap::new(),
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
    #[must_use]
    pub fn files(mut self, vocab: String) -> Self {
        self.config.files = Some(vocab);
        self
    }

    /// Set the vocab (token -> ID) mapping.
    #[must_use]
    pub fn vocab<V: Into<AHashMap<String, u32>>>(mut self, vocab: V) -> Self {
        self.config.vocab = vocab.into();
        self
    }

    /// The the `UNK` token for the vocab.
    #[must_use]
    pub fn unk_token(mut self, unk_token: String) -> Self {
        self.config.unk_token = unk_token;
        self
    }

    /// Set the prefix for continuing subwords.
    #[must_use]
    pub fn continuing_subword_prefix(mut self, continuing_subword_prefix: String) -> Self {
        self.config.continuing_subword_prefix = continuing_subword_prefix;
        self
    }

    /// Set the maximum number of input characters per word.
    #[must_use]
    pub fn max_input_chars_per_word(mut self, max_input_chars_per_word: usize) -> Self {
        self.config.max_input_chars_per_word = max_input_chars_per_word;
        self
    }

    /// Constructs a `WordPiece` model that uses the `WordPieceBuilder`'s configuration.
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

        Ok(WordPiece {
            vocab: self.config.vocab,
            vocab_r,
            unk_token: self.config.unk_token,
            continuing_subword_prefix: self.config.continuing_subword_prefix,
            max_input_chars_per_word: self.config.max_input_chars_per_word,
        })
    }
}

/// A
/// [WordPiece](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)
/// model.
#[derive(Clone, PartialEq, Eq)]
pub struct WordPiece {
    pub vocab: Vocab,
    pub vocab_r: VocabR,
    pub unk_token: String,
    pub continuing_subword_prefix: String,
    pub max_input_chars_per_word: usize,
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
            vocab: AHashMap::new(),
            vocab_r: AHashMap::new(),
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

        let mut vocab = AHashMap::new();
        for (index, line) in file.lines().enumerate() {
            let line = line?;
            vocab.insert(line.trim_end().to_owned(), index as u32);
        }

        Ok(vocab)
    }

    pub fn read_bytes(vocab: &[u8]) -> Result<Vocab> {
        let file = BufReader::new(vocab);

        let mut vocab = AHashMap::new();
        for (index, line) in file.lines().enumerate() {
            let line = line?;
            vocab.insert(line.trim_end().to_owned(), index as u32);
        }

        Ok(vocab)
    }

    pub fn from_bytes<P: AsRef<[u8]>>(bytes: P) -> Result<Self> {
        let tokenizer = serde_json::from_slice(bytes.as_ref())?;
        Ok(tokenizer)
    }

    /// Initialize a `WordPiece` model from a vocab mapping file.
    pub fn from_file(vocab: &str) -> WordPieceBuilder {
        WordPiece::builder().files(vocab.to_owned())
    }

    /// Create a `WordPiece` model from a `BPE` model.
    pub fn from_bpe(bpe: &BPE) -> Self {
        let mut wp = Self::builder()
            .vocab(bpe.get_vocab().into_iter().collect::<AHashMap<_, _>>())
            .build()
            .unwrap();
        if let Some(unk) = bpe.get_unk_token() {
            unk.clone_into(&mut wp.unk_token);
        }
        if let Some(prefix) = bpe.get_continuing_subword_prefix() {
            prefix.clone_into(&mut wp.continuing_subword_prefix);
        }
        wp
    }
}

impl Model for WordPiece {
    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.clone().into_iter().collect()
    }

    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        let char_len = sequence.chars().count();
        if char_len > self.max_input_chars_per_word {
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
        let mut start = 0;
        let mut sub_tokens: Vec<Token> = vec![];

        while start < sequence.len() {
            let mut end = sequence.len();
            let mut cur_str = None;

            while start < end {
                let mut substr: Cow<str> = Cow::Borrowed(&sequence[start..end]);

                if start > 0 {
                    substr = Cow::Owned(format!("{}{}", self.continuing_subword_prefix, substr));
                }
                if self.vocab.contains_key(substr.as_ref()) {
                    cur_str = Some(Token {
                        id: self.vocab[substr.as_ref()],
                        value: substr.to_string(),
                        offsets: (start, end),
                    });
                    break;
                }
                end -= substr.chars().last().map_or(1, |c| c.len_utf8());
            }

            if cur_str.is_none() {
                is_bad = true;
                break;
            }

            sub_tokens.push(cur_str.unwrap());
            start = end;
        }

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
            Some(name) => format!("{name}-vocab.txt"),
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
                .flat_map(|(token, _)| format!("{token}\n").as_bytes().to_owned())
                .collect::<Vec<_>>()[..],
        )?;

        Ok(vec![vocab_path])
    }
}

impl pipeline::Model for WordPiece {
    fn tokenize_bytes(
        &self,
        bytes: &[u8],
        output: &mut Vec<pipeline::PipelineToken>,
    ) -> Result<()> {
        // todo: implement
        // if bytes.is_empty() {
        //     return Ok(());
        // }
        // // todo: maybe we can use unchecked here (unsafe)
        // let char_len = str::from_utf8(bytes)?.chars().count();
        // if char_len > self.max_input_chars_per_word {
        //     let unk_id = *self
        //         .vocab
        //         .get(&self.unk_token)
        //         .ok_or(Error::MissingUnkToken)?;
        //     output.push(PipelineToken { id: unk_id });
        //     return Ok(());
        // }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        assert!(format!("{}", Error::MissingUnkToken).contains("Missing [UNK] token"));
    }

    const INVARIANT_CORPUS: &[&str] = &[
        "",
        "un",
        "unaff",
        "unaffable",
        "b",
        "ab",
        "ba",
        "xyz",
        "café",
        "日本",
        "日本語",
        "🙂",
        "e\u{301}",
    ];

    fn invariant_model() -> WordPiece {
        let vocab: Vocab = [
            ("[UNK]", 0u32),
            ("un", 1),
            ("##aff", 2),
            ("##able", 3),
            ("日", 4),
            ("##本", 5),
            ("ca", 6),
            ("##fé", 7),
            ("a", 8),
            ("##b", 9),
            ("b", 10),
        ]
        .iter()
        .map(|&(token, id)| (token.to_string(), id))
        .collect();
        WordPiece::builder().vocab(vocab).build().unwrap()
    }

    #[test]
    fn test_offsets_partition_input() {
        let model = invariant_model();
        for word in INVARIANT_CORPUS {
            let tokens = model.tokenize(word).unwrap();
            let mut pos = 0;
            for token in &tokens {
                let (start, end) = token.offsets;
                assert_eq!(start, pos, "gap or overlap in {word:?}");
                assert!(start < end, "empty piece in {:?}", word);
                assert!(
                    word.is_char_boundary(end),
                    "offset {} splits a char in {:?}",
                    end,
                    word
                );
                pos = end;
            }
            assert_eq!(pos, word.len(), "input not covered: {word:?}");
        }
    }

    #[test]
    fn test_pieces_match_input_slices() {
        let model = invariant_model();
        for word in INVARIANT_CORPUS {
            let tokens = model.tokenize(word).unwrap();
            if tokens.len() == 1 && tokens[0].value == model.unk_token {
                continue;
            }
            for token in &tokens {
                let (start, end) = token.offsets;
                let expected = if start == 0 {
                    word[start..end].to_string()
                } else {
                    format!("{}{}", model.continuing_subword_prefix, &word[start..end])
                };
                assert_eq!(token.value, expected, "{word:?}");
            }
        }
    }

    #[test]
    fn test_unk_is_all_or_nothing() {
        let model = invariant_model();
        for word in INVARIANT_CORPUS {
            let tokens = model.tokenize(word).unwrap();
            if tokens.len() > 1 {
                for token in &tokens {
                    assert_ne!(token.value, model.unk_token, "unk fragment in {word:?}");
                }
            }
        }
    }

    #[test]
    fn test_max_input_chars_gives_single_unk() {
        let model = WordPiece::builder()
            .vocab(invariant_model().vocab)
            .max_input_chars_per_word(4)
            .build()
            .unwrap();
        let tokens = model.tokenize("unaffable").unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].value, "[UNK]");
        assert_eq!(tokens[0].offsets, (0, "unaffable".len()));
    }

    #[test]
    fn test_tokenize_bytes_matches_tokenize() {
        let model = invariant_model();
        for word in INVARIANT_CORPUS {
            let expected: Vec<u32> = model
                .tokenize(word)
                .unwrap()
                .iter()
                .map(|token| token.id)
                .collect();
            let mut output = vec![];
            pipeline::Model::tokenize_bytes(&model, word.as_bytes(), &mut output).unwrap();
            let got: Vec<u32> = output.iter().map(|token| token.id).collect();
            assert_eq!(expected, got, "{word:?}");
        }
    }
}
