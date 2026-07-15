//! [WordPiece](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)
//! model.

use crate::models::bpe::BPE;
use crate::pipeline::{self, PipelineToken};
use crate::tokenizer::{Model, Result, Token};
use ahash::AHashMap;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::{
    borrow::Cow,
    fs::File,
    io::prelude::*,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};
use yada::builder::DoubleArrayBuilder;
use yada::DoubleArray;

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

pub struct PipelineWordPiece {
    vocab_trie: yada::DoubleArray<Vec<u8>>,
    unk_token: Option<u32>,
    continuing_subword_prefix: String,
    max_input_chars_per_word: usize,
    /// Longest vocab key in bytes. A `common_prefix_search` can never match past
    /// this depth, so continuing-subword candidates only need the first
    /// `max_key_len` bytes of the remaining tail — never the whole tail.
    max_key_len: usize,
}

impl TryFrom<WordPiece> for PipelineWordPiece {
    type Error = crate::Error;
    fn try_from(value: WordPiece) -> Result<Self> {
        let WordPiece {
            vocab,
            unk_token,
            continuing_subword_prefix,
            max_input_chars_per_word,
            ..
        } = value;
        let unk_token = vocab.get(&unk_token).copied();

        // yada requires the keyset sorted by key bytes.
        let mut keyset: Vec<_> = vocab.into_iter().collect();
        keyset.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
        let max_key_len = keyset.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
        let vocab_trie = DoubleArray::new(DoubleArrayBuilder::build(&keyset)?)?;

        Ok(Self {
            continuing_subword_prefix,
            max_input_chars_per_word,
            unk_token,
            vocab_trie,
            max_key_len,
        })
    }
}

impl pipeline::Model for PipelineWordPiece {
    fn tokenize_pipeline(
        &self,
        sequence: &str,
        output: &mut Vec<pipeline::PipelineToken>,
    ) -> Result<()> {
        // Max-chars cap. The cap is on *characters*, but a string's byte length is
        // an upper bound on its char count, so we only pay for a full UTF-8 decode
        // in the rare case where the byte length already exceeds the cap.
        if sequence.len() > self.max_input_chars_per_word
            && sequence.chars().count() > self.max_input_chars_per_word
        {
            let unk_id = self.unk_token.ok_or(Error::MissingUnkToken)?;
            output.push(PipelineToken { id: unk_id });
            return Ok(());
        }

        let bytes = sequence.as_bytes();
        let prefix = self.continuing_subword_prefix.as_bytes();

        // Continuing-subword candidate buffer: `prefix` + a bounded slice of the
        // tail. Bounding to `max_key_len` avoids copying the whole remaining tail
        // every token, and byte slicing (vs `&str`) means we never have to land on
        // a char boundary — the trie matches on bytes. The buffer lives on the
        // stack for the common case (no per-word heap allocation — the dominant
        // cost of the previous impl); only a vocab whose keys exceed the stack
        // buffer falls back to a one-off heap `Vec`.
        const CAND_STACK_CAP: usize = 256;
        let cand_len = prefix.len() + self.max_key_len;
        let mut stack_buf = [0u8; CAND_STACK_CAP];
        let mut heap_buf: Vec<u8> = Vec::new();
        let buf: &mut [u8] = if cand_len <= CAND_STACK_CAP {
            &mut stack_buf[..]
        } else {
            heap_buf.resize(cand_len, 0);
            &mut heap_buf[..]
        };
        buf[..prefix.len()].copy_from_slice(prefix);

        // Push directly to `output`, remembering where this word started so a
        // mid-word failure can roll the partial tokens back and emit a single UNK
        // (WordPiece's all-or-nothing semantics), without a temp Vec.
        let rollback = output.len();
        let mut start = 0;

        while start < bytes.len() {
            let (search, prefix_len): (&[u8], usize) = if start == 0 {
                // First subword: search the raw tail directly — zero-copy, and the
                // walk stops itself once the trie runs out of transitions.
                (&bytes[start..], 0)
            } else {
                // Continuing subword: `prefix` is already in `buf`; append the
                // bounded tail after it (no key can match past `max_key_len`).
                let end = start + (bytes.len() - start).min(self.max_key_len);
                let cand = prefix.len() + (end - start);
                buf[prefix.len()..cand].copy_from_slice(&bytes[start..end]);
                (&buf[..cand], prefix.len())
            };

            // Matches must extend past the continuing-subword prefix: the prefix
            // alone (or a fragment of it) is not a valid subword here, even if it
            // happens to be in the vocab. `.last()` = the longest match.
            let Some((token_id, match_len)) = self
                .vocab_trie
                .common_prefix_search(search)
                .filter(|(_, len)| *len > prefix_len)
                .last()
            else {
                output.truncate(rollback);
                let unk_id = self.unk_token.ok_or(Error::MissingUnkToken)?;
                output.push(PipelineToken { id: unk_id });
                return Ok(());
            };
            output.push(PipelineToken { id: token_id });
            start += match_len - prefix_len;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::Model as _;

    #[test]
    fn test_error_display() {
        assert!(format!("{}", Error::MissingUnkToken).contains("Missing [UNK] token"));
    }

    fn wp(vocab: &[(&str, u32)]) -> PipelineWordPiece {
        let vocab: Vocab = vocab.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        PipelineWordPiece::try_from(
            WordPieceBuilder::new()
                .vocab(vocab)
                .unk_token("[UNK]".into())
                .build()
                .unwrap(),
        )
        .unwrap()
    }

    fn ids(model: &PipelineWordPiece, s: &str) -> Vec<u32> {
        let mut out = Vec::new();
        model.tokenize_pipeline(s, &mut out).unwrap();
        out.into_iter().map(|t| t.id).collect()
    }

    #[test]
    fn pipeline_tokenize_matches_greedy_wordpiece() {
        let model = wp(&[
            ("[UNK]", 0),
            ("un", 1),
            ("##want", 2),
            ("##ed", 3),
            ("play", 4),
            ("##ing", 5),
            ("hello", 6),
        ]);
        // First subword (no prefix) + two continuing subwords (matched via "##").
        assert_eq!(ids(&model, "unwanted"), vec![1, 2, 3]);
        assert_eq!(ids(&model, "playing"), vec![4, 5]);
        assert_eq!(ids(&model, "hello"), vec![6]);
    }

    #[test]
    fn pipeline_tokenize_rolls_back_partial_word_to_single_unk() {
        // "un" matches, but the remaining "zzz" has no continuing subword: the
        // whole word must collapse to a single [UNK], not "un" + [UNK].
        let model = wp(&[("[UNK]", 0), ("un", 1), ("##ed", 3)]);
        assert_eq!(ids(&model, "unzzz"), vec![0]);
        // A prior real token in the output stays untouched by the rollback.
        let mut out = vec![PipelineToken { id: 99 }];
        model.tokenize_pipeline("unzzz", &mut out).unwrap();
        assert_eq!(out.into_iter().map(|t| t.id).collect::<Vec<_>>(), vec![99, 0]);
    }
}
