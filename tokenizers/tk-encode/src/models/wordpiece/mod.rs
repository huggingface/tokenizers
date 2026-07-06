//! [WordPiece](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)
//! model.

use crate::models::bpe::BPE;
use crate::tokenizer::{Model, Result, Token};
use crate::vocab_store::VocabStore;
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

struct Config {
    files: Option<String>,
    vocab: VocabStore,
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
                vocab: VocabStore::new(),
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
        let string_vocab: AHashMap<String, u32> = vocab.into();
        self.config.vocab = VocabStore::build(
            string_vocab
                .into_iter()
                .map(|(token_str, token_id)| (token_str.into_bytes(), token_id))
                .collect(),
        );
        self
    }

    pub fn vocab_store(mut self, vocab_store: VocabStore) -> Self {
        self.config.vocab = vocab_store;
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

        Ok(WordPiece {
            vocab: self.config.vocab,
            unk_token: self.config.unk_token,
            continuing_subword_prefix: self.config.continuing_subword_prefix,
            max_input_chars_per_word: self.config.max_input_chars_per_word,
        })
    }
}

/// A
/// [WordPiece](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)
/// model.
#[derive(Clone, PartialEq)]
pub struct WordPiece {
    pub vocab: VocabStore,
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
            vocab: VocabStore::new(),
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
    pub fn read_file(vocab: &str) -> Result<VocabStore> {
        let file = File::open(vocab)?;
        let file = BufReader::new(file);

        let mut vocab_raw: Vec<(Vec<u8>, u32)> = vec![];
        for (index, line) in file.lines().enumerate() {
            let line = line?;
            vocab_raw.push((line.trim_end().to_owned().into_bytes(), index as u32));
        }

        let vocab = VocabStore::build(vocab_raw);

        Ok(vocab)
    }

    pub fn read_bytes(vocab: &[u8]) -> Result<VocabStore> {
        let file = BufReader::new(vocab);

        let mut vocab_raw: Vec<(Vec<u8>, u32)> = vec![];
        for (index, line) in file.lines().enumerate() {
            let line = line?;
            vocab_raw.push((line.trim_end().to_owned().into_bytes(), index as u32));
        }
        let vocab = VocabStore::build(vocab_raw);

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
        self.vocab.get_vocab().into_iter().collect()
    }

    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        let char_len = sequence.chars().count();

        if char_len > self.max_input_chars_per_word {
            return Ok(vec![Token {
                value: self.unk_token.clone(),
                id: self
                    .vocab
                    .token_to_id(&self.unk_token)
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
                if let Some(token) = self.vocab.token_to_id(substr.as_ref()) {
                    cur_str = Some(Token {
                        id: token,
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
                id: self
                    .vocab
                    .token_to_id(&self.unk_token)
                    .ok_or(Error::MissingUnkToken)?,
                offsets: (0, sequence.len()),
            }])
        } else {
            Ok(sub_tokens)
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab.id_to_token(id)
    }

    fn save(&self, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>> {
        let vocab_file_name = match name {
            Some(name) => format!("{name}-vocab.txt"),
            _ => "vocab.txt".to_string(),
        };

        // Write vocab.txt
        let vocab_path: PathBuf = [folder, Path::new(vocab_file_name.as_str())]
            .iter()
            .collect();
        let mut vocab_file = File::create(&vocab_path)?;
        let mut vocab: Vec<(String, u32)> = self.vocab.get_vocab().into_iter().collect();
        vocab.sort_unstable_by_key(|(_, rank)| *rank);
        vocab_file.write_all(
            &vocab
                .into_iter()
                .flat_map(|(token, _)| format!("{token}\n").as_bytes().to_owned())
                .collect::<Vec<_>>()[..],
        )?;

        Ok(vec![vocab_path])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vocab_store(tokens: &[&str]) -> VocabStore {
        VocabStore::build(
            tokens
                .iter()
                .enumerate()
                .map(|(id, token)| (token.to_string().into_bytes(), id as u32))
                .collect(),
        )
    }

    fn model(tokens: &[&str]) -> WordPiece {
        WordPiece::builder()
            .vocab_store(make_vocab_store(tokens))
            .build()
            .unwrap()
    }

    fn tok(id: u32, value: &str, offsets: (usize, usize)) -> Token {
        Token::new(id, value.to_string(), offsets)
    }

    #[test]
    fn test_error_display() {
        assert!(format!("{}", Error::MissingUnkToken).contains("Missing [UNK] token"));
    }

    #[test]
    fn whole_word_in_vocab() {
        let wp = model(&["[UNK]", "hello"]);
        assert_eq!(wp.tokenize("hello").unwrap(), vec![tok(1, "hello", (0, 5))]);
    }

    #[test]
    fn greedy_longest_match_splits_word() {
        let wp = model(&["[UNK]", "un", "##aff", "##able"]);
        assert_eq!(
            wp.tokenize("unaffable").unwrap(),
            vec![
                tok(1, "un", (0, 2)),
                tok(2, "##aff", (2, 5)),
                tok(3, "##able", (5, 9)),
            ]
        );
    }

    #[test]
    fn longest_piece_wins_over_shorter_ones() {
        let wp = model(&["[UNK]", "un", "unaff", "##aff", "##able"]);
        assert_eq!(
            wp.tokenize("unaffable").unwrap(),
            vec![tok(2, "unaff", (0, 5)), tok(4, "##able", (5, 9))]
        );
    }

    #[test]
    fn single_char_pieces() {
        let wp = model(&["[UNK]", "a", "##b", "##c"]);
        assert_eq!(
            wp.tokenize("abc").unwrap(),
            vec![
                tok(1, "a", (0, 1)),
                tok(2, "##b", (1, 2)),
                tok(3, "##c", (2, 3)),
            ]
        );
    }

    #[test]
    fn continuation_piece_requires_prefix() {
        let wp = model(&["[UNK]", "un", "able"]);
        assert_eq!(
            wp.tokenize("unable").unwrap(),
            vec![tok(0, "[UNK]", (0, 6))]
        );
    }

    #[test]
    fn prefixed_piece_never_matches_at_word_start() {
        let wp = model(&["[UNK]", "##able"]);
        assert_eq!(wp.tokenize("able").unwrap(), vec![tok(0, "[UNK]", (0, 4))]);
    }

    #[test]
    fn literal_prefix_in_input_matches_prefixed_vocab_entry() {
        let wp = model(&["[UNK]", "##able"]);
        assert_eq!(
            wp.tokenize("##able").unwrap(),
            vec![tok(1, "##able", (0, 6))]
        );
    }

    #[test]
    fn unmatchable_suffix_collapses_whole_word_to_unk() {
        let wp = model(&["[UNK]", "un", "##aff", "##able"]);
        assert_eq!(
            wp.tokenize("unaffordable").unwrap(),
            vec![tok(0, "[UNK]", (0, 12))]
        );
    }

    #[test]
    fn greedy_choice_is_never_backtracked() {
        let wp = model(&["[UNK]", "abc", "a", "##bcd"]);
        assert_eq!(wp.tokenize("abcd").unwrap(), vec![tok(0, "[UNK]", (0, 4))]);
    }

    #[test]
    fn unknown_first_char_is_unk() {
        let wp = model(&["[UNK]", "a"]);
        assert_eq!(wp.tokenize("xa").unwrap(), vec![tok(0, "[UNK]", (0, 2))]);
    }

    #[test]
    fn unk_id_is_looked_up_in_vocab() {
        let wp = model(&["a", "b", "[UNK]"]);
        assert_eq!(wp.tokenize("z").unwrap(), vec![tok(2, "[UNK]", (0, 1))]);
    }

    #[test]
    fn empty_input_yields_no_tokens() {
        let wp = model(&["[UNK]"]);
        assert_eq!(wp.tokenize("").unwrap(), vec![]);
    }

    #[test]
    fn multibyte_offsets_are_byte_positions() {
        let wp = model(&["[UNK]", "猫", "##です"]);
        assert_eq!(
            wp.tokenize("猫です").unwrap(),
            vec![tok(1, "猫", (0, 3)), tok(2, "##です", (3, 9))]
        );
    }

    #[test]
    fn multibyte_shrinking_stays_on_char_boundaries() {
        let wp = model(&["[UNK]", "a"]);
        assert_eq!(wp.tokenize("a€b").unwrap(), vec![tok(0, "[UNK]", (0, 5))]);
    }

    #[test]
    fn word_longer_than_max_chars_is_unk_even_if_in_vocab() {
        let wp = WordPiece::builder()
            .vocab_store(make_vocab_store(&["[UNK]", "abcde"]))
            .max_input_chars_per_word(4)
            .build()
            .unwrap();
        assert_eq!(wp.tokenize("abcde").unwrap(), vec![tok(0, "[UNK]", (0, 5))]);
    }

    #[test]
    fn word_at_max_chars_is_tokenized() {
        let wp = WordPiece::builder()
            .vocab_store(make_vocab_store(&["[UNK]", "abcd"]))
            .max_input_chars_per_word(4)
            .build()
            .unwrap();
        assert_eq!(wp.tokenize("abcd").unwrap(), vec![tok(1, "abcd", (0, 4))]);
    }

    #[test]
    fn max_chars_counts_chars_not_bytes() {
        let wp = WordPiece::builder()
            .vocab_store(make_vocab_store(&["[UNK]", "éééé"]))
            .max_input_chars_per_word(4)
            .build()
            .unwrap();
        assert_eq!(wp.tokenize("éééé").unwrap(), vec![tok(1, "éééé", (0, 8))]);
    }

    #[test]
    fn missing_unk_token_is_an_error_when_needed() {
        let wp = model(&["a"]);
        let err = wp.tokenize("b").unwrap_err();
        assert!(err.to_string().contains("Missing [UNK] token"));
    }

    #[test]
    fn missing_unk_token_is_an_error_for_overlong_words() {
        let wp = WordPiece::builder()
            .vocab_store(make_vocab_store(&["ab"]))
            .max_input_chars_per_word(1)
            .build()
            .unwrap();
        assert!(wp.tokenize("ab").is_err());
    }

    #[test]
    fn missing_unk_token_is_not_an_error_when_unneeded() {
        let wp = model(&["hello"]);
        assert_eq!(wp.tokenize("hello").unwrap(), vec![tok(0, "hello", (0, 5))]);
    }

    #[test]
    fn custom_unk_token() {
        let wp = WordPiece::builder()
            .vocab_store(make_vocab_store(&["<unk>", "a"]))
            .unk_token("<unk>".into())
            .build()
            .unwrap();
        assert_eq!(wp.tokenize("z").unwrap(), vec![tok(0, "<unk>", (0, 1))]);
    }

    #[test]
    fn custom_continuing_subword_prefix() {
        let wp = WordPiece::builder()
            .vocab_store(make_vocab_store(&["[UNK]", "foo", "@@bar"]))
            .continuing_subword_prefix("@@".into())
            .build()
            .unwrap();
        assert_eq!(
            wp.tokenize("foobar").unwrap(),
            vec![tok(1, "foo", (0, 3)), tok(2, "@@bar", (3, 6))]
        );
    }

    #[test]
    fn regular_words_tokenize_like_bert() {
        let wp = model(&[
            "[UNK]",
            "the",
            "token",
            "##izer",
            "##ization",
            "un",
            "##believ",
            "##able",
            "run",
            "##ning",
            "hugging",
            "##face",
            "transform",
            "##ers",
            "inter",
            "##national",
            "in",
            "##ter",
            "##nation",
            "##al",
            "##iz",
            "##ation",
            "fast",
            "##er",
            ".",
            ",",
        ]);
        let cases: &[(&str, &[&str])] = &[
            ("the", &["the"]),
            ("tokenizer", &["token", "##izer"]),
            ("tokenization", &["token", "##ization"]),
            ("unbelievable", &["un", "##believ", "##able"]),
            ("running", &["run", "##ning"]),
            ("huggingface", &["hugging", "##face"]),
            ("transformers", &["transform", "##ers"]),
            (
                "internationalization",
                &["inter", "##national", "##ization"],
            ),
            ("faster", &["fast", "##er"]),
            (".", &["."]),
            (",", &[","]),
            ("xylophone", &["[UNK]"]),
        ];
        for (word, expected) in cases {
            let values: Vec<String> = wp
                .tokenize(word)
                .unwrap()
                .into_iter()
                .map(|t| t.value)
                .collect();
            assert_eq!(&values, expected, "word: {word}");
        }

        assert_eq!(
            wp.tokenize("internationalization").unwrap(),
            vec![
                tok(14, "inter", (0, 5)),
                tok(15, "##national", (5, 13)),
                tok(4, "##ization", (13, 20)),
            ]
        );
    }
}
