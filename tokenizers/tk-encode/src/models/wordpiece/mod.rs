//! [WordPiece](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37842.pdf)
//! model.

use crate::pipeline::{self, PipelineToken};
use crate::tokenizer::{Model, Result, Token};
use crate::utils::cache::DEFAULT_CACHE_CAPACITY;
use crate::utils::word_cache::{Lookup, WordCache};
use ahash::AHashMap;
use std::borrow::Cow;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::fs::File;
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use yada::DoubleArray;
use yada::builder::DoubleArrayBuilder;

#[cfg(feature = "serde")]
mod serialization;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("WordPiece error: Missing [UNK] token from the vocabulary")]
    MissingUnkToken,
}

type Vocab = AHashMap<String, u32>;
type VocabR = AHashMap<u32, String>;

struct Config {
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
    pub fn build(self) -> Result<WordPiece> {
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

    // Loading a vocabulary is `tk-convert`'s job now, in every spelling: `read_file` /
    // `read_bytes` / `from_file` for a `vocab.txt`, `from_bytes` for a serialized model (which
    // needs serde), and `from_bpe`, which cannot be here at all because the config-shaped `BPE` it
    // names is not in this crate any more. See `tk_convert::models::wordpiece`.
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

    /// The one model whose file is not serde output: a `vocab.txt` is one token per line, so this
    /// stays where the model is.
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

pub struct WordPieceScratch {
    candidate_str: String,
    /// Outlives the encode call that fills it, or it would never see a word twice.
    word_cache: WordCache,
}

impl pipeline::ModelScratch for WordPieceScratch {}

pub struct PipelineWordPiece {
    vocab_trie: yada::DoubleArray<Vec<u8>>,
    vocab_r: Box<[Option<Box<str>>]>,
    unk_token: Option<u32>,
    continuing_subword_prefix: String,
    max_input_chars_per_word: usize,
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
        let vocab_trie = DoubleArray::new(DoubleArrayBuilder::build(&keyset)?)?;
        let max_id = keyset.iter().map(|&(_, id)| id).max().unwrap_or(0) as usize;
        let mut vocab_r = vec![None; max_id + 1];
        for (token, id) in keyset {
            vocab_r[id as usize] = Some(token.into_boxed_str());
        }

        Ok(Self {
            continuing_subword_prefix,
            max_input_chars_per_word,
            unk_token,
            vocab_trie,
            vocab_r: vocab_r.into_boxed_slice(),
        })
    }
}

impl PipelineWordPiece {
    /// One word, greedily: the longest vocabulary entry it starts with, then the
    /// longest entry the rest of it starts with once the continuing-subword prefix
    /// is put in front, and so on. A piece with no entry at all anywhere in the
    /// word makes the whole word one unk token.
    fn tokenize_word(
        &self,
        sequence: &str,
        candidate: &mut String,
        output: &mut Vec<PipelineToken>,
    ) -> Result<()> {
        let checkpoint = output.len();

        let char_len = sequence.chars().count();
        if char_len > self.max_input_chars_per_word {
            let unk_id = self.unk_token.ok_or(Error::MissingUnkToken)?;
            output.push(PipelineToken::from(unk_id));
            return Ok(());
        }

        let mut start = 0;

        while start < sequence.len() {
            candidate.clear();
            let prefix_len = if start > 0 {
                candidate.push_str(&self.continuing_subword_prefix);
                self.continuing_subword_prefix.len()
            } else {
                0
            };
            candidate.push_str(&sequence[start..]);

            // Matches must extend past the continuing-subword prefix: the
            // prefix alone (or a fragment of it) is not a valid subword here,
            // even if it happens to be in the vocab.
            let Some((token_id, match_len)) = self
                .vocab_trie
                .common_prefix_search(&candidate)
                .filter(|(_, len)| *len > prefix_len)
                .last()
            else {
                let unk_id = self.unk_token.ok_or(Error::MissingUnkToken)?;
                output.truncate(checkpoint);
                output.push(PipelineToken::from(unk_id));
                return Ok(());
            };
            output.push(PipelineToken::from(token_id));
            start += match_len - prefix_len;
        }
        Ok(())
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab_r.get(id as usize)?.as_deref().map(str::to_owned)
    }
}

impl pipeline::Model for PipelineWordPiece {
    type Scratch = WordPieceScratch;

    fn init_scratch(&self) -> Self::Scratch {
        Self::Scratch {
            candidate_str: String::with_capacity(self.max_input_chars_per_word),
            word_cache: WordCache::new(DEFAULT_CACHE_CAPACITY),
        }
    }

    /// A hit skips `tokenize_word`: one trie search per piece of the word,
    /// each over a fresh copy of what is left to match.
    fn tokenize_pipeline(
        &self,
        sequence: &str,
        scratch: &mut Self::Scratch,
        output: &mut Vec<pipeline::PipelineToken>,
    ) -> Result<()> {
        if sequence.is_empty() {
            return Ok(());
        }
        let WordPieceScratch {
            candidate_str,
            word_cache,
        } = scratch;

        let placement = match word_cache.lookup(sequence.as_bytes()) {
            Lookup::Hit(ids) => {
                output.extend(ids.iter().copied().map(PipelineToken::from));
                return Ok(());
            }
            Lookup::Miss(at) => at,
        };

        let start = output.len();
        self.tokenize_word(sequence, candidate_str, output)?;

        word_cache.insert(placement, output[start..].iter().map(|token| token.id()));
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

    /// `hello` is in the vocabulary whole and as `hell` + `##o`, so the
    /// longest-match walk has something to choose; `world` gives a second
    /// one-token word.
    fn pipeline_wordpiece() -> PipelineWordPiece {
        let vocab: Vocab = [
            ("[UNK]", 0u32),
            ("hell", 1),
            ("##o", 2),
            ("hello", 3),
            ("world", 4),
        ]
        .into_iter()
        .map(|(token, id)| (token.to_string(), id))
        .collect();
        let model = WordPiece::builder()
            .vocab(vocab)
            .max_input_chars_per_word(8)
            .build()
            .unwrap();
        PipelineWordPiece::try_from(model).unwrap()
    }

    fn pipeline_ids(
        model: &PipelineWordPiece,
        sequence: &str,
        scratch: &mut WordPieceScratch,
    ) -> Vec<u32> {
        let mut output = vec![];
        pipeline::Model::tokenize_pipeline(model, sequence, scratch, &mut output).unwrap();
        output.iter().map(|token| token.id()).collect()
    }

    #[test]
    fn pipeline_remembers_what_a_word_encoded_to() {
        let model = pipeline_wordpiece();
        let mut scratch = pipeline::Model::init_scratch(&model);

        let ids = pipeline_ids(&model, "hello", &mut scratch);

        assert_eq!(scratch.word_cache.lookup(b"hello").hit(), Some(&ids[..]));
    }

    #[test]
    fn cache_hits_agree_with_a_cold_run() {
        let model = pipeline_wordpiece();
        let long = "hello".repeat(300);
        let corpus = [
            "hello",
            // No id for `##w`, so the whole word is one unk token.
            "hellow",
            // Both again, so these two are served from the cache.
            "hello",
            "hellow",
            // Past `max_input_chars_per_word`, which is another unk token.
            "hellohello",
            "hellohello",
            // Out of the vocabulary, and multibyte.
            "東京",
            "東京",
            // 1500 bytes, past the longest word the cache will store.
            long.as_str(),
            long.as_str(),
        ];

        let mut warm_scratch = pipeline::Model::init_scratch(&model);
        let warm = corpus.map(|sequence| pipeline_ids(&model, sequence, &mut warm_scratch));
        let cold = corpus.map(|sequence| {
            let mut scratch = pipeline::Model::init_scratch(&model);
            pipeline_ids(&model, sequence, &mut scratch)
        });

        assert_eq!(warm, cold);
    }

    #[test]
    fn caches_only_the_ids_this_word_produced() {
        // Every word the pipeline hands the model appends to one output buffer,
        // so a word has to remember its own ids, not everything the buffer holds.
        let model = pipeline_wordpiece();
        let mut scratch = pipeline::Model::init_scratch(&model);
        let mut output = vec![];
        pipeline::Model::tokenize_pipeline(&model, "hello", &mut scratch, &mut output).unwrap();
        pipeline::Model::tokenize_pipeline(&model, "world", &mut scratch, &mut output).unwrap();

        let ids: Vec<u32> = output.iter().map(|token| token.id()).collect();
        assert_eq!(ids, [3, 4]);
        assert_eq!(scratch.word_cache.lookup(b"world").hit(), Some(&[4u32][..]));
    }
}
