use crate::pipeline::{self, ModelScratch, PipelineToken};
use crate::tokenizer::{Model, Result, Token};
use ahash::AHashMap;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Only the tests name this now: reading a `vocab.json` into one is `tk-convert`'s job.
#[cfg(test)]
type Vocab = AHashMap<String, u32>;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("WordLevel error: Missing [UNK] token from the vocabulary")]
    MissingUnkToken,
    #[error("Bad vocabulary json file")]
    BadVocabulary,
}

struct Config {
    vocab: AHashMap<String, u32>,
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
                vocab: AHashMap::new(),
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
    #[must_use]
    pub fn vocab(mut self, vocab: AHashMap<String, u32>) -> Self {
        self.config.vocab = vocab;
        self
    }

    /// The the `UNK` token for the vocab.
    #[must_use]
    pub fn unk_token(mut self, unk_token: String) -> Self {
        self.config.unk_token = unk_token;
        self
    }

    /// Constructs a `WordLevel` model that uses the `WordLevelBuilder`'s configuration.
    pub fn build(self) -> Result<WordLevel> {
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

#[derive(PartialEq, Clone, Eq)]
pub struct WordLevel {
    pub vocab: AHashMap<String, u32>,
    pub vocab_r: AHashMap<u32, String>,
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
}

impl Default for WordLevel {
    fn default() -> Self {
        Self {
            vocab: AHashMap::new(),
            vocab_r: AHashMap::new(),
            unk_token: String::from("<unk>"),
        }
    }
}

impl Model for WordLevel {
    fn tokenize(&self, token: &str) -> Result<Vec<Token>> {
        if let Some(&id) = self.vocab.get(token) {
            Ok(vec![Token {
                id,
                value: token.to_owned(),
                offsets: (0, token.len()),
            }])
        } else if let Some(&unk_id) = self.vocab.get(&self.unk_token) {
            Ok(vec![Token {
                id: unk_id,
                value: self.unk_token.to_owned(),
                offsets: (0, token.len()),
            }])
        } else {
            Err(Box::new(Error::MissingUnkToken))
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab_r.get(&id).cloned()
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.clone().into_iter().collect()
    }

    fn get_vocab_size(&self) -> usize {
        self.vocab.keys().len()
    }

    /// A `vocab.json` is serde output — a `{token: id}` object in id order — and this crate links
    /// no serde, so the writing lives in `tk_convert::models::wordlevel::save`. It cannot move
    /// *with* the method, because `Model` and `WordLevel` are both defined here and the orphan rule
    /// forbids another crate from writing `impl Model for WordLevel`. Every real caller reaches
    /// `save` through `tk_convert::ModelWrapper`, which writes the file itself.
    fn save(&self, _folder: &Path, _name: Option<&str>) -> Result<Vec<PathBuf>> {
        Err("writing a WordLevel vocabulary is config-layer work: use \
             `tk_convert::ModelWrapper::save`, or `tk_convert::models::wordlevel::save` for a bare \
             model"
            .into())
    }
}

type WordLevelScratch = ();
impl ModelScratch for WordLevelScratch {}

impl pipeline::Model for WordLevel {
    type Scratch = WordLevelScratch;
    fn init_scratch(&self) -> Self::Scratch {}
    fn tokenize_pipeline(
        &self,
        sequence: &str,
        _scratch: &mut Self::Scratch,
        output: &mut Vec<pipeline::PipelineToken>,
    ) -> Result<()> {
        if let Some(&id) = self.vocab.get(sequence) {
            output.push(PipelineToken::from(id))
        } else if let Some(&unk_id) = self.vocab.get(&self.unk_token) {
            output.push(PipelineToken::from(unk_id));
        } else {
            return Err(Box::new(Error::MissingUnkToken));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_unk() {
        let vocab: Vocab = [("<unk>".into(), 0), ("a".into(), 1), ("b".into(), 2)]
            .iter()
            .cloned()
            .collect();
        let wordlevel = WordLevelBuilder::default()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .build()
            .unwrap();
        let tokens = wordlevel.tokenize("c").unwrap();
        assert_eq!(tokens, vec![Token::new(0u32, "<unk>".into(), (0, 1)),]);

        let tokens = wordlevel.tokenize("a").unwrap();
        assert_eq!(tokens, vec![Token::new(1u32, "a".into(), (0, 1)),]);
    }

    #[test]
    fn test_tokenize_missing_unk_token() {
        let vocab: Vocab = [("a".into(), 0), ("b".into(), 1)].iter().cloned().collect();
        let wordlevel = WordLevelBuilder::default().vocab(vocab).build().unwrap();
        let tokens = wordlevel.tokenize("a").unwrap();
        assert_eq!(tokens, vec![Token::new(0u32, "a".into(), (0, 1)),]);

        let error = wordlevel.tokenize("c").err().unwrap();
        assert!(error.is::<Error>());
    }
}
