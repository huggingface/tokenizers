use super::WordPiece;
use crate::models::bpe::{BpeTrainer, BpeTrainerBuilder};
use crate::tokenizer::{Model, Result, Trainer};
use std::collections::{HashMap, HashSet};

/// A `WordPieceTrainerBuilder` can be used to create a `WordPieceTrainer` with a custom
/// configuration.
pub struct WordPieceTrainerBuilder {
    bpe_trainer_builder: BpeTrainerBuilder,
}

impl Default for WordPieceTrainerBuilder {
    fn default() -> Self {
        Self {
            bpe_trainer_builder: BpeTrainerBuilder::new().continuing_subword_prefix("##".into()),
        }
    }
}

impl WordPieceTrainerBuilder {
    /// Constructs a new `WordPieceTrainerBuilder`
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the expected minimum frequency
    pub fn min_frequency(mut self, frequency: u32) -> Self {
        self.bpe_trainer_builder = self.bpe_trainer_builder.min_frequency(frequency);
        self
    }

    /// Set the vocabulary size
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.bpe_trainer_builder = self.bpe_trainer_builder.vocab_size(size);
        self
    }

    /// Set whether to show progress
    pub fn show_progress(mut self, show: bool) -> Self {
        self.bpe_trainer_builder = self.bpe_trainer_builder.show_progress(show);
        self
    }

    /// Set the special tokens
    pub fn special_tokens(mut self, tokens: Vec<String>) -> Self {
        self.bpe_trainer_builder = self.bpe_trainer_builder.special_tokens(tokens);
        self
    }

    /// Set whether to limit the alphabet
    pub fn limit_alphabet(mut self, limit: usize) -> Self {
        self.bpe_trainer_builder = self.bpe_trainer_builder.limit_alphabet(limit);
        self
    }

    /// Set the initial alphabet
    pub fn initial_alphabet(mut self, alphabet: HashSet<char>) -> Self {
        self.bpe_trainer_builder = self.bpe_trainer_builder.initial_alphabet(alphabet);
        self
    }

    /// Set the continuing_subword_prefix
    pub fn continuing_subword_prefix(mut self, prefix: String) -> Self {
        self.bpe_trainer_builder = self.bpe_trainer_builder.continuing_subword_prefix(prefix);
        self
    }

    /// Set the end_of_word_suffix
    pub fn end_of_word_suffix(mut self, suffix: String) -> Self {
        self.bpe_trainer_builder = self.bpe_trainer_builder.end_of_word_suffix(suffix);
        self
    }

    /// Constructs the final BpeTrainer
    pub fn build(self) -> WordPieceTrainer {
        let bpe_trainer = self.bpe_trainer_builder.build();
        WordPieceTrainer { bpe_trainer }
    }
}

/// Trains a `WordPiece` model.
#[derive(Default)]
pub struct WordPieceTrainer {
    bpe_trainer: BpeTrainer,
}

impl WordPieceTrainer {
    pub fn builder() -> WordPieceTrainerBuilder {
        WordPieceTrainerBuilder::default()
    }

    pub fn train(&self, word_counts: HashMap<String, u32>) -> Result<(WordPiece, Vec<String>)> {
        let (bpe, tokens) = self.bpe_trainer.train(word_counts)?;
        Ok((WordPiece::from_bpe(&bpe), tokens))
    }
}

impl Trainer for WordPieceTrainer {
    fn train(
        &self,
        word_counts: HashMap<String, u32>,
    ) -> Result<(Box<dyn Model + Sync>, Vec<String>)> {
        let (wp, tokens) = self.train(word_counts)?;
        Ok((Box::new(wp), tokens))
    }

    fn process_tokens(&self, mut words: &mut HashMap<String, u32>, tokens: Vec<String>) {
        self.bpe_trainer.process_tokens(&mut words, tokens)
    }

    fn should_show_progress(&self) -> bool {
        self.bpe_trainer.should_show_progress()
    }
}
