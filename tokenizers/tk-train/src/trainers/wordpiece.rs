use crate::error::Result;
use crate::trainer::{ModelTrainer, TrainingParams};
use crate::trainers::bpe::{BpeTrainer, BpeTrainerBuilder};
use ahash::AHashSet;
use serde::Deserialize;
use tk_encode::models::wordpiece::WordPiece;

#[derive(Debug)]
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
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn min_frequency(mut self, frequency: u64) -> Self {
        self.bpe_trainer_builder = self.bpe_trainer_builder.min_frequency(frequency);
        self
    }

    /// See [`BpeTrainerBuilder::limit_alphabet`].
    #[must_use]
    pub fn limit_alphabet(mut self, limit: usize) -> Self {
        self.bpe_trainer_builder = self.bpe_trainer_builder.limit_alphabet(limit);
        self
    }

    /// See [`BpeTrainerBuilder::initial_alphabet`].
    #[must_use]
    pub fn initial_alphabet(mut self, alphabet: impl IntoIterator<Item = char>) -> Self {
        self.bpe_trainer_builder = self.bpe_trainer_builder.initial_alphabet(alphabet);
        self
    }

    #[must_use]
    pub fn continuing_subword_prefix(mut self, prefix: String) -> Self {
        self.bpe_trainer_builder = self.bpe_trainer_builder.continuing_subword_prefix(prefix);
        self
    }

    #[must_use]
    pub fn end_of_word_suffix(mut self, suffix: String) -> Self {
        self.bpe_trainer_builder = self.bpe_trainer_builder.end_of_word_suffix(suffix);
        self
    }

    pub fn build(self) -> WordPieceTrainer {
        let bpe_trainer = self.bpe_trainer_builder.build();
        WordPieceTrainer { bpe_trainer }
    }
}

/// Trains a `WordPiece` model.
#[derive(Debug, Deserialize)]
pub struct WordPieceTrainer {
    bpe_trainer: BpeTrainer,
}

impl WordPieceTrainer {
    pub fn builder() -> WordPieceTrainerBuilder {
        WordPieceTrainerBuilder::default()
    }

    /// A builder with this trainer's settings. The words fed so far are dropped.
    pub fn to_builder(self) -> WordPieceTrainerBuilder {
        WordPieceTrainerBuilder {
            bpe_trainer_builder: self.bpe_trainer.to_builder(),
        }
    }

    pub fn min_frequency(&self) -> u64 {
        self.bpe_trainer.min_frequency()
    }

    pub fn limit_alphabet(&self) -> Option<usize> {
        self.bpe_trainer.limit_alphabet()
    }

    pub fn initial_alphabet(&self) -> &AHashSet<char> {
        self.bpe_trainer.initial_alphabet()
    }

    pub fn continuing_subword_prefix(&self) -> Option<&str> {
        self.bpe_trainer.continuing_subword_prefix()
    }

    pub fn end_of_word_suffix(&self) -> Option<&str> {
        self.bpe_trainer.end_of_word_suffix()
    }
}

impl ModelTrainer for WordPieceTrainer {
    type Model = WordPiece;

    fn train_model(&self, params: &TrainingParams) -> Result<WordPiece> {
        // WordPiece reinterprets a trained BPE's vocabulary as its own pieces; the merge list has no
        // meaning here and is dropped. This used to go through `tk_convert`'s `from_bpe`, which
        // built a whole `BPE` to read its vocabulary back off -- `train_vocab` hands over the same
        // vocabulary without building anything.
        let (vocab, _merges) = self.bpe_trainer.train_vocab(params)?;

        let mut model = WordPiece {
            vocab_r: vocab.iter().map(|(t, id)| (*id, t.clone())).collect(),
            vocab,
            ..Default::default()
        };
        // The continuing_subword_prefix is the only other option to be overridden by the trainer
        if let Some(prefix) = self.bpe_trainer.continuing_subword_prefix() {
            model.continuing_subword_prefix = prefix.to_owned();
        }
        if let Some(unk_token) = &params.unk_token {
            model.unk_token = unk_token.clone();
        }

        Ok(model)
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        self.bpe_trainer.feed(iterator, process)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_builder_keeps_settings() {
        let mut trainer = WordPieceTrainer::builder().min_frequency(2).build();
        trainer
            .feed(["hello world"].iter(), |s| {
                Ok(s.split(' ').map(str::to_owned).collect())
            })
            .unwrap();

        let trainer = trainer.to_builder().build();

        assert_eq!(trainer.continuing_subword_prefix(), Some("##"));
        assert_eq!(trainer.min_frequency(), 2);
    }
}
