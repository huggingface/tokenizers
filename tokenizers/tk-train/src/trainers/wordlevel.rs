use crate::error::Result;
use crate::trainer::{ModelTrainer, TrainingParams};
use ahash::AHashMap;
use serde::Deserialize;
use std::cmp::Ordering;
use tk_encode::models::wordlevel::WordLevel;
use tk_encode::utils::parallelism::*;

#[derive(Debug, Deserialize)]
pub struct WordLevelTrainer {
    min_frequency: u64,

    words: AHashMap<String, u64>,
}

impl Default for WordLevelTrainer {
    fn default() -> Self {
        Self {
            min_frequency: 0,
            words: AHashMap::new(),
        }
    }
}

#[derive(Debug, Default)]
pub struct WordLevelTrainerBuilder {
    trainer: WordLevelTrainer,
}

impl WordLevelTrainerBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// The minimum frequency a word must have to be part of the vocabulary.
    #[must_use]
    pub fn min_frequency(mut self, min_frequency: u64) -> Self {
        self.trainer.min_frequency = min_frequency;
        self
    }

    pub fn build(self) -> WordLevelTrainer {
        self.trainer
    }
}

impl WordLevelTrainer {
    pub fn builder() -> WordLevelTrainerBuilder {
        WordLevelTrainerBuilder::default()
    }

    /// A builder with this trainer's settings. The words fed so far are dropped.
    pub fn to_builder(mut self) -> WordLevelTrainerBuilder {
        self.words = AHashMap::new();
        WordLevelTrainerBuilder { trainer: self }
    }

    pub fn min_frequency(&self) -> u64 {
        self.min_frequency
    }

    fn do_train(
        &self,
        word_counts: &AHashMap<String, u64>,
        params: &TrainingParams,
    ) -> Result<WordLevel> {
        let mut ordered_counts = word_counts.iter().collect::<Vec<_>>();

        //sort the word counts first by inverse counts and then by word, in order
        //to keep the sorting deterministic in case of equal counts
        let cmp = |l: &(&String, &u64), r: &(&String, &u64)| -> Ordering {
            let count_comp: Ordering = l.1.cmp(r.1);
            if count_comp != Ordering::Equal {
                return count_comp.reverse();
            }
            l.0.cmp(r.0)
        };

        ordered_counts.sort_by(cmp);

        let mut word_level = WordLevel::builder();
        if let Some(unk_token) = &params.unk_token {
            word_level = word_level.unk_token(unk_token.clone());
        }
        let word_level = word_level
            .vocab(
                params
                    .special_tokens
                    .iter()
                    .map(|token| token.content.clone())
                    .chain(
                        ordered_counts
                            .into_iter()
                            .filter(|(_, n)| **n >= self.min_frequency)
                            .map(|(w, _)| w.to_owned()),
                    )
                    .take(params.vocab_size)
                    .enumerate()
                    .map(|(i, w)| (w, i as u32))
                    .collect(),
            )
            .build()?;

        Ok(word_level)
    }
}

impl ModelTrainer for WordLevelTrainer {
    type Model = WordLevel;

    /// Train a WordLevel model
    fn train_model(&self, params: &TrainingParams) -> Result<WordLevel> {
        self.do_train(&self.words, params)
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        let words: Result<AHashMap<String, u64>> = iterator
            .maybe_par_bridge()
            .map(|sequence| {
                let words = process(sequence.as_ref())?;
                let mut map = AHashMap::new();
                for word in words {
                    *map.entry(word).or_default() += 1;
                }
                Ok(map)
            })
            .reduce(
                || Ok(AHashMap::new()),
                |acc, ws| {
                    let mut acc = acc?;
                    for (k, v) in ws? {
                        *acc.entry(k).or_default() += v;
                    }
                    Ok(acc)
                },
            );

        self.words = words?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train() {
        let word_counts: AHashMap<String, u64> = [
            ("the".into(), 25),
            ("roses".into(), 22),
            ("are".into(), 24),
            ("red".into(), 12),
            ("voilets".into(), 10),
            ("blue".into(), 16),
        ]
        .iter()
        .cloned()
        .collect();

        let mut trainer = WordLevelTrainer::default();
        let params = TrainingParams::for_tests(5);

        let model = trainer.do_train(&word_counts, &params).unwrap();
        let expected_vocab: AHashMap<String, u32> = [
            ("the".into(), 0),
            ("are".into(), 1),
            ("roses".into(), 2),
            ("blue".into(), 3),
            ("red".into(), 4),
        ]
        .iter()
        .cloned()
        .collect();
        assert_eq!(model.vocab, expected_vocab);

        // If we specify a min_frequency
        trainer.min_frequency = 15;
        let model = trainer.do_train(&word_counts, &params).unwrap();
        let expected_vocab: AHashMap<String, u32> = [
            ("the".into(), 0),
            ("are".into(), 1),
            ("roses".into(), 2),
            ("blue".into(), 3),
        ]
        .iter()
        .cloned()
        .collect();

        assert_eq!(model.vocab, expected_vocab);
    }

    #[test]
    fn to_builder_keeps_settings_and_drops_words() {
        let mut trainer = WordLevelTrainer::builder().min_frequency(2).build();
        trainer
            .feed(["hello world"].iter(), |s| {
                Ok(s.split(' ').map(str::to_owned).collect())
            })
            .unwrap();

        let trainer = trainer.to_builder().build();

        assert_eq!(trainer.min_frequency(), 2);
        assert!(trainer.words.is_empty());
    }
}
