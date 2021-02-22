use super::WordLevel;
use crate::utils::parallelism::*;
use crate::{AddedToken, Result, Trainer};
use std::collections::HashMap;

#[non_exhaustive]
#[derive(Debug, Clone, Builder)]
pub struct WordLevelTrainer {
    /// The minimum frequency a word must have to be part of the vocabulary
    #[builder(default)]
    pub min_frequency: u32,
    /// The target vocabulary size
    #[builder(default)]
    pub vocab_size: usize,
    /// Whether to show progress while training
    #[builder(default)]
    pub show_progress: bool,
    /// A list of special tokens that the model should know of
    #[builder(default)]
    pub special_tokens: Vec<AddedToken>,

    #[builder(default, private)]
    words: HashMap<String, u32>,
}

impl Default for WordLevelTrainer {
    fn default() -> Self {
        Self {
            min_frequency: 0,
            vocab_size: 30_000,
            show_progress: true,
            special_tokens: vec![],
            words: HashMap::new(),
        }
    }
}

impl WordLevelTrainer {
    pub fn builder() -> WordLevelTrainerBuilder {
        WordLevelTrainerBuilder::default()
    }

    fn do_train(
        &self,
        word_counts: &HashMap<String, u32>,
        model: &mut WordLevel,
    ) -> Result<Vec<AddedToken>> {
        let mut ordered_counts = word_counts.iter().collect::<Vec<_>>();
        ordered_counts.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        let word_level = WordLevel::builder()
            .vocab(
                self.special_tokens
                    .iter()
                    .map(|token| token.content.clone())
                    .chain(
                        ordered_counts
                            .into_iter()
                            .filter(|(_, n)| **n >= self.min_frequency)
                            .map(|(w, _)| w.to_owned()),
                    )
                    .take(self.vocab_size)
                    .enumerate()
                    .map(|(i, w)| (w, i as u32))
                    .collect(),
            )
            .build()?;

        // Transfer the vocab
        model.vocab = word_level.vocab;
        model.vocab_r = word_level.vocab_r;

        Ok(self.special_tokens.clone())
    }
}

impl Trainer for WordLevelTrainer {
    type Model = WordLevel;

    /// Train a WordLevel model
    fn train(&self, model: &mut WordLevel) -> Result<Vec<AddedToken>> {
        self.do_train(&self.words, model)
    }

    /// Whether we should show progress
    fn should_show_progress(&self) -> bool {
        self.show_progress
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        let words: Result<HashMap<String, u32>> = iterator
            .maybe_par_bridge()
            .map(|sequence| {
                let words = process(sequence.as_ref())?;
                let mut map = HashMap::new();
                for word in words {
                    map.entry(word).and_modify(|c| *c += 1).or_insert(1);
                }
                Ok(map)
            })
            .reduce(
                || Ok(HashMap::new()),
                |acc, ws| {
                    let mut acc = acc?;
                    for (k, v) in ws? {
                        acc.entry(k).and_modify(|c| *c += v).or_insert(v);
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
        let word_counts: HashMap<String, u32> = [
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
        trainer.vocab_size = 5;

        let mut model = WordLevel::default();
        trainer.do_train(&word_counts, &mut model).unwrap();
        let expected_vocab: HashMap<String, u32> = [
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
        let mut model = WordLevel::default();
        trainer.do_train(&word_counts, &mut model).unwrap();
        let expected_vocab: HashMap<String, u32> = [
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
}
