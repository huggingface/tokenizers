#![allow(clippy::map_entry)]

use super::{Pair, Word, BPE};
use crate::tokenizer::{Model, Result, Trainer};
use std::{
    collections::{HashMap, HashSet},
    time::Instant,
};

pub struct BpeTrainerConfig {
    min_frequency: u32,
    vocab_size: usize,
}

impl BpeTrainerConfig {
    pub fn new(min_frequency: u32, vocab_size: usize) -> Self {
        BpeTrainerConfig {
            min_frequency,
            vocab_size,
        }
    }

    pub fn set_vocab_size(&mut self, value: usize) {
        self.vocab_size = value;
    }

    pub fn set_min_frequency(&mut self, value: u32) {
        self.min_frequency = value;
    }
}

impl Default for BpeTrainerConfig {
    fn default() -> Self {
        BpeTrainerConfig::new(0, 30000)
    }
}

/// In charge of training a BPE model from a mapping of words to word counts.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
/// use tokenizers::tokenizer::Trainer;
/// use tokenizers::models::bpe::BpeTrainer;
///
/// let word_counts: HashMap<String, u32> = [
///     (String::from("Hello"), 1),
///     (String::from("World"), 1),
/// ].iter().cloned().collect();
/// let trainer = BpeTrainer::default();
/// let model = trainer.train(word_counts);
/// ```
#[derive(Default)]
pub struct BpeTrainer {
    // Training parameters
    config: BpeTrainerConfig,
}

impl BpeTrainer {
    pub fn new(config: BpeTrainerConfig) -> Self {
        BpeTrainer { config }
    }
}

impl Trainer for BpeTrainer {
    /// Train a BPE model
    fn train(&self, word_counts: HashMap<String, u32>) -> Result<Box<dyn Model + Sync>> {
        let mut words: Vec<Word> = vec![];
        let mut counts: Vec<i32> = vec![];
        let mut word_to_id: HashMap<String, u32> = HashMap::new();
        let mut id_to_word: Vec<String> = vec![];

        //
        // 1. Tokenize words
        //
        let timer = Instant::now();
        for (word, count) in &word_counts {
            let mut current_word = Word::new();
            counts.push(*count as i32);

            for c in word.chars() {
                let s = c.to_string();
                if !word_to_id.contains_key(&s) {
                    id_to_word.push(s.clone());
                    word_to_id.insert(s.clone(), (id_to_word.len() - 1) as u32);
                }
                current_word.add(word_to_id[&s]);
            }
            words.push(current_word);
        }
        println!("[{:?}] Tokenized {} words", timer.elapsed(), words.len());

        //
        // 2. Count pairs in words
        //
        let timer = Instant::now();
        let mut pair_counts: HashMap<Pair, (i32, Pair)> = HashMap::new();
        let mut where_to_update: HashMap<Pair, HashSet<usize>> = HashMap::new();
        for (index, word) in words.iter().enumerate() {
            for window in word.get_chars().windows(2) {
                let cur_pair: Pair = (window[0], window[1]);

                // Initialize pair_counts and where_to_update for this pair if we just saw it
                if !pair_counts.contains_key(&cur_pair) {
                    let pair = (0, cur_pair);
                    pair_counts.insert(cur_pair, pair);
                    if !where_to_update.contains_key(&cur_pair) {
                        where_to_update.insert(cur_pair, HashSet::new());
                    }
                }

                // Then update counts
                let count = counts[index];
                if count > 0 {
                    where_to_update.get_mut(&cur_pair).unwrap().insert(index);
                } else {
                    where_to_update.get_mut(&cur_pair).unwrap().remove(&index);
                }
                pair_counts.get_mut(&cur_pair).unwrap().0 += count;
            }
        }
        println!(
            "[{:?}] Counted {} pairs with {} unique tokens",
            timer.elapsed(),
            pair_counts.len(),
            word_to_id.len()
        );

        //
        // 3. Do merges
        //
        let mut merges: Vec<(Pair, u32)> = vec![];
        let timer = Instant::now();
        loop {
            // Stop as soon as we have a big enough vocabulary
            if word_to_id.len() >= self.config.vocab_size {
                break;
            }

            // Find the best pair
            let mut best_count = 0;
            let mut best_pair = (std::u32::MAX, std::u32::MAX);
            for x in pair_counts.values() {
                if x.0 > best_count {
                    best_count = x.0;
                    best_pair = x.1;
                } else if x.0 == best_count && x.1 < best_pair {
                    best_pair = x.1;
                }
            }
            // Stop if we reached the minimum frequency
            if best_count < 1 || self.config.min_frequency > best_count as u32 {
                break;
            }

            let new_token = format!(
                "{}{}",
                id_to_word[best_pair.0 as usize], id_to_word[best_pair.1 as usize]
            );

            // Insert new token
            let new_token_id = id_to_word.len() as u32;
            id_to_word.push(new_token.clone());
            word_to_id.insert(new_token.clone(), new_token_id);
            merges.push((best_pair, new_token_id));

            // Reset count for the current best pair
            pair_counts.get_mut(&best_pair).unwrap().0 = 0;

            // We have to clone below, because the change_count closure keeps a mutable borrow
            let where_to = where_to_update.get(&best_pair).unwrap().clone();

            let mut change_count = |pair: Pair, count: i32, word_index: usize| {
                if pair_counts.contains_key(&pair) {
                    pair_counts.get_mut(&pair).unwrap().0 += count;
                } else if count > 0 {
                    pair_counts.insert(pair, (count, pair));
                    if !where_to_update.contains_key(&pair) {
                        where_to_update.insert(pair, HashSet::new());
                    }
                }

                if count > 0 {
                    where_to_update.get_mut(&pair).unwrap().insert(word_index);
                }
            };

            // Change all other counts
            for word_index in where_to {
                let cur_word = &mut words[word_index];
                let changes = cur_word.merge(best_pair.0, best_pair.1, new_token_id);

                for change in changes {
                    change_count(change.0, change.1 * counts[word_index], word_index);
                }
            }
        }
        println!("[{:?}] Computed {} merges", timer.elapsed(), merges.len());

        Ok(Box::new(BPE::new(
            word_to_id.clone(),
            word_to_id
                .into_iter()
                .map(|(token, id)| (id, token))
                .collect(),
            merges
                .into_iter()
                .enumerate()
                .map(|(index, (pair, new_id))| (pair, (index as u32, new_id)))
                .collect(),
        )))
    }

    /// Process a bunch of tokens, counting them
    fn process_tokens(&self, words: &mut HashMap<String, u32>, tokens: Vec<String>) {
        for token in tokens {
            words
                .entry(token.clone())
                .and_modify(|c| *c += 1)
                .or_insert(1);
        }
    }
}
