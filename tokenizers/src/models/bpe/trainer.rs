#![allow(clippy::map_entry)]

use super::{Pair, Word, BPE};
use crate::tokenizer::{Model, Result, Trainer};
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::{HashMap, HashSet};

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
///     (String::from(","), 1),
///     (String::from("ĠWorld"), 1),
///     (String::from("!"), 1),
/// ].iter().cloned().collect();
/// let trainer = BpeTrainer::default();
/// let model = trainer.train(word_counts);
/// ```
pub struct BpeTrainer {
    pub min_frequency: u32,
    pub vocab_size: usize,
    pub show_progress: bool,
    pub special_tokens: Vec<String>,
    pub limit_alphabet: Option<usize>,
}

impl Default for BpeTrainer {
    fn default() -> Self {
        Self {
            min_frequency: 0,
            vocab_size: 30000,
            show_progress: true,
            special_tokens: vec![],
            limit_alphabet: Some(100),
        }
    }
}

impl BpeTrainer {
    pub fn new(min_frequency: u32, vocab_size: usize) -> Self {
        Self {
            min_frequency,
            vocab_size,
            ..Default::default()
        }
    }

    /// Setup a progress bar if asked to show progress
    fn setup_progress(&self) -> Option<ProgressBar> {
        if self.show_progress {
            let p = ProgressBar::new(0);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {msg:30} {bar:40} {pos:>7}/{len:7}"),
            );
            Some(p)
        } else {
            None
        }
    }

    /// Set the progress bar in the finish state
    fn finalize_progress(&self, p: &Option<ProgressBar>, final_len: usize) {
        if let Some(p) = p {
            p.set_length(final_len as u64);
            p.finish();
            println!();
        }
    }

    /// Update the progress bar with the new provided length and message
    fn update_progress(&self, p: &Option<ProgressBar>, len: usize, message: &str) {
        if let Some(p) = p {
            p.set_message(message);
            p.set_length(len as u64);
            p.reset();
        }
    }

    /// Add the provided special tokens to the initial vocabulary
    fn add_special_tokens(&self, w2id: &mut HashMap<String, u32>, id2w: &mut Vec<String>) {
        for token in &self.special_tokens {
            if !w2id.contains_key(token) {
                id2w.push(token.to_owned());
                w2id.insert(token.to_owned(), (id2w.len() - 1) as u32);
            }
        }
    }

    /// Compute the initial alphabet and limit it if relevant
    fn limit_alphabet(
        &self,
        wc: &HashMap<String, u32>,
        w2id: &mut HashMap<String, u32>,
        id2w: &mut Vec<String>,
    ) {
        let mut alphabet: HashMap<char, usize> = HashMap::new();
        for (word, count) in wc {
            for c in word.chars() {
                alphabet
                    .entry(c)
                    .and_modify(|cnt| *cnt += *count as usize)
                    .or_insert(*count as usize);
            }
        }

        let to_remove = if let Some(limit) = self.limit_alphabet {
            alphabet.len() - limit
        } else {
            0
        };

        let mut kept = alphabet.iter().collect::<Vec<_>>();
        kept.sort_unstable_by_key(|k| *k.1);

        // Remove the unwanted chars
        if to_remove > 0 {
            kept.drain(..to_remove);
        }

        // Keep the initial alphabet
        kept.sort_unstable_by_key(|k| (*k.0) as u32);
        kept.into_iter().for_each(|(c, _)| {
            let s = c.to_string();
            if !w2id.contains_key(&s) {
                id2w.push(s.clone());
                w2id.insert(s, (id2w.len() - 1) as u32);
            }
        });
    }
}

impl Trainer for BpeTrainer {
    /// Train a BPE model
    fn train(&self, word_counts: HashMap<String, u32>) -> Result<Box<dyn Model + Sync>> {
        let mut words: Vec<Word> = vec![];
        let mut counts: Vec<i32> = vec![];
        let mut word_to_id: HashMap<String, u32> = HashMap::new();
        let mut id_to_word: Vec<String> = vec![];

        let progress = self.setup_progress();

        //
        // 1. Add all special tokens to the vocabulary
        //
        self.add_special_tokens(&mut word_to_id, &mut id_to_word);

        //
        // 2. Limit the initial alphabet if relevant
        //
        self.limit_alphabet(&word_counts, &mut word_to_id, &mut id_to_word);

        //
        // 3. Tokenize words
        //
        self.update_progress(&progress, word_counts.len(), "Tokenize words");
        for (word, count) in &word_counts {
            let mut current_word = Word::new();
            counts.push(*count as i32);

            for c in word.chars() {
                let s = c.to_string();
                if let Some(id) = word_to_id.get(&s) {
                    current_word.add(*id);
                }
            }
            words.push(current_word);

            if let Some(p) = &progress {
                p.inc(1);
            }
        }
        self.finalize_progress(&progress, words.len());

        //
        // 4. Count pairs in words
        //
        self.update_progress(&progress, words.len(), "Count pairs");
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

            if let Some(p) = &progress {
                p.inc(1);
            }
        }
        self.finalize_progress(&progress, words.len());

        //
        // 5. Do merges
        //
        self.update_progress(&progress, self.vocab_size, "Compute merges");
        let mut merges: Vec<(Pair, u32)> = vec![];
        loop {
            // Stop as soon as we have a big enough vocabulary
            if word_to_id.len() >= self.vocab_size {
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
            if best_count < 1 || self.min_frequency > best_count as u32 {
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

            if let Some(p) = &progress {
                p.inc(1);
            }
        }
        self.finalize_progress(&progress, merges.len());

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
