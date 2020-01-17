#![allow(clippy::map_entry)]

use super::{Pair, WithFirstLastIterator, Word, BPE};
use crate::tokenizer::{Model, Result, Trainer};
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::{HashMap, HashSet};

struct Config {
    min_frequency: u32,
    vocab_size: usize,
    show_progress: bool,
    special_tokens: Vec<String>,
    limit_alphabet: Option<usize>,
    initial_alphabet: HashSet<char>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
}

/// A `BpeTrainerBuilder` can be used to create a `BpeTrainer` with a custom
/// configuration.
pub struct BpeTrainerBuilder {
    config: Config,
}

impl Default for BpeTrainerBuilder {
    fn default() -> Self {
        Self {
            config: Config {
                min_frequency: 0,
                vocab_size: 30000,
                show_progress: true,
                special_tokens: vec![],
                limit_alphabet: None,
                initial_alphabet: HashSet::new(),
                continuing_subword_prefix: None,
                end_of_word_suffix: None,
            },
        }
    }
}

impl BpeTrainerBuilder {
    /// Constructs a new `BpeTrainerBuilder`
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the expected minimum frequency
    pub fn min_frequency(mut self, frequency: u32) -> Self {
        self.config.min_frequency = frequency;
        self
    }

    /// Set the vocabulary size
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.config.vocab_size = size;
        self
    }

    /// Set whether to show progress
    pub fn show_progress(mut self, show: bool) -> Self {
        self.config.show_progress = show;
        self
    }

    /// Set the special tokens
    pub fn special_tokens(mut self, tokens: Vec<String>) -> Self {
        self.config.special_tokens = tokens;
        self
    }

    /// Set whether to limit the alphabet
    pub fn limit_alphabet(mut self, limit: usize) -> Self {
        self.config.limit_alphabet = Some(limit);
        self
    }

    /// Set the initial alphabet
    pub fn initial_alphabet(mut self, alphabet: HashSet<char>) -> Self {
        self.config.initial_alphabet = alphabet;
        self
    }

    /// Set the continuing_subword_prefix
    pub fn continuing_subword_prefix(mut self, prefix: String) -> Self {
        self.config.continuing_subword_prefix = Some(prefix);
        self
    }

    /// Set the end_of_word_suffix
    pub fn end_of_word_suffix(mut self, suffix: String) -> Self {
        self.config.end_of_word_suffix = Some(suffix);
        self
    }

    /// Constructs the final BpeTrainer
    pub fn build(self) -> BpeTrainer {
        BpeTrainer {
            min_frequency: self.config.min_frequency,
            vocab_size: self.config.vocab_size,
            show_progress: self.config.show_progress,
            special_tokens: self.config.special_tokens,
            limit_alphabet: self.config.limit_alphabet,
            initial_alphabet: self.config.initial_alphabet,
            continuing_subword_prefix: self.config.continuing_subword_prefix,
            end_of_word_suffix: self.config.end_of_word_suffix,
        }
    }
}

/// In charge of training a `BPE` model from a mapping of words to word counts.
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
pub struct BpeTrainer {
    /// The minimum frequency a pair must have to produce a merge operation
    min_frequency: u32,
    /// The target vocabulary size
    vocab_size: usize,
    /// Whether to show progress while training
    show_progress: bool,
    /// A list of special tokens that the model should know of
    special_tokens: Vec<String>,
    /// Whether to limit the number of initial tokens that can be kept before computing merges
    limit_alphabet: Option<usize>,
    /// The initial alphabet we want absolutely to include. This allows to cover
    /// some characters that are not necessarily in the training set
    initial_alphabet: HashSet<char>,
    /// An optional prefix to use on any subword that exist only behind another one
    continuing_subword_prefix: Option<String>,
    /// An optional suffix to caracterize and end-of-word subword
    end_of_word_suffix: Option<String>,
}

impl Default for BpeTrainer {
    fn default() -> Self {
        Self::builder().build()
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

    pub fn builder() -> BpeTrainerBuilder {
        BpeTrainerBuilder::new()
    }

    /// Setup a progress bar if asked to show progress
    fn setup_progress(&self) -> Option<ProgressBar> {
        if self.show_progress {
            let p = ProgressBar::new(0);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {msg:<40!} {wide_bar} {pos:<9!}/{len:>9!}"),
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
    fn compute_alphabet(
        &self,
        wc: &HashMap<String, u32>,
        w2id: &mut HashMap<String, u32>,
        id2w: &mut Vec<String>,
    ) {
        // Compute the alphabet from seen words
        let mut alphabet: HashMap<char, usize> = HashMap::new();
        for (word, count) in wc {
            for c in word.chars() {
                alphabet
                    .entry(c)
                    .and_modify(|cnt| *cnt += *count as usize)
                    .or_insert(*count as usize);
            }
        }

        // Also include anything from the provided initial alphabet
        for c in &self.initial_alphabet {
            alphabet
                .entry(*c)
                .and_modify(|cnt| *cnt = std::usize::MAX)
                .or_insert(std::usize::MAX);
        }

        let mut kept = alphabet.iter().collect::<Vec<_>>();

        // Compute the number of chars to remove from the alphabet
        // If `limit_alphabet < initial_alphabet.len()`, some of these initial characters
        // will be removed
        let to_remove = self
            .limit_alphabet
            .map(|limit| {
                if alphabet.len() > limit {
                    alphabet.len() - limit
                } else {
                    0
                }
            })
            .unwrap_or(0);

        // Remove the unwanted chars
        if to_remove > 0 {
            kept.sort_unstable_by_key(|k| *k.1);
            kept.drain(..to_remove);
        }

        // Keep the initial alphabet (sorted for determinism)
        kept.sort_unstable_by_key(|k| (*k.0) as u32);
        kept.into_iter().for_each(|(c, _)| {
            let s = c.to_string();
            if !w2id.contains_key(&s) {
                id2w.push(s.clone());
                w2id.insert(s, (id2w.len() - 1) as u32);
            }
        });
    }

    /// Tokenize words and add subwords to the vocabulary when relevant
    fn tokenize_words(
        &self,
        wc: &HashMap<String, u32>,
        w2id: &mut HashMap<String, u32>,
        id2w: &mut Vec<String>,
        p: &Option<ProgressBar>,
    ) -> (Vec<Word>, Vec<i32>) {
        let mut words: Vec<Word> = vec![];
        let mut counts: Vec<i32> = vec![];

        for (word, count) in wc {
            let mut current_word = Word::new();
            counts.push(*count as i32);

            for (is_first, is_last, c) in word.chars().with_first_and_last() {
                let mut s = c.to_string();
                //if let Some(id) = word_to_id.get(&s) {
                if w2id.contains_key(&s) {
                    // Found the initial char in the authorized alphabet

                    // Add the `continuing_subword_prefix` if relevant
                    if !is_first {
                        if let Some(prefix) = &self.continuing_subword_prefix {
                            s = format!("{}{}", prefix, s);
                        }
                    }
                    // Add the `end_of_word_suffix` if relevant
                    if is_last {
                        if let Some(suffix) = &self.end_of_word_suffix {
                            s = format!("{}{}", s, suffix);
                        }
                    }

                    // Insert the new formed string if necessary
                    if !w2id.contains_key(&s) {
                        id2w.push(s.clone());
                        w2id.insert(s.clone(), (id2w.len() - 1) as u32);
                    }
                    current_word.add(w2id[&s]);
                }
            }
            words.push(current_word);

            if let Some(p) = p {
                p.inc(1);
            }
        }

        (words, counts)
    }

    pub fn train(&self, word_counts: HashMap<String, u32>) -> Result<(BPE, Vec<String>)> {
        let mut word_to_id: HashMap<String, u32> = HashMap::new();
        let mut id_to_word: Vec<String> = vec![];

        let progress = self.setup_progress();

        //
        // 1. Add all special tokens to the vocabulary
        //
        self.add_special_tokens(&mut word_to_id, &mut id_to_word);

        //
        // 2. Compute the initial alphabet
        //
        self.compute_alphabet(&word_counts, &mut word_to_id, &mut id_to_word);

        //
        // 3. Tokenize words
        //
        self.update_progress(&progress, word_counts.len(), "Tokenize words");
        let (mut words, counts) =
            self.tokenize_words(&word_counts, &mut word_to_id, &mut id_to_word, &progress);
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

            // Build new token
            let part_a = &id_to_word[best_pair.0 as usize];
            let mut part_b = id_to_word[best_pair.1 as usize].to_owned();
            if let Some(prefix) = &self.continuing_subword_prefix {
                if part_b.starts_with(prefix) {
                    let prefix_byte_len = prefix.chars().map(|c| c.len_utf8()).sum();
                    part_b = part_b[prefix_byte_len..].to_string();
                }
            }
            let new_token = format!("{}{}", part_a, part_b);

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

        let mut builder = BPE::builder().vocab_and_merges(
            word_to_id,
            merges
                .into_iter()
                .enumerate()
                .map(|(index, (pair, new_id))| (pair, (index as u32, new_id)))
                .collect(),
        );
        if let Some(prefix) = &self.continuing_subword_prefix {
            builder = builder.continuing_subword_prefix(prefix.to_owned());
        }
        if let Some(suffix) = &self.end_of_word_suffix {
            builder = builder.end_of_word_suffix(suffix.to_owned());
        }
        Ok((
            builder
                .build()
                .expect("Trainer should know how to build BPE"),
            self.special_tokens.clone(),
        ))
    }
}

impl Trainer for BpeTrainer {
    /// Train a BPE model
    fn train(
        &self,
        word_counts: HashMap<String, u32>,
    ) -> Result<(Box<dyn Model + Sync>, Vec<String>)> {
        let (bpe, tokens) = self.train(word_counts)?;
        Ok((Box::new(bpe), tokens))
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

    /// Whether we should show progress
    fn should_show_progress(&self) -> bool {
        self.show_progress
    }
}
