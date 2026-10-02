#![allow(clippy::map_entry)]

#[cfg(feature = "parity-aware-bpe")]
mod parity_trainer;
mod word;
#[cfg(feature = "parity-aware-bpe")]
pub use parity_trainer::{ParityBpeTrainer, ParityBpeTrainerBuilder, ParityVariant};

use crate::trainer::{ModelTrainer, TrainingParams};
use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use dary_heap::OctonaryHeap;
use serde::Deserialize;
use std::cmp::Ordering;
use tk_encode::vocab::bucket_added_vocabulary::AddedToken;
// The `Word` machinery a trainer merges into is training-only, so it lives here rather than in the
// inference crate. `PipelineBPE` is the only BPE left; a trainer reaches it through
// `from_config`, the same serde-free door a reader walks through, because its fields are
// private to `tk-encode`.
use word::{WithFirstLastIterator, Word};

use crate::error::{Result, TrainingError};
use crate::progress::{ProgressBar, ProgressFormat, ProgressStyle};
use tk_encode::models::bpe::{BpeConfig, Merges, Pair, PipelineBPE, Vocab};
use tk_encode::parallelism::*;
use tk_encode::utils::byte_level::BYTES_CHAR_LOOKUP;

#[derive(Debug, Eq)]
struct Merge {
    pair: Pair,
    count: u64,
    pos: AHashSet<usize>,
}
impl PartialEq for Merge {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}
impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Merge {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.count != other.count {
            self.count.cmp(&other.count)
        } else {
            // Here we want ascending order
            other.pair.cmp(&self.pair)
        }
    }
}

#[derive(Debug, Default)]
pub struct BpeTrainerBuilder {
    trainer: BpeTrainer,
}

impl BpeTrainerBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// The minimum frequency a pair must have to produce a merge.
    #[must_use]
    pub fn min_frequency(mut self, frequency: u64) -> Self {
        self.trainer.min_frequency = frequency;
        self
    }

    /// The maximum number of characters kept in the initial alphabet, before any merge.
    #[must_use]
    pub fn limit_alphabet(mut self, limit: usize) -> Self {
        self.trainer.limit_alphabet = Some(limit);
        self
    }

    /// Characters added to the alphabet even when the training data does not contain them.
    #[must_use]
    pub fn initial_alphabet(mut self, alphabet: impl IntoIterator<Item = char>) -> Self {
        self.trainer.initial_alphabet = alphabet.into_iter().collect();
        self
    }

    /// A prefix on every subword that does not start a word, like WordPiece's `##`.
    #[must_use]
    pub fn continuing_subword_prefix(mut self, prefix: String) -> Self {
        self.trainer.continuing_subword_prefix = Some(prefix);
        self
    }

    /// A suffix on every subword that ends a word, like `</w>`.
    #[must_use]
    pub fn end_of_word_suffix(mut self, suffix: String) -> Self {
        self.trainer.end_of_word_suffix = Some(suffix);
        self
    }

    /// The maximum length of a learned token, in characters.
    #[must_use]
    pub fn max_token_length(mut self, max_token_length: Option<usize>) -> Self {
        self.trainer.max_token_length = max_token_length;
        self
    }

    /// Whether the trained model emits one unk token for a run of unknown characters, rather than
    /// one per character.
    #[must_use]
    pub fn fuse_unk(mut self, fuse_unk: bool) -> Self {
        self.trainer.fuse_unk = fuse_unk;
        self
    }

    /// Whether the trained model emits a word that is a vocab entry as that entry, without
    /// applying the merges.
    #[must_use]
    pub fn ignore_merges(mut self, ignore_merges: bool) -> Self {
        self.trainer.ignore_merges = ignore_merges;
        self
    }

    pub fn build(self) -> BpeTrainer {
        self.trainer
    }
}

/// In charge of training a `BPE` model
///
/// # Examples
///
/// ```
/// use tk_train::{BpeTrainer, ModelTrainer, ProgressFormat, TrainingParams};
///
/// let sequences = vec![ "Hello", "World" ];
///
/// let mut trainer = BpeTrainer::default();
/// trainer.feed(sequences.iter(), |s| Ok(vec![s.to_owned()])).unwrap();
///
/// let params = TrainingParams {
///     vocab_size: 30_000,
///     special_tokens: vec![],
///     unk_token: None,
///     progress: ProgressFormat::Silent,
///     byte_level: false,
/// };
/// let model = trainer.train_model(&params).unwrap();
/// ```
#[derive(Debug, Deserialize)]
pub struct BpeTrainer {
    /// Pairs with lower occurence than this are not considered as merge candidates
    min_frequency: u64,
    max_token_length: Option<usize>,
    /// Limits the size of the initial alphabet, least frequent symbols are dropped
    limit_alphabet: Option<usize>,
    /// Seeds the symbol alphabet
    initial_alphabet: AHashSet<char>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    // Model options in 0.23.2, so its trainer JSON does not have them.
    #[serde(default)]
    fuse_unk: bool,
    #[serde(default)]
    ignore_merges: bool,
    words: AHashMap<CompactString, u64>,
}

impl Default for BpeTrainer {
    fn default() -> Self {
        Self {
            min_frequency: 0,
            limit_alphabet: None,
            initial_alphabet: AHashSet::new(),
            continuing_subword_prefix: None,
            end_of_word_suffix: None,
            max_token_length: None,
            fuse_unk: false,
            ignore_merges: false,
            words: AHashMap::new(),
        }
    }
}

impl BpeTrainer {
    pub fn builder() -> BpeTrainerBuilder {
        BpeTrainerBuilder::new()
    }

    /// A builder with this trainer's settings. The words fed so far are dropped.
    pub fn to_builder(mut self) -> BpeTrainerBuilder {
        self.words = AHashMap::new();
        BpeTrainerBuilder { trainer: self }
    }

    pub fn min_frequency(&self) -> u64 {
        self.min_frequency
    }

    pub fn limit_alphabet(&self) -> Option<usize> {
        self.limit_alphabet
    }

    pub fn initial_alphabet(&self) -> &AHashSet<char> {
        &self.initial_alphabet
    }

    pub fn continuing_subword_prefix(&self) -> Option<&str> {
        self.continuing_subword_prefix.as_deref()
    }

    pub fn end_of_word_suffix(&self) -> Option<&str> {
        self.end_of_word_suffix.as_deref()
    }

    pub fn max_token_length(&self) -> Option<usize> {
        self.max_token_length
    }

    pub fn fuse_unk(&self) -> bool {
        self.fuse_unk
    }

    pub fn ignore_merges(&self) -> bool {
        self.ignore_merges
    }

    /// Setup a progress bar if asked to show progress (only for Indicatif format)
    fn setup_progress(&self, progress: ProgressFormat) -> Option<ProgressBar> {
        if progress == ProgressFormat::Indicatif {
            let p = ProgressBar::new(0);
            p.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {msg:<30!} {wide_bar} {pos:<9!}/{len:>9!}")
                    .expect("Invalid progress template"),
            );
            Some(p)
        } else {
            None
        }
    }

    /// Emit JSON progress line to stderr (for JsonLines format)
    fn emit_json_progress(
        &self,
        progress: ProgressFormat,
        stage: &str,
        current: usize,
        total: usize,
    ) {
        if progress == ProgressFormat::JsonLines {
            eprintln!(
                r#"{{"stage":"{}","current":{},"total":{}}}"#,
                stage, current, total
            );
        }
    }

    /// Set the progress bar in the finish state
    fn finalize_progress(
        &self,
        progress: ProgressFormat,
        p: &Option<ProgressBar>,
        final_len: usize,
        stage: &str,
    ) {
        if let Some(p) = p {
            p.set_length(final_len as u64);
            p.finish();
            println!();
        }
        self.emit_json_progress(progress, stage, final_len, final_len);
    }

    /// Update the progress bar with the new provided length and message
    fn update_progress(
        &self,
        progress: ProgressFormat,
        p: &Option<ProgressBar>,
        len: usize,
        message: &'static str,
    ) {
        if let Some(p) = p {
            p.set_message(message);
            p.set_length(len as u64);
            p.reset();
        }
        // Emit initial JSON progress for this stage
        self.emit_json_progress(progress, message, 0, len);
    }

    /// Add the provided special tokens to the initial vocabulary
    fn add_special_tokens(
        &self,
        special_tokens: &[AddedToken],
        w2id: &mut AHashMap<CompactString, u32>,
        id2w: &mut Vec<CompactString>,
    ) {
        for token in special_tokens {
            // get hash of content
            if !w2id.contains_key(&CompactString::from(&token.content)) {
                id2w.push(CompactString::from(&token.content));
                w2id.insert(CompactString::from(&token.content), (id2w.len() - 1) as u32);
            }
        }
    }

    /// Compute the initial alphabet and limit it if relevant
    fn compute_alphabet(
        &self,
        wc: &AHashMap<CompactString, u64>,
        byte_level: bool,
        w2id: &mut AHashMap<CompactString, u32>,
        id2w: &mut Vec<CompactString>,
    ) {
        // Compute the alphabet from seen words
        let mut alphabet: AHashMap<char, usize> = AHashMap::new();
        for (word, count) in wc {
            for c in word.chars() {
                *alphabet.entry(c).or_default() += *count as usize;
            }
        }

        // Also include anything from the provided initial alphabet
        for c in &self.initial_alphabet {
            *alphabet.entry(*c).or_default() = usize::MAX;
        }
        // `PipelineBPE` refuses a byte-level vocabulary that cannot spell every byte.
        if byte_level {
            for c in BYTES_CHAR_LOOKUP.iter() {
                *alphabet.entry(*c).or_default() = usize::MAX;
            }
        }

        let mut kept = alphabet.iter().collect::<Vec<_>>();

        // Compute the number of chars to remove from the alphabet
        // If `limit_alphabet < initial_alphabet.len()`, some of these initial characters
        // will be removed
        let to_remove = self
            .limit_alphabet
            .map(|limit| alphabet.len().saturating_sub(limit))
            .unwrap_or(0);

        // Remove the unwanted chars
        if to_remove > 0 {
            kept.sort_unstable_by_key(|k| *k.1);
            kept.drain(..to_remove);
        }

        // Keep the initial alphabet (sorted for determinism)
        kept.sort_unstable_by_key(|k| *k.0 as u32);
        kept.into_iter().for_each(|(c, _)| {
            let s = c.to_string();
            /*
            if !w2id.contains_key(&s) {
                id2w.push(s.clone());
                w2id.insert(s, (id2w.len() - 1) as u32);
            }
            */
            // u64 hash version
            if !w2id.contains_key(&CompactString::from(&s)) {
                id2w.push(CompactString::from(&s));
                w2id.insert(CompactString::from(&s), (id2w.len() - 1) as u32);
            }
        });
    }

    /// Tokenize words and add subwords to the vocabulary when relevant
    fn tokenize_words(
        &self,
        wc: &AHashMap<CompactString, u64>,
        w2id: &mut AHashMap<CompactString, u32>,
        id2w: &mut Vec<CompactString>,
        p: &Option<ProgressBar>,
    ) -> (Vec<Word>, Vec<u64>) {
        let mut words: Vec<Word> = Vec::with_capacity(wc.len());
        let mut counts: Vec<u64> = Vec::with_capacity(wc.len());

        for (word, count) in wc {
            let mut current_word = Word::new();
            counts.push(*count);

            for (is_first, is_last, c) in word.chars().with_first_and_last() {
                let mut s = c.to_string();
                if w2id.contains_key(&CompactString::from(&s)) {
                    // Found the initial char in the authorized alphabet

                    // Add the `continuing_subword_prefix` if relevant
                    if !is_first && let Some(prefix) = &self.continuing_subword_prefix {
                        s.insert_str(0, prefix);
                    }
                    // Add the `end_of_word_suffix` if relevant
                    if is_last && let Some(suffix) = &self.end_of_word_suffix {
                        s.push_str(suffix);
                    }

                    // Insert the new formed string if necessary
                    if !w2id.contains_key(&CompactString::from(&s)) {
                        id2w.push(CompactString::from(&s));
                        w2id.insert(CompactString::from(&s), (id2w.len() - 1) as u32);
                    }
                    current_word.add(w2id[&CompactString::from(&s)], 1); // We do not care about the len here
                }
            }
            words.push(current_word);

            if let Some(p) = p {
                p.inc(1);
            }
        }

        (words, counts)
    }

    fn count_pairs(
        &self,
        words: &[Word],
        counts: &[u64],
        p: &Option<ProgressBar>,
    ) -> (AHashMap<Pair, i32>, AHashMap<Pair, AHashSet<usize>>) {
        words
            .maybe_par_iter()
            .enumerate()
            .map(|(i, word)| {
                let mut pair_counts = AHashMap::new();
                let mut where_to_update: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();

                for window in word.get_chars().windows(2) {
                    let cur_pair: Pair = (window[0], window[1]);

                    // Initialize pair_counts and where_to_update for this pair if we just saw it
                    // Then update counts
                    *pair_counts.entry(cur_pair).or_default() += counts[i] as i32;
                    where_to_update.entry(cur_pair).or_default().insert(i);
                }

                if let Some(p) = &p {
                    p.inc(1);
                }

                (pair_counts, where_to_update)
            })
            .reduce(
                || (AHashMap::new(), AHashMap::new()),
                |(mut pair_counts, mut where_to_update), (pc, wtu)| {
                    for (k, v) in pc {
                        *pair_counts.entry(k).or_default() += v;
                    }
                    for (k, v) in wtu {
                        where_to_update.entry(k).or_default().extend(v);
                    }
                    (pair_counts, where_to_update)
                },
            )
    }

    /// Train and hand back the raw parts, for a caller that wants them rather than a built model.
    ///
    /// The WordPiece trainer is the one caller: it trains a BPE and reinterprets the vocabulary as
    /// WordPiece pieces, so building a `PipelineBPE` first -- merge tables and all -- would be work
    /// thrown away.
    pub(crate) fn train_vocab(&self, params: &TrainingParams) -> Result<(Vocab, Merges)> {
        self.do_train(&self.words, params)
    }

    /// The runtime options a trained model is built with.
    ///
    /// Training never reads `fuse_unk` and `ignore_merges`. The trainer carries them because a
    /// [`PipelineBPE`] takes them when it is built and cannot change them afterwards.
    fn model_options(&self, byte_level: bool) -> BpeConfig {
        BpeConfig {
            continuing_subword_prefix: self.continuing_subword_prefix.clone(),
            end_of_word_suffix: self.end_of_word_suffix.clone(),
            fuse_unk: self.fuse_unk,
            ignore_merges: self.ignore_merges,
            byte_level,
            ..Default::default()
        }
    }

    /// Runs the training and returns the vocabulary and merge list it produced.
    ///
    /// It hands back `(Vocab, Merges)` rather than filling in a model because that pair *is* what a
    /// `tokenizer.json` stores, and `PipelineBPE` can only be built from it -- see
    /// [`PipelineBPE::from_config`]. The WordPiece trainer wants the vocabulary alone, so
    /// splitting the two also saves it building merge tables it would throw away.
    fn do_train(
        &self,
        word_counts: &AHashMap<CompactString, u64>,
        params: &TrainingParams,
    ) -> Result<(Vocab, Merges)> {
        // A byte-level model reads text as bytes, so it learns from each word spelled with one
        // visible character per byte, the spelling its vocabulary is written in.
        let byte_level_counts;
        let word_counts = if params.byte_level {
            byte_level_counts = spell_as_bytes(word_counts);
            &byte_level_counts
        } else {
            word_counts
        };
        let mut word_to_id: AHashMap<CompactString, u32> =
            AHashMap::with_capacity(params.vocab_size);
        let mut id_to_word: Vec<CompactString> = Vec::with_capacity(params.vocab_size);
        let max_token_length: usize = self.max_token_length.unwrap_or(usize::MAX);

        let progress = self.setup_progress(params.progress);

        //
        // 1. Add all special tokens to the vocabulary
        //
        self.add_special_tokens(&params.special_tokens, &mut word_to_id, &mut id_to_word);

        //
        // 2. Compute the initial alphabet
        //
        self.compute_alphabet(
            word_counts,
            params.byte_level,
            &mut word_to_id,
            &mut id_to_word,
        );

        //
        // 3. Tokenize words
        //
        self.update_progress(
            params.progress,
            &progress,
            word_counts.len(),
            "Tokenize words",
        );
        let (mut words, counts) =
            self.tokenize_words(word_counts, &mut word_to_id, &mut id_to_word, &progress);
        self.finalize_progress(params.progress, &progress, words.len(), "Tokenize words");

        //
        // 4. Count pairs in words
        //
        self.update_progress(params.progress, &progress, words.len(), "Count pairs");
        let (mut pair_counts, mut where_to_update) = self.count_pairs(&words, &counts, &progress);
        // Insert them in the queue
        let mut queue = OctonaryHeap::with_capacity(pair_counts.len());
        where_to_update.drain().for_each(|(pair, pos)| {
            let count = pair_counts[&pair];
            if count > 0 {
                queue.push(Merge {
                    pair,
                    count: count as u64,
                    pos,
                });
            }
        });
        self.finalize_progress(params.progress, &progress, words.len(), "Count pairs");

        //
        // 5. Do merges
        //
        self.update_progress(
            params.progress,
            &progress,
            params.vocab_size,
            "Compute merges",
        );
        let mut merges: Vec<(Pair, u32)> = vec![];
        loop {
            // Stop as soon as we have a big enough vocabulary
            if word_to_id.len() >= params.vocab_size {
                break;
            }

            let Some(mut top) = queue.pop() else {
                break;
            };

            if top.count != pair_counts[&top.pair] as u64 {
                top.count = pair_counts[&top.pair] as u64;
                queue.push(top);
                continue;
            }

            if top.count < 1 || self.min_frequency > top.count {
                break;
            }

            let part_a = &id_to_word[top.pair.0 as usize];
            let mut part_b = id_to_word[top.pair.1 as usize].as_str();

            // Build new token
            if let Some(prefix) = &self.continuing_subword_prefix
                && let Some(rest) = part_b.strip_prefix(prefix)
            {
                part_b = rest;
            }

            // Insert new token if it does not already exist
            let new_token = format!("{part_a}{part_b}");
            let new_token_id = word_to_id
                .get(&CompactString::from(&new_token))
                .copied()
                .unwrap_or(id_to_word.len() as u32);
            if !word_to_id.contains_key(&CompactString::from(&new_token)) {
                id_to_word.push(CompactString::from(&new_token));
                word_to_id.insert(CompactString::from(&new_token), new_token_id);
            }
            merges.push((top.pair, new_token_id));

            // Merge the new pair in every words
            // Safety: This is just a type assertion, the code below may no longer be safe
            // if the type of `pos` changes
            let pos: &AHashSet<usize> = &top.pos;

            let words_len = words.len();
            // FIXME: doesn't look great
            struct WordPtr(*mut Word);
            // Safety: We do not actually use this for concurrent access to the same memory,
            // only to different chunks within the same allocation.
            unsafe impl Sync for WordPtr {}
            let word_start = WordPtr(words.as_mut_ptr());

            let changes = pos
                .maybe_par_iter()
                .flat_map(|&i| {
                    // We can merge each of these words in parallel here because each position
                    // can be there only once (AHashSet). So this is safe.
                    unsafe {
                        // Edition ≥2021 closures capture the `.0` field (a non-Sync raw
                        // pointer) unless we force whole-struct capture of the Sync wrapper.
                        let word_start = &word_start;
                        assert!(i < words_len);
                        // This is words[i], but avoids needing to go through &T (which triggers UB)
                        let word = word_start.0.add(i);
                        // let word: &mut Word = &mut (*word);
                        (*word)
                            .merge(top.pair.0, top.pair.1, new_token_id, max_token_length)
                            .into_iter()
                            .map(|c| (c, i))
                            .collect::<Vec<_>>()
                    }
                })
                .collect::<Vec<_>>();

            // Introduce new formed pairs
            for ((pair, change), iw) in changes {
                let count = change * counts[iw] as i32;
                *pair_counts.entry(pair).or_default() += count;
                if change > 0 {
                    where_to_update.entry(pair).or_default().insert(iw);
                }
            }
            where_to_update.drain().for_each(|(pair, pos)| {
                let count = pair_counts[&pair];
                if count > 0 {
                    queue.push(Merge {
                        pair,
                        count: count as u64,
                        pos,
                    });
                }
            });

            if let Some(p) = &progress {
                p.inc(1);
            }
            self.emit_json_progress(
                params.progress,
                "Compute merges",
                merges.len(),
                params.vocab_size,
            );
        }
        self.finalize_progress(params.progress, &progress, merges.len(), "Compute merges");

        // The vocabulary, keyed by the token string rather than by `word_to_id`'s hash: we have to
        // look the string up in `id_to_word` either way.
        let vocab: Vocab = word_to_id
            .into_iter()
            .map(|(_key, val)| (id_to_word[val as usize].to_string(), val))
            .collect();

        // `merges` holds id pairs, highest priority first; the on-disk form is the two token
        // strings, which is also what `from_config` re-derives its ranks from. Order is
        // the rank, so it has to be preserved.
        let merges: Merges = merges
            .into_iter()
            .map(|(pair, _new_token_id)| {
                (
                    id_to_word[pair.0 as usize].to_string(),
                    id_to_word[pair.1 as usize].to_string(),
                )
            })
            .collect();

        Ok((vocab, merges))
    }
}

fn spell_as_bytes(word_counts: &AHashMap<CompactString, u64>) -> AHashMap<CompactString, u64> {
    word_counts
        .iter()
        .map(|(word, count)| {
            let word = word
                .bytes()
                .map(|b| BYTES_CHAR_LOOKUP[b as usize])
                .collect();
            (word, *count)
        })
        .collect()
}

impl ModelTrainer for BpeTrainer {
    type Model = PipelineBPE;

    /// Train a BPE model
    fn train_model(&self, params: &TrainingParams) -> Result<PipelineBPE> {
        let (vocab, merges) = self.train_vocab(params)?;
        Ok(PipelineBPE::from_config(BpeConfig {
            vocab,
            merges,
            unk_token: params.unk_token.clone(),
            ..self.model_options(params.byte_level)
        })?)
    }

    fn check(&self, params: &TrainingParams) -> Result<()> {
        // A byte-level vocabulary holds all 256 byte tokens before its first merge.
        let reserved = params.special_tokens.len() + 256;
        if params.byte_level && params.vocab_size <= reserved {
            return Err(TrainingError::VocabTooSmall {
                vocab_size: params.vocab_size,
                reserved,
            });
        }
        if params.byte_level
            && let Some(limit_alphabet) = self.limit_alphabet
            && limit_alphabet < 256
        {
            return Err(TrainingError::AlphabetTooSmall { limit_alphabet });
        }
        Ok(())
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        let words: Result<AHashMap<CompactString, u64>> = iterator
            .maybe_par_bridge()
            .map(|sequence| {
                let words = process(sequence.as_ref())?;
                let mut map = AHashMap::new();
                for word in words {
                    *map.entry(CompactString::from(word)).or_default() += 1;
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
    use super::{BpeTrainer, Merges};
    use crate::trainer::{ModelTrainer, TrainingParams};
    use ahash::AHashMap;
    use compact_str::CompactString;
    use tk_encode::vocab::bucket_added_vocabulary::AddedToken;

    fn params() -> TrainingParams {
        TrainingParams::for_tests(30_000)
    }

    #[test]
    fn test_train() {
        let word_counts: AHashMap<CompactString, u64> = [
            ("roses".into(), 1),
            ("are".into(), 2),
            ("red".into(), 1),
            ("voilets".into(), 1),
            ("blue".into(), 1),
            ("BERT".into(), 1),
            ("is".into(), 2),
            ("big".into(), 1),
            ("and".into(), 1),
            ("so".into(), 1),
            ("GPT-2".into(), 1),
        ]
        .iter()
        .cloned()
        .collect();
        let trainer = BpeTrainer::builder().min_frequency(2).build();
        let (trained_vocab, merges) = trainer.do_train(&word_counts, &params()).unwrap();

        // Vocab should contain all of the characters from the `word_counts` mapping
        // as well as three merges: 're', 'are', and 'is'.
        let expected_vocab: AHashMap<String, u32> = [
            ("-".into(), 0),
            ("2".into(), 1),
            ("B".into(), 2),
            ("E".into(), 3),
            ("G".into(), 4),
            ("P".into(), 5),
            ("R".into(), 6),
            ("T".into(), 7),
            ("a".into(), 8),
            ("b".into(), 9),
            ("d".into(), 10),
            ("e".into(), 11),
            ("g".into(), 12),
            ("i".into(), 13),
            ("l".into(), 14),
            ("n".into(), 15),
            ("o".into(), 16),
            ("r".into(), 17),
            ("s".into(), 18),
            ("t".into(), 19),
            ("u".into(), 20),
            ("v".into(), 21),
            ("re".into(), 22),
            ("are".into(), 23),
            ("is".into(), 24),
        ]
        .iter()
        .cloned()
        .collect();
        assert_eq!(trained_vocab, expected_vocab);

        // `merges` is the pair of symbol *strings* per merge, highest priority first -- the on-disk
        // form, and what `PipelineBPE::from_config` re-derives its ranks from. Position in
        // the list is the rank, so the order is part of what is being asserted.
        let expected_merges: Merges = vec![
            ("r".into(), "e".into()),  // 'r' + 'e'  -> 're'
            ("a".into(), "re".into()), // 'a' + 're' -> 'are'
            ("i".into(), "s".into()),  // 'i' + 's'  -> 'is'
        ];
        assert_eq!(merges, expected_merges);
    }
    #[test]
    fn bpe_test_max_token_length_16() {
        /* bpe_test_max_token_length series of tests test the max_token_length flag of bpetrainer
        // this is the more robust version that only tests max length of learned tokens
        // (pre) tokenizer settings or vocab can be easily modified when necessary
         */

        let max_token_length = 16;
        let long_word_counts: AHashMap<CompactString, u64> = [
            ("singlelongtokenwithoutcasechange", 2),
            ("singleLongTokenWithCamelCaseChange", 2),
            ("Longsingletokenwithpunctu@t!onwithin", 2),
            ("Anotherlongsingletokenwithnumberw1th1n", 2),
            ("짧은한글문자열짧은한", 2),             // korean 10 char
            ("긴한글문자열긴한글문자열긴한글문", 2), // korean 16 char
            ("短字符串短字符串短字", 2),             //simplified chinese 10 char
            ("长字符串长字符串长字符串长字符串", 2), // simp. chinese 16 char
            ("短い文字列短い文字列", 2),             // japanese 10 char
            ("長い文字列長い文字列長い文字列長", 2), // japanese 16 char
            ("so", 2),
            ("GPT-2", 2),
        ]
        .iter()
        .map(|(key, value)| (CompactString::from(key.to_string()), *value))
        .collect();
        let trainer = BpeTrainer::builder()
            .max_token_length(Some(max_token_length))
            .min_frequency(0)
            .build();
        let (vocab, _merges) = trainer.do_train(&long_word_counts, &params()).unwrap();
        for token in vocab.keys() {
            assert!(
                token.chars().count() <= max_token_length,
                "token too long : {} , chars().count() = {}",
                token,
                token.chars().count()
            )
        }
    }
    #[test]
    fn bpe_test_max_token_length_direct_assert() {
        /* more direct version of bpe_test_max_token_length test
        // directly compares tokens with known expected values.
        // maybe unstable depending on specific settings or changes.
         */
        let long_word_counts: AHashMap<CompactString, u64> = [
            ("sin", 2),
            ("Sin", 2),
            ("Lon", 2),
            ("Ano", 2),
            ("짧은한", 2),
            ("긴한글", 2),
            ("短字符", 2),
            ("长字符", 2),
            ("短い文", 2),
            ("長い文", 2),
            ("so", 2),
            ("GP", 2),
        ]
        .iter()
        .map(|(key, value)| (CompactString::from(key.to_string()), *value))
        .collect();
        let trainer = BpeTrainer::builder()
            .max_token_length(Some(2))
            .min_frequency(0)
            .build();
        let (trained_vocab, _merges) = trainer.do_train(&long_word_counts, &params()).unwrap();
        let expected_vocab: AHashMap<String, u32> = [
            ("短", 12),
            ("n", 6),
            ("i", 5),
            ("s", 8),
            ("字符", 23),
            ("長", 14),
            ("긴", 17),
            ("い文", 22),
            ("L", 2),
            ("in", 21),
            ("o", 7),
            ("은한", 29),
            ("S", 4),
            ("P", 3),
            ("so", 27),
            ("符", 13),
            ("文", 11),
            ("字", 10),
            ("짧", 19),
            ("GP", 25),
            ("글", 16),
            ("G", 1),
            ("An", 24),
            ("长", 15),
            ("A", 0),
            ("Lo", 26),
            ("긴한", 28),
            ("い", 9),
            ("한", 20),
            ("은", 18),
        ]
        .iter()
        .cloned()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        assert_eq!(trained_vocab, expected_vocab)
    }

    #[test]
    fn trained_model_keeps_the_unk_token() {
        let mut trainer = BpeTrainer::builder().build();
        trainer
            .feed(["hello world"].iter(), |s| {
                Ok(s.split(' ').map(str::to_owned).collect())
            })
            .unwrap();
        let params = TrainingParams {
            special_tokens: vec![AddedToken::from("<unk>", true)],
            unk_token: Some("<unk>".into()),
            ..params()
        };

        let model = trainer.train_model(&params).unwrap();

        assert_eq!(
            model.to_config().unwrap().unk_token.as_deref(),
            Some("<unk>")
        );
    }

    #[test]
    fn trained_model_keeps_fuse_unk_and_ignore_merges() {
        let mut trainer = BpeTrainer::builder()
            .fuse_unk(true)
            .ignore_merges(true)
            .build();
        trainer
            .feed(["hello world"].iter(), |s| {
                Ok(s.split(' ').map(str::to_owned).collect())
            })
            .unwrap();

        let config = trainer.train_model(&params()).unwrap().to_config().unwrap();

        assert!(config.fuse_unk);
        assert!(config.ignore_merges);
    }

    #[test]
    fn to_builder_keeps_settings_and_drops_words() {
        let mut trainer = BpeTrainer::builder()
            .continuing_subword_prefix("##".into())
            .build();
        trainer
            .feed(["hello world"].iter(), |s| {
                Ok(s.split(' ').map(str::to_owned).collect())
            })
            .unwrap();

        let trainer = trainer.to_builder().min_frequency(4).build();

        assert_eq!(trainer.continuing_subword_prefix(), Some("##"));
        assert_eq!(trainer.min_frequency(), 4);
        assert!(trainer.words.is_empty());
    }
}
