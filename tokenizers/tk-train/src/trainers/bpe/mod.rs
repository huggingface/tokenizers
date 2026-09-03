#![allow(clippy::map_entry)]

#[cfg(feature = "parity-aware-bpe")]
pub mod parity_trainer;
mod word;
#[cfg(feature = "parity-aware-bpe")]
pub use parity_trainer::{ParityBpeTrainer, ParityBpeTrainerBuilder, ParityVariant};

use crate::Trainer;
use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use itertools::Itertools;
use dary_heap::OctonaryHeap;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::OnceLock;
use tk_encode::vocab::bucket_added_vocabulary::AddedToken;
// The `Word` machinery a trainer merges into is training-only, so it lives here rather than in the
// inference crate. `PipelineBPE` is the only BPE left; a trainer reaches it through
// `from_vocab_and_merges`, the same serde-free door a reader walks through, because its fields are
// private to `tk-encode`.
use word::{Symbol, WithFirstLastIterator, WordArena, merge_run};

use tk_encode::Result;
use tk_encode::models::bpe::{Merges, Pair, PipelineBPE, BpeConfig, Vocab};
use tk_encode::parallelism::*;
use tk_encode::utils::progress::{ProgressBar, ProgressFormat, ProgressStyle};

/// The default for [`BpeTrainer::parallel_merge_threshold`].
///
/// 1000, chosen from measurements at two scales rather than from a guess. Training a 32 k vocabulary
/// over a 588 MB corpus of code and prose:
///
/// | threshold | 588 MB corpus | 6.5 MB corpus |
/// |-----------|---------------|---------------|
/// | serial    | 29.10 s       | 72 ms         |
/// | 10000     | 21.94 s       | --            |
/// | 4000      | 20.17 s       | 83 ms         |
/// | 1000      | 18.90 s       | 98 ms         |
/// | 0         | 18.92 s       | 280 ms        |
///
/// The two want opposite things -- a big corpus wants to fan out sooner, a small one later -- but
/// the asymmetry settles it: 1000 costs the small corpus 26 ms and saves the large one 10.2 s.
///
/// Fan-out is a proxy for the work a merge represents, and an imperfect one: it counts words, not
/// their lengths, which is why the two corpora disagree at all -- the large one's words are longer,
/// so the same fan-out is more work. Weighting by symbol count would discriminate better; it is not
/// worth the bookkeeping until a corpus is found that this number handles badly.
///
pub fn default_parallel_merge_threshold() -> usize {
    static THRESHOLD: OnceLock<usize> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("TOKENIZERS_TRAIN_PARALLEL_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1_000)
    })
}

/// The two scratch buffers one worker reuses across every merge it handles.
///
/// `scratch` takes a single word's pair deltas from `Word::merge`; `deltas` accumulates the whole
/// chunk's, tagged with the word they came from, to be folded in once the workers are done.
#[derive(Default)]
struct ChunkBuffers {
    scratch: Vec<(Pair, i32)>,
    deltas: Vec<(Pair, i32, u32)>,
}

/// One worker's share of a merge.
///
/// `syms` and `lives` are disjoint slices peeled off the arena, so no two workers can touch the same
/// word. The two bases translate a global word index into this slice: word `iw` has its length at
/// `lives[iw - word_base]` and its run starting at `start[iw] - sym_base` within `syms`.
struct Task<'a> {
    syms: &'a mut [Symbol],
    lives: &'a mut [u32],
    word_base: u32,
    sym_base: usize,
    chunk: &'a [u32],
    buf: &'a mut ChunkBuffers,
}

/// Appends `iw` to a word list unless it is already the last entry.
///
/// Both callers append every index for one word before moving to the next, so an index can only
/// repeat as the immediately preceding entry -- which makes this exactly as strong as the set
/// membership test it replaces, at one comparison instead of a hash and an allocation.
#[inline]
fn push_word(pos: &mut Vec<u32>, iw: u32) {
    if pos.last() != Some(&iw) {
        pos.push(iw);
    }
}

#[derive(Debug, Eq)]
struct Merge {
    pair: Pair,
    count: u64,
    /// Indices of the words containing `pair`, ascending and without duplicates.
    ///
    /// A `Vec<u32>` rather than an `AHashSet<usize>`: the set cost one allocation per pair per
    /// merge, and rayon can only split a hash set by walking its buckets, which is why fanning the
    /// merge loop out measured slower at every threshold. A slice splits in constant time.
    ///
    /// Ascending comes free -- both producers visit words in index order -- and duplicates cannot
    /// appear because every append for one word is contiguous, so comparing against the last entry
    /// is exact. Neither `Ord` nor `PartialEq` for `Merge` looks at this field, so the change cannot
    /// move an id.
    pos: Vec<u32>,
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

struct Config {
    min_frequency: u64,
    vocab_size: usize,
    show_progress: bool,
    progress_format: ProgressFormat,
    special_tokens: Vec<AddedToken>,
    limit_alphabet: Option<usize>,
    initial_alphabet: AHashSet<char>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    max_token_length: Option<usize>,
    parallel_merge_threshold: usize,
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
                progress_format: ProgressFormat::default(),
                special_tokens: vec![],
                limit_alphabet: None,
                initial_alphabet: AHashSet::new(),
                continuing_subword_prefix: None,
                end_of_word_suffix: None,
                max_token_length: None,
                parallel_merge_threshold: default_parallel_merge_threshold(),
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
    #[must_use]
    pub fn min_frequency(mut self, frequency: u64) -> Self {
        self.config.min_frequency = frequency;
        self
    }

    /// Set the vocabulary size
    #[must_use]
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.config.vocab_size = size;
        self
    }

    /// Set whether to show progress
    #[must_use]
    pub fn show_progress(mut self, show: bool) -> Self {
        self.config.show_progress = show;
        self
    }

    /// Set the progress output format
    ///
    /// Controls how progress information is reported during training.
    /// - `Indicatif` (default): Interactive terminal progress bars
    /// - `JsonLines`: Machine-readable JSON lines to stderr
    /// - `Silent`: No progress output
    #[must_use]
    pub fn progress_format(mut self, format: ProgressFormat) -> Self {
        self.config.progress_format = format;
        self
    }

    /// Set the special tokens
    #[must_use]
    pub fn special_tokens(mut self, tokens: Vec<AddedToken>) -> Self {
        self.config.special_tokens = tokens;
        self
    }

    /// Set whether to limit the alphabet
    #[must_use]
    pub fn limit_alphabet(mut self, limit: usize) -> Self {
        self.config.limit_alphabet = Some(limit);
        self
    }

    /// Set the initial alphabet
    #[must_use]
    pub fn initial_alphabet(mut self, alphabet: HashSet<char>) -> Self {
        let mut initial_alphabet = AHashSet::with_capacity(alphabet.len());
        initial_alphabet.extend(alphabet);
        self.config.initial_alphabet = initial_alphabet;
        self
    }

    /// Set the continuing_subword_prefix
    #[must_use]
    pub fn continuing_subword_prefix(mut self, prefix: String) -> Self {
        self.config.continuing_subword_prefix = Some(prefix);
        self
    }

    /// Set the end_of_word_suffix
    #[must_use]
    pub fn end_of_word_suffix(mut self, suffix: String) -> Self {
        self.config.end_of_word_suffix = Some(suffix);
        self
    }
    /// Set max_token_length
    #[must_use]
    pub fn max_token_length(mut self, max_token_length: Option<usize>) -> Self {
        self.config.max_token_length = max_token_length;
        self
    }

    /// Fan-out at or above which one merge is spread across threads.
    ///
    /// See [`default_parallel_merge_threshold`] for why the default is high.
    #[must_use]
    pub fn parallel_merge_threshold(mut self, threshold: usize) -> Self {
        self.config.parallel_merge_threshold = threshold;
        self
    }

    /// Constructs the final BpeTrainer
    pub fn build(self) -> BpeTrainer {
        BpeTrainer {
            min_frequency: self.config.min_frequency,
            vocab_size: self.config.vocab_size,
            show_progress: self.config.show_progress,
            progress_format: self.config.progress_format,
            special_tokens: self.config.special_tokens,
            limit_alphabet: self.config.limit_alphabet,
            initial_alphabet: self.config.initial_alphabet,
            continuing_subword_prefix: self.config.continuing_subword_prefix,
            end_of_word_suffix: self.config.end_of_word_suffix,
            max_token_length: self.config.max_token_length,
            parallel_merge_threshold: self.config.parallel_merge_threshold,
            words: AHashMap::new(),
        }
    }
}

/// In charge of training a `BPE` model
///
/// # Examples
///
/// ```
/// use tk_train::BpeTrainer;
/// use tk_train::Trainer;
/// use tk_encode::models::bpe::{PipelineBPE, BpeConfig};
///
/// let sequences = vec![ "Hello", "World" ];
///
/// let mut trainer = BpeTrainer::default();
/// trainer.feed(sequences.iter(), |s| Ok(vec![s.to_owned()]));
///
/// // `PipelineBPE` has no empty state to train *into* -- it only exists once there is a
/// // vocabulary and a merge list -- so take the parts and build it.
/// let (vocab, merges, special_tokens) = trainer.train_vocab().unwrap();
/// let model = PipelineBPE::from_config(BpeConfig { vocab: vocab, merges: merges, ..BpeConfig::default() }).unwrap();
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Eq)]
pub struct BpeTrainer {
    /// The minimum frequency a pair must have to produce a merge operation
    pub min_frequency: u64,
    /// The target vocabulary size
    pub vocab_size: usize,
    /// Whether to show progress while training
    pub show_progress: bool,
    /// Progress output format (Indicatif, JsonLines, or Silent)
    ///
    /// `ProgressFormat` is a `tk-encode` type and carries no serde of its own; `tk-convert` used to
    /// own its on-disk shape, and that layer is gone. It only decides how progress is *displayed*,
    /// so it is skipped rather than given a shape here, and falls back to its `Default`.
    #[serde(skip)]
    pub progress_format: ProgressFormat,
    /// Fan-out at or above which one merge is spread across threads. See
    /// [`default_parallel_merge_threshold`].
    ///
    /// Skipped by serde for the same reason as `progress_format`: it tunes how the training runs,
    /// not what it produces, and a serialized `0` would silently force the parallel path. The
    /// explicit `default` is what stops that.
    #[serde(skip, default = "default_parallel_merge_threshold")]
    pub parallel_merge_threshold: usize,
    /// A list of special tokens that the model should know of
    #[serde(with = "crate::added_token_serde")]
    pub special_tokens: Vec<AddedToken>,
    /// Whether to limit the number of initial tokens that can be kept before computing merges
    pub limit_alphabet: Option<usize>,
    /// The initial alphabet we want absolutely to include. This allows to cover
    /// some characters that are not necessarily in the training set
    pub initial_alphabet: AHashSet<char>,
    /// An optional prefix to use on any subword that exist only behind another one
    pub continuing_subword_prefix: Option<String>,
    /// An optional suffix to characterize and end-of-word subword
    pub end_of_word_suffix: Option<String>,
    /// An optional parameter to limit the max length of any single token
    pub max_token_length: Option<usize>,

    words: AHashMap<CompactString, u64>,
}

impl Default for BpeTrainer {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl BpeTrainer {
    pub fn new(min_frequency: u64, vocab_size: usize) -> Self {
        Self {
            min_frequency,
            vocab_size,
            ..Default::default()
        }
    }

    pub fn builder() -> BpeTrainerBuilder {
        BpeTrainerBuilder::new()
    }

    /// Returns the number of unique words in the corpus after feeding.
    /// This can be used to estimate training time before starting.
    pub fn get_word_count(&self) -> usize {
        self.words.len()
    }

    /// Setup a progress bar if asked to show progress (only for Indicatif format)
    fn setup_progress(&self) -> Option<ProgressBar> {
        if self.show_progress && self.progress_format == ProgressFormat::Indicatif {
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
    fn emit_json_progress(&self, stage: &str, current: usize, total: usize) {
        if self.progress_format == ProgressFormat::JsonLines {
            eprintln!(
                r#"{{"stage":"{}","current":{},"total":{}}}"#,
                stage, current, total
            );
        }
    }

    /// Set the progress bar in the finish state
    fn finalize_progress(&self, p: &Option<ProgressBar>, final_len: usize, stage: &str) {
        if let Some(p) = p {
            p.set_length(final_len as u64);
            p.finish();
            println!();
        }
        self.emit_json_progress(stage, final_len, final_len);
    }

    /// Update the progress bar with the new provided length and message
    fn update_progress(&self, p: &Option<ProgressBar>, len: usize, message: &'static str) {
        if let Some(p) = p {
            p.set_message(message);
            p.set_length(len as u64);
            p.reset();
        }
        // Emit initial JSON progress for this stage
        self.emit_json_progress(message, 0, len);
    }

    /// Add the provided special tokens to the initial vocabulary
    fn add_special_tokens(
        &self,
        w2id: &mut AHashMap<CompactString, u32>,
        id2w: &mut Vec<CompactString>,
    ) {
        for token in &self.special_tokens {
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
    ) -> (WordArena, Vec<u64>) {
        // One buffer for every word's symbols instead of a `Vec` each. `wc` keys are `CompactString`
        // so their byte length bounds their character count, which bounds the symbols a word can
        // ever hold -- runs only shrink from there, so neither `Vec` grows while filling.
        let symbol_upper_bound = wc.keys().map(|w| w.len()).sum();
        let mut words = WordArena::with_capacity(wc.len(), symbol_upper_bound);
        // Reused whenever a character needs a prefix/suffix; see the note below.
        let mut decorated = String::new();
        let mut counts: Vec<u64> = Vec::with_capacity(wc.len());

        for (word, count) in wc {
            words.open_word();
            counts.push(*count);

            for (is_first, is_last, c) in word.chars().with_first_and_last() {
                // The character as a `&str` in a stack buffer. This used to be `c.to_string()` --
                // one heap allocation per character of every unique word, which was the single
                // largest source of allocation in a training run (~925k of 1.08M) even though it
                // was only 5% of the time. The four `CompactString::from(&s)` rebuilds that
                // followed it are gone the same way: `AHashMap<CompactString, _>` is `Borrow<str>`,
                // so it can be probed with the `&str` directly, once.
                let mut buf = [0u8; 4];
                let plain: &str = c.encode_utf8(&mut buf);

                // Found the initial char in the authorized alphabet
                if !w2id.contains_key(plain) {
                    continue;
                }

                // Add the `continuing_subword_prefix` / `end_of_word_suffix` if relevant
                let prefix = if is_first {
                    None
                } else {
                    self.continuing_subword_prefix.as_deref()
                };
                let suffix = if is_last {
                    self.end_of_word_suffix.as_deref()
                } else {
                    None
                };

                let key: &str = if prefix.is_some() || suffix.is_some() {
                    decorated.clear();
                    if let Some(prefix) = prefix {
                        decorated.push_str(prefix);
                    }
                    decorated.push_str(plain);
                    if let Some(suffix) = suffix {
                        decorated.push_str(suffix);
                    }
                    &decorated
                } else {
                    plain
                };

                // Insert the new formed string if necessary
                let id = match w2id.get(key) {
                    Some(&id) => id,
                    None => {
                        let id = id2w.len() as u32;
                        id2w.push(CompactString::from(key));
                        w2id.insert(CompactString::from(key), id);
                        id
                    }
                };
                words.push_symbol(id, 1); // We do not care about the len here
            }
            words.close_word();

            if let Some(p) = p {
                p.inc(1);
            }
        }

        (words, counts)
    }

    fn count_pairs(
        &self,
        words: &WordArena,
        counts: &[u64],
        p: &Option<ProgressBar>,
    ) -> (AHashMap<Pair, i32>, AHashMap<Pair, Vec<u32>>) {
        // Counted straight into one pair of maps, serially.
        //
        // It used to build a `Pair -> count` map and a `Pair -> {word}` map *per word* and reduce
        // them in parallel, which meant the combine step merged hash maps all the way up the
        // reduction tree -- more work than the counting it was spreading out. Measured on a 6.5 MB
        // corpus, the parallel version cost 48 ms of a 244 ms training run. Accumulating in place
        // also drops one map pair, and one `get_chars` `Vec`, per word.
        let mut pair_counts: AHashMap<Pair, i32> = AHashMap::new();
        let mut where_to_update: AHashMap<Pair, Vec<u32>> = AHashMap::new();

        for i in 0..words.len() {
            let count = counts[i] as i32;
            let iw = i as u32;
            for pair in words.chars(i).tuple_windows::<(u32, u32)>() {
                *pair_counts.entry(pair).or_default() += count;
                // `push_word` keeps `pos` duplicate-free; a pair can repeat inside one word.
                push_word(where_to_update.entry(pair).or_default(), iw);
            }

            if let Some(p) = &p {
                p.inc(1);
            }
        }

        (pair_counts, where_to_update)
    }

    /// Train and hand back the raw parts, for a caller that wants them rather than a built model.
    ///
    /// The WordPiece trainer is the one caller: it trains a BPE and reinterprets the vocabulary as
    /// WordPiece pieces, so building a `PipelineBPE` first -- merge tables and all -- would be work
    /// thrown away.
    pub fn train_vocab(&self) -> Result<(Vocab, Merges, Vec<AddedToken>)> {
        self.do_train(&self.words)
    }

    /// The runtime options a trained model is built with.
    ///
    /// The two affixes are the only settings a BPE trainer decides: everything else in
    /// [`BpeConfig`] describes how to *read* a model (unknown-token handling, dropout,
    /// caching) and is the reader's business, not the trainer's, so it stays at its default.
    fn model_options(&self) -> BpeConfig {
        BpeConfig {
            continuing_subword_prefix: self.continuing_subword_prefix.clone(),
            end_of_word_suffix: self.end_of_word_suffix.clone(),
            ..Default::default()
        }
    }

    /// Runs the training and returns the vocabulary and merge list it produced, plus the special
    /// tokens the caller has to add alongside them.
    ///
    /// It hands back `(Vocab, Merges)` rather than filling in a model because that pair *is* what a
    /// `tokenizer.json` stores, and `PipelineBPE` can only be built from it -- see
    /// [`PipelineBPE::from_vocab_and_merges`]. The WordPiece trainer wants the vocabulary alone, so
    /// splitting the two also saves it building merge tables it would throw away.
    pub fn do_train(
        &self,
        word_counts: &AHashMap<CompactString, u64>,
    ) -> Result<(Vocab, Merges, Vec<AddedToken>)> {
        let mut word_to_id: AHashMap<CompactString, u32> = AHashMap::with_capacity(self.vocab_size);
        let mut id_to_word: Vec<CompactString> = Vec::with_capacity(self.vocab_size);
        let max_token_length: usize = self.max_token_length.unwrap_or(usize::MAX);

        let progress = self.setup_progress();

        //
        // 1. Add all special tokens to the vocabulary
        //
        self.add_special_tokens(&mut word_to_id, &mut id_to_word);

        //
        // 2. Compute the initial alphabet
        //
        self.compute_alphabet(word_counts, &mut word_to_id, &mut id_to_word);

        //
        // 3. Tokenize words
        //
        self.update_progress(&progress, word_counts.len(), "Tokenize words");
        let (mut words, counts) =
            self.tokenize_words(word_counts, &mut word_to_id, &mut id_to_word, &progress);
        self.finalize_progress(&progress, words.len(), "Tokenize words");

        //
        // 4. Count pairs in words
        //
        self.update_progress(&progress, words.len(), "Count pairs");
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
        self.finalize_progress(&progress, words.len(), "Count pairs");

        //
        // 5. Do merges
        //
        self.update_progress(&progress, self.vocab_size, "Compute merges");
        let mut merges: Vec<(Pair, u32)> = vec![];
        // Reused by every `Word::merge` below; see the note in the loop.
        let mut changes: Vec<(Pair, i32)> = Vec::new();
        // One set of buffers per worker, reused for every merge that takes the parallel path.
        let threads = current_num_threads().max(1);
        let mut chunk_buffers: Vec<ChunkBuffers> =
            (0..threads).map(|_| ChunkBuffers::default()).collect();
        loop {
            // Stop as soon as we have a big enough vocabulary
            if word_to_id.len() >= self.vocab_size {
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

            // Merge the new pair into every word that contains it, then fold in the pair deltas.
            //
            // Two paths, chosen by fan-out. `pos` is ascending and duplicate-free, so a contiguous
            // run of it addresses a contiguous run of `words` -- which is what lets the parallel
            // path hand each worker a real `&mut [Word]` from `split_at_mut` instead of the raw
            // pointer this code used to smuggle past rayon's `Sync` bound.
            //
            // The serial path is allocation-free: one reused `changes` buffer for every word, and
            // the deltas applied inline. It used to take a fresh `Vec` from `Word::merge`, collect a
            // second to attach the word index and a third for all of them -- two allocations for
            // each of 800632 word-visits.
            //
            // Applying deltas in any order is safe: `+=` into `pair_counts` and `push_word` into a
            // word list are both commutative, and `push_word`'s dedup only needs each *word*'s
            // appends to be contiguous, which every path preserves.
            if top.pos.len() < self.parallel_merge_threshold {
                for &iw in &top.pos {
                    let i = iw as usize;
                    changes.clear();
                    words.merge(
                        i,
                        top.pair.0,
                        top.pair.1,
                        new_token_id,
                        max_token_length,
                        &mut changes,
                    );

                    let word_count = counts[i] as i32;
                    for &(pair, change) in changes.iter() {
                        *pair_counts.entry(pair).or_default() += change * word_count;
                        if change > 0 {
                            push_word(where_to_update.entry(pair).or_default(), iw);
                        }
                    }
                }
            } else {
                // Cut `pos` into runs, then peel the matching `&mut [Word]` off the front for each.
                // `base` is the word index the peeled slice starts at, so a worker indexes with
                // `iw - base`.
                let run = top.pos.len().div_ceil(threads);

                // Each worker gets a real `&mut [Word]`, peeled off the front, plus the two buffers
                // it reuses. `base` is the word index its slice starts at, so it indexes with
                // `iw - base`.
                //
                // The buffers live outside the merge loop. Allocating them per merge cost 38.4 GB of
                // churn against the serial path's 7.2 GB on a 588 MB corpus -- the per-chunk lists
                // are short-lived but there are two of them per chunk per merge.
                // Every buffer, not just the ones this merge hands out: a merge with fewer chunks
                // than threads would otherwise leave the tail holding the previous merge's deltas,
                // and the fold below walks all of them.
                for buf in chunk_buffers.iter_mut() {
                    buf.deltas.clear();
                }

                // Destructured so the layout stays readable while the storage is carved up: three
                // separate field borrows, not one borrow of the arena.
                let WordArena {
                    symbols,
                    start,
                    live,
                } = &mut words;
                let starts: &[u32] = start;

                let mut tasks: Vec<Task<'_>> = Vec::with_capacity(threads);
                let mut sym_rest: &mut [Symbol] = symbols;
                let mut live_rest: &mut [u32] = live;
                let mut buffers: &mut [ChunkBuffers] = &mut chunk_buffers[..];
                let mut word_base: u32 = 0;
                let mut sym_base: usize = 0;
                for chunk in top.pos.chunks(run) {
                    // +1 because the run has to reach past the last word it names. `start` is
                    // indexed one past that, which is why it carries a sentinel.
                    let upto = chunk[chunk.len() - 1] + 1;
                    let take_words = (upto - word_base) as usize;
                    let take_syms = starts[upto as usize] as usize - sym_base;

                    let (sym_head, sym_tail) = sym_rest.split_at_mut(take_syms);
                    let (live_head, live_tail) = live_rest.split_at_mut(take_words);
                    let (buf_head, buf_tail) = buffers.split_at_mut(1);
                    tasks.push(Task {
                        syms: sym_head,
                        lives: live_head,
                        word_base,
                        sym_base,
                        chunk,
                        buf: &mut buf_head[0],
                    });

                    sym_rest = sym_tail;
                    live_rest = live_tail;
                    buffers = buf_tail;
                    sym_base += take_syms;
                    word_base = upto;
                }

                tasks.into_maybe_par_iter().for_each(|task| {
                    let Task {
                        syms,
                        lives,
                        word_base,
                        sym_base,
                        chunk,
                        buf,
                    } = task;
                    for &iw in chunk {
                        let local = (iw - word_base) as usize;
                        let begin = starts[iw as usize] as usize - sym_base;
                        let live = &mut lives[local];
                        let run = &mut syms[begin..begin + *live as usize];

                        buf.scratch.clear();
                        merge_run(
                            run,
                            live,
                            top.pair.0,
                            top.pair.1,
                            new_token_id,
                            max_token_length,
                            &mut buf.scratch,
                        );
                        for &(pair, change) in buf.scratch.iter() {
                            buf.deltas.push((pair, change, iw));
                        }
                    }
                });

                // Folded back in chunk order, which is `pos` order, so each word's appends stay
                // contiguous and `push_word` stays exact.
                for buf in chunk_buffers.iter() {
                    for &(pair, change, iw) in buf.deltas.iter() {
                        *pair_counts.entry(pair).or_default() += change * counts[iw as usize] as i32;
                        if change > 0 {
                            push_word(where_to_update.entry(pair).or_default(), iw);
                        }
                    }
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
            self.emit_json_progress("Compute merges", merges.len(), self.vocab_size);
        }
        self.finalize_progress(&progress, merges.len(), "Compute merges");

        // The vocabulary, keyed by the token string rather than by `word_to_id`'s hash: we have to
        // look the string up in `id_to_word` either way.
        let vocab: Vocab = word_to_id
            .into_iter()
            .map(|(_key, val)| (id_to_word[val as usize].to_string(), val))
            .collect();

        // `merges` holds id pairs, highest priority first; the on-disk form is the two token
        // strings, which is also what `from_vocab_and_merges` re-derives its ranks from. Order is
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

        Ok((vocab, merges, self.special_tokens.clone()))
    }
}

impl Trainer for BpeTrainer {
    type Model = PipelineBPE;

    /// Train a BPE model
    fn train(&self, model: &mut PipelineBPE) -> Result<Vec<AddedToken>> {
        let (vocab, merges, special_tokens) = self.do_train(&self.words)?;
        *model = PipelineBPE::from_config(BpeConfig { vocab: vocab, merges: merges, ..self.model_options() })?;
        Ok(special_tokens)
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
    use ahash::AHashMap;
    use compact_str::CompactString;

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
        let trainer = BpeTrainer::builder()
            .show_progress(false)
            .min_frequency(2)
            .build();
        let (trained_vocab, merges, _special_tokens) = trainer.do_train(&word_counts).unwrap();

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
        // form, and what `PipelineBPE::from_vocab_and_merges` re-derives its ranks from. Position in
        // the list is the rank, so the order is part of what is being asserted.
        let expected_merges: Merges = vec![
            ("r".into(), "e".into()),  // 'r' + 'e'  -> 're'
            ("a".into(), "re".into()), // 'a' + 're' -> 'are'
            ("i".into(), "s".into()),  // 'i' + 's'  -> 'is'
        ];
        assert_eq!(merges, expected_merges);
    }
    /// The parallel merge path has to produce the same vocabulary and the same merge list, in the
    /// same order, as the serial one. It splits `words` with `split_at_mut` over runs of an
    /// ascending `pos` and folds each run's deltas back in run order, so `push_word`'s
    /// "duplicates can only be adjacent" invariant has to survive the split -- this is what checks
    /// that it does.
    #[test]
    fn parallel_merges_agree_with_serial() {
        let word_counts: AHashMap<CompactString, u64> = [
            ("roses", 3), ("are", 5), ("red", 2), ("violets", 3), ("blue", 4),
            ("bert", 6), ("is", 7), ("big", 2), ("and", 4), ("so", 3),
            ("rosier", 2), ("reddish", 2), ("bluer", 2), ("bigger", 3), ("blurred", 2),
            ("aaaa", 4), ("aaab", 3), ("abab", 3), ("baba", 2), ("bbbb", 2),
        ]
        .iter()
        .map(|(w, c)| (CompactString::from(*w), *c as u64))
        .collect();

        let train_with = |threshold: usize| {
            BpeTrainer::builder()
                .show_progress(false)
                .min_frequency(0)
                .vocab_size(120)
                .parallel_merge_threshold(threshold)
                .build()
                .do_train(&word_counts)
                .unwrap()
        };

        // `0` forces every merge through the parallel path; `usize::MAX` forces every one serial.
        let (par_vocab, par_merges, _) = train_with(0);
        let (ser_vocab, ser_merges, _) = train_with(usize::MAX);

        assert_eq!(par_vocab, ser_vocab, "vocabularies differ");
        assert_eq!(par_merges, ser_merges, "merge lists differ");
        // Guard against the test passing because nothing was learned.
        assert!(!ser_merges.is_empty(), "no merges were produced");
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
            .show_progress(false)
            .min_frequency(0)
            .build();
        let (vocab, _merges, _special_tokens) = trainer.do_train(&long_word_counts).unwrap();
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
            .show_progress(false)
            .min_frequency(0)
            .build();
        let (trained_vocab, _merges, _special_tokens) =
            trainer.do_train(&long_word_counts).unwrap();
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
}
