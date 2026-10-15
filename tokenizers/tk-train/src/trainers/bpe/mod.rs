#![allow(clippy::map_entry)]

mod feed;
mod word_counts;
use word_counts::{WordCounts, WordCountsView};

mod engine;
#[cfg(feature = "parity-aware-bpe")]
pub mod parity_trainer;
#[cfg(test)]
mod reference;
#[cfg(any(test, feature = "parity-aware-bpe"))]
mod word;
#[cfg(feature = "parity-aware-bpe")]
pub use parity_trainer::{ParityBpeTrainer, ParityBpeTrainerBuilder, ParityVariant};

use crate::Trainer;
use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use tk_encode::vocab::bucket_added_vocabulary::AddedToken;
// The reference and optional parity trainer use `Word`; production training uses
// the engine's fixed-coordinate corpus. Both representations are training-only.
#[cfg(any(test, feature = "parity-aware-bpe"))]
use word::{WithFirstLastIterator, Word};

use tk_encode::Result;
#[cfg(any(test, feature = "parity-aware-bpe"))]
use tk_encode::models::bpe::Pair;
use tk_encode::models::bpe::{BpeConfig, Merges, PipelineBPE, Vocab};
use tk_encode::parallelism::*;
use tk_encode::utils::progress::ProgressFormat;

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

    /// Set the initial alphabet. See [`BpeTrainer::initial_alphabet`] for truncation.
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
    /// Set the exclusive span limit for newly created adjacent pairs.
    ///
    /// See [`BpeTrainer::max_token_length`] for units and the initial-pair exception.
    #[must_use]
    pub fn max_token_length(mut self, max_token_length: Option<usize>) -> Self {
        self.config.max_token_length = max_token_length;
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
            words: WordCounts::default(),
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
/// let model = PipelineBPE::from_config(BpeConfig { vocab, merges, ..BpeConfig::default() }).unwrap();
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
    /// Progress display is not serialized; deserialization uses its default.
    /// It does not change training results.
    #[serde(skip)]
    pub progress_format: ProgressFormat,
    /// A list of special tokens that the model should know of
    #[serde(with = "crate::added_token_serde")]
    pub special_tokens: Vec<AddedToken>,
    /// Whether to limit the number of initial tokens that can be kept before computing merges
    pub limit_alphabet: Option<usize>,
    /// Characters prioritized during alphabet selection, including characters
    /// absent from the training input. If `limit_alphabet` is smaller than this
    /// set, some of these characters can still be removed.
    pub initial_alphabet: AHashSet<char>,
    /// An optional prefix to use on any subword that exist only behind another one
    pub continuing_subword_prefix: Option<String>,
    /// An optional suffix to characterize and end-of-word subword
    pub end_of_word_suffix: Option<String>,
    /// An exclusive span limit for newly created adjacent pairs, measured in
    /// retained input characters. Affix text and UTF-8 byte widths do not count.
    /// `None` disables this limit.
    ///
    /// Initial pairs bypass the limit, and selected pairs have no additional
    /// length check. For example, `Some(3)` rejects a newborn pair spanning three
    /// characters, while `Some(1)` still permits an initial two-character merge.
    pub max_token_length: Option<usize>,

    words: WordCounts,
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

    /// Select the alphabet with the existing frequency-tie and codepoint order.
    fn select_alphabet(&self, wc: WordCountsView<'_>) -> Vec<char> {
        // Compute the alphabet from seen words
        let mut alphabet: AHashMap<char, usize> = AHashMap::new();
        for (word, count) in wc.iter() {
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
        kept.into_iter().map(|(&character, _)| character).collect()
    }

    #[cfg(test)]
    fn compute_alphabet(
        &self,
        wc: &AHashMap<CompactString, u64>,
        w2id: &mut AHashMap<CompactString, u32>,
        id2w: &mut Vec<CompactString>,
    ) {
        for character in self.select_alphabet(WordCountsView::from_map(wc)) {
            let mut utf8 = [0; 4];
            let text: &str = character.encode_utf8(&mut utf8);
            let token = CompactString::from(text);
            if !w2id.contains_key(&token) {
                id2w.push(token.clone());
                w2id.insert(token, (id2w.len() - 1) as u32);
            }
        }
    }

    /// Train the collected weighted words and return vocabulary entries, ordered
    /// merges, and special tokens.
    ///
    /// Stored counts remain available for subsequent calls. Execution policy and
    /// numeric limits are the same as for [`Self::do_train`]. The WordPiece trainer
    /// uses these parts to reinterpret the vocabulary without building BPE merge tables.
    ///
    /// # Errors
    ///
    /// Returns the training errors described in [`Self::do_train`], including the
    /// signed input limits for nonempty affixes even when no merge is needed.
    pub fn train_vocab(&self) -> Result<(Vocab, Merges, Vec<AddedToken>)> {
        self.train_counts(self.words.view())
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

    /// Train weighted words and return vocabulary entries, ordered merges, and
    /// special tokens for registration by the caller.
    ///
    /// These parts populate [`BpeConfig`] for [`PipelineBPE::from_config`]. The
    /// WordPiece trainer consumes the vocabulary without building BPE merge tables.
    /// The input map is borrowed and remains unchanged.
    ///
    /// Training uses a dedicated Rayon pool sized by
    /// [`tk_encode::parallelism::num_threads`], or one worker when parallelism is
    /// disabled. [`Trainer::feed`] uses the ambient pool, including one installed
    /// by the caller.
    ///
    /// # Errors
    ///
    /// Returns an error on overflow or underflow in checked `u64` pair, birth,
    /// or removal arithmetic, or in checked `i64` active-reuse ledger updates.
    /// Nonempty affixes and active reuse also require the maximum word weight and
    /// initial weighted edge mass (the sum of each word's weight times its retained
    /// adjacent-pair count) to fit in `i64::MAX`. The affix check applies even when
    /// the initial vocabulary already meets the target size. Plain first-activation
    /// input has no total-`u64` mass limit when every individual pair count fits.
    ///
    /// Pool creation, progress setup, vocabulary or corpus size bounds, and fallible position
    /// storage operations can also return errors. Feed counting and limited-alphabet
    /// frequency accumulation use ordinary addition rather than these checked rules.
    pub fn do_train(
        &self,
        word_counts: &AHashMap<CompactString, u64>,
    ) -> Result<(Vocab, Merges, Vec<AddedToken>)> {
        self.train_counts(WordCountsView::from_map(word_counts))
    }

    fn train_counts(
        &self,
        word_counts: WordCountsView<'_>,
    ) -> Result<(Vocab, Merges, Vec<AddedToken>)> {
        let workers = if get_parallelism() {
            num_threads().max(1)
        } else {
            1
        };
        engine::train(
            self,
            word_counts,
            workers,
            #[cfg(test)]
            None,
        )
    }
}

impl Trainer for BpeTrainer {
    type Model = PipelineBPE;

    /// Train the collected words and replace the model using the trainer's affixes.
    /// Return special tokens for registration by the caller.
    fn train(&self, model: &mut PipelineBPE) -> Result<Vec<AddedToken>> {
        let (vocab, merges, special_tokens) = self.train_counts(self.words.view())?;
        *model = PipelineBPE::from_config(BpeConfig {
            vocab,
            merges,
            ..self.model_options()
        })?;
        Ok(special_tokens)
    }

    /// Whether we should show progress
    fn should_show_progress(&self) -> bool {
        self.show_progress
    }

    /// Apply `process` to each input and collect the resulting weighted words.
    /// Successful collection replaces the words used by `train` and `train_vocab`.
    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        self.words = feed::count(iterator, &process)?;
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
        // form, and what `PipelineBPE::from_config` derives its ranks from. Position in
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
