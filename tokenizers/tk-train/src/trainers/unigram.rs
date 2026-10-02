use crate::error::{Result, TrainingError};
use crate::progress::{ProgressBar, ProgressFormat, ProgressStyle};
use crate::trainer::{ModelTrainer, TrainingParams};
use ahash::{AHashMap, AHashSet};
use log::debug;
use serde::Deserialize;
use std::cmp::Reverse;
use tk_encode::models::unigram::{lattice::Lattice, model::Unigram};
use tk_encode::utils::parallelism::*;

// A token and a score
type SentencePiece = (String, f64);

// A full sentence or word + it's count within the dataset
type Sentence = (String, u64);

fn digamma(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 7.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    x -= 1.0 / 2.0;
    let xx = 1.0 / x;
    let xx2 = xx * xx;
    let xx4 = xx2 * xx2;
    result += x.ln() + (1.0 / 24.0) * xx2 - 7.0 / 960.0 * xx4 + (31.0 / 8064.0) * xx4 * xx2
        - (127.0 / 30720.0) * xx4 * xx4;
    result
}

#[derive(thiserror::Error, Debug)]
pub(crate) enum UnigramTrainerError {
    #[error("The vocabulary is not large enough to contain all chars")]
    VocabularyTooSmall,
}

fn to_log_prob(pieces: &mut [SentencePiece]) {
    let sum: f64 = pieces.iter().map(|(_, score)| score).sum();
    let logsum = sum.ln();
    for (_, score) in pieces.iter_mut() {
        *score = score.ln() - logsum;
    }
}

/// Trains a [`Unigram`] model.
#[derive(Debug, Deserialize)]
pub struct UnigramTrainer {
    n_sub_iterations: u32,
    shrinking_factor: f64,
    initial_alphabet: AHashSet<char>,
    max_piece_length: usize,
    seed_size: usize,
    words: AHashMap<String, u64>,
}

impl Default for UnigramTrainer {
    fn default() -> Self {
        Self {
            n_sub_iterations: 2,
            shrinking_factor: 0.75,
            initial_alphabet: AHashSet::new(),
            max_piece_length: 16,
            seed_size: 1_000_000,
            words: AHashMap::new(),
        }
    }
}

#[derive(Debug, Default)]
pub struct UnigramTrainerBuilder {
    trainer: UnigramTrainer,
}

impl UnigramTrainerBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of EM iterations run between two pruning steps.
    #[must_use]
    pub fn n_sub_iterations(mut self, n_sub_iterations: u32) -> Self {
        self.trainer.n_sub_iterations = n_sub_iterations;
        self
    }

    /// The share of pieces each pruning step keeps.
    #[must_use]
    pub fn shrinking_factor(mut self, shrinking_factor: f64) -> Self {
        self.trainer.shrinking_factor = shrinking_factor;
        self
    }

    /// Characters added to the alphabet even when the training data does not contain them.
    #[must_use]
    pub fn initial_alphabet(mut self, alphabet: impl IntoIterator<Item = char>) -> Self {
        self.trainer.initial_alphabet = alphabet.into_iter().collect();
        self
    }

    /// The maximum length of a piece, in characters.
    #[must_use]
    pub fn max_piece_length(mut self, max_piece_length: usize) -> Self {
        self.trainer.max_piece_length = max_piece_length;
        self
    }

    /// The number of substrings kept as candidate pieces before EM starts.
    #[must_use]
    pub fn seed_size(mut self, seed_size: usize) -> Self {
        self.trainer.seed_size = seed_size;
        self
    }

    pub fn build(self) -> UnigramTrainer {
        self.trainer
    }
}

impl UnigramTrainer {
    pub fn builder() -> UnigramTrainerBuilder {
        UnigramTrainerBuilder::default()
    }

    /// A builder with this trainer's settings. The words fed so far are dropped.
    pub fn to_builder(mut self) -> UnigramTrainerBuilder {
        self.words = AHashMap::new();
        UnigramTrainerBuilder { trainer: self }
    }

    pub fn n_sub_iterations(&self) -> u32 {
        self.n_sub_iterations
    }

    pub fn shrinking_factor(&self) -> f64 {
        self.shrinking_factor
    }

    pub fn initial_alphabet(&self) -> &AHashSet<char> {
        &self.initial_alphabet
    }

    pub fn max_piece_length(&self) -> usize {
        self.max_piece_length
    }

    pub fn seed_size(&self) -> usize {
        self.seed_size
    }

    /// Setup a progress bar if asked to show progress
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

    fn is_valid_sentencepiece(&self, char_string: &[char]) -> bool {
        // Checks string length
        // Space not in the substring, numbers, hiragana and more should be taken
        // care of within pre_tokenizers.
        // https://github.com/google/sentencepiece/blob/26be9516cd81d5315ee31c48d2438018e0eab879/src/trainer_interface.cc#L203
        let n = char_string.len();
        if char_string.is_empty() || n > self.max_piece_length {
            return false;
        }

        true
    }

    fn finalize(
        &self,
        model: Unigram,
        required_chars: AHashSet<String>,
        params: &TrainingParams,
    ) -> Result<Unigram> {
        let mut min_score_penalty = 0.0;
        let min_score_penalty_delta = 0.0001;

        let mut pieces: Vec<(String, f64)> = vec![];
        let mut inserted: AHashSet<String> = AHashSet::new();

        // We don't want to include the <UNK> that was used to train
        inserted.insert("<UNK>".into());

        let existing_pieces: AHashMap<String, f64> = model.iter().cloned().collect();
        for c in required_chars {
            if let Some(t) = existing_pieces.get(&c) {
                inserted.insert(c.clone());
                pieces.push((c, *t));
            } else {
                let score = model.min_score + min_score_penalty;

                inserted.insert(c.clone());
                pieces.push((c, score));
                min_score_penalty += min_score_penalty_delta;
            }
        }

        let (unk_id, need_add_unk) = if let Some(ref unk) = params.unk_token {
            let unk_id = params
                .special_tokens
                .iter()
                .enumerate()
                .find_map(|(i, t)| if t.content == *unk { Some(i) } else { None });
            match unk_id {
                Some(id) => (Some(id), false),
                None => (Some(0), true),
            }
        } else {
            (None, false)
        };

        let vocab_size_without_special_tokens = if need_add_unk {
            params.vocab_size - params.special_tokens.len() - 1
        } else {
            params.vocab_size - params.special_tokens.len()
        };
        for (token, score) in model.iter() {
            if inserted.contains::<str>(token) {
                continue;
            }
            inserted.insert(token.to_string());
            pieces.push((token.to_string(), if score.is_nan() { 0.0 } else { *score }));

            if pieces.len() == vocab_size_without_special_tokens {
                break;
            }
        }
        pieces.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        // Insert the necessary tokens
        let mut special_tokens = params
            .special_tokens
            .iter()
            .map(|t| (t.content.clone(), 0.0))
            .collect::<Vec<_>>();
        if need_add_unk {
            special_tokens.insert(0, (params.unk_token.clone().unwrap(), 0.0));
        }

        Ok(Unigram::from(
            special_tokens.into_iter().chain(pieces).collect(),
            unk_id,
            model.byte_fallback(),
        )?)
    }

    fn required_chars(&self, word_counts: &[Sentence]) -> AHashSet<String> {
        word_counts
            .iter()
            .flat_map(|(s, _count)| s.chars())
            .chain(self.initial_alphabet.iter().copied())
            .map(|c| c.to_string())
            .collect()
    }
    fn make_seed_sentence_pieces(
        &self,
        sentences: &[Sentence],
        _progress: &Option<ProgressBar>,
    ) -> Vec<SentencePiece> {
        // Put all sentences in a string, separated by \0
        let total: usize = sentences
            .iter()
            .map(|(s, _)| s.chars().count())
            .sum::<usize>()
            + sentences.len();
        let mut flat_string = String::with_capacity(total);
        let mut all_chars: AHashMap<char, u64> = AHashMap::new();
        let c_sentence_boundary = '\0';
        let k_sentence_boundary = '\0'.to_string();
        for (string, n) in sentences {
            if string.is_empty() {
                continue;
            }
            flat_string.push_str(string);
            // XXX
            // Comment suggests we add sentence boundary, but it seems to be missing from actual
            // code in spm.
            flat_string.push_str(&k_sentence_boundary);
            for c in string.chars() {
                if c != c_sentence_boundary {
                    *all_chars.entry(c).or_insert(0) += n;
                }
            }
        }
        flat_string.shrink_to_fit();
        #[cfg(feature = "esaxx_fast")]
        let suffix = esaxx_rs::suffix(&flat_string).unwrap();
        #[cfg(not(feature = "esaxx_fast"))]
        let suffix = esaxx_rs::suffix_rs(&flat_string).unwrap();

        //  Basic chars need to be in sentence pieces.
        let mut seed_sentencepieces: Vec<SentencePiece> = vec![];

        let mut sall_chars: Vec<_> = all_chars.into_iter().map(|(a, b)| (b, a)).collect();
        // Reversed order
        sall_chars.sort_by_key(|&a| Reverse(a));
        let mut substr_index: Vec<_> = suffix
            .iter()
            .filter_map(|(string, freq)| {
                if string.len() <= 1 {
                    return None;
                }
                if string.contains(&c_sentence_boundary) {
                    return None;
                }
                if !self.is_valid_sentencepiece(string) {
                    return None;
                }
                let score = freq * string.len() as u32;
                // if let Some(p) = &progress {
                //     p.inc(1);
                // }
                Some((score, string))
            })
            .collect();

        // Fill seed_sentencepieces
        for (count, character) in sall_chars {
            seed_sentencepieces.push((character.to_string(), count as f64));
        }

        // sort by decreasing score
        substr_index.sort_by_key(|&a| Reverse(a));
        for (score, char_string) in substr_index {
            // Just in case
            assert!(self.is_valid_sentencepiece(char_string));
            let string: String = char_string.iter().collect();
            seed_sentencepieces.push((string, score.into()));
            if seed_sentencepieces.len() >= self.seed_size {
                break;
            }
        }
        to_log_prob(&mut seed_sentencepieces);
        seed_sentencepieces
    }
    fn prune_sentence_pieces(
        &self,
        model: &Unigram,
        pieces: &[SentencePiece],
        sentences: &[Sentence],
        params: &TrainingParams,
    ) -> Vec<SentencePiece> {
        let mut always_keep = vec![true; pieces.len()];
        let mut alternatives: Vec<Vec<usize>> = vec![Vec::new(); pieces.len()];

        let bos_id = pieces.len() + 1;
        let eos_id = pieces.len() + 2;

        // First, segments the current sentencepieces to know
        // how each sentencepiece is resegmented if this sentencepiece is removed
        // from the vocabulary.
        // To do so, we take the second best segmentation of sentencepiece[i].
        // alternatives[i] stores the sequence of second best sentencepieces.
        for (id, (token, _score)) in pieces.iter().enumerate() {
            // Always keep unk.
            if id == 0 {
                always_keep[id] = false;
                continue;
            }
            let mut lattice = Lattice::from(token, bos_id, eos_id);
            model.populate_nodes(&mut lattice);

            let nbests = lattice.nbest(2);
            if nbests.len() == 1 {
                always_keep[id] = true;
            } else if nbests[0].len() >= 2 {
                always_keep[id] = false;
            } else if nbests[0].len() == 1 {
                always_keep[id] = true;
                for node in &nbests[1] {
                    let alt_id = node.borrow().id;
                    alternatives[id].push(alt_id);
                }
            }
        }

        // Second, segments all sentences to compute likelihood
        // with a unigram language model. inverted[i] stores
        // the set of sentence index where the sentencepieces[i] appears.
        let chunk_size = std::cmp::max(sentences.len() / current_num_threads(), 1);
        let indexed_sentences: Vec<(usize, &Sentence)> = sentences.iter().enumerate().collect();
        let collected: (f64, Vec<f64>, Vec<Vec<usize>>) = indexed_sentences
            .maybe_par_chunks(chunk_size)
            .map(|enumerated_sentence_count_chunk| {
                let mut vsum = 0.0;
                let mut freq: Vec<f64> = vec![0.0; pieces.len()];
                let mut inverted: Vec<Vec<usize>> = vec![Vec::new(); pieces.len()];

                for (i, (sentence, count)) in enumerated_sentence_count_chunk {
                    let mut lattice = Lattice::from(sentence, bos_id, eos_id);
                    model.populate_nodes(&mut lattice);
                    vsum += *count as f64;
                    for node_ref in lattice.viterbi() {
                        let id = node_ref.borrow().id;
                        freq[id] += *count as f64;
                        inverted[id].push(*i);
                    }
                }
                (vsum, freq, inverted)
            })
            .reduce(
                || (0.0, vec![0.0; pieces.len()], vec![Vec::new(); pieces.len()]),
                |(vsum, freq, inverted), (lvsum, lfreq, linverted)| {
                    (
                        vsum + lvsum,
                        freq.iter()
                            .zip(lfreq)
                            .map(|(global_el, local_el)| global_el + local_el)
                            .collect(),
                        inverted
                            .iter()
                            .zip(linverted)
                            .map(|(global_el, local_el)| [&global_el[..], &local_el[..]].concat())
                            .collect(),
                    )
                },
            );

        let (vsum, freq, inverted) = collected;

        let sum: f64 = freq.iter().sum();
        let logsum = sum.ln();
        let mut candidates: Vec<(usize, f64)> = vec![];
        let mut new_pieces: Vec<SentencePiece> = Vec::with_capacity(params.vocab_size);
        new_pieces.push(pieces[0].clone());

        // Finally, computes how likely the LM likelihood is reduced if
        // the sentencepiece[i] is removed from the vocabulary.
        // Since the exact computation of loss is difficult, we compute the
        // loss approximately by assuming that all sentencepiece[i] in the sentences
        // are replaced with alternatives[i] when sentencepiece[i] is removed.
        for (id, (token, score)) in pieces.iter().enumerate() {
            if id == 0 {
                continue;
            }
            if freq[id] == 0.0 && !always_keep[id] {
                // not found in Viterbi path. Can remove this entry safely.
                continue;
            } else if alternatives[id].is_empty() {
                // no alternatives. Keeps this entry.
                new_pieces.push((token.to_string(), *score));
            } else {
                let mut f = 0.0; // the frequency of pieces[i];

                for n in &inverted[id] {
                    let score = sentences[*n].1 as f64;
                    f += score;
                }
                // TODO: Temporary hack to avoid Nans.
                if f == 0.0 || f.is_nan() {
                    // new_pieces.push((token.to_string(), *score));
                    continue;
                }
                f /= vsum; // normalizes by all sentence frequency.
                let logprob_sp = freq[id].ln() - logsum;

                // After removing the sentencepiece[i], its frequency freq[i] is
                // re-assigned to alternatives.
                // new_sum = current_sum - freq[i] + freq[i] * alternatives.size()
                //         = current_sum + freq[i] (alternatives - 1)

                let logsum_alt = (sum + freq[id] * (alternatives.len() - 1) as f64).ln();

                // The frequencies of alternatives are increased by freq[i].
                let mut logprob_alt = 0.0;
                for n in &alternatives[id] {
                    logprob_alt += (freq[*n] + freq[id]).ln() - logsum_alt;
                }

                // loss: the diff of likelihood after removing the sentencepieces[i].
                let loss = f * (logprob_sp - logprob_alt);
                if loss.is_nan() {
                    panic!("");
                }

                candidates.push((id, loss));
            }
        }
        let desired_vocab_size: usize = (params.vocab_size * 11) / 10; // * 1.1
        let pruned_size: usize = ((pieces.len() as f64) * self.shrinking_factor) as usize;
        let pruned_size = desired_vocab_size.max(pruned_size);

        candidates.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        for (id, _score) in candidates {
            if new_pieces.len() == pruned_size {
                break;
            }
            new_pieces.push(pieces[id].clone());
        }

        new_pieces
    }

    /// Update the progress bar with the new provided length and message
    fn update_progress(&self, p: &Option<ProgressBar>, len: usize, message: &'static str) {
        if let Some(p) = p {
            p.set_message(message);
            p.set_length(len as u64);
            p.reset();
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

    fn run_e_step(&self, model: &Unigram, sentences: &[Sentence]) -> (f64, u32, Vec<f64>) {
        let all_sentence_freq: u64 = sentences.iter().map(|(_a, b)| *b).sum();

        let chunk_size = std::cmp::max(sentences.len() / current_num_threads(), 1);
        let collected: (f64, u32, Vec<f64>) = sentences
            .maybe_par_chunks(chunk_size)
            .map(|sentences_chunk| {
                let mut expected: Vec<f64> = vec![0.0; model.len()];
                let mut objs: f64 = 0.0;
                let mut ntokens: u32 = 0;

                for (string, freq) in sentences_chunk {
                    let mut lattice = Lattice::from(string, model.bos_id, model.eos_id);
                    model.populate_nodes(&mut lattice);

                    let z: f64 = lattice.populate_marginal(*freq as f64, &mut expected);
                    if z.is_nan() {
                        panic!("likelihood is NAN. Input sentence may be too long.");
                    }
                    ntokens += lattice.viterbi().len() as u32;
                    objs -= z / (all_sentence_freq as f64);
                }
                (objs, ntokens, expected)
            })
            .reduce(
                || (0.0, 0, vec![0.0; model.len()]),
                |(objs, ntokens, expected), (lobjs, lntokens, lexpected)| {
                    (
                        objs + lobjs,
                        ntokens + lntokens,
                        expected
                            .iter()
                            .zip(lexpected)
                            .map(|(global_el, local_el)| global_el + local_el)
                            .collect(),
                    )
                },
            );

        collected
    }
    fn run_m_step(
        &self,
        pieces: &[SentencePiece],
        expected: &[f64],
        params: &TrainingParams,
    ) -> Vec<SentencePiece> {
        if pieces.len() != expected.len() {
            panic!(
                "Those two iterators are supposed to be the same length ({} vs {})",
                pieces.len(),
                expected.len()
            );
        }
        let mut new_pieces: Vec<SentencePiece> = Vec::with_capacity(params.vocab_size);

        let mut sum = 0.0;
        let expected_frequency_threshold = 0.5;

        for (i, (freq, (piece, _score))) in expected.iter().zip(pieces).enumerate() {
            // Always keep unk.
            if i == 0 {
                new_pieces.push((piece.clone(), f64::NAN));
                continue;
            }
            if *freq < expected_frequency_threshold {
                continue;
            }
            new_pieces.push((piece.clone(), *freq));
            sum += freq;
        }
        // // Here we do not use the original EM, but use the
        // // Bayesianified/DPified EM algorithm.
        // // https://cs.stanford.edu/~pliang/papers/tutorial-acl2007-talk.pdf
        // // This modification will act as a sparse prior.
        let logsum = digamma(sum);
        let new_pieces: Vec<_> = new_pieces
            .into_iter()
            .map(|(s, c)| (s, digamma(c) - logsum))
            .collect();
        new_pieces
    }
    fn do_train(&self, sentences: Vec<Sentence>, params: &TrainingParams) -> Result<Unigram> {
        let progress = self.setup_progress(params.progress);
        //
        // 1. Compute frequent substrings
        self.update_progress(&progress, sentences.len(), "Suffix array seeds");
        let mut pieces: Vec<SentencePiece> = Vec::with_capacity(params.vocab_size);

        // We use a UNK token when training, whatever the `params.unk_token`
        pieces.push(("<UNK>".into(), f64::NAN));
        pieces.extend(self.make_seed_sentence_pieces(&sentences, &progress));
        self.finalize_progress(&progress, sentences.len());

        // Useful to check compatibility with spm.
        debug!(
            "Using {} pieces on {} sentences for EM training",
            pieces.len(),
            sentences.len()
        );

        let desired_vocab_size: usize = (params.vocab_size * 11) / 10; // * 1.1

        // 2. Run E-M Loops to fine grain the pieces.
        // We will shrink the vocab by shrinking_factor every loop on average
        // Some other pieces are dropped if logprob is too small
        // V = N * (f)**k
        // k = log(V / N) / log(f)
        let expected_loops = (((desired_vocab_size as f64).ln() - (pieces.len() as f64).ln())
            / self.shrinking_factor.ln()) as usize
            + 1;
        let expected_updates = expected_loops * self.n_sub_iterations as usize;
        self.update_progress(&progress, expected_updates, "EM training");
        let required_chars = self.required_chars(&sentences);
        if required_chars.len() > params.vocab_size {
            return Err(TrainingError::Pipeline(Box::new(
                UnigramTrainerError::VocabularyTooSmall,
            )));
        }
        let mut new_model = Unigram::from(pieces.clone(), Some(0), false)?;
        loop {
            // Sub-EM iteration.
            for _iter in 0..self.n_sub_iterations {
                // Executes E step
                let (_objective, _num_tokens, expected) = self.run_e_step(&new_model, &sentences);

                // Executes M step.
                pieces = self.run_m_step(&pieces, &expected, params);
                new_model = Unigram::from(pieces.clone(), Some(0), false)?;

                // Useful comment for checking compatibility with spm
                debug!(
                    "Em iter={} size={} obj={} num_tokens={} num_tokens/piece={}",
                    _iter,
                    new_model.len(),
                    _objective,
                    _num_tokens,
                    _num_tokens as f64 / new_model.len() as f64
                );
                if let Some(p) = &progress {
                    p.inc(1);
                }
            } // end of Sub EM iteration

            // Stops the iteration when the size of sentences reaches to the
            // desired symbol size.
            if pieces.len() <= desired_vocab_size {
                break;
            }

            // Prunes pieces.
            pieces = self.prune_sentence_pieces(&new_model, &pieces, &sentences, params);
            new_model = Unigram::from(pieces.clone(), Some(0), false)?;
        }
        self.finalize_progress(&progress, expected_updates);

        // Finally, adjusts the size of sentencepices to be |vocab_size|.
        self.finalize(new_model, required_chars, params)
    }
}

impl ModelTrainer for UnigramTrainer {
    type Model = Unigram;

    /// Train a Unigram model
    fn train_model(&self, params: &TrainingParams) -> Result<Unigram> {
        let sentences: Vec<_> = self.words.iter().map(|(s, i)| (s.to_owned(), *i)).collect();
        self.do_train(sentences, params)
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
    use assert_approx_eq::assert_approx_eq;
    use std::iter::FromIterator;
    use tk_encode::vocab::bucket_added_vocabulary::AddedToken;

    fn params() -> TrainingParams {
        TrainingParams::for_tests(8000)
    }

    #[test]
    fn test_unigram_chars() {
        let trainer = UnigramTrainerBuilder::default().build();

        let sentences = vec![
            ("This is a".to_string(), 1),
            ("こんにちは友達".to_string(), 1),
        ];

        let required_chars = trainer.required_chars(&sentences);
        assert_eq!(required_chars.len(), 13);

        let progress = None;
        let table = trainer.make_seed_sentence_pieces(&sentences, &progress);

        let target_strings = vec![
            "s", "i", " ", "達", "友", "ん", "は", "に", "ち", "こ", "h", "a", "T", "is ", "s ",
        ];

        let strings: Vec<_> = table.iter().map(|(string, _)| string).collect();
        assert_eq!(strings, target_strings);

        let scores = table.iter().map(|(_, score)| score);
        let target_scores = vec![
            -2.5649493574615367, // 2.0
            -2.5649493574615367, // 2.0
            -2.5649493574615367, // 2.0
            -3.258096538021482,  // 1.0
            -3.258096538021482,  // 1.0
            -3.258096538021482,  // 1.0
            -3.258096538021482,  // 1.0
            -3.258096538021482,  // 1.0
            -3.258096538021482,  // 1.0
            -3.258096538021482,  // 1.0
            -3.258096538021482,  // 1.0
            -3.258096538021482,  // 1.0
            -3.258096538021482,  // 1.0
            -1.4663370687934272, // 6.0
            -1.8718021769015916, // 4.0
        ];

        for (score, target_score) in scores.zip(target_scores) {
            assert_approx_eq!(*score, target_score, 0.01);
        }
    }

    #[test]
    fn trains_on_word_counts_beyond_u32() {
        let trainer = UnigramTrainerBuilder::default().build();
        let params = TrainingParams {
            vocab_size: 5,
            ..params()
        };

        let model = trainer
            .do_train(
                vec![
                    ("abc".to_string(), 3_000_000_000),
                    ("bcd".to_string(), 3_000_000_000),
                ],
                &params,
            )
            .unwrap();

        assert!(model.token_to_id("a").is_some());
    }

    #[test]
    fn test_initial_alphabet() {
        let trainer = UnigramTrainerBuilder::default()
            .initial_alphabet(AHashSet::from_iter(vec!['a', 'b', 'c', 'd', 'e', 'f']))
            .build();

        let sentences = vec![("こんにちは友達".to_string(), 1)];
        let required_chars = trainer.required_chars(&sentences);
        assert_eq!(
            required_chars,
            vec![
                "こ", "ん", "に", "ち", "は", "友", "達", "a", "b", "c", "d", "e", "f"
            ]
            .into_iter()
            .map(|s| s.to_owned())
            .collect::<AHashSet<_>>()
        );
    }

    #[test]
    fn test_unk_token() {
        // 1. Should add `unk_token` as first special token
        let trainer = UnigramTrainerBuilder::default().build();
        let params1 = TrainingParams {
            special_tokens: vec![
                AddedToken::from("[SEP]", true),
                AddedToken::from("[CLS]", true),
            ],
            unk_token: Some("[UNK]".into()),
            ..params()
        };

        let unigram = trainer
            .do_train(vec![("The".into(), 12), ("are".into(), 11)], &params1)
            .unwrap();

        let mut pieces = unigram.iter();
        assert_eq!(pieces.next(), Some(&("[UNK]".into(), 0.0)));
        assert_eq!(pieces.next(), Some(&("[SEP]".into(), 0.0)));
        assert_eq!(pieces.next(), Some(&("[CLS]".into(), 0.0)));

        // 2. Let it where it is
        let params2 = TrainingParams {
            special_tokens: vec![
                AddedToken::from("[SEP]", true),
                AddedToken::from("[CLS]", true),
                AddedToken::from("[UNK]", true),
            ],
            unk_token: Some("[UNK]".into()),
            ..params()
        };

        let unigram = trainer
            .do_train(vec![("The".into(), 12), ("are".into(), 11)], &params2)
            .unwrap();

        let mut pieces = unigram.iter();
        assert_eq!(pieces.next(), Some(&("[SEP]".into(), 0.0)));
        assert_eq!(pieces.next(), Some(&("[CLS]".into(), 0.0)));
        assert_eq!(pieces.next(), Some(&("[UNK]".into(), 0.0)));

        // 3. Don't put it there if not needed
        let unigram = trainer
            .do_train(vec![("The".into(), 12), ("are".into(), 11)], &params())
            .unwrap();

        let mut pieces = unigram.iter();
        assert_eq!(pieces.next().unwrap().0, "e".to_string());
    }

    #[test]
    fn test_special_tokens() {
        let trainer = UnigramTrainerBuilder::default().build();
        let params = TrainingParams {
            special_tokens: vec![
                AddedToken::from("[SEP]", true),
                AddedToken::from("[CLS]", true),
            ],
            ..params()
        };

        let unigram = trainer
            .do_train(vec![("The".into(), 12), ("are".into(), 11)], &params)
            .unwrap();

        let mut pieces = unigram.iter();
        assert_eq!(pieces.next(), Some(&("[SEP]".into(), 0.0)));
        assert_eq!(pieces.next(), Some(&("[CLS]".into(), 0.0)));
    }

    #[test]
    fn test_to_log_prob() {
        let mut a = vec![("".to_string(), 1.0), ("".to_string(), 2.0)];
        to_log_prob(&mut a);
        let scores = a.iter().map(|(_, score)| *score).collect::<Vec<_>>();
        // ln(1) - ln(3)
        assert_approx_eq!(scores[0], -1.098, 0.01);
        // ln(2) - ln(3)
        assert_approx_eq!(scores[1], -0.405, 0.01);
    }

    #[test]
    fn to_builder_keeps_settings_and_drops_words() {
        let mut trainer = UnigramTrainer::builder().max_piece_length(8).build();
        trainer
            .feed(["hello world"].iter(), |s| {
                Ok(s.split(' ').map(str::to_owned).collect())
            })
            .unwrap();

        let trainer = trainer.to_builder().shrinking_factor(0.5).build();

        assert_eq!(trainer.max_piece_length(), 8);
        assert_eq!(trainer.shrinking_factor(), 0.5);
        assert!(trainer.words.is_empty());
    }
}
