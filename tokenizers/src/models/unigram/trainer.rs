use crate::models::unigram::{lattice::Lattice, model::Unigram};
use crate::tokenizer::{AddedToken, Result, Trainer};
use crate::utils::parallelism::*;
use crate::utils::progress::{ProgressBar, ProgressStyle};
use log::debug;
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;

// A token and a score
type SentencePiece = (String, f64);

// A full sentence or word + it's count within the dataset
type Sentence = (String, u32);

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

fn to_log_prob(pieces: &mut [SentencePiece]) {
    let sum: f64 = pieces.iter().map(|(_, score)| score).sum();
    let logsum = sum.ln();
    for (_, score) in pieces.iter_mut() {
        *score = score.ln() - logsum;
    }
}

/// A `UnigramTrainer` can train a `Unigram` model from `word_counts`.
#[non_exhaustive]
#[derive(Builder, Debug, Clone)]
pub struct UnigramTrainer {
    #[builder(default = "true")]
    pub show_progress: bool,
    #[builder(default = "8000")]
    pub vocab_size: u32,
    #[builder(default = "2")]
    pub n_sub_iterations: u32,
    #[builder(default = "0.75")]
    pub shrinking_factor: f64,
    #[builder(default = "vec![]")]
    pub special_tokens: Vec<AddedToken>,
    #[builder(default = "HashSet::new()")]
    pub initial_alphabet: HashSet<char>,

    #[builder(default = "None")]
    pub unk_token: Option<String>,

    #[builder(default = "16")]
    pub max_piece_length: usize,
    #[builder(default = "1_000_000")]
    seed_size: usize,
    #[builder(default = "HashMap::new()")]
    words: HashMap<String, u32>,
}

impl Default for UnigramTrainer {
    fn default() -> Self {
        Self::builder().build().unwrap()
    }
}

impl UnigramTrainer {
    pub fn builder() -> UnigramTrainerBuilder {
        UnigramTrainerBuilder::default()
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

    fn finalize(&self, model: Unigram, required_chars: HashSet<String>) -> Result<Unigram> {
        let mut min_score_penalty = 0.0;
        let min_score_penalty_delta = 0.0001;

        let mut pieces: Vec<(String, f64)> = vec![];
        let mut inserted: HashSet<String> = HashSet::new();

        // We don't want to include the <UNK> that was used to train
        inserted.insert("<UNK>".into());

        let existing_pieces: HashMap<String, f64> = model.iter().cloned().collect();
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
        for (token, score) in model.iter() {
            if inserted.contains::<str>(token) {
                continue;
            }
            inserted.insert(token.to_string());
            pieces.push((token.to_string(), if score.is_nan() { 0.0 } else { *score }));
            if pieces.len() == self.vocab_size as usize {
                break;
            }
        }
        pieces.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        // Insert the necessary tokens
        let (unk_id, need_add_unk) = if let Some(ref unk) = self.unk_token {
            let unk_id = self.special_tokens.iter().enumerate().find_map(|(i, t)| {
                if t.content == *unk {
                    Some(i)
                } else {
                    None
                }
            });
            match unk_id {
                Some(id) => (Some(id), false),
                None => (Some(0), true),
            }
        } else {
            (None, false)
        };
        let mut special_tokens = self
            .special_tokens
            .iter()
            .map(|t| (t.content.clone(), 0.0))
            .collect::<Vec<_>>();
        if need_add_unk {
            special_tokens.insert(0, (self.unk_token.clone().unwrap(), 0.0));
        }

        Unigram::from(special_tokens.into_iter().chain(pieces).collect(), unk_id)
    }

    fn required_chars(&self, word_counts: &[Sentence]) -> HashSet<String> {
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
    ) -> Result<Vec<SentencePiece>> {
        // Put all sentences in a string, separated by \0
        let total: usize = sentences
            .iter()
            .map(|(s, _)| s.chars().count())
            .sum::<usize>()
            + sentences.len();
        let mut flat_string = String::with_capacity(total);
        let mut all_chars: HashMap<char, u32> = HashMap::new();
        let c_sentence_boundary = '\0';
        let k_sentence_boundary = '\0'.to_string();
        for (string, n) in sentences {
            flat_string.push_str(&string);
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
        let suffix = esaxx_rs::suffix(&flat_string).unwrap();

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
            seed_sentencepieces.push((character.to_string(), count.into()));
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
        Ok(seed_sentencepieces)
    }
    fn prune_sentence_pieces(
        &self,
        model: &Unigram,
        pieces: &[SentencePiece],
        sentences: &[Sentence],
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
        let mut vsum = 0.0;
        let mut freq: Vec<f64> = vec![0.0; pieces.len()];
        let mut inverted: Vec<Vec<usize>> = vec![Vec::new(); pieces.len()];
        // TODO reparallelize this
        for (i, (sentence, count)) in sentences.iter().enumerate() {
            let mut lattice = Lattice::from(sentence, bos_id, eos_id);
            model.populate_nodes(&mut lattice);
            vsum += *count as f64;
            for node_ref in lattice.viterbi() {
                let id = node_ref.borrow().id;
                freq[id] += *count as f64;
                inverted[id].push(i);
            }
        }

        let sum: f64 = freq.iter().sum();
        let logsum = sum.ln();
        let mut candidates: Vec<(usize, f64)> = vec![];
        let mut new_pieces: Vec<SentencePiece> = Vec::with_capacity(self.vocab_size as usize);
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

                // The frequencies of altenatives are increased by freq[i].
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
        let desired_vocab_size: usize = (self.vocab_size as usize * 11) / 10; // * 1.1
        let pruned_size: usize = ((pieces.len() as f64) * self.shrinking_factor) as usize;
        let pruned_size = desired_vocab_size.max(pruned_size);

        candidates.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        for (id, _score) in candidates {
            if new_pieces.len() == pruned_size {
                break;
            }
            new_pieces.push(pieces[id].clone());
        }

        new_pieces.to_vec()
    }

    /// Update the progress bar with the new provided length and message
    fn update_progress(&self, p: &Option<ProgressBar>, len: usize, message: &str) {
        if let Some(p) = p {
            p.set_message(message);
            p.set_length(len as u64);
            p.set_draw_delta(len as u64 / 100);
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
        let mut expected: Vec<f64> = vec![0.0; model.len()];
        let mut objs: f64 = 0.0;
        let mut ntokens: u32 = 0;

        let all_sentence_freq: u32 = sentences.iter().map(|(_a, b)| *b).sum();

        // TODO reparallelize this.
        for (string, freq) in sentences {
            let mut lattice = Lattice::from(string, model.bos_id, model.eos_id);
            model.populate_nodes(&mut lattice);
            let z: f64 = lattice.populate_marginal(*freq as f64, &mut expected);
            ntokens += lattice.viterbi().len() as u32;
            if z.is_nan() {
                panic!("likelihood is NAN. Input sentence may be too long.");
            }

            objs -= z / (all_sentence_freq as f64);
        }

        (objs, ntokens, expected)
    }
    fn run_m_step(&self, pieces: &[SentencePiece], expected: &[f64]) -> Vec<SentencePiece> {
        if pieces.len() != expected.len() {
            panic!(
                "Those two iterators are supposed to be the same length ({} vs {})",
                pieces.len(),
                expected.len()
            );
        }
        let mut new_pieces: Vec<SentencePiece> =
            Vec::with_capacity(self.vocab_size.try_into().unwrap());

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
    pub fn do_train(
        &self,
        sentences: Vec<Sentence>,
        model: &mut Unigram,
    ) -> Result<Vec<AddedToken>> {
        let progress = self.setup_progress();
        //
        // 1. Compute frequent substrings
        // TODO Should be able to upgrade to u64 when needed
        self.update_progress(&progress, sentences.len(), "Suffix array seeds");
        let mut pieces: Vec<SentencePiece> =
            Vec::with_capacity(self.vocab_size.try_into().unwrap());

        // We use a UNK token when training, whatever the `self.unk_token`
        pieces.push(("<UNK>".into(), f64::NAN));
        pieces.extend(self.make_seed_sentence_pieces(&sentences, &progress)?);
        self.finalize_progress(&progress, sentences.len());

        // Useful to check compatibility with spm.
        debug!(
            "Using {} pieces on {} sentences for EM training",
            pieces.len(),
            sentences.len()
        );

        let desired_vocab_size: usize = (self.vocab_size as usize * 11) / 10; // * 1.1

        // 2. Run E-M Loops to fine grain the pieces.
        // We will shrink the vocab by shrinking_factor every loop on average
        // Some other pieces are dropped if logprob is too small
        // V = N * (f)**k
        // k = log(V / N) / log(f)
        let expected_loops = (((desired_vocab_size as f64).ln() - (pieces.len() as f64).ln())
            / self.shrinking_factor.ln()) as usize
            + 1;
        let expected_updates = expected_loops as usize * self.n_sub_iterations as usize;
        self.update_progress(&progress, expected_updates, "EM training");
        let required_chars = self.required_chars(&sentences);
        let mut new_model = Unigram::from(pieces.clone(), Some(0))?;
        loop {
            // Sub-EM iteration.
            for _iter in 0..self.n_sub_iterations {
                // Executes E step
                let (_objective, _num_tokens, expected) = self.run_e_step(&new_model, &sentences);

                // Executes M step.
                pieces = self.run_m_step(&pieces, &expected);
                new_model = Unigram::from(pieces.clone(), Some(0))?;

                // Useful comment for checking compatibility with spm
                debug!(
                    "Em iter={} size={} obj={} num_tokens={} num_tokens/piece={}",
                    _iter,
                    new_model.len(),
                    _objective,
                    _num_tokens,
                    _num_tokens as f64 / model.len() as f64
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
            pieces = self.prune_sentence_pieces(&new_model, &pieces, &sentences);
            new_model = Unigram::from(pieces.clone(), Some(0))?;
        }
        self.finalize_progress(&progress, expected_updates);

        // Finally, adjusts the size of sentencepices to be |vocab_size|.
        *model = self.finalize(new_model, required_chars)?;

        Ok(self.special_tokens.clone())
    }
}

impl Trainer for UnigramTrainer {
    type Model = Unigram;

    /// Train a Unigram model
    fn train(&self, model: &mut Unigram) -> Result<Vec<AddedToken>> {
        let sentences: Vec<_> = self.words.iter().map(|(s, i)| (s.to_owned(), *i)).collect();
        self.do_train(sentences, model)
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
    use assert_approx_eq::assert_approx_eq;
    use std::iter::FromIterator;

    #[test]
    fn test_unigram_chars() {
        let trainer = UnigramTrainerBuilder::default()
            .show_progress(false)
            .build()
            .unwrap();

        let sentences = vec![
            ("This is a".to_string(), 1),
            ("こんにちは友達".to_string(), 1),
        ];

        let required_chars = trainer.required_chars(&sentences);
        assert_eq!(required_chars.len(), 13);

        let progress = None;
        let table = trainer
            .make_seed_sentence_pieces(&sentences, &progress)
            .unwrap();

        let target_strings = vec![
            "s", "i", " ", "達", "友", "ん", "は", "に", "ち", "こ", "h", "a", "T", "is ", "s ",
        ];

        let strings: Vec<_> = table.iter().map(|(string, _)| string).collect();
        assert_eq!(strings, target_strings);

        let scores: Vec<_> = table.iter().map(|(_, score)| score).collect();
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

        for (score, target_score) in scores.into_iter().zip(target_scores) {
            assert_approx_eq!(*score, target_score, 0.01);
        }
    }

    #[test]
    fn test_initial_alphabet() {
        let trainer = UnigramTrainerBuilder::default()
            .show_progress(false)
            .initial_alphabet(HashSet::from_iter(vec!['a', 'b', 'c', 'd', 'e', 'f']))
            .build()
            .unwrap();

        let sentences = vec![("こんにちは友達".to_string(), 1)];
        let required_chars = trainer.required_chars(&sentences);
        assert_eq!(
            required_chars,
            HashSet::from_iter(
                vec!["こ", "ん", "に", "ち", "は", "友", "達", "a", "b", "c", "d", "e", "f"]
                    .into_iter()
                    .map(|s| s.to_owned())
            )
        );
    }

    #[test]
    fn test_unk_token() {
        // 1. Should add `unk_token` as first special token
        let trainer = UnigramTrainerBuilder::default()
            .show_progress(false)
            .special_tokens(vec![
                AddedToken::from("[SEP]", true),
                AddedToken::from("[CLS]", true),
            ])
            .unk_token(Some("[UNK]".into()))
            .build()
            .unwrap();

        let mut unigram = Unigram::default();
        trainer
            .do_train(vec![("The".into(), 12), ("are".into(), 11)], &mut unigram)
            .unwrap();

        let mut pieces = unigram.iter();
        assert_eq!(pieces.next(), Some(&("[UNK]".into(), 0.0)));
        assert_eq!(pieces.next(), Some(&("[SEP]".into(), 0.0)));
        assert_eq!(pieces.next(), Some(&("[CLS]".into(), 0.0)));

        // 2. Let it where it is
        let trainer = UnigramTrainerBuilder::default()
            .show_progress(false)
            .special_tokens(vec![
                AddedToken::from("[SEP]", true),
                AddedToken::from("[CLS]", true),
                AddedToken::from("[UNK]", true),
            ])
            .unk_token(Some("[UNK]".into()))
            .build()
            .unwrap();

        let mut unigram = Unigram::default();
        trainer
            .do_train(vec![("The".into(), 12), ("are".into(), 11)], &mut unigram)
            .unwrap();

        let mut pieces = unigram.iter();
        assert_eq!(pieces.next(), Some(&("[SEP]".into(), 0.0)));
        assert_eq!(pieces.next(), Some(&("[CLS]".into(), 0.0)));
        assert_eq!(pieces.next(), Some(&("[UNK]".into(), 0.0)));

        // 3. Don't put it there if not needed
        let trainer = UnigramTrainerBuilder::default()
            .show_progress(false)
            .build()
            .unwrap();

        let mut unigram = Unigram::default();
        trainer
            .do_train(vec![("The".into(), 12), ("are".into(), 11)], &mut unigram)
            .unwrap();

        let mut pieces = unigram.iter();
        assert_eq!(pieces.next().unwrap().0, "e".to_string());
    }

    #[test]
    fn test_special_tokens() {
        let trainer = UnigramTrainerBuilder::default()
            .show_progress(false)
            .special_tokens(vec![
                AddedToken::from("[SEP]", true),
                AddedToken::from("[CLS]", true),
            ])
            .build()
            .unwrap();

        let mut unigram = Unigram::default();
        trainer
            .do_train(vec![("The".into(), 12), ("are".into(), 11)], &mut unigram)
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
}
