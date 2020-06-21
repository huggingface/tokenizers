use crate::models::unigram::{lattice::Lattice, model::Unigram};
use crate::tokenizer::{AddedToken, Model, Result, Trainer};
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;

type SentencePiece = (String, f64);

fn digamma(x: f64) -> f64 {
    let mut x = x;
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

pub struct UnigramTrainerBuilder {
    show_progress: bool,
}

impl Default for UnigramTrainerBuilder {
    fn default() -> Self {
        UnigramTrainerBuilder {
            show_progress: true,
        }
    }
}

impl UnigramTrainerBuilder {
    pub fn with_progress(mut self, progress: bool) -> Self {
        self.show_progress = progress;
        self
    }

    pub fn build(&self) -> UnigramTrainer {
        UnigramTrainer::new(self.show_progress)
    }
}

pub struct UnigramTrainer {
    show_progress: bool,
    vocab_size: u32,
    n_iterations: u32,
    special_tokens: Vec<AddedToken>,
}

impl Default for UnigramTrainer {
    fn default() -> Self {
        Self {
            show_progress: true,
            vocab_size: 8_000,
            n_iterations: 10,
            special_tokens: vec![],
        }
    }
}

fn is_valid_sentencepiece(_char_string: &[char]) -> bool {
    // TODO
    // Checks string length, space not in the substring, numbers, hiragana and more
    // https://github.com/google/sentencepiece/blob/26be9516cd81d5315ee31c48d2438018e0eab879/src/trainer_interface.cc#L203
    true
}

impl UnigramTrainer {
    pub fn new(show_progress: bool) -> UnigramTrainer {
        UnigramTrainer {
            show_progress,
            vocab_size: 8_000,
            n_iterations: 10,
            special_tokens: vec![],
        }
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

    fn finalize(&self) {}

    fn required_chars(&self, word_counts: &HashMap<String, u32>) -> HashSet<String> {
        // TODO more logic needed if this required chars > vocab_size
        word_counts
            .iter()
            .map(|(s, _count)| s.chars())
            .flatten()
            .map(|c| c.to_string())
            .collect()
    }
    fn make_seed_sentence_pieces(
        &self,
        word_counts: &HashMap<String, u32>,
    ) -> Result<Vec<(String, f64)>> {
        let vocab_size: usize = self.vocab_size.try_into()?;
        let progress = self.setup_progress();
        // TODO This is unused, figure out why.
        let _required_chars = self.required_chars(word_counts);
        // Put all sentences in a string, separated by \0
        let total: usize = word_counts
            .iter()
            .map(|(s, _)| s.chars().count())
            .sum::<usize>()
            + word_counts.len();
        let mut flat_string = String::with_capacity(total);
        let mut all_chars: HashMap<char, u32> = HashMap::new();
        let k_sentence_boundary = '\0';
        for string in word_counts.keys() {
            flat_string.push_str(&string);
            // Comment suggests we add sentence boundary, but it seems to be missing from actual
            // code.
            // flat_string.push_str(k_sentence_boundary);
            for c in string.chars() {
                if c != k_sentence_boundary {
                    *all_chars.entry(c).or_insert(0) += 1;
                }
            }
        }
        let chars: Vec<_> = flat_string.chars().collect();
        let n = chars.len();
        let mut sa = vec![0; n];
        let mut l = vec![0; n];
        let mut r = vec![0; n];
        let mut d = vec![0; n];
        let mut node_num = 0;
        let alphabet_size = 0x110000; // All UCS4 range.
        esaxx_rs::esaxx(
            &chars,
            &mut sa,
            &mut l,
            &mut r,
            &mut d,
            alphabet_size,
            &mut node_num,
        )
        .unwrap();

        self.update_progress(&progress, vocab_size, "Updating frequent sub strings...");
        let mut substr_index: Vec<(u32, usize)> = vec![];
        //  Basic chars need to be in sentence pieces.
        let mut seed_sentencepieces: Vec<(String, f64)> = vec![];

        let mut sall_chars: Vec<_> = all_chars.into_iter().map(|(a, b)| (b, a)).collect();
        // Reversed order
        sall_chars.sort_by(|a, b| b.cmp(a));
        for (count, character) in sall_chars {
            seed_sentencepieces.push((character.to_string(), count.into()));
            if let Some(p) = &progress {
                p.inc(1);
            }
        }
        for i in 0..node_num {
            let index: usize = i.try_into()?;
            let left: usize = l[index].try_into()?;
            let offset: usize = sa[left].try_into()?;
            let len: usize = d[index].try_into()?;
            if len <= 1 {
                continue;
            }
            let string = &chars[offset..offset + len];
            if string.contains(&k_sentence_boundary) {
                continue;
            }
            if !is_valid_sentencepiece(string) {
                continue;
            }

            let freq: u32 = (r[index] - l[index]).try_into()?;
            let len_u32: u32 = len.try_into()?;
            let score = freq * len_u32;
            substr_index.push((score, index));
            if let Some(p) = &progress {
                p.inc(1);
            }
        }

        // sort by decreasing score
        substr_index.sort_by(|a, b| b.cmp(a));

        for (score, i) in substr_index {
            let left: usize = l[i].try_into()?;
            let offset: usize = sa[left].try_into()?;
            let len: usize = d[i].try_into()?;
            assert!(len > 0);
            let char_string = &chars[offset..offset + len];
            // Just in case
            assert!(is_valid_sentencepiece(char_string));
            let string: String = char_string.iter().collect();
            seed_sentencepieces.push((string, score.into()));
            if seed_sentencepieces.len() >= vocab_size {
                break;
            }

            // C++ code uses strings, we kept chars
            //assert_eq!(all_chars.get(string), None);
        }
        // TODO
        // ToLogProb(seed_sentencepieces.begin(), seed_sentencepieces.end());
        self.finalize_progress(&progress, vocab_size);
        Ok(seed_sentencepieces)
    }
    fn prune_sentence_pieces(&self) {}
    fn get_piece_size(&self) -> u32 {
        0
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

    fn run_e_step(&self, model: &mut Unigram, sentences: &[(String, f64)]) -> (f64, u32, Vec<f64>) {
        let mut expected: Vec<f64> = vec![0.0; self.vocab_size as usize];
        let mut objs: f64 = 0.0;
        let mut ntokens: u32 = 0;

        let all_sentence_freq: f64 = sentences.iter().map(|(_a, b)| *b).sum();

        // TODO reparallelize this.
        for (string, freq) in sentences {
            let mut lattice = Lattice::from(string);
            model.populate_nodes(&mut lattice);
            let z: f64 = lattice.populate_marginal(*freq, &mut expected);
            ntokens += lattice.viterbi().len() as u32;
            if z.is_nan() {
                panic!("likelihood is NAN. Input sentence may be too long.");
            }

            objs -= z / (all_sentence_freq as f64);
        }

        (objs, ntokens, expected)
    }
    // TODO: Model is not really used...
    fn run_m_step(
        &self,
        _model: &mut Unigram,
        sentences: &[(String, f64)],
        expected: &[f64],
    ) -> Vec<SentencePiece> {
        // const auto &sentencepieces = model.GetSentencePieces();
        // CHECK_EQ(sentencepieces.size(), expected.size());
        // TrainerModel::SentencePieces new_sentencepieces;
        //
        let mut new_sentencepieces: Vec<(String, f64)> =
            Vec::with_capacity(self.vocab_size.try_into().unwrap());

        let mut sum = 0.0;
        let expected_frequency_threshold = 0.5;
        for (i, freq) in expected.iter().enumerate() {
            if freq < &expected_frequency_threshold {
                continue;
            }
            new_sentencepieces.push((sentences[i].0.clone(), *freq));
            sum += freq;
        }
        // // Here we do not use the original EM, but use the
        // // Bayesianified/DPified EM algorithm.
        // // https://cs.stanford.edu/~pliang/papers/tutorial-acl2007-talk.pdf
        // // This modification will act as a sparse prior.
        let logsum = digamma(sum);
        let new_sentencepieces: Vec<_> = new_sentencepieces
            .into_iter()
            .map(|(s, c)| (s, digamma(c) - logsum))
            .collect();
        new_sentencepieces
    }
    pub fn _train(
        &self,
        word_counts: HashMap<String, u32>,
    ) -> Result<(Box<dyn Model>, Vec<AddedToken>)> {
        // TODO handle progress bar.
        let _progress = self.setup_progress();
        //
        // 1. Compute frequent substrings
        // TODO should be either i64 or i32
        let table: Vec<(String, f64)> = self.make_seed_sentence_pieces(&word_counts)?;

        // Probably not implementing this, pre-tokenizer should handle that beforehand.
        // if (trainer_spec_.split_by_whitespace()) {
        //   SplitSentencesByWhitespace();
        // }

        // LOG(INFO) << "Using " << sentences_.size() << " sentences for EM training";

        let desired_vocab_size_ = (self.vocab_size * 11) / 10; // * 1.1

        // TODO make the model correctly ?
        let mut model = Unigram::from(&table);

        loop {
            // Sub-EM iteration.
            for _iter in 0..self.n_iterations {
                // Executes E step
                let (_objective, _num_tokens, expected) = self.run_e_step(&mut model, &table);

                // // Executes M step.
                let new_sentencepieces = self.run_m_step(&mut model, &table, &expected);
                model = Unigram::from(&new_sentencepieces);
                // self.sentence_pieces = new_sentencepieces;

                // LOG(INFO) << "EM sub_iter=" << iter << " size=" << model.GetPieceSize()
                //           << " obj=" << objective << " num_tokens=" << num_tokens
                //           << " num_tokens/piece="
                //           << 1.0 * num_tokens / model.GetPieceSize();
            } // end of Sub EM iteration

            // Stops the iteration when the size of sentences reaches to the
            // desired symbol size.
            if self.get_piece_size() <= desired_vocab_size_ {
                break;
            }

            // Prunes pieces.
            self.prune_sentence_pieces();
        }

        // Finally, adjusts the size of sentencepices to be |vocab_size|.
        self.finalize();

        Ok((Box::new(model), self.special_tokens.clone()))
    }
}

impl Trainer for UnigramTrainer {
    /// Train a Unigram model
    fn train(
        &self,
        word_counts: HashMap<String, u32>,
    ) -> Result<(Box<dyn Model>, Vec<AddedToken>)> {
        self._train(word_counts)
        // §let (unigram, tokens) = self._train(word_counts)?;
        // §Ok((unigram, tokens))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unigram_chars() {
        let trainer = UnigramTrainerBuilder::new().with_progress(false).build();
        let mut word_count: HashMap<String, u32> = HashMap::new();
        word_count.insert("This is a".to_string(), 1);
        word_count.insert("こんにちは友達".to_string(), 1);

        let required_chars = trainer.required_chars(&word_count);
        assert_eq!(required_chars.len(), 13);

        let table = trainer.make_seed_sentence_pieces(&word_count).unwrap();
        assert_eq!(
            table,
            vec![
                ("s".to_string(), 2.0),
                ("i".to_string(), 2.0),
                (" ".to_string(), 2.0),
                ("達".to_string(), 1.0),
                ("友".to_string(), 1.0),
                ("ん".to_string(), 1.0),
                ("は".to_string(), 1.0),
                ("に".to_string(), 1.0),
                ("ち".to_string(), 1.0),
                ("こ".to_string(), 1.0),
                ("h".to_string(), 1.0),
                ("a".to_string(), 1.0),
                ("T".to_string(), 1.0),
                ("is ".to_string(), 6.0),
                ("s ".to_string(), 4.0)
            ]
        );
    }
}
