use crate::tokenizer::{AddedToken, Model, Offsets, Result, Token, Trainer};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

type SentencePiece = (String, f64);
type Vocab = HashMap<String, u32>;

#[derive(PartialEq, Serialize, Deserialize)]
struct Unigram {
    vocab: Vocab,
}
impl std::fmt::Debug for Unigram {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("BPE")
            .field("vocab", &self.vocab.len())
            .finish()
    }
}

#[typetag::serde]
impl Model for Unigram {
    fn get_vocab(&self) -> &HashMap<String, u32> {
        &self.vocab
    }

    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn tokenize(&self, sentence: Vec<(String, Offsets)>) -> Result<Vec<Token>> {
        Ok(vec![])
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        None
    }

    fn save(&self, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>> {
        Ok(vec![])
    }
}

struct UnigramTrainer {
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

impl UnigramTrainer {
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
    fn make_seed_sentence_pieces(&self, word_counts: &HashMap<String, u32>) {
        let required_chars = self.required_chars(word_counts);
        // Put all sentences in a string, separated by \0
        let total: usize = word_counts
            .iter()
            .map(|(s, _)| s.chars().count())
            .sum::<usize>()
            + word_counts.len();
        let mut flat_string = String::with_capacity(total);
        for (string, _) in word_counts {
            flat_string.push_str(&string);
            // flat_string.push_str("\0");
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
        println!("Sa {:?}", sa);
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

    fn run_e_step(&self, objective: &f64, num_tokens: &u32) -> Vec<f64> {
        vec![]
    }
    fn run_m_step(&self, expected: Vec<f64>) -> Vec<SentencePiece> {
        vec![]
    }
    pub fn _train(&self, word_counts: HashMap<String, u32>) -> Result<(Unigram, Vec<AddedToken>)> {
        let progress = self.setup_progress();
        //
        // 1. Compute frequent substrings
        self.make_seed_sentence_pieces(&word_counts);

        // if (trainer_spec_.split_by_whitespace()) {
        //   SplitSentencesByWhitespace();
        // }

        // LOG(INFO) << "Using " << sentences_.size() << " sentences for EM training";

        let desired_vocab_size_ = (self.vocab_size * 11) / 10; // * 1.1

        loop {
            // Sub-EM iteration.
            for iter in 0..self.n_iterations {
                // Executes E step
                let objective = 0.0;
                let num_tokens = 0;
                let expected = self.run_e_step(&objective, &num_tokens);

                // // Executes M step.
                let new_sentencepieces = self.run_m_step(expected);
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

        Ok((
            Unigram {
                vocab: HashMap::new(),
            },
            self.special_tokens.clone(),
        ))
    }
}

impl Trainer for UnigramTrainer {
    /// Train a Unigram model
    fn train(
        &self,
        word_counts: HashMap<String, u32>,
    ) -> Result<(Box<dyn Model>, Vec<AddedToken>)> {
        let (unigram, tokens) = self._train(word_counts)?;
        Ok((Box::new(unigram), tokens))
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
    fn test_chars() {
        let trainer = UnigramTrainer::default();
        let mut word_count: HashMap<String, u32> = HashMap::new();
        word_count.insert("This is a".to_string(), 1);
        word_count.insert("こんにちは友達".to_string(), 1);

        let required_chars = trainer.required_chars(&word_count);
        assert_eq!(required_chars.len(), 13);

        let table = trainer.make_seed_sentence_pieces(&word_count);
    }
}
