use super::{Pair, Word};
use crate::tokenizer::{Model, Token};
use std::collections::HashMap;

pub struct BPE {
    /// The vocabulary assigns a number to each token
    vocab: HashMap<String, u32>,
    /// Reversed vocabulary, to rebuild sentences
    vocab_r: HashMap<u32, String>,
    /// Contains the mapping between Pairs and their (rank, new_id)
    merges: HashMap<Pair, (u32, u32)>,
}

impl BPE {
    pub fn new(
        vocab: HashMap<String, u32>,
        vocab_r: HashMap<u32, String>,
        merges: HashMap<Pair, (u32, u32)>,
    ) -> Self {
        BPE {
            vocab,
            vocab_r,
            merges,
        }
    }
}

impl Model for BPE {
    fn tokenize(&self, sentence: Vec<String>) -> Vec<Token> {
        let mut encoded: Vec<Token> = Vec::with_capacity(sentence.len());

        for w in sentence {
            let mut word = Word::new();
            for c in w.chars() {
                match self.vocab.get(&c.to_string()) {
                    // TODO: Handle UNK
                    None => println!("{} is an unknown character. Skip it.", c.escape_unicode()),
                    Some(id) => word.add(*id),
                }
            }

            loop {
                if word.get_chars().len() < 2 {
                    break;
                }

                let ((rank, new_id), pair) = word
                    .get_chars()
                    .windows(2)
                    .map(|window| {
                        let pair = (window[0], window[1]);
                        let rank = self
                            .merges
                            .get(&pair)
                            .unwrap_or(&(std::u32::MAX, std::u32::MAX));
                        (rank, pair)
                    })
                    .min()
                    .unwrap();

                if *rank == std::u32::MAX {
                    // We are done merging this word
                    break;
                }

                // Let's merge
                word.merge(pair.0, pair.1, *new_id);
            }

            // Offsets are word-based, we need to translate them to be sentence-based
            let last_offset = encoded.last().map(|token| token.offsets.1).unwrap_or(0);

            let tokens = word
                .get_chars()
                .iter()
                .zip(word.get_offsets())
                .map(|(id, offsets)| {
                    Token::new(
                        *id,
                        self.vocab_r[id].clone(),
                        (last_offset + offsets.0, last_offset + offsets.1),
                    )
                })
                .collect::<Vec<_>>();

            encoded.extend(tokens);
        }

        encoded
    }
}
