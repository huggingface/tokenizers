use daachorse::{DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder};

use crate::models::bpe::{MergeMap, Vocab, VocabR, Word};
use crate::utils::byte_level::BYTES_CHAR_LOOKUP;

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Copy)]
pub(crate) struct CanonicalTokenId(u32);

impl From<usize> for CanonicalTokenId {
    fn from(value: usize) -> Self {
        // todo: is this safe?
        Self(value as u32)
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Copy)]
struct HfToken {
    token_id: u32,
    byte_len: usize,
}

#[derive(PartialEq, Clone)]
pub(crate) struct BacktrackingEngine {
    canonical_to_hf: Vec<HfToken>,
    pattern_matcher: DoubleArrayAhoCorasick<CanonicalTokenId>,
}

impl BacktrackingEngine {
    pub(crate) fn new(vocab: &Vocab, vocab_r: &VocabR, merges: &MergeMap) -> Self {
        // Step 1: build the Aho-Corasick matcher over a canonical (merge-only) vocabulary
        let mut canonicalized: Vec<(Vec<u8>, HfToken)> = Vec::with_capacity(vocab.len());

        for character in BYTES_CHAR_LOOKUP.iter() {
            let pattern = character.to_string();
            if let Some(hf_token_id) = vocab.get(&pattern) {
                // Supports partial coverage of single bytes in the vocab
                canonicalized.push((
                    pattern.into_bytes(),
                    HfToken {
                        token_id: *hf_token_id,
                        byte_len: 1,
                    },
                ));
            }
        }

        let mut ordered_merges = vec![None; merges.len()];
        for (_, (rank, new_id)) in merges.iter() {
            let merged_token_bytes = vocab_r
                .get(new_id)
                .expect("Merged token not in vocabulary")
                .clone()
                .into_bytes();
            let hf_token_info = HfToken {
                token_id: *new_id,
                byte_len: merged_token_bytes.len(),
            };
            ordered_merges[*rank as usize] = Some((merged_token_bytes, hf_token_info));
        }
        let ordered_merges: Option<Vec<_>> = ordered_merges.into_iter().collect();

        let (patterns, canonical_to_hf): (Vec<_>, _) = canonicalized
            .into_iter()
            .chain(ordered_merges.expect("Invalid merge table: missing rank"))
            .unzip();

        let pattern_matcher = DoubleArrayAhoCorasickBuilder::new()
            .match_kind(daachorse::MatchKind::Standard)
            .build(patterns)
            .expect("Failed to build Aho-Corasick automaton from vocab");

        Self {
            pattern_matcher,
            canonical_to_hf,
        }
    }

    pub(crate) fn build_token_suffix_array(&self, sequence_bytes: &[u8]) -> Vec<CanonicalTokenId> {
        let mut last_tokens: Vec<CanonicalTokenId> = Vec::with_capacity(sequence_bytes.len());

        for matched_pattern in self.pattern_matcher.find_overlapping_iter(sequence_bytes) {
            if matched_pattern.end() != last_tokens.len() + 1 {
                continue;
            }
            let canonical_token_id = matched_pattern.value();

            if matched_pattern.start() == 0
                || self.is_valid_token_pair(
                    last_tokens[matched_pattern.start() - 1],
                    canonical_token_id,
                )
            {
                last_tokens.push(matched_pattern.value());
            }
        }

        last_tokens
    }

    pub(crate) fn backtrack_token_suffix_array(
        &self,
        token_suffixes: Vec<CanonicalTokenId>,
        word: &mut Word,
    ) {
    }

    fn is_valid_token_pair(&self, left: CanonicalTokenId, right: CanonicalTokenId) -> bool {
        false
    }
}
