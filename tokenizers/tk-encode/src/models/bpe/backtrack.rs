use std::hash::Hash;

use ahash::AHashMap;
use daachorse::{DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder};

use crate::models::bpe::{MergeMap, Vocab, VocabR, Word};
use crate::utils::byte_level::BYTES_CHAR_LOOKUP;

/// Flip to `true` to trace `is_valid_token_pair` (the seam check) to stderr. Compiled out
/// (dead-code-eliminated) when `false`, so it costs nothing in normal builds.
const TRACE_SEAM: bool = false;

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Copy, Hash)]
pub(crate) struct CanonicalTokenId(u32);

impl From<usize> for CanonicalTokenId {
    fn from(value: usize) -> Self {
        // todo: is this safe?
        Self(value as u32)
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone, Copy)]
struct HfToken(u32);

#[derive(PartialEq, Clone)]
pub(crate) struct BacktrackingEngine {
    /// Bytes of all tokens, contiguous
    token_bytes: Vec<u8>,
    /// An array giving the index of the first byte of the Canonical Token ID (u32) in [`self.token_bytes`]
    /// Can derive the token's length easily
    canonical_token_starts: Vec<usize>,
    /// maps the hash of the token bytes to its Canonical Token ID
    bytes_hash_to_canonical: AHashMap<u32, CanonicalTokenId>,
    /// Perfect hash constant, guaranteed collision-free in u32
    hash_factor: u64,

    /// Maps Canonical token ID -> HF token ID
    canonical_to_hf: Vec<HfToken>,

    /// Maps a Canonical Token ID to its left and right predecessors
    split_table: Vec<(CanonicalTokenId, CanonicalTokenId)>,

    /// Aho-Corasick pattern matcher
    pattern_matcher: DoubleArrayAhoCorasick<CanonicalTokenId>,
}

#[derive(Copy, Clone, Debug)]
pub enum PeelDirection {
    Left,
    Right,
}

impl BacktrackingEngine {
    pub(crate) fn new(vocab: &Vocab, vocab_r: &VocabR, merges: &MergeMap) -> Self {
        // Phase 1: collect the canonical token list — atoms (single byte-level chars present
        // in the vocab) in byte-char order, then merge results in rank order. No hashing yet.
        let mut canonicalized: Vec<(Vec<u8>, HfToken)> = Vec::with_capacity(vocab.len());

        for character in BYTES_CHAR_LOOKUP.iter() {
            let pattern = character.to_string();
            if let Some(hf_token_id) = vocab.get(&pattern) {
                // Supports partial coverage of single bytes in the vocab
                canonicalized.push((pattern.into_bytes(), HfToken(*hf_token_id)));
            }
        }
        let atom_count = canonicalized.len();

        let mut merges: Vec<_> = merges.iter().collect();
        merges.sort_unstable_by_key(|(_, (rank, _))| *rank);
        let mut merge_predecessors: Vec<(u32, u32)> = Vec::with_capacity(merges.len());
        for ((left_predecessor, right_predecessor), (_merged_rank, merged_id)) in merges.iter() {
            let merged_token_bytes = vocab_r
                .get(merged_id)
                .expect("Merged token not in vocabulary")
                .clone()
                .into_bytes();
            canonicalized.push((merged_token_bytes, HfToken(*merged_id)));
            merge_predecessors.push((*left_predecessor, *right_predecessor));
        }

        // Phase 2: find a multiplier that hashes every token to a distinct u32.
        let hash_factor = Self::find_hash_factor(canonicalized.iter().map(|(b, _)| b.as_slice()));

        // Phase 3: build the perfect-hash map (token bytes -> canonical id).
        let mut bytes_hash_to_canonical = AHashMap::with_capacity(canonicalized.len());
        for (index, (bytes, _)) in canonicalized.iter().enumerate() {
            bytes_hash_to_canonical
                .insert(Self::hash_bytes(bytes, hash_factor), CanonicalTokenId(index as u32));
        }

        // Phase 4: split table. Atoms point to themselves; merges to their two predecessors.
        let canonical_of = |token_id: u32| {
            let bytes = vocab_r
                .get(&token_id)
                .expect("Predecessor token not in vocabulary")
                .as_bytes();
            *bytes_hash_to_canonical
                .get(&Self::hash_bytes(bytes, hash_factor))
                .expect("Invalid merges: predecessor is not a token at this point")
        };
        let mut split_table: Vec<(CanonicalTokenId, CanonicalTokenId)> =
            Vec::with_capacity(canonicalized.len());
        for index in 0..atom_count {
            let id = CanonicalTokenId(index as u32);
            split_table.push((id, id));
        }
        for (left_id, right_id) in merge_predecessors {
            split_table.push((canonical_of(left_id), canonical_of(right_id)));
        }

        let (patterns, canonical_to_hf): (Vec<_>, _) = canonicalized.into_iter().unzip();

        let mut token_bytes: Vec<u8> = vec![];
        let mut canonical_token_starts = Vec::with_capacity(vocab.len());

        for pattern in patterns.iter() {
            canonical_token_starts.push(token_bytes.len());
            token_bytes.extend_from_slice(pattern);
        }

        let pattern_matcher = DoubleArrayAhoCorasickBuilder::new()
            .match_kind(daachorse::MatchKind::Standard)
            .build(patterns)
            .expect("Failed to build Aho-Corasick automaton from vocab");

        Self {
            pattern_matcher,
            canonical_to_hf,
            token_bytes,
            bytes_hash_to_canonical,
            canonical_token_starts,
            hash_factor,
            split_table,
        }
    }

    /// Per-component heap footprint of the engine. `capacity`-based for the owned containers;
    /// exact for the Aho-Corasick automaton via daachorse's `heap_bytes`.
    pub(crate) fn memory_breakdown(&self) -> Vec<(&'static str, usize)> {
        let slot = std::mem::size_of::<u32>() + std::mem::size_of::<CanonicalTokenId>();
        vec![
            ("token_bytes", self.token_bytes.capacity()),
            (
                "canonical_token_starts",
                self.canonical_token_starts.capacity() * std::mem::size_of::<usize>(),
            ),
            (
                "bytes_hash_to_canonical",
                self.bytes_hash_to_canonical.capacity() * slot,
            ),
            (
                "split_table",
                self.split_table.capacity()
                    * std::mem::size_of::<(CanonicalTokenId, CanonicalTokenId)>(),
            ),
            (
                "canonical_to_hf",
                self.canonical_to_hf.capacity() * std::mem::size_of::<HfToken>(),
            ),
            ("aho_corasick", self.pattern_matcher.heap_bytes()),
        ]
    }

    pub(crate) fn build_token_suffix_array(&self, sequence_bytes: &[u8]) -> Vec<CanonicalTokenId> {
        let n = sequence_bytes.len();
        // `last_token[p - 1]` is the token ending at byte position `p` in the unique valid
        // tokenization of `sequence_bytes[..p]`. Indexed by byte position (not push count) so
        // that multi-byte atoms — e.g. "Ġ" spans two bytes — don't stall the scan at a byte
        // position where no token ends. `reachable[p]` marks positions an encoding can reach.
        let mut last_token = vec![CanonicalTokenId(0); n];
        let mut reachable = vec![false; n + 1];
        reachable[0] = true;

        for matched_pattern in self.pattern_matcher.find_overlapping_iter(sequence_bytes) {
            let start = matched_pattern.start();
            let end = matched_pattern.end();
            // Skip if the prefix before this match isn't reachable, or we already recorded the
            // token ending here (first valid match wins — greedy BPE's output is unique).
            if !reachable[start] || reachable[end] {
                continue;
            }
            let canonical_token_id = matched_pattern.value();
            if start == 0 || self.is_valid_token_pair(last_token[start - 1], canonical_token_id) {
                last_token[end - 1] = canonical_token_id;
                reachable[end] = true;
            }
        }

        last_token
    }

    pub(crate) fn backtrack_token_suffix_array(
        &self,
        token_suffixes: Vec<CanonicalTokenId>,
        word: &mut Word,
    ) {
        let mut reversed_tokens = vec![];
        let mut idx = token_suffixes.len();
        while idx > 0 {
            let canonical_token = token_suffixes[idx - 1];
            let token_length = self
                .get_token_byte_length(canonical_token)
                .expect("Defined by construction");
            let hf_token = self.canonical_to_hf[canonical_token.0 as usize];
            reversed_tokens.push((hf_token, token_length));
            idx -= token_length;
        }
        for (token, byte_length) in reversed_tokens.into_iter().rev() {
            word.add(token.0, byte_length);
        }
    }

    fn peel(&self, token: CanonicalTokenId, direction: PeelDirection) -> Option<CanonicalTokenId> {
        let (left, right) = self.split_table[token.0 as usize];
        let peeled = match direction {
            PeelDirection::Left => left,
            PeelDirection::Right => right,
        };
        if peeled == token {
            return None;
        }
        Some(peeled)
    }

    fn get_token_byte_length(&self, token: CanonicalTokenId) -> Option<usize> {
        let index = token.0 as usize;
        let start = self.canonical_token_starts.get(index).copied()?;
        let end = self
            .canonical_token_starts
            .get(index + 1)
            .copied()
            .unwrap_or_else(|| self.token_bytes.len());
        Some(end - start)
    }

    fn get_token_bytes(&self, token: CanonicalTokenId) -> Option<&[u8]> {
        let index = token.0 as usize;
        let start = self.canonical_token_starts.get(index).copied()?;
        let end = self
            .canonical_token_starts
            .get(index + 1)
            .copied()
            .unwrap_or_else(|| self.token_bytes.len());
        Some(&self.token_bytes[start..end])
    }

    fn hash_bytes<'a, T: IntoIterator<Item = &'a u8>>(range: T, hash_factor: u64) -> u32 {
        let mut hash = 0u64;
        for &byte in range {
            hash = hash.wrapping_add(byte as u64).wrapping_mul(hash_factor);
        }
        (hash >> 32) as u32
    }

    /// Searches for a multiplier that hashes every token to a distinct u32, i.e. a perfect
    /// hash over the vocabulary. The candidate stream is deterministic (splitmix64) so builds
    /// are reproducible, and each candidate is forced odd so the multiplicative hash stays a
    /// bijection mod 2^64 (otherwise short tokens collapse to the same high bits).
    fn find_hash_factor<'a>(tokens: impl Iterator<Item = &'a [u8]> + Clone) -> u64 {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        loop {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let factor = (z ^ (z >> 31)) | 1;

            let mut hashes: Vec<u32> =
                tokens.clone().map(|t| Self::hash_bytes(t, factor)).collect();
            hashes.sort_unstable();
            if hashes.windows(2).all(|w| w[0] != w[1]) {
                return factor;
            }
        }
    }

    fn get_merged(
        &self,
        left: CanonicalTokenId,
        right: CanonicalTokenId,
    ) -> Option<CanonicalTokenId> {
        let left_range = self.get_token_bytes(left).expect("Invalid left token");
        let right_range = self.get_token_bytes(right).expect("Invalid right token");
        // Hash over both slices in sequence instead of allocating a concatenation — same bytes,
        // same hash.
        let hash = Self::hash_bytes(left_range.iter().chain(right_range.iter()), self.hash_factor);
        let candidate_id = *self.bytes_hash_to_canonical.get(&hash)?;
        if let Some(candidate) = self.get_token_bytes(candidate_id) {
            // Check the candidate token's bytes match the concatenation of its left and right constituent
            if candidate.len() == left_range.len() + right_range.len()
                && &candidate[..left_range.len()] == left_range
                && &candidate[left_range.len()..] == right_range
            {
                return Some(candidate_id);
            }
        }
        None
    }

    /// Renders a canonical token as `id:"bytes"` for trace output. Only used when
    /// [`TRACE_SEAM`] is on.
    #[allow(dead_code)]
    fn dbg_tok(&self, token: CanonicalTokenId) -> String {
        let bytes = self.get_token_bytes(token).unwrap_or(b"");
        format!("{}:{:?}", token.0, String::from_utf8_lossy(bytes))
    }

    fn is_valid_token_pair(
        &self,
        left_rank: CanonicalTokenId,
        right_rank: CanonicalTokenId,
    ) -> bool {
        let mut left = left_rank;
        let mut right = right_rank;
        // Note: a token with a lower rank gets merged / materialized earlier when BPEing greedily
        let mut highest_merge_rank = u32::MAX;

        if TRACE_SEAM {
            eprintln!(
                "[seam] check left={} right={}",
                self.dbg_tok(left),
                self.dbg_tok(right)
            );
        }

        loop {
            // First: check whether the tokens at the frontier would merge together
            if let Some(merged) = self.get_merged(left, right) {
                if TRACE_SEAM {
                    eprintln!(
                        "[seam]   frontier {}+{} would merge into {} (rank {}) vs limit {}",
                        self.dbg_tok(left),
                        self.dbg_tok(right),
                        self.dbg_tok(merged),
                        merged.0,
                        highest_merge_rank
                    );
                }
                // The two adjacent tokens would merge AND the merge has higher priority
                // => the token pair is not valid
                if merged.0 < highest_merge_rank {
                    if TRACE_SEAM {
                        eprintln!("[seam] -> INVALID ({} crosses the seam first)", self.dbg_tok(merged));
                    }
                    return false;
                }
            }

            // Then, repeatedly "peel" the tokens inwards until we find a better-ranked merge, or we peel down to atomic tokens
            // We peel the token with the highest rank first.
            //
            // The limit is asymmetric: peeling the left token uses its rank directly, but
            // peeling the right token uses `rank + 1`. This encodes greedy BPE's left-to-right
            // preference — a crossing merge that *ties* the right token's rank still fires first
            // (invalid), while a tie with the left token's rank does not. Without the `+1`,
            // symmetric runs like `ab|bbbb` and `abbbb|b` collapse to the same state.
            if left > right {
                highest_merge_rank = left.0;
                match self.peel(left, PeelDirection::Right) {
                    Some(peeled) => {
                        if TRACE_SEAM {
                            eprintln!(
                                "[seam]   peel left {} -> {} (limit now {})",
                                self.dbg_tok(left),
                                self.dbg_tok(peeled),
                                highest_merge_rank
                            );
                        }
                        left = peeled;
                    }
                    None => {
                        // left has a higher rank than right: right is necessarily an atomic token too
                        if TRACE_SEAM {
                            eprintln!("[seam] -> VALID (left {} is atomic)", self.dbg_tok(left));
                        }
                        return true;
                    }
                }
            } else {
                highest_merge_rank = right.0 + 1;
                match self.peel(right, PeelDirection::Left) {
                    Some(peeled) => {
                        if TRACE_SEAM {
                            eprintln!(
                                "[seam]   peel right {} -> {} (limit now {})",
                                self.dbg_tok(right),
                                self.dbg_tok(peeled),
                                highest_merge_rank
                            );
                        }
                        right = peeled;
                    }
                    None => {
                        // right has a higher rank than left: left is necessarily an atomic token too
                        if TRACE_SEAM {
                            eprintln!("[seam] -> VALID (right {} is atomic)", self.dbg_tok(right));
                        }
                        return true;
                    }
                }
            }
        }
    }
}
