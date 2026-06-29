use std::cell::RefCell;
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
    /// Byte length of each canonical token, indexed by Canonical Token ID. This is all the
    /// backtrack step needs to walk back through the suffix array (`idx -= len`); the token
    /// bytes themselves are no longer stored since `get_merged` is structure-based.
    token_byte_lengths: Vec<u32>,
    /// Maps an adjacent (left, right) canonical token pair to the token they merge into — i.e.
    /// the merge rules in canonical-id space. Absent key = the pair does not merge.
    pair_lookup: AHashMap<(CanonicalTokenId, CanonicalTokenId), CanonicalTokenId>,

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

/// Per-thread reusable scratch buffers for [`BacktrackingEngine::encode_run`], so encoding a
/// word allocates nothing on the hot path after warmup. Thread-local because `encode` runs on
/// `&self` (often across rayon workers); each thread gets its own.
#[derive(Default)]
struct Scratch {
    /// `last_token[p - 1]` = token ending at byte position `p`. Guarded by `reachable`, so
    /// stale entries from a previous word are never read — no need to clear it.
    last_token: Vec<CanonicalTokenId>,
    /// `reachable[p]` = whether a valid tokenization reaches byte position `p`. Reset per word.
    reachable: Vec<bool>,
    /// Tokens collected while backtracking from the end, in reverse order.
    reversed: Vec<(HfToken, usize)>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
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

        // Phase 2: HF token id -> canonical id. Only needed to resolve merge predecessors
        // during construction, so it's a local map, not a stored field.
        let mut hf_to_canonical: AHashMap<u32, CanonicalTokenId> =
            AHashMap::with_capacity(canonicalized.len());
        for (index, (_, hf)) in canonicalized.iter().enumerate() {
            hf_to_canonical.insert(hf.0, CanonicalTokenId(index as u32));
        }
        let canonical_of = |hf_id: u32| {
            *hf_to_canonical
                .get(&hf_id)
                .expect("Invalid merges: predecessor is not a token")
        };

        // Phase 3: split table (atoms point to themselves; merges to their predecessors) and
        // pair_lookup ((left, right) -> merged) in one pass over the merges.
        let mut split_table: Vec<(CanonicalTokenId, CanonicalTokenId)> =
            Vec::with_capacity(canonicalized.len());
        for index in 0..atom_count {
            let id = CanonicalTokenId(index as u32);
            split_table.push((id, id));
        }
        let mut pair_lookup: AHashMap<(CanonicalTokenId, CanonicalTokenId), CanonicalTokenId> =
            AHashMap::with_capacity(merge_predecessors.len());
        for (merge_index, (left_id, right_id)) in merge_predecessors.into_iter().enumerate() {
            let pair = (canonical_of(left_id), canonical_of(right_id));
            let merged = CanonicalTokenId((atom_count + merge_index) as u32);
            split_table.push(pair);
            pair_lookup.insert(pair, merged);
        }

        let (patterns, canonical_to_hf): (Vec<_>, _) = canonicalized.into_iter().unzip();

        let token_byte_lengths: Vec<u32> = patterns.iter().map(|p| p.len() as u32).collect();

        let pattern_matcher = DoubleArrayAhoCorasickBuilder::new()
            .match_kind(daachorse::MatchKind::Standard)
            .build(patterns)
            .expect("Failed to build Aho-Corasick automaton from vocab");

        Self {
            pattern_matcher,
            canonical_to_hf,
            token_byte_lengths,
            pair_lookup,
            split_table,
        }
    }

    /// Per-component heap footprint of the engine. `capacity`-based for the owned containers;
    /// exact for the Aho-Corasick automaton via daachorse's `heap_bytes`.
    pub(crate) fn memory_breakdown(&self) -> Vec<(&'static str, usize)> {
        let pair_slot = std::mem::size_of::<(CanonicalTokenId, CanonicalTokenId)>()
            + std::mem::size_of::<CanonicalTokenId>();
        vec![
            (
                "token_byte_lengths",
                self.token_byte_lengths.capacity() * std::mem::size_of::<u32>(),
            ),
            ("pair_lookup", self.pair_lookup.capacity() * pair_slot),
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

    /// Encodes one coverable byte run into `word`. Builds the byte-position-indexed suffix
    /// array, then backtracks from the end — all in per-thread reusable scratch, so after
    /// warmup the hot path allocates nothing (only `word.add`, whose buffer is pre-sized).
    ///
    /// `last_token[p - 1]` is the token ending at byte position `p` in the unique valid
    /// tokenization of `sequence_bytes[..p]`. Indexing by byte position (not push count) lets
    /// multi-byte atoms — e.g. "Ġ" spans two bytes — not stall the scan at a position where no
    /// token ends. `reachable[p]` marks positions an encoding can reach.
    pub(crate) fn encode_run(&self, sequence_bytes: &[u8], word: &mut Word) {
        let n = sequence_bytes.len();
        SCRATCH.with(|cell| {
            let mut guard = cell.borrow_mut();
            let scratch = &mut *guard;

            // `last_token` is guarded by `reachable`, so stale entries need not be cleared;
            // `reachable` must be reset to all-false (except position 0).
            scratch.last_token.resize(n, CanonicalTokenId(0));
            scratch.reachable.clear();
            scratch.reachable.resize(n + 1, false);
            scratch.reachable[0] = true;

            for matched_pattern in self.pattern_matcher.find_overlapping_iter(sequence_bytes) {
                let start = matched_pattern.start();
                let end = matched_pattern.end();
                // Skip if the prefix before this match isn't reachable, or we already recorded
                // the token ending here (first valid match wins — greedy BPE's output is unique).
                if !scratch.reachable[start] || scratch.reachable[end] {
                    continue;
                }
                let canonical_token_id = matched_pattern.value();
                if start == 0
                    || self.is_valid_token_pair(scratch.last_token[start - 1], canonical_token_id)
                {
                    scratch.last_token[end - 1] = canonical_token_id;
                    scratch.reachable[end] = true;
                }
            }

            scratch.reversed.clear();
            let mut idx = n;
            while idx > 0 {
                let canonical_token = scratch.last_token[idx - 1];
                let token_length = self
                    .get_token_byte_length(canonical_token)
                    .expect("Defined by construction");
                let hf_token = self.canonical_to_hf[canonical_token.0 as usize];
                scratch.reversed.push((hf_token, token_length));
                idx -= token_length;
            }
            for &(token, byte_length) in scratch.reversed.iter().rev() {
                word.add(token.0, byte_length);
            }
        });
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
        self.token_byte_lengths
            .get(token.0 as usize)
            .map(|&len| len as usize)
    }

    /// Whether `left` and `right` merge into a single token, and which — a direct lookup of the
    /// merge rules in canonical-id space (no byte hashing).
    fn get_merged(
        &self,
        left: CanonicalTokenId,
        right: CanonicalTokenId,
    ) -> Option<CanonicalTokenId> {
        self.pair_lookup.get(&(left, right)).copied()
    }

    /// Renders a canonical token as `id(lenN)` for trace output. Only used when [`TRACE_SEAM`]
    /// is on (token bytes are no longer stored, so the trace shows id + byte length).
    #[allow(dead_code)]
    fn dbg_tok(&self, token: CanonicalTokenId) -> String {
        format!("{}(len{})", token.0, self.get_token_byte_length(token).unwrap_or(0))
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
