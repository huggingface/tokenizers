//! Stable token endpoints over one fixed slot plane.
//!
//! A live token retains its ID at its first and last slots. Its span locates the
//! next token; the preceding endpoint locates the previous token. Rewrites never
//! shift a word's suffix. Unequal identity reuse materializes occurrence spans
//! before changing the corpus, while immutable boundaries retain word identity.
use super::storage::{IntervalCursor, IntervalIndex};
mod prepare;
mod slots;
use super::WORD_SEPARATOR_ID;
#[cfg(test)]
use crate::progress::TrainingProgress;
pub(super) use prepare::CorpusPlan;
pub(super) use slots::{PackedU24Slots, SlotStorage, U16Slots, U32Slots, slot_bits};
use std::ops::Range;
#[cfg(test)]
use std::sync::atomic::{AtomicU32, Ordering};
use tk_encode::models::bpe::Pair;

pub(super) struct Corpus<S: SlotStorage = U32Slots> {
    slots: S,
    word_starts: Vec<u64>,
    weights: IntervalIndex<u64>,
    unit_weight: Option<(u64, u64)>,
    spans_by_id: Vec<u64>,
    occurrence_spans: Option<Vec<u64>>,
    scan_whole_words: bool,
}
#[cfg(test)]
pub(super) struct InitialCorpus<'a> {
    pub(super) token_ids: &'a [AtomicU32],
    pub(super) word_weights: &'a IntervalIndex<u64>,
}
/// Initial routing visits complete keys in ascending physical-coordinate order.
/// A range owns left endpoints; its final edge may read one token past the range.
/// During a build, repeated scans of a range emit the same (position, key)
/// sequence, and `edge_count` returns its length. A true `compact_keys` result
/// guarantees that both emitted IDs fit in u16.
pub(super) trait InitialPairSource: Sync {
    fn len(&self) -> usize;
    fn word_weights(&self) -> &IntervalIndex<u64>;
    fn for_each_edge(&self, range: Range<usize>, emit: impl FnMut(usize, u64));
    fn edge_count(&self, range: Range<usize>) -> usize {
        let mut count = 0;
        self.for_each_edge(range, |_, _| count += 1);
        count
    }
    /// True only when both IDs in every emitted pair fit in sixteen bits.
    /// Unknown sources keep the full-width path.
    fn compact_keys(&self) -> bool {
        false
    }
    /// Sorted distinct initial IDs covering every symbol in emitted edges.
    /// Unknown sources keep generic records. IDs are original vocabulary IDs,
    /// not narrowed ordinals; unused symbols may conservatively be included.
    fn bounded_initial_ids(&self) -> Option<Vec<u32>> {
        None
    }
    /// Exact range cardinality from retained word geometry, without a symbol
    /// scan. Unknown sources decline cost admission and use generic records.
    fn bounded_edge_count(&self, _range: Range<usize>) -> Option<usize> {
        None
    }
}
#[cfg(test)]
impl InitialPairSource for InitialCorpus<'_> {
    fn len(&self) -> usize {
        self.token_ids.len()
    }
    fn word_weights(&self) -> &IntervalIndex<u64> {
        self.word_weights
    }
    fn compact_keys(&self) -> bool {
        self.token_ids.iter().all(|id| {
            let id = id.load(Ordering::Relaxed);
            id == WORD_SEPARATOR_ID || id <= u16::MAX as u32
        })
    }
    fn for_each_edge(&self, range: Range<usize>, mut emit: impl FnMut(usize, u64)) {
        for position in range {
            let left = self.token_ids[position].load(Ordering::Relaxed);
            let right = self.token_ids[position + 1].load(Ordering::Relaxed);
            if left != WORD_SEPARATOR_ID && right != WORD_SEPARATOR_ID {
                emit(position, super::pair_index::pair_key((left, right)));
            }
        }
    }
}
#[cfg(test)]
impl<S: SlotStorage> InitialPairSource for &Corpus<S> {
    fn len(&self) -> usize {
        self.slots.len()
    }
    fn word_weights(&self) -> &IntervalIndex<u64> {
        &self.weights
    }
    fn compact_keys(&self) -> bool {
        (0..self.slots.len()).all(|position| {
            let id = self.slots.load(position);
            id == WORD_SEPARATOR_ID || id <= u16::MAX as u32
        })
    }
    fn for_each_edge(&self, range: Range<usize>, mut emit: impl FnMut(usize, u64)) {
        for position in range {
            let left = self.slots.load(position);
            let right = self.slots.load(position + 1);
            if left != WORD_SEPARATOR_ID && right != WORD_SEPARATOR_ID {
                emit(position, super::pair_index::pair_key((left, right)));
            }
        }
    }
}
#[derive(Clone, Copy)]
pub(super) struct PairMatch {
    pub(super) left_start: u64,
    pub(super) right_start: u64,
    pub(super) next_start: u64,
    pub(super) merged_span: u64,
}
pub(super) struct WordWeightCursor<'a> {
    intervals: IntervalCursor<'a, u64>,
    uniform_weight: Option<u64>,
    unit_weight: Option<(u64, u64)>,
}
impl WordWeightCursor<'_> {
    #[inline]
    pub(super) fn weight(&mut self, position: u64) -> u64 {
        if let Some(weight) = self.uniform_weight {
            return weight;
        }
        // PERF: Weight ordering gives one complete range for weight one. The
        // original trainer checks it before touching the general cursor.
        if self
            .unit_weight
            .is_some_and(|(start, length)| position.wrapping_sub(start) < length)
        {
            return 1;
        }
        *self
            .intervals
            .get(position)
            .expect("a matched edge belongs to a word")
    }
}
impl<S: SlotStorage> Corpus<S> {
    #[cfg(test)]
    pub(super) fn initial_view(&self) -> impl InitialPairSource + '_ {
        self
    }
    pub(super) fn len(&self) -> usize {
        self.slots.len()
    }
    #[inline]
    pub(super) fn token(&self, position: u64) -> u32 {
        self.slots.load(position as usize)
    }
    #[inline]
    pub(super) fn span(&self, position: u64) -> u64 {
        match &self.occurrence_spans {
            Some(spans) => spans[position as usize],
            None => self.spans_by_id[self.token(position) as usize],
        }
    }
    pub(super) fn span_by_id(&self, id: u32) -> u64 {
        self.spans_by_id[id as usize]
    }
    pub(super) fn matcher(&self, pair: Pair) -> PairMatcher<'_, S> {
        PairMatcher {
            corpus: self,
            pair,
            left_span: self.spans_by_id[pair.0 as usize],
            right_span: self.spans_by_id[pair.1 as usize],
        }
    }
    pub(super) fn word_containing(&self, position: u64) -> usize {
        self.word_starts.partition_point(|&start| start <= position) - 1
    }
    pub(super) fn word_end(&self, word: usize) -> u64 {
        self.word_starts
            .get(word + 1)
            .copied()
            .unwrap_or(self.len() as u64)
    }
    pub(super) fn word_start(&self, word: usize) -> u64 {
        self.word_starts[word]
    }

    pub(super) fn weight_cursor(&self) -> WordWeightCursor<'_> {
        WordWeightCursor {
            intervals: self.weights.cursor(),
            uniform_weight: match self.weights.values() {
                [weight] => Some(*weight),
                _ => None,
            },
            unit_weight: self.unit_weight,
        }
    }
    pub(super) fn needs_word_scan(&self) -> bool {
        self.scan_whole_words
    }
    pub(super) fn prepare_spans(&mut self, pair: Pair, replacement: u32, reused_active: bool) {
        self.scan_whole_words |= reused_active;
        let span = self.spans_by_id[pair.0 as usize] + self.spans_by_id[pair.1 as usize];
        if replacement as usize == self.spans_by_id.len() {
            self.spans_by_id.push(if self.occurrence_spans.is_some() {
                0
            } else {
                span
            });
        } else if self.occurrence_spans.is_none() {
            let previous = self.spans_by_id[replacement as usize];
            if previous == 0 || previous == span {
                self.spans_by_id[replacement as usize] = span;
                return;
            }
            let mut spans = vec![0; self.len()];
            for &start in &self.word_starts {
                let mut position = start;
                while self.token(position) != WORD_SEPARATOR_ID {
                    let span = self.spans_by_id[self.token(position) as usize];
                    spans[position as usize] = span;
                    spans[(position + span - 1) as usize] = span;
                    position += span;
                }
            }
            self.occurrence_spans = Some(spans);
        }
    }
    #[inline]
    /// # Safety
    /// No token readers may overlap this joined write phase. The prepared
    /// geometry must own disjoint endpoint slots across all active writers.
    pub(super) unsafe fn write_endpoints(&self, matched: PairMatch, replacement: u32) {
        // SAFETY: the caller supplies the write-phase and ownership proof.
        unsafe {
            write_endpoints(&self.slots, matched, replacement);
        }
    }
    pub(super) fn word_writers(
        &mut self,
        regions: &[std::ops::Range<u64>],
    ) -> Option<Vec<WordWriter<'_, S>>> {
        let spans = self.occurrence_spans.as_mut()?;
        let mut remaining = spans.as_mut_slice();
        let mut writers = Vec::with_capacity(regions.len());
        for region in regions {
            let (spans, next) = remaining.split_at_mut((region.end - region.start) as usize);
            remaining = next;
            writers.push(WordWriter {
                base: region.start,
                slots: &self.slots,
                spans,
            });
        }
        Some(writers)
    }
    pub(super) fn has_occurrence_spans(&self) -> bool {
        self.occurrence_spans.is_some()
    }
    #[inline]
    pub(super) fn prefetch(&self, position: u64) {
        #[cfg(not(target_arch = "x86_64"))]
        let _ = position;
        #[cfg(target_arch = "x86_64")]
        if position < self.len() as u64 {
            // SAFETY: this address is inside the live slot allocation. Prefetch
            // does not load a value or change the phase-ordering contract.
            unsafe {
                std::arch::x86_64::_mm_prefetch(
                    self.slots.prefetch_pointer(position as usize),
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
        }
    }
}

// Prepared plans own disjoint live-token spans. Pool joins delimit the
// Relaxed stores; this helper also serves exclusive whole-word regions.
#[inline]
unsafe fn write_endpoints<S: SlotStorage>(slots: &S, matched: PairMatch, replacement: u32) {
    // SAFETY: caller guarantees a reader-free joined phase and disjoint spans.
    unsafe {
        slots.store(matched.left_start as usize, replacement);
        if matched.right_start + 1 != matched.next_start {
            slots.store(matched.right_start as usize, WORD_SEPARATOR_ID);
        }
        slots.store((matched.next_start - 1) as usize, replacement);
    }
}
/// A complete immutable word region owns its occurrence-span writes as well.
pub(super) struct WordWriter<'a, S: SlotStorage> {
    base: u64,
    slots: &'a S,
    spans: &'a mut [u64],
}
impl<S: SlotStorage> WordWriter<'_, S> {
    pub(super) fn merge(&mut self, position: u64, replacement: u32) {
        let left = (position - self.base) as usize;
        let right = left + self.spans[left] as usize;
        let after = right + self.spans[right] as usize;
        let merged = (after - left) as u64;
        // SAFETY: word_writers borrows Corpus mutably until all writers drop.
        // Their checked occurrence-span regions are disjoint; merge never reads
        // token IDs and changes only endpoints inside this writer's region.
        unsafe {
            write_endpoints(
                self.slots,
                PairMatch {
                    left_start: self.base + left as u64,
                    right_start: self.base + right as u64,
                    next_start: self.base + after as u64,
                    merged_span: merged,
                },
                replacement,
            );
        }
        if right + 1 != after {
            self.spans[right] = 0;
        }
        self.spans[left] = merged;
        self.spans[after - 1] = merged;
    }
}
/// Fixed rule geometry is cached once. Unequal identity reuse reads its
/// occurrence plane from the same immutable snapshot instead.
pub(super) struct PairMatcher<'a, S: SlotStorage> {
    corpus: &'a Corpus<S>,
    pair: Pair,
    left_span: u64,
    right_span: u64,
}
impl<S: SlotStorage> PairMatcher<'_, S> {
    #[inline]
    pub(super) fn geometry(&self, left_start: u64) -> PairMatch {
        let left_span = self
            .corpus
            .occurrence_spans
            .as_ref()
            .map_or(self.left_span, |spans| spans[left_start as usize]);
        let right_start = left_start + left_span;
        let right_span = self
            .corpus
            .occurrence_spans
            .as_ref()
            .map_or(self.right_span, |spans| spans[right_start as usize]);
        PairMatch {
            left_start,
            right_start,
            next_start: right_start + right_span,
            merged_span: left_span + right_span,
        }
    }
    #[inline]
    pub(super) fn get(&self, left_start: u64) -> Option<PairMatch> {
        if self.corpus.token(left_start) != self.pair.0 {
            return None;
        }
        let left_span = self
            .corpus
            .occurrence_spans
            .as_ref()
            .map_or(self.left_span, |spans| spans[left_start as usize]);
        let right_start = left_start + left_span;
        if left_span == 0
            || right_start >= self.corpus.len() as u64
            || self.corpus.token(right_start) != self.pair.1
        {
            return None;
        }
        let right_span = self
            .corpus
            .occurrence_spans
            .as_ref()
            .map_or(self.right_span, |spans| spans[right_start as usize]);
        let next_start = right_start + right_span;
        if right_span == 0 || next_start >= self.corpus.len() as u64 {
            return None;
        }
        Some(PairMatch {
            left_start,
            right_start,
            next_start,
            merged_span: left_span + right_span,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        IdentityPolicy,
        execution::Execution,
        initial_pairs,
        merge::{self, MergeRule},
        pair_index::{PairIndex, key_pair},
    };
    use super::*;

    #[test]
    fn unit_weight_range_preserves_boundaries_and_cursor_resets() {
        let start = 1_u64 << 32;
        let weights = IntervalIndex::new(vec![1, start, start + 4], vec![3, 1, 0]);
        let mut cursor = WordWeightCursor {
            intervals: weights.cursor(),
            uniform_weight: None,
            unit_weight: Some((start, 4)),
        };
        for (position, expected) in [
            (start, 1),
            (start + 3, 1),
            (start + 4, 0),
            (start - 1, 3),
            (start + 1, 1),
            (u64::MAX, 0),
            (1, 3),
        ] {
            assert_eq!(cursor.weight(position), expected);
        }
        for weight in [0, 2, u64::MAX] {
            let weights = IntervalIndex::new(vec![1], vec![weight]);
            let mut cursor = WordWeightCursor {
                intervals: weights.cursor(),
                uniform_weight: Some(weight),
                unit_weight: None,
            };
            for position in [1, start, u64::MAX, 2] {
                assert_eq!(cursor.weight(position), weight);
            }
        }
    }
    use super::super::storage::AllocationArena;
    use tk_encode::utils::progress::ProgressFormat;

    #[test]
    fn cohort_cohorts_keep_distinct_words_and_unequal_occurrence_spans() {
        // This synthetic identity state tests the corpus/cohort contract directly.
        // Plain concatenation cannot reuse an active ID this way.
        let mut corpus = Corpus {
            slots: [
                WORD_SEPARATOR_ID,
                0,
                1,
                2,
                WORD_SEPARATOR_ID,
                3,
                2,
                WORD_SEPARATOR_ID,
            ]
            .into_iter()
            .map(AtomicU32::new)
            .collect::<Vec<_>>(),
            word_starts: vec![1, 5],
            weights: IntervalIndex::new(vec![1, 5], vec![3, 1]),
            unit_weight: Some((5, 4)),
            spans_by_id: vec![1; 4],
            occurrence_spans: None,
            scan_whole_words: false,
        };
        let execution = Execution::new(2).unwrap();
        let arena = AllocationArena::new(2, 3);
        let progress = TrainingProgress::new(false, ProgressFormat::Silent).unwrap();
        execution.pool.install(|| {
            let initial = initial_pairs::InitialPairTable::build(
                corpus.initial_view(),
                0,
                &execution,
                &arena,
                &progress,
            )
            .unwrap();
            let mut index =
                PairIndex::from_initial_pairs(initial, IdentityPolicy::AllowActiveReuse, 2)
                    .unwrap();
            for (round, (pair, count, position, replacement, reused)) in [
                ((0, 1), 3, 1, 3, true),
                ((3, 2), 4, 1, 4, false),
                ((3, 2), 4, 5, 4, true),
            ]
            .into_iter()
            .enumerate()
            {
                let priority = index.best().unwrap();
                assert_eq!(
                    (key_pair(priority.key), priority.priority_count),
                    (pair, count)
                );
                let candidate = index.take_best();
                assert_eq!(candidate.positions.iter().collect::<Vec<_>>(), [position]);
                corpus.prepare_spans(pair, replacement, reused);
                let rules = [MergeRule { pair, replacement }];
                let (prepared, births) = merge::prepare_merges_with_births(
                    &corpus,
                    &rules,
                    &[candidate],
                    IdentityPolicy::AllowActiveReuse,
                    corpus.spans_by_id.len(),
                    usize::MAX,
                    &execution,
                    &arena,
                    1,
                    merge::MergeOptions::default(),
                )
                .unwrap();
                assert!(births.is_empty());
                let events = prepared.apply(&mut corpus);
                index
                    .commit_merges(&events, corpus.spans_by_id.len(), &execution, &arena)
                    .unwrap();
                drop(events);
                if round == 0 {
                    assert_eq!((corpus.span(1), corpus.span(5)), (2, 1));
                }
                if round == 1 {
                    assert_eq!((corpus.token(5), corpus.span(1)), (3, 3));
                }
            }
            assert_eq!((corpus.token(1), corpus.span(1)), (4, 3));
            assert_eq!((corpus.token(5), corpus.span(5)), (4, 2));
            assert!(index.best().is_none());
        });
    }

    #[test]
    fn intermediate_birth_survives_its_later_boundary_removal() {
        let mut corpus = Corpus {
            slots: [WORD_SEPARATOR_ID, 0, 0, 0, 0, WORD_SEPARATOR_ID]
                .into_iter()
                .map(AtomicU32::new)
                .collect::<Vec<_>>(),
            word_starts: vec![1],
            weights: IntervalIndex::new(vec![1], vec![1]),
            unit_weight: Some((1, 5)),
            spans_by_id: vec![1],
            occurrence_spans: None,
            scan_whole_words: true,
        };
        // Independent mainline Word semantics: the first rewrite births a
        // length-three boundary; the next rewrite removes it. Its cohort still
        // owns the word even though the final length-four boundary is gated out.
        let mut reference = super::super::super::word::Word::new();
        for _ in 0..4 {
            reference.add(0, 1);
        }
        assert_eq!(
            reference.merge(0, 0, 0, 4),
            [((0, 0), -1), ((0, 0), 1), ((0, 0), -1)]
        );
        let execution = Execution::new(2).unwrap();
        let arena = AllocationArena::new(2, 3);
        let progress = TrainingProgress::new(false, ProgressFormat::Silent).unwrap();
        execution.pool.install(|| {
            let initial = initial_pairs::InitialPairTable::build(
                corpus.initial_view(),
                0,
                &execution,
                &arena,
                &progress,
            )
            .unwrap();
            let mut index =
                PairIndex::from_initial_pairs(initial, IdentityPolicy::AllowActiveReuse, 1)
                    .unwrap();
            assert_eq!(index.best().unwrap().priority_count, 3);
            let candidate = index.take_best();
            corpus.prepare_spans((0, 0), 0, true);
            let rules = [MergeRule {
                pair: (0, 0),
                replacement: 0,
            }];
            let (prepared, births) = merge::prepare_merges_with_births(
                &corpus,
                &rules,
                &[candidate],
                IdentityPolicy::AllowActiveReuse,
                1,
                4,
                &execution,
                &arena,
                1,
                merge::MergeOptions::default(),
            )
            .unwrap();
            assert!(births.is_empty());
            let events = prepared.apply(&mut corpus);
            let changes: Vec<_> = events
                .chunks
                .iter()
                .flat_map(|chunk| chunk.changes.iter())
                .collect();
            assert_eq!(
                changes
                    .iter()
                    .map(|change| change.removed_weight)
                    .sum::<u64>(),
                2
            );
            assert_eq!(
                changes.iter().map(|change| change.born_weight).sum::<u64>(),
                1
            );
            let births: Vec<_> = events
                .chunks
                .iter()
                .flat_map(|chunk| {
                    chunk
                        .changes
                        .iter()
                        .flat_map(|change| chunk.chains.reversed(change.positions))
                })
                .collect();
            assert_eq!(births, [1]);
            index
                .commit_merges(&events, corpus.spans_by_id.len(), &execution, &arena)
                .unwrap();
            drop(events);
            assert_eq!(index.best().unwrap().priority_count, 2);
            let cohort = index.take_best();
            assert_eq!(cohort.positions.iter().collect::<Vec<_>>(), [1]);
            assert_eq!((corpus.span(1), corpus.span(3)), (2, 2));
        });
    }
}
