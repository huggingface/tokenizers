//! Immutable corpus planning, checkpoint seeking and deferred materialization.
use super::super::{
    IdentityPolicy, WORD_SEPARATOR_ID,
    pair_index::pair_key,
    storage::IntervalIndex,
    vocabulary::{InitialTokenIds, Vocabulary},
};
use super::{Corpus, InitialPairSource, SlotStorage};
use crate::progress::TrainingProgress;
use crate::trainers::bpe::word_counts::WordCountsView;
use compact_str::CompactString;
use rayon::prelude::*;
use std::{
    mem::{ManuallyDrop, MaybeUninit},
    ops::{ControlFlow, Range},
    sync::atomic::AtomicU32,
};
use tk_encode::Result;

// A word keeps its original byte coordinates and its measured global start.
// Initial pair construction borrows this plan; no mutable slot allocation exists
// until all raw records have retired.
struct PlannedWord<'input> {
    word: &'input CompactString,
    start: u64,
}
struct SymbolCheckpoint {
    slot_position: usize,
    byte_offset: usize,
}
pub(in super::super) struct CorpusPlan<'input> {
    words: Vec<PlannedWord<'input>>,
    checkpoints: Vec<SymbolCheckpoint>,
    initial_ids: InitialTokenIds,
    compact_keys: bool,
    len: usize,
    weights: IntervalIndex<u64>,
    unit_weight: Option<(u64, u64)>,
    spans_by_id: Vec<u64>,
    scan_whole_words: bool,
    edges: usize,
    weighted_mass: u128,
}
impl InitialPairSource for &CorpusPlan<'_> {
    fn len(&self) -> usize {
        self.len
    }
    fn word_weights(&self) -> &IntervalIndex<u64> {
        &self.weights
    }
    fn compact_keys(&self) -> bool {
        self.compact_keys
    }
    fn bounded_initial_ids(&self) -> Option<Vec<u32>> {
        // Require the resolver's conservative ID bound to fit compact keys first.
        // Then IDs activated by retained input symbols define the small alphabet,
        // including affix aliases. Unobserved entries have no initial span.
        if !self.compact_keys {
            return None;
        }
        let mut ids = Vec::new();
        for (id, &span) in self.spans_by_id.iter().enumerate() {
            if span != 0 {
                if ids.len() == 256 {
                    return None;
                }
                ids.push(id as u32);
            }
        }
        Some(ids)
    }
    fn bounded_edge_count(&self, range: Range<usize>) -> Option<usize> {
        Some(
            if range.start == 0 && range.end == self.len.saturating_sub(1) {
                self.edges
            } else {
                self.edge_count(range)
            },
        )
    }
    fn edge_count(&self, range: Range<usize>) -> usize {
        // The plan has already measured retained symbols. Each word owns its
        // left endpoints through the penultimate symbol, including wave cuts.
        let first = self
            .words
            .partition_point(|word| word.start <= range.start as u64)
            .saturating_sub(1);
        let mut count = 0;
        for index in first..self.words.len() {
            let start = self.words[index].start as usize;
            if start >= range.end {
                break;
            }
            let after = self
                .words
                .get(index + 1)
                .map_or(self.len, |word| word.start as usize);
            let end = after.saturating_sub(2).max(start);
            count += end.min(range.end).saturating_sub(start.max(range.start));
        }
        count
    }
    fn for_each_edge(&self, range: Range<usize>, mut emit: impl FnMut(usize, u64)) {
        let first = self
            .words
            .partition_point(|word| word.start <= range.start as u64)
            .saturating_sub(1);
        for index in first..self.words.len() {
            let planned = &self.words[index];
            let word_start = planned.start as usize;
            if word_start >= range.end {
                break;
            }
            let separator = self
                .words
                .get(index + 1)
                .map_or(self.len, |word| word.start as usize)
                - 1;
            if separator <= range.start {
                continue;
            }
            let (mut position, byte_start) = self.scan_start(index, range.start);
            let mut previous = None;
            self.initial_ids
                .scan_symbols(planned.word, byte_start, |id| {
                    if let Some((left_position, left)) = previous
                        && left_position >= range.start
                        && left_position < range.end
                    {
                        emit(left_position, pair_key((left, id)));
                    }
                    previous = Some((position, id));
                    // The right endpoint at range.end has supplied the lookahead.
                    if position >= range.end {
                        return ControlFlow::Break(());
                    }
                    position += 1;
                    ControlFlow::Continue(())
                });
        }
    }
}
/// Symbol counts and seek anchors share one measurement of long UTF-8 words.
/// Byte chunks stay private; planning only consumes lengths and global checkpoints.
struct WordMeasure {
    lengths: Vec<usize>,
    chunks: Vec<(usize, Range<usize>)>,
    counts: Vec<usize>,
}
impl WordMeasure {
    fn new(words: &[(&CompactString, u64)], initial_ids: &InitialTokenIds) -> Self {
        // Long words are measured by independent UTF-8 byte chunks. Retain
        // these counts for checkpoints instead of rescanning their characters.
        const CHECKPOINT_BYTES: usize = 4096;
        let mut byte_chunks = Vec::new();
        for (index, (word, _)) in words.iter().enumerate() {
            if word.len() <= CHECKPOINT_BYTES {
                continue;
            }
            for byte in (0..word.len()).step_by(CHECKPOINT_BYTES) {
                let mut begin = byte;
                while !word.is_char_boundary(begin) {
                    begin += 1;
                }
                if begin == word.len() {
                    break;
                }
                let mut end = (byte + CHECKPOINT_BYTES).min(word.len());
                while !word.is_char_boundary(end) {
                    end += 1;
                }
                byte_chunks.push((index, begin..end));
            }
        }
        let counts: Vec<usize> = byte_chunks
            .par_iter()
            .map(|(index, range)| initial_ids.symbol_count(&words[*index].0[range.clone()]))
            .collect();
        let mut lengths: Vec<usize> = words
            .par_iter()
            .map(|(word, _)| {
                if word.len() <= CHECKPOINT_BYTES {
                    initial_ids.symbol_count(word)
                } else {
                    0
                }
            })
            .collect();
        for ((index, _), count) in byte_chunks.iter().zip(&counts) {
            lengths[*index] += count;
        }
        Self {
            lengths,
            chunks: byte_chunks,
            counts,
        }
    }
    fn checkpoints(self, words: &[PlannedWord<'_>]) -> Vec<SymbolCheckpoint> {
        // Ordered per-word prefixes restore global anchors, including repeated
        // positions across byte chunks whose characters were all filtered out.
        let mut checkpoints = Vec::new();
        let mut current_word = None;
        let mut retained = 0;
        for ((index, range), count) in self.chunks.into_iter().zip(self.counts) {
            if current_word != Some(index) {
                current_word = Some(index);
                retained = 0;
            }
            if range.start != 0 {
                checkpoints.push(SymbolCheckpoint {
                    slot_position: words[index].start as usize + retained,
                    byte_offset: range.start,
                });
            }
            retained += count;
        }
        checkpoints
    }
}

impl<'input> CorpusPlan<'input> {
    pub(in super::super) fn build(
        word_counts: WordCountsView<'input>,
        vocabulary: &mut Vocabulary,
        policy: IdentityPolicy,
        length_limited: bool,
        progress: &TrainingProgress,
    ) -> Result<Self> {
        let work = progress.stage("Resolve initial IDs", word_counts.len());
        let initial_ids = vocabulary.initial_ids(word_counts, &work)?;
        let mut words: Vec<_> = word_counts
            .iter()
            .map(|(word, &weight)| (word, weight))
            .collect();
        let work = progress.stage("Arrange weighted words", words.len());
        words.par_sort_unstable_by(|left, right| right.1.cmp(&left.1));
        work.complete(words.len());
        let work = progress.stage("Measure corpus", words.len());
        let measured = WordMeasure::new(&words, &initial_ids);
        work.complete(words.len());
        let mut word_starts = Vec::with_capacity(words.len());
        let mut interval_starts = Vec::new();
        let mut interval_weights = Vec::new();
        let mut slots = 1_usize;
        let mut edges = 0_usize;
        let mut weighted_mass = 0_u128;
        let mut maximum_weight = 0_u64;
        for ((_, weight), &symbols) in words.iter().zip(&measured.lengths) {
            weighted_mass += u128::from(*weight) * symbols.saturating_sub(1) as u128;
            maximum_weight = maximum_weight.max(*weight);
            word_starts.push(slots as u64);
            if interval_weights.last() != Some(weight) {
                interval_starts.push(slots as u64);
                interval_weights.push(*weight);
            }
            slots = slots
                .checked_add(symbols)
                .and_then(|size| size.checked_add(1))
                .ok_or("BPE corpus exceeds resident index bounds")?;
            edges = edges
                .checked_add(symbols.saturating_sub(1))
                .ok_or("BPE edge count exceeds usize")?;
        }
        if slots > isize::MAX as usize / std::mem::size_of::<AtomicU32>() {
            return Err("BPE corpus exceeds resident allocation bounds".into());
        }
        // Preserve the signed input bound for configurations that may reuse
        // an identity, even while their first attempt accepts fresh IDs only.
        if (policy == IdentityPolicy::AllowActiveReuse || !initial_ids.plain())
            && (weighted_mass > i64::MAX as u128 || maximum_weight > i64::MAX as u64)
        {
            return Err("BPE identity-reuse weighted edge mass or word weight exceeds i64".into());
        }
        let words: Vec<_> = words
            .into_iter()
            .zip(word_starts)
            .map(|((word, _), start)| PlannedWord { word, start })
            .collect();
        let checkpoints = measured.checkpoints(&words);
        let unit_weight = interval_weights
            .iter()
            .position(|&weight| weight == 1)
            .map(|index| {
                let start = interval_starts[index];
                let end = interval_starts
                    .get(index + 1)
                    .copied()
                    .unwrap_or(slots as u64);
                (start, end - start)
            });
        // Initial ID range determines compact-key eligibility.
        let compact_keys = initial_ids.compact_pair_keys();
        let spans_by_id = vocabulary.initial_spans();
        Ok(Self {
            words,
            checkpoints,
            initial_ids,
            compact_keys,
            len: slots,
            weights: IntervalIndex::new(interval_starts, interval_weights),
            unit_weight,
            spans_by_id,
            scan_whole_words: length_limited,
            edges,
            weighted_mass,
        })
    }
    pub(in super::super) fn initial_counts_fit_u64(&self) -> bool {
        self.weighted_mass <= u128::from(u64::MAX)
    }
    pub(in super::super) fn initial_edges(&self) -> usize {
        self.edges
    }
    pub(in super::super) fn materialize<S: SlotStorage>(
        self,
        workers: usize,
        policy: IdentityPolicy,
        progress: &TrainingProgress,
    ) -> Result<Corpus<S>> {
        let work = progress.stage("Fill corpus", self.len - 1);
        let tokens = S::from_prepared(&self, workers, &work)?;
        let word_starts = if policy == IdentityPolicy::AllowActiveReuse {
            self.words.iter().map(|word| word.start).collect()
        } else {
            Vec::new()
        };
        Ok(Corpus {
            slots: tokens,
            word_starts,
            weights: self.weights,
            unit_weight: self.unit_weight,
            spans_by_id: self.spans_by_id,
            occurrence_spans: None,
            scan_whole_words: self.scan_whole_words,
        })
    }
    fn scan_start(&self, index: usize, slot_position: usize) -> (usize, usize) {
        let word_start = self.words[index].start as usize;
        if word_start < slot_position {
            let after = self
                .checkpoints
                .partition_point(|point| point.slot_position <= slot_position);
            if let Some(point) = after.checked_sub(1).map(|index| &self.checkpoints[index])
                && point.slot_position >= word_start
            {
                return (point.slot_position, point.byte_offset);
            }
        }
        (word_start, 0)
    }
    fn for_each_word_token(
        &self,
        index: usize,
        range: Range<usize>,
        mut emit: impl FnMut(usize, u32),
    ) {
        let planned = &self.words[index];
        let separator = self
            .words
            .get(index + 1)
            .map_or(self.len, |word| word.start as usize)
            - 1;
        if range.start < separator {
            let (mut position, byte_start) = self.scan_start(index, range.start);
            self.initial_ids
                .scan_symbols(planned.word, byte_start, |id| {
                    if position >= range.end {
                        return ControlFlow::Break(());
                    }
                    if position >= range.start {
                        emit(position, id);
                    }
                    position += 1;
                    ControlFlow::Continue(())
                });
        }
        if range.contains(&separator) {
            emit(separator, WORD_SEPARATOR_ID);
        }
    }
    pub(super) fn fill_tokens<T: Send>(
        &self,
        workers: usize,
        work: &crate::progress::WorkProgress,
        make: impl Fn(usize, u32) -> T + Sync,
    ) -> Result<Vec<T>> {
        let slots = self.len;
        let words = &self.words;
        let mut tokens = Vec::<MaybeUninit<T>>::new();
        tokens
            .try_reserve_exact(slots.checked_add(1).ok_or("BPE corpus size overflow")?)
            .map_err(|_| "BPE corpus allocation failed")?;
        // PERF: Fill each slot once in its owning word job. Preinitializing the
        // whole plane would add a serial store pass before the parallel writes.
        // SAFETY: MaybeUninit elements may be uninitialized. The prefix and every
        // disjoint job region are fully written before conversion below.
        unsafe {
            tokens.set_len(slots);
        }
        tokens[0].write(make(0, WORD_SEPARATOR_ID));
        let chunk = (slots - 1).div_ceil(workers * 8).max(1);
        let mut specs = Vec::new();
        let mut start = 0;
        while start < words.len() {
            let base = words[start].start as usize;
            let after = words
                .get(start + 1)
                .map_or(slots, |word| word.start as usize);
            if after - base > chunk {
                for begin in (base..after).step_by(chunk) {
                    specs.push((begin, (begin + chunk).min(after), start, start + 1, true));
                }
                start += 1;
                continue;
            }
            let target = base + chunk;
            let mut end = words.partition_point(|word| (word.start as usize) < target);
            let last = end - 1;
            let last_after = words.get(end).map_or(slots, |word| word.start as usize);
            // Leave a crossing large word for the next iteration to split.
            if last > start && last_after - words[last].start as usize > chunk {
                end -= 1;
            }
            let after = words.get(end).map_or(slots, |word| word.start as usize);
            specs.push((base, after, start, end, false));
            start = end;
        }
        let mut jobs = Vec::with_capacity(specs.len());
        let mut remaining = &mut tokens[1..];
        for (base, after, start, end, segment) in specs {
            let (region, next) = remaining.split_at_mut(after - base);
            remaining = next;
            jobs.push((base, start, end, segment, region));
        }
        debug_assert!(remaining.is_empty());
        jobs.into_par_iter()
            .for_each(|(base, start, end, segment, region)| {
                if segment {
                    let mut written = 0;
                    self.for_each_word_token(start, base..base + region.len(), |position, id| {
                        region[position - base].write(make(position, id));
                        written += 1;
                    });
                    debug_assert_eq!(written, region.len());
                    work.complete(region.len());
                    return;
                }
                let mut position = 0;
                for planned in &words[start..end] {
                    self.initial_ids.scan_symbols(planned.word, 0, |id| {
                        region[position].write(make(base + position, id));
                        position += 1;
                        ControlFlow::Continue(())
                    });
                    // Empty filtered words also own one initialized separator.
                    region[position].write(make(base + position, WORD_SEPARATOR_ID));
                    position += 1;
                }
                debug_assert_eq!(position, region.len());
                work.complete(region.len());
            });
        let mut tokens = ManuallyDrop::new(tokens);
        // SAFETY: All word jobs joined after writing every measured token and
        // separator. MaybeUninit<T> has the same layout as T;
        // the new vector takes sole ownership of the allocation.
        let tokens = unsafe {
            Vec::from_raw_parts(
                tokens.as_mut_ptr().cast::<T>(),
                tokens.len(),
                tokens.capacity(),
            )
        };
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ahash::AHashMap;
    #[test]
    fn bounded_domain_counts_resolved_affixes_aliases_and_observed_symbols() {
        use crate::trainers::bpe::BpeTrainer;
        use tk_encode::vocab::bucket_added_vocabulary::AddedToken;
        for (prefix, suffix, letters) in [
            (false, false, 256),
            (true, false, 128),
            (false, true, 128),
            (true, true, 85),
        ] {
            for extra in [false, true] {
                let first = char::from_u32(0x3400).unwrap();
                let mut words: AHashMap<CompactString, u64> = (0..letters)
                    .map(|i| {
                        let ch = char::from_u32(0x3400 + i).unwrap();
                        (
                            ch.to_string()
                                .repeat(if prefix && suffix { 3 } else { 2 })
                                .into(),
                            0,
                        )
                    })
                    .collect();
                if prefix && suffix {
                    words.insert(first.to_string().into(), 0);
                }
                if extra {
                    words.insert(
                        char::from_u32(0x3400 + letters).unwrap().to_string().into(),
                        0,
                    );
                }
                let mut builder = BpeTrainer::builder()
                    .vocab_size(2048)
                    .initial_alphabet(['\u{e000}'].into_iter().collect())
                    .show_progress(false);
                if prefix {
                    builder = builder.continuing_subword_prefix("##".into());
                }
                if suffix {
                    builder = builder.end_of_word_suffix("</w>".into());
                }
                // Reuse a reserved decorated ID without changing the symbol count.
                if prefix {
                    builder =
                        builder.special_tokens(vec![AddedToken::from(format!("##{first}"), true)]);
                }
                let trainer = builder.build();
                let progress = TrainingProgress::new(false, trainer.progress_format).unwrap();
                let mut retained = None;
                let mut vocabulary = Vocabulary::initialize(
                    &trainer,
                    WordCountsView::from_map(&words),
                    1,
                    &progress,
                    &mut retained,
                )
                .unwrap();
                let plan = CorpusPlan::build(
                    WordCountsView::from_map(&words),
                    &mut vocabulary,
                    IdentityPolicy::FirstActivationOnly,
                    false,
                    &progress,
                )
                .unwrap();
                let actual: Vec<_> = plan
                    .spans_by_id
                    .iter()
                    .enumerate()
                    .filter_map(|(id, &span)| (span != 0).then_some(id as u32))
                    .collect();
                assert_eq!(actual.len(), if extra { 257 } else { 256 });
                assert_eq!(
                    (&plan).bounded_initial_ids(),
                    (!extra).then_some(actual.clone())
                );
                let emitted = plan
                    .fill_tokens(1, &progress.stage("test", plan.len - 1), |_, id| id)
                    .unwrap();
                assert!(
                    emitted
                        .into_iter()
                        .filter(|&id| id != WORD_SEPARATOR_ID)
                        .all(|id| actual.binary_search(&id).is_ok())
                );
            }
        }
    }

    #[test]
    fn initial_id_bounds_preserve_full_ids_and_collector_admission() {
        use crate::trainers::bpe::BpeTrainer;
        use tk_encode::vocab::bucket_added_vocabulary::AddedToken;
        let words = [(CompactString::from("ab"), 1)].into_iter().collect();
        for reserved in [65_535, 65_536] {
            let trainer = BpeTrainer::builder()
                .vocab_size(reserved + 2)
                .special_tokens(
                    (0..reserved)
                        .map(|index| {
                            AddedToken::from(
                                if index == 7 {
                                    "b".to_owned()
                                } else {
                                    format!("reserved:{index}")
                                },
                                true,
                            )
                        })
                        .collect(),
                )
                .show_progress(false)
                .build();
            let progress = TrainingProgress::new(false, trainer.progress_format).unwrap();
            let mut retained = None;
            let mut vocabulary = Vocabulary::initialize(
                &trainer,
                WordCountsView::from_map(&words),
                1,
                &progress,
                &mut retained,
            )
            .unwrap();
            let plan = CorpusPlan::build(
                WordCountsView::from_map(&words),
                &mut vocabulary,
                IdentityPolicy::FirstActivationOnly,
                false,
                &progress,
            )
            .unwrap();
            assert_eq!(
                (&plan).bounded_initial_ids(),
                (reserved == 65_535).then(|| vec![7, reserved as u32])
            );
            let mut edges = Vec::new();
            (&plan).for_each_edge(0..3, |position, key| edges.push((position, key)));
            assert_eq!(edges, [(1, pair_key((reserved as u32, 7)))]);
            let mut lookahead = Vec::new();
            (&plan).for_each_edge(1..2, |position, key| lookahead.push((position, key)));
            assert_eq!(lookahead, edges);
            super::super::super::tests::check(&trainer, &words);
            let expected = [WORD_SEPARATOR_ID, reserved as u32, 7, WORD_SEPARATOR_ID];
            assert_eq!(
                plan.fill_tokens(1, &progress.stage("test", 3), |_, id| id)
                    .unwrap(),
                expected
            );
        }
    }

    #[test]
    fn deferred_symbols_preserve_multiword_filtered_and_boundary_ranges() {
        use crate::trainers::bpe::{BpeTrainer, word_counts::WordCountsView};
        use std::collections::HashSet;
        use tk_encode::vocab::bucket_added_vocabulary::AddedToken;

        let mut entries = vec![
            ("ab".to_owned(), 10_000),
            ("baba".to_owned(), 9_000),
            ("aa".to_owned(), 8_000),
            ("bb".to_owned(), 7_000),
            ("".to_owned(), 6_000),
            ("x".to_owned(), 5_000),
        ];
        entries.push(("abab".repeat(1500), 1));
        entries.extend((0..32).map(|index| {
            let filtered = char::from_u32(0x4e00 + index).unwrap();
            (format!("x{filtered}"), 100 - u64::from(index))
        }));
        let words: AHashMap<CompactString, u64> = entries
            .into_iter()
            .map(|(word, weight)| (CompactString::from(word), weight))
            .collect();

        // Real single-character IDs sit at both ends of the compact domain;
        // all other reserved entries are multicharacter strings and cannot
        // become emitted initial IDs. Alphabet truncation filters x and CJK.
        let reserved = usize::from(u16::MAX) + 1;
        let trainer = BpeTrainer::builder()
            .vocab_size(reserved)
            .special_tokens(
                (0..reserved)
                    .map(|index| {
                        AddedToken::from(
                            match index {
                                0 => "a".to_owned(),
                                index if index == usize::from(u16::MAX) => "b".to_owned(),
                                _ => format!("reserved:{index}"),
                            },
                            true,
                        )
                    })
                    .collect(),
            )
            .limit_alphabet(2)
            .initial_alphabet(HashSet::from(['a', 'b']))
            .show_progress(false)
            .build();
        let progress = TrainingProgress::new(false, trainer.progress_format).unwrap();
        let mut retained_alphabet = None;
        let mut vocabulary = Vocabulary::initialize(
            &trainer,
            WordCountsView::from_map(&words),
            1,
            &progress,
            &mut retained_alphabet,
        )
        .unwrap();
        let plan = CorpusPlan::build(
            WordCountsView::from_map(&words),
            &mut vocabulary,
            IdentityPolicy::FirstActivationOnly,
            false,
            &progress,
        )
        .unwrap();

        let scanner_tokens = plan
            .fill_tokens(
                1,
                &progress.stage("Scanner fixture", plan.len - 1),
                |position, id| (position, id),
            )
            .unwrap();
        assert!(scanner_tokens.iter().any(|&(_, id)| id == 0));
        assert!(scanner_tokens.iter().any(|&(_, id)| id == u16::MAX as u32));
        assert!(
            scanner_tokens
                .iter()
                .all(|&(_, id)| [WORD_SEPARATOR_ID, 0, u16::MAX as u32].contains(&id))
        );

        let collect = |plan: &CorpusPlan<'_>, range: Range<usize>| {
            let expected = plan.edge_count(range.clone());
            let mut edges = Vec::new();
            plan.for_each_edge(range, |position, key| edges.push((position, key)));
            assert_eq!(edges.len(), expected);
            edges
        };
        let ab_start = plan.words[0].start as usize;
        assert_eq!(plan.words[0].word.as_str(), "ab");
        assert_eq!(plan.words[1].word.as_str(), "baba");
        let baba_start = plan
            .words
            .iter()
            .find(|word| word.word.as_str() == "baba")
            .unwrap()
            .start as usize;
        let ranges = [
            0..plan.len,
            baba_start + 2..baba_start + 3,
            baba_start + 1..baba_start + 3,
            ab_start..ab_start + 1,
        ];
        let scanner_edges: Vec<_> = ranges
            .iter()
            .map(|range| collect(&plan, range.clone()))
            .collect();
        assert_eq!(
            scanner_edges[3],
            [(ab_start, pair_key((0, u16::MAX as u32)))]
        );
        assert_eq!(
            scanner_edges[1],
            [(baba_start + 2, pair_key((u16::MAX as u32, 0)))]
        );

        assert!((&plan).compact_keys());
        assert_eq!(
            (&plan).bounded_initial_ids(),
            Some(vec![0, u16::MAX as u32])
        );
        let parallel_tokens = plan
            .fill_tokens(
                4,
                &progress.stage("Parallel fixture", plan.len - 1),
                |position, id| (position, id),
            )
            .unwrap();
        assert_eq!(parallel_tokens, scanner_tokens);
        for (range, actual) in ranges.iter().zip(scanner_edges) {
            let expected: Vec<_> = range
                .clone()
                .filter_map(|position| {
                    let left = scanner_tokens[position].1;
                    let right = scanner_tokens.get(position + 1)?.1;
                    (left != WORD_SEPARATOR_ID && right != WORD_SEPARATOR_ID)
                        .then_some((position, pair_key((left, right))))
                })
                .collect();
            assert_eq!(actual, expected);
        }
    }
}
