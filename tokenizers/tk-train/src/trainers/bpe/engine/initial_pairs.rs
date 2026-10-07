//! Initial pair counting and position construction from a read-only corpus.
//!
//! Generic collectors cache the complete pair key and a wave-local coordinate:
//! two 16-bit token IDs use eight-byte records; other keys use twelve bytes.
//! The bounded collector stores four-byte offsets per occurrence and one original
//! full key per group. Both retain incoming spatial order and share subsequent
//! frequency counting, position encoding and checked publication.
//! Adding the wave base restores the full-width global coordinate.
//! A wave is bounded to 2^28 physical slots. Its boundary does not
//! end a word or discard an edge: scanning reads the next corpus slot.
//! Each key is filtered after its complete frequency is known. Multiwave counts
//! accumulate before filtering; a single wave can filter before encoding.
mod bounded;

use super::storage::{AllocationArena, IntervalIndex, SortedPositions, radix};
use super::{corpus::InitialPairSource, execution::Execution, pair_index::PairState};
use crate::progress::{TrainingProgress, WorkProgress};
use ahash::AHashMap;
use radix::{CompactKeyedValue, KeyedValue, RadixRecord};
use rayon::prelude::*;
use std::{mem::MaybeUninit, ops::Range};
use tk_encode::Result;

pub(super) struct InitialPairTable<'arena> {
    pub(super) shards: Vec<AHashMap<u64, PairState<'arena>>>,
    pub(super) weighted_mass: u128,
    pub(super) maximum_word_weight: u64,
}
struct InitialGroup {
    begin: u32,
    end: u32,
    frequency: u64,
}
#[derive(Default)]
struct RecordBuffer<'a, R> {
    records: &'a mut [MaybeUninit<R>],
    used: usize,
}
impl<R> RecordBuffer<'_, R> {
    fn push(&mut self, record: R) {
        self.records[self.used].write(record);
        self.used += 1;
    }
    fn finish(self) {
        assert_eq!(self.used, self.records.len(), "initial pair scan changed");
    }
}
fn allocate_records<R>(count: usize) -> Vec<MaybeUninit<R>> {
    let mut records = Vec::with_capacity(count);
    // SAFETY: MaybeUninit permits uninitialized elements. Producers only receive
    // disjoint slices; initialized values are exposed after all slices finish.
    unsafe { records.set_len(count) };
    records
}
/// # Safety
/// Every counted slice must have finished, and all writers must have joined.
unsafe fn initialized_records<R>(records: Vec<MaybeUninit<R>>) -> Vec<R> {
    let mut records = std::mem::ManuallyDrop::new(records);
    // SAFETY: The caller proves initialization; MaybeUninit<R> and R have the
    // same layout, so the allocation's pointer, length and capacity are preserved.
    unsafe {
        Vec::from_raw_parts(
            records.as_mut_ptr().cast(),
            records.len(),
            records.capacity(),
        )
    }
}
struct RecordJob<'a, R> {
    range: Range<usize>,
    buffers: Vec<RecordBuffer<'a, R>>,
}
// Wave/tile limits give P <= 2^28 / 2^18 = 1024 producers. On 64-bit
// targets, count8 + RecordBuffer24 gives 32*P*W <= 32 KiB*W directory
// payload bytes; vector headers, allocator capacity and records are separate.
const ROUTE_TILE_SLOTS: usize = 1 << 18;
// Dispatch proves the chosen record can represent each emitted key and wave
// offset; construction and access then preserve both values exactly.
trait PositionRecord: Copy + Send + Sync {
    fn local_offset(self) -> u32;
}
impl PositionRecord for u32 {
    fn local_offset(self) -> u32 {
        self
    }
}
impl PositionRecord for KeyedValue {
    fn local_offset(self) -> u32 {
        self.value()
    }
}
impl PositionRecord for CompactKeyedValue {
    fn local_offset(self) -> u32 {
        self.value()
    }
}
trait InitialRecord: RadixRecord + PositionRecord {
    fn new(key: u64, value: u32) -> Self;
}
impl InitialRecord for KeyedValue {
    fn new(key: u64, value: u32) -> Self {
        KeyedValue::new(key, value)
    }
}
impl InitialRecord for CompactKeyedValue {
    fn new(key: u64, value: u32) -> Self {
        CompactKeyedValue::new(key, value)
    }
}
fn global_position<R: PositionRecord>(base: usize, record: R) -> u64 {
    // The producer bounds the offset by its wave. Both parts fit a resident
    // corpus coordinate; retaining the base keeps positions above u32 intact.
    base as u64 + u64::from(record.local_offset())
}
// Generic records carry their key; the bounded collector stores one key per
// group and only u32 offsets per occurrence. Both feed the same count, encode,
// checked multiwave append and publication lifecycle.
trait GroupKeys<R> {
    fn key(&self, records: &[R], group: usize, begin: usize) -> u64;
}
impl<R: InitialRecord> GroupKeys<R> for () {
    fn key(&self, records: &[R], _group: usize, begin: usize) -> u64 {
        records[begin].key()
    }
}
impl GroupKeys<u32> for Vec<u64> {
    fn key(&self, _records: &[u32], group: usize, _begin: usize) -> u64 {
        self[group]
    }
}
struct GroupedWave<R, K> {
    records: Vec<R>,
    groups: Vec<InitialGroup>,
    keys: K,
    mass: u128,
}

/// Collect fully initialized owner streams for one bounded wave. Count, slice
/// assignment, producer writes and their join stay within this safety boundary.
fn collect_wave_records<R: InitialRecord>(
    corpus: &impl InitialPairSource,
    range: Range<usize>,
    execution: &Execution,
    progress: &TrainingProgress,
) -> Vec<Vec<R>> {
    let workers = execution.workers();
    let router = execution.router();
    let base = range.start;
    let end = range.end;
    if workers == 1 {
        // A sole owner needs neither a key-dependent capacity scan nor
        // route directories. The immutable plan counts edges from word
        // geometry, then one symbol scan fills the exact allocation.
        let work = progress.stage("Route initial pairs", end - base);
        let count = corpus.edge_count(base..end);
        let mut records = Vec::with_capacity(count);
        corpus.for_each_edge(base..end, |position, key| {
            records.push(R::new(key, (position - base) as u32));
        });
        debug_assert_eq!(records.len(), count);
        work.complete(end - base);
        vec![records]
    } else {
        // Fixed slot tiles keep producer count independent of pool size.
        // Every producer uses the same direct-index directory at every pool size.
        let chunk = ROUTE_TILE_SLOTS;
        let ranges: Vec<_> = (base..end)
            .step_by(chunk)
            .map(|start| start..(start + chunk).min(end))
            .collect();
        let route_work = progress.stage("Route initial pairs", (end - base) * 2);
        let sizes: Vec<Vec<usize>> = ranges
            .par_iter()
            .map(|range| {
                let mut sizes = vec![0_usize; workers];
                corpus.for_each_edge(range.clone(), |_, key| {
                    sizes[router.owner(key)] += 1;
                });
                route_work.complete(range.len());
                sizes
            })
            .collect();
        let mut shard_sizes = vec![0_usize; workers];
        for counts in &sizes {
            for (total, count) in shard_sizes.iter_mut().zip(counts) {
                *total += count;
            }
        }
        // Each owner owns its record allocation and releases it as soon as
        // its position lists are complete.
        let mut record_buffers: Vec<_> = shard_sizes
            .iter()
            .map(|&count| allocate_records::<R>(count))
            .collect();
        let mut remaining: Vec<_> = record_buffers.iter_mut().map(Vec::as_mut_slice).collect();
        // Ranges are ascending. Each owner gets earlier ranges first, and each
        // producer preserves source order. Task completion cannot reorder records.
        let jobs: Vec<_> = ranges
            .into_iter()
            .zip(sizes)
            .map(|(range, counts)| {
                let mut buffers: Vec<RecordBuffer<'_, R>> =
                    (0..workers).map(|_| RecordBuffer::default()).collect();
                for (shard, count) in counts.into_iter().enumerate() {
                    if count == 0 {
                        continue;
                    }
                    let buffer = std::mem::take(&mut remaining[shard]);
                    let (records, next) = buffer.split_at_mut(count);
                    remaining[shard] = next;
                    buffers[shard] = RecordBuffer { records, used: 0 };
                }
                RecordJob { range, buffers }
            })
            .collect();
        debug_assert!(remaining.iter().all(|buffer| buffer.is_empty()));
        drop(remaining);
        jobs.into_par_iter().for_each(|mut job| {
            corpus.for_each_edge(job.range.clone(), |position, key| {
                let shard = router.owner(key);
                let buffer = &mut job.buffers[shard];
                buffer.push(R::new(key, (position - base) as u32));
            });
            job.buffers.into_iter().for_each(RecordBuffer::finish);
            route_work.complete(job.range.len());
        });
        // Each emitted offset is below records_per_wave <= 2^28. The u32
        // payload is local to this wave; the global coordinate is never narrowed.
        // SAFETY: Source scans and router ownership are repeatable. Counts
        // allocate exact cells, split_at_mut gives disjoint producer slices,
        // and all producers joined after filling their counted slices.
        // A producer panic unwinds before conversion, dropping MaybeUninit buffers.
        // MaybeUninit<R> and R have the same layout and allocation size.
        record_buffers
            .into_iter()
            .map(|records| unsafe { initialized_records(records) })
            .collect()
    }
}

/// Count stable equal-key runs and retain those admitted by this wave's floor.
/// Coordinates in each run remain sorted for the immutable interval weights.
fn count_groups<R: InitialRecord>(
    records: &[R],
    wave_base: usize,
    weights: &IntervalIndex<u64>,
    uniform_weight: Option<u64>,
    frequency_floor: u64,
    work: &WorkProgress,
) -> Result<(Vec<InitialGroup>, u128)> {
    let mut mass = 0_u128;
    let mut begin = 0;
    let mut groups = Vec::new();
    let mut completed = 0;
    while begin < records.len() {
        let key = records[begin].key();
        let mut end = begin + 1;
        while end < records.len() && records[end].key() == key {
            end += 1;
        }
        let frequency = group_frequency(&records[begin..end], wave_base, weights, uniform_weight)?;
        // Resident edges times u64 weights fit u128. Per-key counts
        // remain checked u64 values, without a global u64 mass cap.
        mass += u128::from(frequency);
        if frequency >= frequency_floor {
            groups.push(InitialGroup {
                begin: begin as u32,
                end: end as u32,
                frequency,
            });
        }
        completed += end - begin;
        if completed >= 1 << 16 {
            work.complete(completed);
            completed = 0;
        }
        begin = end;
    }
    work.complete(completed);
    Ok((groups, mass))
}

fn group_frequency<R: PositionRecord>(
    records: &[R],
    wave_base: usize,
    weights: &IntervalIndex<u64>,
    uniform_weight: Option<u64>,
) -> Result<u64> {
    if let Some(weight) = uniform_weight {
        return weight
            .checked_mul(records.len() as u64)
            .ok_or_else(|| "BPE initial pair frequency exceeds u64".into());
    }
    let mut frequency = 0_u64;
    for (count, weight) in
        weights.runs_for_sorted(records, |&record| global_position(wave_base, record))
    {
        let weight = *weight.expect("every initial edge belongs to a word interval");
        frequency = weight
            .checked_mul(count as u64)
            .and_then(|part| frequency.checked_add(part))
            .ok_or("BPE initial pair frequency exceeds u64")?;
    }
    Ok(frequency)
}

impl<'arena> InitialPairTable<'arena> {
    fn admitted_bounded_alphabet(
        corpus: &impl InitialPairSource,
        workers: usize,
        records_per_wave: usize,
    ) -> Option<bounded::Alphabet> {
        let alphabet = bounded::Alphabet::new(corpus.bounded_initial_ids()?)?;
        alphabet
            .admits_source(corpus, workers, records_per_wave)
            .then_some(alphabet)
    }

    #[cfg(test)]
    pub(super) fn admits_bounded_for_test(
        corpus: &impl InitialPairSource,
        workers: usize,
        records_per_wave: usize,
    ) -> bool {
        Self::admitted_bounded_alphabet(corpus, workers, records_per_wave).is_some()
    }

    pub(super) fn build(
        corpus: impl InitialPairSource,
        minimum_frequency: u64,
        execution: &Execution,
        arena: &'arena AllocationArena,
        progress: &TrainingProgress,
    ) -> Result<InitialPairTable<'arena>> {
        // PERF: 2^28 bounds raw records while favoring complete-count filtering in
        // one wave. Smaller waves can encode low-frequency partial runs that later
        // disappear, repeat append and table growth, and retain retired arena buffers.
        Self::build_in_waves(
            corpus,
            minimum_frequency,
            execution,
            arena,
            progress,
            1 << 28,
        )
    }

    pub(super) fn build_in_waves(
        corpus: impl InitialPairSource,
        minimum_frequency: u64,
        execution: &Execution,
        arena: &'arena AllocationArena,
        progress: &TrainingProgress,
        records_per_wave: usize,
    ) -> Result<InitialPairTable<'arena>> {
        assert!(records_per_wave > 1 && records_per_wave <= radix::MAX_RECORDS);
        InitialCollector {
            corpus,
            minimum_frequency,
            execution,
            arena,
            progress,
            records_per_wave,
        }
        .build()
    }

    /// Exercise the actual bounded collector and shared publication on tiny
    /// semantic fixtures. Production admission remains unchanged; this bypasses
    /// only its cost heuristic, retaining the source/alphabet contracts.
    #[cfg(test)]
    pub(super) fn build_bounded_for_test(
        corpus: impl InitialPairSource,
        minimum_frequency: u64,
        execution: &Execution,
        arena: &'arena AllocationArena,
        progress: &TrainingProgress,
        records_per_wave: usize,
    ) -> Result<InitialPairTable<'arena>> {
        let alphabet = bounded::Alphabet::new(
            corpus
                .bounded_initial_ids()
                .expect("forced bounded fixture supplies its actual IDs"),
        )
        .expect("forced bounded fixture has a legal alphabet");
        InitialCollector {
            corpus,
            minimum_frequency,
            execution,
            arena,
            progress,
            records_per_wave,
        }
        .bounded(alphabet)
    }
}

/// Route, count, encode and publish waves under one frequency and resource policy.
/// Collector variants supply grouped occurrences; publication stays generic.
struct InitialCollector<'work, 'arena, C> {
    corpus: C,
    minimum_frequency: u64,
    execution: &'work Execution,
    arena: &'arena AllocationArena,
    progress: &'work TrainingProgress,
    records_per_wave: usize,
}
impl<'arena, C: InitialPairSource> InitialCollector<'_, 'arena, C> {
    fn build(&self) -> Result<InitialPairTable<'arena>> {
        if let Some(alphabet) = InitialPairTable::admitted_bounded_alphabet(
            &self.corpus,
            self.execution.workers(),
            self.records_per_wave,
        ) {
            return self.bounded(alphabet);
        }
        if self.corpus.compact_keys() {
            self.keyed::<CompactKeyedValue>()
        } else {
            self.keyed::<KeyedValue>()
        }
    }

    fn bounded(&self, alphabet: bounded::Alphabet) -> Result<InitialPairTable<'arena>> {
        self.collect(|corpus, range, wave_floor, uniform_weight| {
            bounded::collect_wave(
                corpus,
                range,
                &alphabet,
                self.execution,
                self.progress,
                wave_floor,
                uniform_weight,
            )
        })
    }

    fn keyed<R: InitialRecord>(&self) -> Result<InitialPairTable<'arena>> {
        let execution = self.execution;
        let progress = self.progress;
        self.collect(|corpus, range, wave_floor, uniform_weight| {
            let mut buffers = collect_wave_records::<R>(corpus, range.clone(), execution, progress);
            let records = buffers.iter().map(Vec::len).sum();
            let sort_work = progress.stage("Sort initial pairs", records);
            buffers.par_iter_mut().for_each(|records| {
                radix::sort_by_key(records);
                sort_work.complete(records.len());
            });
            let group_work = progress.stage("Count initial pairs", records);
            buffers
                .into_par_iter()
                .map(|records| {
                    let (groups, mass) = count_groups(
                        &records,
                        range.start,
                        corpus.word_weights(),
                        uniform_weight,
                        wave_floor,
                        &group_work,
                    )?;
                    Ok(GroupedWave {
                        records,
                        groups,
                        keys: (),
                        mass,
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
    }

    fn collect<R: PositionRecord, K: GroupKeys<R> + Send>(
        &self,
        collect: impl Fn(&C, Range<usize>, u64, Option<u64>) -> Result<Vec<GroupedWave<R, K>>>,
    ) -> Result<InitialPairTable<'arena>> {
        let corpus = &self.corpus;
        let minimum_frequency = self.minimum_frequency;
        let records_per_wave = self.records_per_wave;
        assert!(records_per_wave > 1 && records_per_wave <= radix::MAX_RECORDS);
        let workers = self.execution.workers();
        let mut shards: Vec<_> = (0..workers)
            .map(|_| AHashMap::<u64, PairState<'arena>>::new())
            .collect();
        let mut weighted_mass = 0_u128;
        let maximum_word_weight = corpus
            .word_weights()
            .values()
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        let uniform_weight =
            (corpus.word_weights().values().len() == 1).then(|| corpus.word_weights().values()[0]);
        let single_wave = corpus.len() <= records_per_wave;
        let wave_floor = if single_wave {
            minimum_frequency
        } else {
            minimum_frequency.min(1)
        };
        for base in (0..corpus.len().saturating_sub(1)).step_by(records_per_wave) {
            let end = (base + records_per_wave).min(corpus.len() - 1);
            let mut wave_tables: Vec<_> = (0..workers)
                .map(|_| AHashMap::<u64, PairState<'arena>>::new())
                .collect();
            let grouped = collect(corpus, base..end, wave_floor, uniform_weight)?;
            weighted_mass += self.encode_wave(base, single_wave, &mut wave_tables, grouped)?;
            self.publish_wave(&mut shards, wave_tables)?;
        }
        if minimum_frequency != 0 {
            for shard in &mut shards {
                shard.retain(|_, state| state.ledger_count_bits >= minimum_frequency);
            }
        }
        Ok(InitialPairTable {
            shards,
            weighted_mass,
            maximum_word_weight,
        })
    }
    fn encode_wave<R: PositionRecord, K: GroupKeys<R> + Send>(
        &self,
        base: usize,
        single_wave: bool,
        tables: &mut [AHashMap<u64, PairState<'arena>>],
        grouped: Vec<GroupedWave<R, K>>,
    ) -> Result<u128> {
        let records = grouped.iter().map(|wave| wave.records.len()).sum();
        let group_work = self.progress.stage("Encode initial positions", records);
        let masses = tables
            .par_iter_mut()
            .zip(grouped.into_par_iter())
            .map(|(table, wave)| -> Result<u128> {
                let GroupedWave {
                    records,
                    groups,
                    keys,
                    mass,
                } = wave;
                let worker = self.execution.current_worker();
                let lease = self.arena.lease(worker);
                let mut scratch = self.execution.encoding(worker);
                table.reserve(groups.len());
                for (
                    group,
                    InitialGroup {
                        begin,
                        end,
                        frequency,
                    },
                ) in groups.into_iter().enumerate()
                {
                    let positions = records[begin as usize..end as usize]
                        .iter()
                        .map(|&record| global_position(base, record));
                    table.insert(
                        keys.key(&records, group, begin as usize),
                        PairState {
                            ledger_count_bits: frequency,
                            // A complete zero count has no candidate
                            // payload. A partial zero count may share
                            // its key with a positive wave elsewhere.
                            positions: if single_wave && frequency == 0 {
                                SortedPositions::new()
                            } else {
                                SortedPositions::from_sorted_iter(positions, &mut scratch, &lease)?
                            },
                        },
                    );
                }
                // Group scanning and list installation each account
                // for one pass. Publication remains visible work after
                // the complete frequencies have been counted.
                group_work.complete(records.len());
                // This owner's raw stream is released when its task returns.
                Ok(mass)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(masses.into_iter().sum())
    }
    fn publish_wave(
        &self,
        shards: &mut [AHashMap<u64, PairState<'arena>>],
        wave_tables: Vec<AHashMap<u64, PairState<'arena>>>,
    ) -> Result<()> {
        // Each wave owns exact lists before publication. Repeated keys append
        // those compressed lists with the original measure/replay lifecycle.
        // Moving an empty owner's table avoids duplicating its map allocation.
        let publish_work = self.progress.stage(
            "Publish initial pairs",
            wave_tables.iter().map(|table| table.len()).sum(),
        );
        shards
            .par_iter_mut()
            .zip(wave_tables.into_par_iter())
            .map(|(table, wave)| -> Result<()> {
                let keys = wave.len();
                if table.is_empty() {
                    *table = wave;
                } else {
                    let worker = self.execution.current_worker();
                    let lease = self.arena.lease(worker);
                    let mut scratch = self.execution.encoding(worker);
                    for (key, state) in wave {
                        use std::collections::hash_map::Entry;
                        match table.entry(key) {
                            Entry::Vacant(entry) => {
                                entry.insert(state);
                            }
                            Entry::Occupied(mut entry) => {
                                let old = entry.get_mut();
                                old.ledger_count_bits = old
                                    .ledger_count_bits
                                    .checked_add(state.ledger_count_bits)
                                    .ok_or("BPE initial pair frequency exceeds u64")?;
                                old.positions
                                    .append(state.positions, &mut scratch, &lease)?;
                            }
                        }
                    }
                }
                publish_work.complete(keys);
                Ok(())
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(())
    }
}

#[cfg(all(test, target_pointer_width = "64"))]
mod tests {
    use super::*;

    #[test]
    fn wave_offsets_restore_positions_above_u32() {
        let base = (1_usize << 32) + 17;
        let mut records = [
            KeyedValue::new(u64::MAX, 0),
            KeyedValue::new(u64::MAX, (1 << 28) - 1),
        ];
        radix::sort_by_key(&mut records);
        let arena = AllocationArena::new(1, 2);
        let lease = arena.lease(0);
        let mut scratch = super::super::storage::PositionEncodingScratch::default();
        let positions = SortedPositions::from_sorted_iter(
            records
                .into_iter()
                .map(|record| global_position(base, record)),
            &mut scratch,
            &lease,
        )
        .unwrap();
        assert_eq!(
            positions.iter().collect::<Vec<_>>(),
            [(1_u64 << 32) + 17, (1_u64 << 32) + (1 << 28) + 16]
        );
    }
}
