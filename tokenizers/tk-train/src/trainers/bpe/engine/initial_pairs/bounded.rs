//! Stable direct pair partition for a bounded initial alphabet.
//!
//! Temporary occurrence records use u32 offsets; original IDs determine keys
//! and owner shards. Corpus materialization follows initial pair construction.
use super::super::pair_index::pair_key;
use super::*;

const MAX_PRODUCERS: usize = 64;
const DIRECTORY_BUDGET: usize = 16 << 20;
// Generic compact sorting may hold raw records and equally sized scratch;
// subtract the bounded collector's u32 offsets before budgeting directories.
const DIRECTORY_BYTES_PER_EDGE: usize =
    2 * std::mem::size_of::<CompactKeyedValue>() - std::mem::size_of::<u32>();

pub(super) struct Alphabet {
    ids: Vec<u32>,
    first: u32,
    ordinals: Vec<u8>,
}
impl Alphabet {
    pub(super) fn new(ids: Vec<u32>) -> Option<Self> {
        if ids.is_empty()
            || ids.len() > 256
            || !ids.windows(2).all(|pair| pair[0] < pair[1])
            || ids.last().copied()? > u16::MAX as u32
        {
            return None;
        }
        let first = ids[0];
        let mut ordinals = vec![0; (ids[ids.len() - 1] - first) as usize + 1];
        for (ordinal, &id) in ids.iter().enumerate() {
            ordinals[(id - first) as usize] = ordinal as u8;
        }
        Some(Self {
            ids,
            first,
            ordinals,
        })
    }
    fn pairs(&self) -> usize {
        self.ids.len() * self.ids.len()
    }
    /// Admit the source against one maximum-wave directory working set. Large
    /// sources keep their bounded waves even when the final tail is very small.
    pub(super) fn admits_source(
        &self,
        corpus: &impl InitialPairSource,
        workers: usize,
        wave_slots: usize,
    ) -> bool {
        let slots = corpus.len().saturating_sub(1);
        slots != 0
            && corpus.bounded_edge_count(0..slots).is_some_and(|edges| {
                edges <= slots && self.admits_records(workers, slots.min(wave_slots), edges)
            })
    }
    fn metadata_bound(&self, workers: usize, slots: usize) -> usize {
        let pairs = self.pairs();
        let producers = producer_count(pairs, workers, slots);
        // Each producer has a fixed count row and at most n² nonempty slice
        // descriptors. Include their outer vectors/ranges during construction,
        // plus owner/key totals and the already allocated ordinal lookup. This
        // estimates bounded-specific overhead; owner-sized record/slice wrappers
        // common to the generic collector are excluded.
        pairs
            * (producers
                * (std::mem::size_of::<u32>() + std::mem::size_of::<RecordBuffer<'_, u32>>())
                + 2 * std::mem::size_of::<usize>())
            + producers
                * (std::mem::size_of::<Vec<u32>>()
                    + std::mem::size_of::<Vec<RecordBuffer<'_, u32>>>()
                    + std::mem::size_of::<Range<usize>>()
                    + std::mem::size_of::<Producer<'_>>())
            + self.ordinals.capacity() * std::mem::size_of::<u8>()
            + self.ids.capacity() * std::mem::size_of::<u32>()
    }
    fn admits_records(&self, workers: usize, slots: usize, edges: usize) -> bool {
        // Conservative sorter-working-set heuristic: subtract our offset bytes
        // from generic raw + scratch records. A constant-key sort may need no
        // scratch, so this is not a bound on every distribution or overall RSS.
        // Division avoids multiplying a possibly large edge count.
        edges
            >= self
                .metadata_bound(workers, slots)
                .div_ceil(DIRECTORY_BYTES_PER_EDGE)
    }
    #[inline]
    fn bucket(&self, key: u64) -> usize {
        let left = (key >> 32) as u32;
        let right = key as u32;
        let a = self.ordinals[(left - self.first) as usize] as usize;
        let b = self.ordinals[(right - self.first) as usize] as usize;
        debug_assert_eq!(self.ids[a], left, "source alphabet covers emitted IDs");
        debug_assert_eq!(self.ids[b], right, "source alphabet covers emitted IDs");
        a * self.ids.len() + b
    }
    fn key(&self, bucket: usize) -> u64 {
        pair_key((
            self.ids[bucket / self.ids.len()],
            self.ids[bucket % self.ids.len()],
        ))
    }
}

struct Producer<'a> {
    range: Range<usize>,
    // Count rows become bucket-to-buffer indices once slices are assigned.
    // Only nonempty buckets hold a fat slice descriptor.
    directory: Vec<u32>,
    buffers: Vec<RecordBuffer<'a, u32>>,
}

fn producer_count(pairs: usize, workers: usize, slots: usize) -> usize {
    assert!(pairs > 0 && workers > 0);
    let per_producer =
        pairs * (std::mem::size_of::<u32>() + std::mem::size_of::<RecordBuffer<'_, u32>>());
    workers
        .min(MAX_PRODUCERS)
        .min((DIRECTORY_BUDGET / per_producer).max(1))
        .min(slots.div_ceil(ROUTE_TILE_SLOTS).max(1))
}

pub(super) fn collect_wave(
    corpus: &impl InitialPairSource,
    range: Range<usize>,
    alphabet: &Alphabet,
    execution: &Execution,
    progress: &TrainingProgress,
    floor: u64,
    uniform_weight: Option<u64>,
) -> Result<Vec<GroupedWave<u32, Vec<u64>>>> {
    let base = range.start;
    let pairs = alphabet.pairs();
    let workers = execution.workers();
    let router = execution.router();
    // Count/index rows have fixed n² capacity; descriptor vectors reserve their
    // exact nonempty bucket count before insertion and never grow. At most 64
    // producers and 16 MiB of these backing allocations overlap the raw records.
    // Avoid one dense row per 2^18-slot tile of a full 2^28-slot wave.
    let producers = producer_count(pairs, workers, range.len());
    let chunk = range.len().div_ceil(producers).max(1);
    let ranges: Vec<_> = (range.start..range.end)
        .step_by(chunk)
        .map(|begin| begin..(begin + chunk).min(range.end))
        .collect();
    let work = progress.stage("Partition bounded initial pairs", range.len() * 2);
    let mut rows: Vec<Vec<u32>> = ranges
        .par_iter()
        .map(|range| {
            let mut counts = vec![0_u32; pairs];
            corpus.for_each_edge(range.clone(), |_, key| {
                counts[alphabet.bucket(key)] += 1;
            });
            work.complete(range.len());
            counts
        })
        .collect();
    let owners: Vec<_> = (0..pairs)
        .map(|bucket| router.owner(alphabet.key(bucket)))
        .collect();
    let mut totals = vec![0_usize; pairs];
    for row in &rows {
        for (total, &count) in totals.iter_mut().zip(row) {
            *total += count as usize;
        }
    }
    let mut owner_sizes = vec![0_usize; workers];
    for (bucket, &count) in totals.iter().enumerate() {
        owner_sizes[owners[bucket]] += count;
    }
    let mut records: Vec<_> = owner_sizes
        .iter()
        .map(|&count| allocate_records::<u32>(count))
        .collect();
    let mut buffers: Vec<Vec<RecordBuffer<'_, u32>>> = rows
        .iter()
        .map(|row| Vec::with_capacity(row.iter().filter(|&&count| count != 0).count()))
        .collect();
    let metadata_bytes = rows
        .iter()
        .map(|row| row.capacity() * std::mem::size_of::<u32>())
        .sum::<usize>()
        + buffers
            .iter()
            .map(|buffer| buffer.capacity() * std::mem::size_of::<RecordBuffer<'_, u32>>())
            .sum::<usize>();
    assert!(
        metadata_bytes <= DIRECTORY_BUDGET,
        "bounded pair metadata exceeds its capacity budget"
    );
    let mut remaining: Vec<_> = records.iter_mut().map(Vec::as_mut_slice).collect();
    // Ascending true IDs give ascending full keys. Inside each key, ascending
    // producer ranges get consecutive slices, independent of completion order.
    for (bucket, &total) in totals.iter().enumerate() {
        if total == 0 {
            continue;
        }
        let owner = owners[bucket];
        let owner_rest = std::mem::take(&mut remaining[owner]);
        let (mut pair_rest, owner_next) = owner_rest.split_at_mut(total);
        remaining[owner] = owner_next;
        for (row, buffers) in rows.iter_mut().zip(&mut buffers) {
            let count = row[bucket] as usize;
            if count == 0 {
                continue;
            }
            let (records, next) = pair_rest.split_at_mut(count);
            pair_rest = next;
            row[bucket] = buffers.len() as u32;
            buffers.push(RecordBuffer { records, used: 0 });
        }
        debug_assert!(pair_rest.is_empty());
    }
    debug_assert!(remaining.iter().all(|slice| slice.is_empty()));
    drop(remaining);
    ranges
        .into_iter()
        .zip(rows)
        .zip(buffers)
        .map(|((range, directory), buffers)| Producer {
            range,
            directory,
            buffers,
        })
        .collect::<Vec<_>>()
        .into_par_iter()
        .for_each(|mut producer| {
            corpus.for_each_edge(producer.range.clone(), |position, key| {
                let bucket = alphabet.bucket(key);
                let buffer = &mut producer.buffers[producer.directory[bucket] as usize];
                buffer.push((position - base) as u32);
            });
            // Repeated source scans have identical edge/key order and cardinality.
            producer.buffers.into_iter().for_each(RecordBuffer::finish);
            work.complete(producer.range.len());
        });
    // SAFETY: Producers own disjoint, counted slices and joined after every
    // counted cell was initialized. A source panic unwinds before conversion.
    // u32 and MaybeUninit<u32> share layout; each allocation keeps its capacity.
    let records: Vec<Vec<u32>> = records
        .into_iter()
        .map(|values| unsafe { initialized_records(values) })
        .collect();
    let mut ranges: Vec<Vec<(u64, usize, usize)>> = (0..workers).map(|_| Vec::new()).collect();
    let mut ends = vec![0_usize; workers];
    for (bucket, &count) in totals.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let owner = owners[bucket];
        let begin = ends[owner];
        ends[owner] += count;
        ranges[owner].push((alphabet.key(bucket), begin, ends[owner]));
    }
    drop(owners);
    drop(totals);
    let group_work = progress.stage("Count bounded initial pairs", owner_sizes.iter().sum());
    records
        .into_par_iter()
        .zip(ranges.into_par_iter())
        .map(|(records, ranges)| {
            let mut groups = Vec::new();
            let mut keys = Vec::new();
            let mut mass = 0_u128;
            for (key, begin, end) in ranges {
                let frequency = group_frequency(
                    &records[begin..end],
                    base,
                    corpus.word_weights(),
                    uniform_weight,
                )?;
                mass += u128::from(frequency);
                if frequency >= floor {
                    groups.push(InitialGroup {
                        begin: begin as u32,
                        end: end as u32,
                        frequency,
                    });
                    keys.push(key);
                }
            }
            group_work.complete(records.len());
            Ok(GroupedWave {
                records,
                groups,
                keys,
                mass,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn alphabet_admission_preserves_high_ids_and_rejects_oversized_domains() {
        assert!(Alphabet::new(Vec::new()).is_none());
        let alphabet = Alphabet::new((0..256).map(|i| 65000 + i).collect()).unwrap();
        assert_eq!(alphabet.bucket(pair_key((65000, 65255))), 255);
        assert_eq!(alphabet.key(255), pair_key((65000, 65255)));
        assert_eq!(alphabet.bucket(pair_key((65255, 65255))), 65535);
        assert!(Alphabet::new((0..257).collect()).is_none());
        assert!(Alphabet::new(vec![1, 65536]).is_none());
        assert!(Alphabet::new(vec![u32::MAX - 1]).is_none());
        assert!(Alphabet::new(vec![1, 1]).is_none());
        assert!(Alphabet::new(vec![2, 1]).is_none());
    }
    struct AdmissionSource {
        slots: usize,
        edges: Option<usize>,
        weights: IntervalIndex<u64>,
    }
    impl InitialPairSource for AdmissionSource {
        fn len(&self) -> usize {
            self.slots
        }
        fn word_weights(&self) -> &IntervalIndex<u64> {
            &self.weights
        }
        fn for_each_edge(&self, _: Range<usize>, _: impl FnMut(usize, u64)) {
            panic!("cost admission must not scan symbols or positions");
        }
        fn bounded_edge_count(&self, range: Range<usize>) -> Option<usize> {
            assert_eq!(range, 0..self.slots.saturating_sub(1));
            self.edges
        }
    }
    #[test]
    fn cost_admission_rejects_sparse_small_sources_and_keeps_large_tails() {
        let alphabet = Alphabet::new((0..256).collect()).unwrap();
        for workers in [1, 8, 64, 65] {
            for edges in [0, 100, 1000, 10000] {
                let source = AdmissionSource {
                    slots: edges + 1,
                    edges: Some(edges),
                    weights: IntervalIndex::new(vec![0], vec![1]),
                };
                assert!(!alphabet.admits_source(&source, workers, 1 << 28));
            }
            let slots = 1 << 28;
            let threshold = alphabet
                .metadata_bound(workers, slots)
                .div_ceil(DIRECTORY_BYTES_PER_EDGE);
            assert!(!alphabet.admits_records(workers, slots, threshold - 1));
            assert!(alphabet.admits_records(workers, slots, threshold));
            assert!(alphabet.admits_records(workers, slots, usize::MAX));
            let source = AdmissionSource {
                slots: slots + 2,
                edges: Some(slots + 1),
                weights: IntervalIndex::new(vec![0], vec![0]),
            };
            assert!(
                alphabet.admits_source(&source, workers, slots),
                "a one-edge tail must not reject the large source"
            );
        }
        let sparse = AdmissionSource {
            slots: 1 << 28,
            edges: Some(1000),
            weights: IntervalIndex::new(vec![0], vec![1]),
        };
        assert!(!alphabet.admits_source(&sparse, 8, 1 << 28));
        let high_ids = Alphabet::new(vec![7, 65535]).unwrap();
        let low_ids = Alphabet::new(vec![7, 8]).unwrap();
        assert!(high_ids.metadata_bound(1, 100) > low_ids.metadata_bound(1, 100));
        let unknown = AdmissionSource {
            slots: 1 << 28,
            edges: None,
            weights: IntervalIndex::new(vec![0], vec![1]),
        };
        assert!(!alphabet.admits_source(&unknown, 8, 1 << 28));
        let empty = AdmissionSource {
            slots: 0,
            edges: Some(0),
            weights: IntervalIndex::new(vec![], vec![]),
        };
        assert!(!alphabet.admits_source(&empty, 1, 1 << 28));
        let small = Alphabet::new(vec![7]).unwrap();
        let source = AdmissionSource {
            slots: 101,
            edges: Some(100),
            weights: IntervalIndex::new(vec![0], vec![0]),
        };
        assert!(small.admits_source(&source, 8, 1 << 28));
    }

    #[test]
    fn producer_capacity_bound_includes_slice_layout_and_large_pools() {
        for symbols in [1, 2, 16, 128, 256] {
            for workers in [1, 4, 8, 64, 65, 1024] {
                for slots in [0, 1, 1 << 18, 1 << 28] {
                    let pairs = symbols * symbols;
                    let producers = producer_count(pairs, workers, slots);
                    assert!(producers > 0 && producers <= workers.min(MAX_PRODUCERS));
                    assert!(
                        producers
                            * pairs
                            * (std::mem::size_of::<u32>()
                                + std::mem::size_of::<RecordBuffer<'_, u32>>())
                            <= DIRECTORY_BUDGET
                    );
                }
            }
        }
    }
}
