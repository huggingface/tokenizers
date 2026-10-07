//! Coordinate planning, full pair keys, wave boundaries, and materialization.
use super::*;
use super::{corpus::InitialPairSource, pair_index::pair_key};

struct ForcedFullWidth<'a>(corpus::InitialCorpus<'a>);
impl corpus::InitialPairSource for ForcedFullWidth<'_> {
    fn len(&self) -> usize {
        corpus::InitialPairSource::len(&self.0)
    }

    fn word_weights(&self) -> &storage::IntervalIndex<u64> {
        corpus::InitialPairSource::word_weights(&self.0)
    }

    fn for_each_edge(&self, range: std::ops::Range<usize>, emit: impl FnMut(usize, u64)) {
        corpus::InitialPairSource::for_each_edge(&self.0, range, emit);
    }
}

fn initial_snapshot(
    table: &initial_pairs::InitialPairTable<'_>,
) -> std::collections::BTreeMap<u64, (u64, Vec<u64>)> {
    table
        .shards
        .iter()
        .flat_map(|shard| {
            shard.iter().map(|(&key, state)| {
                (
                    key,
                    (state.ledger_count_bits, state.positions.iter().collect()),
                )
            })
        })
        .collect()
}

struct BoundedInitialSource<'a> {
    inner: corpus::InitialCorpus<'a>,
    ids: Vec<u32>,
    edges: usize,
}
impl<'a> BoundedInitialSource<'a> {
    fn new(inner: corpus::InitialCorpus<'a>, ids: Vec<u32>) -> Self {
        // Synthetic fixtures precompute their hint before invoking production
        // admission. Real CorpusPlan already stores this exact geometry count.
        let edges = inner
            .token_ids
            .windows(2)
            .filter(|pair| {
                pair.iter()
                    .all(|id| id.load(std::sync::atomic::Ordering::Relaxed) != WORD_SEPARATOR_ID)
            })
            .count();
        Self { inner, ids, edges }
    }
}
impl corpus::InitialPairSource for BoundedInitialSource<'_> {
    fn len(&self) -> usize {
        corpus::InitialPairSource::len(&self.inner)
    }
    fn word_weights(&self) -> &storage::IntervalIndex<u64> {
        self.inner.word_weights
    }
    fn for_each_edge(&self, range: std::ops::Range<usize>, emit: impl FnMut(usize, u64)) {
        corpus::InitialPairSource::for_each_edge(&self.inner, range, emit);
    }
    fn bounded_initial_ids(&self) -> Option<Vec<u32>> {
        Some(self.ids.clone())
    }
    fn bounded_edge_count(&self, range: std::ops::Range<usize>) -> Option<usize> {
        assert_eq!(range, 0..self.inner.token_ids.len().saturating_sub(1));
        Some(self.edges)
    }
}

#[test]
fn cost_rejected_high_ids_match_forced_bounded_weights_and_wave_append() {
    use std::sync::atomic::AtomicU32;
    let ids = [
        WORD_SEPARATOR_ID,
        7,
        7,
        7,
        WORD_SEPARATOR_ID,
        7,
        7,
        65535,
        7,
        WORD_SEPARATOR_ID,
        65535,
        65535,
        WORD_SEPARATOR_ID,
        WORD_SEPARATOR_ID,
        7,
        7,
        WORD_SEPARATOR_ID,
    ];
    let slots: Vec<_> = ids.into_iter().map(AtomicU32::new).collect();
    let weights = storage::IntervalIndex::new(vec![1, 5, 10, 14], vec![5, 0, 3, 1]);
    let source = || {
        BoundedInitialSource::new(
            corpus::InitialCorpus {
                token_ids: &slots,
                word_weights: &weights,
            },
            vec![7, 65535],
        )
    };
    // A two-slot wave at 12 has no edge, between populated waves. The
    // forced collector must handle this without publishing unwritten offsets.
    let mut empty_wave_edges = 0;
    source().for_each_edge(12..14, |_, _| empty_wave_edges += 1);
    assert_eq!(empty_wave_edges, 0);
    let progress =
        TrainingProgress::new(false, tk_encode::utils::progress::ProgressFormat::Silent).unwrap();
    for workers in [1, 4, 65] {
        let execution = execution::Execution::new(workers).unwrap();
        let arena = AllocationArena::new(workers, slots.len() - 1);
        execution.pool.install(|| {
            for floor in [0, 1, 6, 11, 12] {
                for wave in [2, 3, 6, 1 << 28] {
                    // These few edges cannot pay for the high-ID lookup. Keep
                    // the admission regression, separately from collector semantics.
                    assert!(!initial_pairs::InitialPairTable::admits_bounded_for_test(
                        &source(),
                        workers,
                        wave,
                    ));
                    let fallback = initial_pairs::InitialPairTable::build_in_waves(
                        source(),
                        floor,
                        &execution,
                        &arena,
                        &progress,
                        wave,
                    )
                    .unwrap();
                    // This entry explicitly selects the same bounded collect
                    // and publication helpers used by the admitted production arm.
                    let bounded = initial_pairs::InitialPairTable::build_bounded_for_test(
                        source(),
                        floor,
                        &execution,
                        &arena,
                        &progress,
                        wave,
                    )
                    .unwrap();
                    let generic = initial_pairs::InitialPairTable::build_in_waves(
                        ForcedFullWidth(corpus::InitialCorpus {
                            token_ids: &slots,
                            word_weights: &weights,
                        }),
                        floor,
                        &execution,
                        &arena,
                        &progress,
                        wave,
                    )
                    .unwrap();
                    let snapshot = initial_snapshot(&bounded);
                    assert_eq!(snapshot, initial_snapshot(&generic));
                    assert_eq!(snapshot, initial_snapshot(&fallback));
                    assert_eq!(snapshot.contains_key(&pair_key((7, 7))), floor <= 11);
                    if floor <= 11 {
                        // Offset 5 contributes zero mass to a positive key. It
                        // survives a complete wave or a mixed positive wave;
                        // a zero-only partial wave is filtered when floor > 0.
                        let positions = if floor == 0 || wave >= 6 {
                            vec![1, 2, 5, 14]
                        } else {
                            vec![1, 2, 14]
                        };
                        assert_eq!(snapshot[&pair_key((7, 7))], (11, positions));
                    }
                    if floor == 0 {
                        let zero_positions = if wave >= slots.len() { vec![] } else { vec![6] };
                        assert_eq!(snapshot[&pair_key((7, 65535))], (0, zero_positions));
                    }
                    assert_eq!(bounded.weighted_mass, 14);
                    assert_eq!(bounded.maximum_word_weight, 5);
                }
            }
        });
    }
}

#[test]
fn cost_rejected_overflow_fixture_checks_forced_bounded_and_generic() {
    use std::sync::atomic::AtomicU32;
    let progress =
        TrainingProgress::new(false, tk_encode::utils::progress::ProgressFormat::Silent).unwrap();
    let execution = execution::Execution::new(1).unwrap();
    execution.pool.install(|| {
        let weights = storage::IntervalIndex::new(vec![1], vec![u64::MAX]);
        for repeated in [false, true] {
            let slots: Vec<_> = [
                WORD_SEPARATOR_ID,
                1,
                2,
                WORD_SEPARATOR_ID,
                if repeated { 1 } else { 2 },
                2,
                WORD_SEPARATOR_ID,
            ]
            .into_iter()
            .map(AtomicU32::new)
            .collect();
            let arena = AllocationArena::new(1, 2);
            for wave in [3, 1 << 28] {
                let source = || {
                    BoundedInitialSource::new(
                        corpus::InitialCorpus {
                            token_ids: &slots,
                            word_weights: &weights,
                        },
                        vec![1, 2],
                    )
                };
                assert!(!initial_pairs::InitialPairTable::admits_bounded_for_test(
                    &source(),
                    1,
                    wave,
                ));
                // A single wave overflows during group counting; wave=3
                // overflows during checked append of two individually valid keys.
                let bounded = initial_pairs::InitialPairTable::build_bounded_for_test(
                    source(),
                    0,
                    &execution,
                    &arena,
                    &progress,
                    wave,
                );
                let fallback = initial_pairs::InitialPairTable::build_in_waves(
                    source(),
                    0,
                    &execution,
                    &arena,
                    &progress,
                    wave,
                );
                if repeated {
                    let message = "BPE initial pair frequency exceeds u64";
                    assert_eq!(bounded.err().unwrap().to_string(), message);
                    assert_eq!(fallback.err().unwrap().to_string(), message);
                } else {
                    let bounded = bounded.unwrap();
                    let fallback = fallback.unwrap();
                    assert_eq!(bounded.weighted_mass, 2 * u128::from(u64::MAX));
                    assert_eq!(initial_snapshot(&bounded), initial_snapshot(&fallback));
                }
            }
        }
    });
}

#[test]
fn no_edge_bounded_hints_are_cost_rejected_and_publish_empty_generic_output() {
    use std::sync::atomic::AtomicU32;
    let execution = execution::Execution::new(4).unwrap();
    let arena = AllocationArena::new(4, 0);
    let progress =
        TrainingProgress::new(false, tk_encode::utils::progress::ProgressFormat::Silent).unwrap();
    execution.pool.install(|| {
        for ids in [
            Vec::new(),
            vec![7],
            vec![WORD_SEPARATOR_ID, 7, WORD_SEPARATOR_ID],
        ] {
            let slots: Vec<_> = ids.into_iter().map(AtomicU32::new).collect();
            let weights = storage::IntervalIndex::new(vec![0], vec![0]);
            let source = BoundedInitialSource::new(
                corpus::InitialCorpus {
                    token_ids: &slots,
                    word_weights: &weights,
                },
                vec![7],
            );
            assert!(!initial_pairs::InitialPairTable::admits_bounded_for_test(
                &source, 4, 3,
            ));
            let table = initial_pairs::InitialPairTable::build_in_waves(
                source, 0, &execution, &arena, &progress, 3,
            )
            .unwrap();
            assert!(initial_snapshot(&table).is_empty());
            assert_eq!(table.weighted_mass, 0);
        }
    });
}

#[test]
fn bounded_directory_preserves_order_across_parallel_spatial_producers() {
    use std::sync::atomic::AtomicU32;
    let tile = 1 << 18;
    let slots: Vec<_> = (0..(2 * tile + 3))
        .map(|i| AtomicU32::new(if i % 3 == 0 { 65535 } else { 7 }))
        .collect();
    let weights = storage::IntervalIndex::new(vec![0], vec![3]);
    let execution = execution::Execution::new(4).unwrap();
    let arena = AllocationArena::new(4, slots.len() - 1);
    let progress =
        TrainingProgress::new(false, tk_encode::utils::progress::ProgressFormat::Silent).unwrap();
    execution.pool.install(|| {
        let source = BoundedInitialSource::new(
            corpus::InitialCorpus {
                token_ids: &slots,
                word_weights: &weights,
            },
            vec![7, 65535],
        );
        assert!(initial_pairs::InitialPairTable::admits_bounded_for_test(
            &source,
            4,
            1 << 28,
        ));
        let bounded =
            initial_pairs::InitialPairTable::build(source, 0, &execution, &arena, &progress)
                .unwrap();
        let generic = initial_pairs::InitialPairTable::build(
            ForcedFullWidth(corpus::InitialCorpus {
                token_ids: &slots,
                word_weights: &weights,
            }),
            0,
            &execution,
            &arena,
            &progress,
        )
        .unwrap();
        assert_eq!(initial_snapshot(&bounded), initial_snapshot(&generic));
        assert_eq!(bounded.weighted_mass, 3 * (slots.len() - 1) as u128);
    });
}

#[test]
fn full_pair_keys_remain_distinct_in_initial_counting() {
    use super::storage::IntervalIndex;
    use std::sync::atomic::AtomicU32;
    let ids = [
        WORD_SEPARATOR_ID,
        1,
        2,
        WORD_SEPARATOR_ID,
        0x1_0001,
        2,
        WORD_SEPARATOR_ID,
        u32::MAX - 1,
        u32::MAX - 2,
        WORD_SEPARATOR_ID,
    ];
    let slots: Vec<_> = ids.into_iter().map(AtomicU32::new).collect();
    let weights = IntervalIndex::new(vec![1], vec![7]);
    for workers in [1, 4, 65] {
        let execution = execution::Execution::new(workers).unwrap();
        let arena = AllocationArena::new(workers, 3);
        let progress =
            TrainingProgress::new(false, tk_encode::utils::progress::ProgressFormat::Silent)
                .unwrap();
        execution.pool.install(|| {
            let initial = initial_pairs::InitialPairTable::build(
                corpus::InitialCorpus {
                    token_ids: &slots,
                    word_weights: &weights,
                },
                1,
                &execution,
                &arena,
                &progress,
            )
            .unwrap();
            assert_eq!(
                initial
                    .shards
                    .iter()
                    .map(|shard| shard.len())
                    .sum::<usize>(),
                3
            );
            for (pair, position) in [
                ((1, 2), 1),
                ((0x1_0001, 2), 4),
                ((u32::MAX - 1, u32::MAX - 2), 7),
            ] {
                let key = pair_index::pair_key(pair);
                let state = &initial.shards[pair_index::shard_for(key, workers)][&key];
                assert_eq!(state.ledger_count_bits, 7);
                assert_eq!(state.positions.iter().collect::<Vec<_>>(), [position]);
            }
        });
    }
}

#[test]
fn low_id_compact_records_match_forced_full_records_across_small_waves() {
    use super::storage::IntervalIndex;
    use std::sync::atomic::AtomicU32;

    // The first and third words contain overlapping AA edges. The second has
    // zero weight, while the other words have distinct positive weights.
    let ids = [
        WORD_SEPARATOR_ID,
        1,
        1,
        1,
        1,
        WORD_SEPARATOR_ID,
        1,
        2,
        1,
        WORD_SEPARATOR_ID,
        3,
        3,
        3,
        WORD_SEPARATOR_ID,
        2,
        2,
        WORD_SEPARATOR_ID,
    ];
    let slots: Vec<_> = ids.into_iter().map(AtomicU32::new).collect();
    let weights = IntervalIndex::new(vec![1, 6, 10, 14], vec![5, 0, 3, 2]);
    let workers = 4;
    let execution = execution::Execution::new(workers).unwrap();
    let arena = AllocationArena::new(workers, slots.len() - 1);
    let progress =
        TrainingProgress::new(false, tk_encode::utils::progress::ProgressFormat::Silent).unwrap();
    execution.pool.install(|| {
        let low_id_source = corpus::InitialCorpus {
            token_ids: &slots,
            word_weights: &weights,
        };
        assert!(corpus::InitialPairSource::compact_keys(&low_id_source));

        for floor in [0, 2] {
            let compact_source = corpus::InitialCorpus {
                token_ids: &slots,
                word_weights: &weights,
            };
            let full_source = ForcedFullWidth(corpus::InitialCorpus {
                token_ids: &slots,
                word_weights: &weights,
            });
            assert!(!corpus::InitialPairSource::compact_keys(&full_source));
            let compact = initial_pairs::InitialPairTable::build_in_waves(
                compact_source,
                floor,
                &execution,
                &arena,
                &progress,
                3,
            )
            .unwrap();
            let compact_snapshot = initial_snapshot(&compact);
            let compact_mass = compact.weighted_mass;
            let compact_max_weight = compact.maximum_word_weight;
            let full = initial_pairs::InitialPairTable::build_in_waves(
                full_source,
                floor,
                &execution,
                &arena,
                &progress,
                3,
            )
            .unwrap();
            assert_eq!(initial_snapshot(&full), compact_snapshot);
            assert_eq!(full.weighted_mass, compact_mass);
            assert_eq!(full.maximum_word_weight, compact_max_weight);
            assert_eq!(compact_mass, 23);
        }
    });
}

#[test]
fn owner_directories_keep_duplicate_keys_stable_across_tiles() {
    use super::storage::IntervalIndex;
    use std::sync::atomic::AtomicU32;

    let tile = 1 << 18;
    let slots: Vec<_> = (0..(2 * tile + 3))
        .map(|index| AtomicU32::new(if index % 2 == 0 { 1 } else { 2 }))
        .collect();
    let weights = IntervalIndex::new(vec![0], vec![7]);
    let mut reference = None;
    for workers in [1, 64, 65] {
        let execution = execution::Execution::new(workers).unwrap();
        let arena = AllocationArena::new(workers, slots.len() - 1);
        let progress =
            TrainingProgress::new(false, tk_encode::utils::progress::ProgressFormat::Silent)
                .unwrap();
        let initial = execution.pool.install(|| {
            initial_pairs::InitialPairTable::build(
                corpus::InitialCorpus {
                    token_ids: &slots,
                    word_weights: &weights,
                },
                1,
                &execution,
                &arena,
                &progress,
            )
            .unwrap()
        });
        let snapshot = initial_snapshot(&initial);
        if let Some(reference) = &reference {
            assert_eq!(&snapshot, reference, "workers={workers}");
        } else {
            reference = Some(snapshot.clone());
        }
        let edges = (slots.len() - 1) as u128;
        assert_eq!(initial.weighted_mass, edges * 7);
        assert_eq!(initial.maximum_word_weight, 7);
        assert_eq!(snapshot.len(), 2);
        let repeated_across_tiles = |pair| {
            snapshot[&pair_index::pair_key(pair)]
                .1
                .iter()
                .any(|&position| position < tile as u64)
                && snapshot[&pair_index::pair_key(pair)]
                    .1
                    .iter()
                    .any(|&position| (tile as u64..(2 * tile) as u64).contains(&position))
                && snapshot[&pair_index::pair_key(pair)]
                    .1
                    .iter()
                    .any(|&position| position >= (2 * tile) as u64)
        };
        assert!(repeated_across_tiles((1, 2)));
        assert!(repeated_across_tiles((2, 1)));
        assert_eq!(
            snapshot[&pair_index::pair_key((1, 2))].1.len(),
            edges as usize / 2
        );
        assert_eq!(
            snapshot[&pair_index::pair_key((2, 1))].1.len(),
            edges as usize / 2
        );
        assert_eq!(
            snapshot[&pair_index::pair_key((1, 2))].1.last(),
            Some(&((2 * tile) as u64))
        );
        assert_eq!(
            snapshot[&pair_index::pair_key((2, 1))].1.last(),
            Some(&((2 * tile + 1) as u64))
        );
    }
}

#[test]
fn initial_waves_preserve_crossing_edges_and_filter_complete_counts() {
    use super::storage::IntervalIndex;
    use std::sync::atomic::AtomicU32;
    let mut slots: Vec<_> = [
        WORD_SEPARATOR_ID,
        1,
        2,
        1,
        2,
        1,
        2,
        WORD_SEPARATOR_ID,
        1,
        2,
        1,
        2,
        WORD_SEPARATOR_ID,
        u32::MAX - 1,
        u32::MAX - 2,
        WORD_SEPARATOR_ID,
    ]
    .into_iter()
    .map(AtomicU32::new)
    .collect();
    slots.extend([1, 2, WORD_SEPARATOR_ID].into_iter().map(AtomicU32::new));
    let weights = IntervalIndex::new(vec![1, 8, 13, 16], vec![3, 5, 0, 0]);
    for workers in [1, 4] {
        let execution = execution::Execution::new(workers).unwrap();
        let arena = AllocationArena::new(workers, 10);
        let progress =
            TrainingProgress::new(false, tk_encode::utils::progress::ProgressFormat::Silent)
                .unwrap();
        execution.pool.install(|| {
            // Use the real wave algorithm with a small resource bound. The
            // (1,2) edge at slot 3 crosses a wave; every partial count is below
            // 12, while its complete weighted count is 19.
            for floor in [0, 12] {
                let initial = initial_pairs::InitialPairTable::build_in_waves(
                    corpus::InitialCorpus {
                        token_ids: &slots,
                        word_weights: &weights,
                    },
                    floor,
                    &execution,
                    &arena,
                    &progress,
                    4,
                )
                .unwrap();
                assert_eq!(initial.weighted_mass, 30);
                let key = pair_index::pair_key((1, 2));
                let state = &initial.shards[pair_index::shard_for(key, workers)][&key];
                assert_eq!(state.ledger_count_bits, 19);
                assert_eq!(
                    state.positions.iter().collect::<Vec<_>>(),
                    if floor == 0 {
                        vec![1, 3, 5, 8, 10, 16]
                    } else {
                        vec![1, 3, 5, 8, 10]
                    }
                );
                assert_eq!(
                    initial
                        .shards
                        .iter()
                        .map(|shard| shard.len())
                        .sum::<usize>(),
                    if floor == 0 { 3 } else { 1 }
                );
                if floor == 0 {
                    let key = pair_index::pair_key((u32::MAX - 1, u32::MAX - 2));
                    let state = &initial.shards[pair_index::shard_for(key, workers)][&key];
                    assert_eq!(state.ledger_count_bits, 0);
                    assert_eq!(state.positions.iter().collect::<Vec<_>>(), [13]);
                }
            }
        });
    }
}

#[test]
fn planned_edges_match_materialized_slots_across_word_and_seek_boundaries() {
    use corpus::InitialPairSource;
    let long = "a测éxb".repeat(2500);
    let mut words = counts(&[("", 1), ("x", 0), ("a测éxb", 7), ("xa", 3), ("aé", 0)]);
    words.insert(long.into(), 2);
    words.insert(
        format!(
            "{}a{}测{}",
            "x".repeat(16000),
            "x".repeat(16000),
            "x".repeat(16000)
        )
        .into(),
        5,
    );
    for affixes in [false, true] {
        for limited in [false, true] {
            let mut trainer = BpeTrainer::builder()
                .vocab_size(100)
                .show_progress(false)
                .build();
            if affixes {
                trainer.continuing_subword_prefix = Some("##".into());
                trainer.end_of_word_suffix = Some("</w>".into());
            }
            if limited {
                trainer.limit_alphabet = Some(2);
                trainer.initial_alphabet = ['a', '测'].into();
            }
            let execution = execution::Execution::new(4).unwrap();
            let progress = TrainingProgress::new(false, trainer.progress_format).unwrap();
            execution.pool.install(|| {
                let mut retained = None;
                let mut vocab = vocabulary::Vocabulary::initialize(
                    &trainer,
                    WordCountsView::from_map(&words),
                    4,
                    &progress,
                    &mut retained,
                )
                .unwrap();
                let plan = corpus::CorpusPlan::build(
                    WordCountsView::from_map(&words),
                    &mut vocab,
                    IdentityPolicy::AllowActiveReuse,
                    false,
                    &progress,
                )
                .unwrap();
                let len = (&plan).len();
                let mut ranges = vec![0..len - 1, 0..1, len - 2..len - 1];
                for start in [1, 2, 3, 4095, 4096, 4097, 8191, 8192, len / 2] {
                    if start < len - 1 {
                        ranges.push(start..(start + 3).min(len - 1));
                        ranges.push(start..start);
                    }
                }
                let observed: Vec<_> = ranges
                    .iter()
                    .map(|range| {
                        let mut pairs = Vec::new();
                        (&plan).for_each_edge(range.clone(), |position, key| {
                            pairs.push((position, key))
                        });
                        assert_eq!(
                            (&plan).edge_count(range.clone()),
                            pairs.len(),
                            "planned capacity for {range:?}"
                        );
                        pairs
                    })
                    .collect();
                let corpus = plan
                    .materialize::<corpus::U32Slots>(4, IdentityPolicy::AllowActiveReuse, &progress)
                    .unwrap();
                let source = corpus.initial_view();
                for (range, actual) in ranges.into_iter().zip(observed) {
                    let mut expected = Vec::new();
                    source.for_each_edge(range.clone(), |position, key| {
                        expected.push((position, key))
                    });
                    assert_eq!(
                        actual, expected,
                        "affixes={affixes} limited={limited} range={range:?}"
                    );
                }
            });
        }
    }
}

#[test]
fn long_word_split_preserves_unicode_filtering_affixes_and_merge_trace() {
    let long = "ab测é".repeat(2049);
    let filtered = format!(
        "{}a{}测{}",
        "x".repeat(8193),
        "x".repeat(8193),
        "x".repeat(8193)
    );
    let words = counts(&[(&long, 3), (&filtered, 2), ("", 1), ("ab", 0)]);
    for filtered in [false, true] {
        for affixes in [false, true] {
            let mut trainer = BpeTrainer::builder()
                .vocab_size(48)
                .min_frequency(2)
                .max_token_length(Some(32))
                .show_progress(false)
                .build();
            if filtered {
                trainer.limit_alphabet = Some(2);
                trainer.initial_alphabet = ['a', '测'].into();
            }
            if affixes {
                trainer.continuing_subword_prefix = Some("##".into());
                trainer.end_of_word_suffix = Some("</w>".into());
            }
            check_with_workers(&trainer, &words, &[1, 4, 16]);
        }
    }
}

// Reconstruct the real first attempt's initial plan, then query the
// very selector used by build_in_waves. No global observer can mix parallel tests.
fn assert_training_plan_bounded_admission(
    trainer: &BpeTrainer,
    words: &AHashMap<CompactString, u64>,
    workers: usize,
    expected: bool,
) {
    let execution = execution::Execution::new(workers).unwrap();
    let progress = TrainingProgress::new(false, trainer.progress_format).unwrap();
    execution.pool.install(|| {
        let mut retained = None;
        let mut vocabulary = vocabulary::Vocabulary::initialize(
            trainer,
            WordCountsView::from_map(words),
            workers,
            &progress,
            &mut retained,
        )
        .unwrap();
        if expected {
            assert!(
                vocabulary.len() < trainer.vocab_size,
                "training must build initial pairs"
            );
        }
        let plan = corpus::CorpusPlan::build(
            WordCountsView::from_map(words),
            &mut vocabulary,
            IdentityPolicy::FirstActivationOnly,
            trainer.max_token_length.is_some(),
            &progress,
        )
        .unwrap();
        assert_eq!(
            initial_pairs::InitialPairTable::admits_bounded_for_test(&&plan, workers, 1 << 28),
            expected,
            "workers={workers}, edges={}",
            plan.initial_edges(),
        );
    });
}

#[test]
fn cost_admitted_real_corpus_plan_preserves_full_training_trace() {
    // Physical edges pay for the directory; increasing weights alone would not.
    // The two-ID domain also keeps this end-to-end oracle fixture inexpensive.
    let words: AHashMap<CompactString, u64> = [
        ("ab".repeat(128).into(), 3),
        ("ba".repeat(128).into(), 1),
        ("aabb".repeat(64).into(), 5),
        ("baaab".repeat(40).into(), 0),
    ]
    .into_iter()
    .collect();
    for floor in [1, 4] {
        let trainer = BpeTrainer::builder()
            .vocab_size(32)
            .min_frequency(floor)
            .show_progress(false)
            .build();
        for workers in [1, 4, 16] {
            assert_training_plan_bounded_admission(&trainer, &words, workers, true);
        }
        check_with_workers(&trainer, &words, &[1, 4, 16]);
    }
}

#[test]
fn small_byte_alphabet_training_preserves_trace_with_cost_rejected_plan() {
    let alphabet = tk_encode::pre_tokenizers::byte_level::ByteLevel::alphabet();
    let words = alphabet
        .into_iter()
        .enumerate()
        .map(|(index, symbol)| (format!("{symbol}{symbol}").into(), [0, 1, 3][index % 3]))
        .collect();
    for target in [256, 270] {
        let trainer = BpeTrainer::builder()
            .vocab_size(target)
            .min_frequency(1)
            .show_progress(false)
            .build();
        // 256 physical edges do not pay for a 65536-pair directory. Target=256
        // can finish before collection; target=270 exercises the generic fallback.
        for workers in [1, 4, 16] {
            assert_training_plan_bounded_admission(&trainer, &words, workers, false);
        }
        check_with_workers(&trainer, &words, &[1, 4, 16]);
    }
}
