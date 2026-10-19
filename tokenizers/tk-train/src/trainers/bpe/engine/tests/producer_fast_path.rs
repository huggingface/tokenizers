//! Complete-producer encoding and partial fallback equivalence.
use super::*;

#[test]
fn producer_encoding_preserves_hf_trace_and_fallbacks() {
    use merge::MergeOptions;
    let fixtures = [
        counts(&[
            ("ab", 1000),
            ("cd", 900),
            ("abcd", 20),
            ("cdab", 15),
            ("ababcdcd", 9),
        ]),
        counts(&[
            ("aaaaaa", 23),
            ("aaabaaa", 17),
            ("abaaab", 11),
            ("cdabcd", 7),
        ]),
        counts(&[
            ("猫猫猫鱼", 19),
            ("猫鱼猫鱼", 13),
            ("abcdef", 7),
            ("abcabc", 3),
        ]),
    ];
    for (fixture_index, words) in fixtures.iter().enumerate() {
        for affixes in [false, true] {
            let mut builder = BpeTrainer::builder()
                .vocab_size(45)
                .min_frequency(3)
                .show_progress(false);
            if affixes {
                builder = builder
                    .continuing_subword_prefix("##".into())
                    .end_of_word_suffix("</w>".into())
                    .max_token_length(Some(9));
            }
            let trainer = builder.build();
            let mut expected_trace = Vec::new();
            let expected = trainer
                .do_train_observed(words, |pair, count, id| {
                    expected_trace.push((pair, count, id))
                })
                .unwrap();

            for single_producer_fast in [false, true] {
                for workers in [1, 4] {
                    let options = MergeOptions {
                        single_producer_fast,
                        ..MergeOptions::default()
                    };
                    let mut trace = Vec::new();
                    let actual = train_with_merge_options(
                        &trainer,
                        WordCountsView::from_map(words),
                        workers,
                        options,
                        Some(&mut |pair, count, id| trace.push((pair, count, id))),
                        None,
                    )
                    .unwrap();
                    assert_eq!(
                        trace, expected_trace,
                        "fixture={fixture_index}, affixes={affixes}, options={options:?}, workers={workers}"
                    );
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}

#[test]
fn owner_single_producer_requires_full_task_and_prunes_complete_births() {
    use merge::MergeOptions;
    let words = counts(&[
        ("ab", 1000),
        ("cd", 900),
        ("abcd", 20),
        ("cdab", 15),
        ("ababcdcd", 9),
    ]);
    let trainer = BpeTrainer::builder()
        .vocab_size(30)
        .min_frequency(2)
        .show_progress(false)
        .build();
    for workers in [1, 4] {
        for floor in [2, 100] {
            let execution = execution::Execution::new(workers).unwrap();
            let progress = TrainingProgress::new(false, trainer.progress_format).unwrap();
            execution.pool.install(|| {
                let mut retained = None;
                let mut vocab = vocabulary::Vocabulary::initialize(
                    &trainer,
                    WordCountsView::from_map(&words),
                    workers,
                    &progress,
                    &mut retained,
                )
                .unwrap();
                let plan = corpus::CorpusPlan::build(
                    WordCountsView::from_map(&words),
                    &mut vocab,
                    IdentityPolicy::FirstActivationOnly,
                    false,
                    &progress,
                )
                .unwrap();
                let arena = AllocationArena::new(workers, plan.initial_edges());
                let initial =
                    initial_pairs::InitialPairTable::build(&plan, 2, &execution, &arena, &progress)
                        .unwrap();
                let mut corpus = plan
                    .materialize::<corpus::U32Slots>(
                        workers,
                        IdentityPolicy::FirstActivationOnly,
                        &progress,
                    )
                    .unwrap();
                let mut index = pair_index::PairIndex::from_initial_pairs(
                    initial,
                    IdentityPolicy::FirstActivationOnly,
                    2,
                )
                .unwrap();
                let mut rules = Vec::new();
                let mut candidates = Vec::new();
                index.begin_selection();
                for pair in [(0, 1), (2, 3)] {
                    assert_eq!(pair_index::key_pair(index.best().unwrap().key), pair);
                    candidates.push(index.take_best());
                    let identity = vocab.resolve_merge(vocab.merge_token(pair)).unwrap();
                    assert!(!identity.reused_active_id);
                    corpus.prepare_spans(pair, identity.id, false);
                    rules.push(merge::MergeRule {
                        pair,
                        replacement: identity.id,
                    });
                }
                index.end_selection();
                let (prepared, births) = merge::prepare_merges_with_births(
                    &corpus,
                    &rules,
                    &candidates,
                    IdentityPolicy::FirstActivationOnly,
                    vocab.len(),
                    usize::MAX,
                    &execution,
                    &arena,
                    floor,
                    MergeOptions {
                        single_producer_fast: true,
                        ..MergeOptions::default()
                    },
                )
                .unwrap();
                if workers == 1 {
                    assert_ne!(
                        prepared.birth_shape,
                        Default::default(),
                        "whole compact producers are observed even when floor prunes births"
                    );
                    if floor == 2 {
                        let mut actual = AHashMap::new();
                        for birth in &births {
                            assert!(
                                actual
                                    .insert(birth.key, (birth.weight, birth.positions.len()))
                                    .is_none()
                            );
                        }
                        assert_eq!(
                            actual[&pair_index::pair_key((
                                rules[0].replacement,
                                rules[1].replacement
                            ))],
                            (29, 2)
                        );
                        assert_eq!(
                            actual[&pair_index::pair_key((
                                rules[1].replacement,
                                rules[0].replacement
                            ))],
                            (15, 1)
                        );
                        assert_eq!(
                            actual[&pair_index::pair_key((
                                rules[0].replacement,
                                rules[0].replacement
                            ))],
                            (9, 1)
                        );
                        assert_eq!(
                            actual[&pair_index::pair_key((
                                rules[1].replacement,
                                rules[1].replacement
                            ))],
                            (9, 1)
                        );
                    } else {
                        assert!(births.is_empty());
                    }
                } else {
                    assert_eq!(
                        prepared.birth_shape,
                        Default::default(),
                        "partial jobs cannot sample the noflush budget delta"
                    );
                    assert!(
                        births.is_empty(),
                        "partial tasks cannot take producer fast path"
                    );
                }
                let events = prepared.apply(&mut corpus);
                if workers == 1 {
                    assert!(
                        events
                            .chunks
                            .iter()
                            .flat_map(|chunk| &chunk.changes)
                            .all(|change| change.positions.is_empty())
                    );
                    assert!(
                        events
                            .chunks
                            .iter()
                            .flat_map(|chunk| &chunk.changes)
                            .any(|change| change.removed_weight != 0)
                    );
                }
                candidates.clear();
                index
                    .commit_merges_with_prepared(&events, vocab.len(), &execution, &arena, births)
                    .unwrap();
                index.begin_selection();
                if workers == 1 && floor == 100 {
                    assert!(index.best().is_none());
                } else {
                    assert_eq!(index.best().unwrap().priority_count, 29);
                }
            });
        }
    }
}

#[test]
fn adaptive_birth_kernels_preserve_dense_tiny_wide_affix_and_reuse_trace() {
    use merge::MergeOptions;
    let dense_words = counts(&[
        (&"ab".repeat(256), 7),
        (&"cd".repeat(160), 6),
        ("xabyabzab", 2),
        ("abcd", 1),
        ("zeroabab🙂", 0),
        ("", 1),
    ]);
    // Sixteen disjoint ordinary keys all fit the existing whole-task cut even
    // at 16 workers: total=16*64, chunk=64. Unlike a single large pair this
    // guarantees actual complete producers and third-birth promotion there.
    let balanced_words: AHashMap<CompactString, u64> = (0..16)
        .map(|i| {
            let a = char::from_u32(0x400 + 2 * i).unwrap();
            let b = char::from_u32(0x401 + 2 * i).unwrap();
            (format!("{a}{b}").repeat(64).into(), 1)
        })
        .collect();
    let affix_words = counts(&[
        (&"xabcdab中abab".repeat(40), 7),
        (&"abababaaaa中文".repeat(20), 11),
        ("zeroaaaa🙂", 0),
        ("", 1),
    ]);
    let reuse_words = counts(&[("baaba", 1), ("xyxyxy", 100)]);
    // The high-weight four-symbol word supplies a tiny complete first batch.
    // Crossed endpoints stop that prefix before the low-weight dense words.
    // Later `ab` supplies linked feedback that can re-enable collection; `bc`
    // cannot share its crossed batch and supplies a subsequent promoted task.
    let feedback_words = counts(&[
        ("pxay", 100_000),
        (&"ab".repeat(256), 1),
        (&"bc".repeat(128), 1),
    ]);
    let scenarios = [
        (
            BpeTrainer::builder()
                .vocab_size(128)
                .min_frequency(2)
                .show_progress(false)
                .build(),
            &balanced_words,
        ),
        (
            BpeTrainer::builder()
                .vocab_size(80)
                .min_frequency(2)
                .show_progress(false)
                .build(),
            &dense_words,
        ),
        // The target chooses packed24 slots without allocating a huge corpus.
        (
            BpeTrainer::builder()
                .vocab_size(65536)
                .min_frequency(2)
                .show_progress(false)
                .build(),
            &dense_words,
        ),
        (
            BpeTrainer::builder()
                .vocab_size(80)
                .min_frequency(2)
                .show_progress(false)
                .max_token_length(Some(7))
                .continuing_subword_prefix("##".into())
                .end_of_word_suffix("</w>".into())
                .special_tokens(vec![AddedToken::from("##ab", true)])
                .build(),
            &affix_words,
        ),
        (
            BpeTrainer::builder()
                .vocab_size(48)
                .min_frequency(1)
                .show_progress(false)
                .end_of_word_suffix("a".into())
                .build(),
            &reuse_words,
        ),
        (
            BpeTrainer::builder()
                .vocab_size(80)
                .min_frequency(1)
                .show_progress(false)
                .build(),
            &feedback_words,
        ),
    ];
    for (scenario, (trainer, words)) in scenarios.iter().enumerate() {
        let mut expected_trace = Vec::new();
        let expected = trainer
            .do_train_observed(words, |pair, count, id| {
                expected_trace.push((pair, count, id))
            })
            .unwrap();
        for (contiguous_births, adaptive_births) in [(false, false), (true, false), (true, true)] {
            for workers in [1, 4, 16] {
                let options = MergeOptions {
                    contiguous_births,
                    adaptive_births,
                    ..MergeOptions::default()
                };
                let mut trace = Vec::new();
                let mut birth_history = Vec::new();
                let actual = train_with_merge_options(
                    trainer,
                    WordCountsView::from_map(words),
                    workers,
                    options,
                    Some(&mut |pair, count, id| trace.push((pair, count, id))),
                    Some(&mut |event| birth_history.push(event)),
                )
                .unwrap();
                if scenario == 0 {
                    let first = birth_history
                        .iter()
                        .find_map(|event| match event {
                            BirthObservation::Round { paths, .. } => Some(paths),
                            _ => None,
                        })
                        .expect("the balanced fixture trains a batch");
                    assert_eq!(first.eligible_tasks, 16, "workers={workers}");
                    if contiguous_births {
                        assert_eq!(first.contiguous_tasks, 16);
                        assert!(first.promoted_groups >= 16, "workers={workers}");
                    } else {
                        assert_eq!(first.contiguous_tasks, 0);
                        assert_eq!(first.promoted_groups, 0);
                    }
                }
                if scenario == 4 {
                    let attempts: Vec<_> = birth_history
                        .iter()
                        .filter_map(|event| match event {
                            BirthObservation::Attempt(policy) => Some(*policy),
                            _ => None,
                        })
                        .collect();
                    assert_eq!(
                        attempts,
                        [
                            IdentityPolicy::FirstActivationOnly,
                            IdentityPolicy::AllowActiveReuse
                        ]
                    );
                    // The rebuilt attempt starts enabled even if the abandoned
                    // attempt's successful batches disabled adaptive collection.
                    let restart = birth_history
                        .iter()
                        .position(|event| {
                            matches!(
                                event,
                                BirthObservation::Attempt(IdentityPolicy::AllowActiveReuse)
                            )
                        })
                        .unwrap();
                    let enabled = birth_history[restart + 1..]
                        .iter()
                        .find_map(|event| match event {
                            BirthObservation::Round {
                                enabled_before,
                                paths,
                                ..
                            } => {
                                assert_eq!(paths.eligible_tasks, 0, "reuse stays buffered");
                                assert_eq!(paths.promoted_groups, 0);
                                Some(*enabled_before)
                            }
                            _ => None,
                        })
                        .unwrap();
                    assert_eq!(enabled, contiguous_births);

                    for event in &birth_history[restart + 1..] {
                        if let BirthObservation::Round { paths, .. } = event {
                            assert_eq!(paths.eligible_tasks, 0, "every reuse round stays buffered");
                            assert_eq!(paths.promoted_groups, 0);
                        }
                    }
                    if workers == 1 && contiguous_births && adaptive_births {
                        let old_mode = birth_history[..restart]
                            .iter()
                            .rev()
                            .find_map(|event| match event {
                                BirthObservation::Round { enabled_after, .. } => {
                                    Some(*enabled_after)
                                }
                                _ => None,
                            })
                            .unwrap();
                        assert!(!old_mode, "restart abandons a disabled policy");
                    }
                }
                if scenario == 5 && workers == 1 && contiguous_births && adaptive_births {
                    let rounds: Vec<_> = birth_history
                        .iter()
                        .filter_map(|event| match event {
                            BirthObservation::Round {
                                enabled_before,
                                enabled_after,
                                paths,
                            } => Some((*enabled_before, *enabled_after, paths)),
                            _ => None,
                        })
                        .collect();
                    assert!(rounds[0].0);
                    let disabled = rounds
                        .iter()
                        .position(|(before, after, _)| *before && !*after)
                        .expect("a tiny complete batch disables collection");
                    let reenabled = rounds
                        .iter()
                        .enumerate()
                        .skip(disabled + 1)
                        .find_map(|(i, (before, after, paths))| {
                            (!*before && *after && paths.eligible_tasks != 0).then_some(i)
                        })
                        .expect("linked complete work re-enables collection");
                    assert_eq!(rounds[reenabled].2.contiguous_tasks, 0);
                    assert!(
                        rounds[reenabled + 1..]
                            .iter()
                            .any(|(before, _, paths)| *before && paths.promoted_groups != 0),
                        "a later batch actually promotes after re-enabling"
                    );
                }
                assert_eq!(
                    trace, expected_trace,
                    "scenario={scenario}, workers={workers}, options={options:?}"
                );
                assert_eq!(
                    actual, expected,
                    "scenario={scenario}, workers={workers}, options={options:?}"
                );
            }
        }
    }
}
