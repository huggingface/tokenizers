use super::*;

#[test]
fn adaptive_birth_policy_updates_next_batch_holds_empty_and_resets() {
    let options = MergeOptions::default();
    let mut policy = ContiguousBirthPolicy::default();
    assert!(policy.options(options).contiguous_births);
    policy.observe(BirthShape::default());
    assert!(policy.options(options).contiguous_births);
    policy.observe(BirthShape {
        births: 31,
        touched_groups: 2,
    });
    assert!(!policy.options(options).contiguous_births);
    policy.observe(BirthShape::default());
    assert!(!policy.options(options).contiguous_births);
    // Linked collection still feeds history and can re-enable exactly at 16.
    policy.observe(BirthShape {
        births: 32,
        touched_groups: 2,
    });
    assert!(policy.options(options).contiguous_births);
    policy.observe(BirthShape {
        births: 0,
        touched_groups: 1,
    });
    assert!(!policy.options(options).contiguous_births);
    let restarted = ContiguousBirthPolicy::default();
    assert!(restarted.options(options).contiguous_births);
    policy.observe(BirthShape {
        births: usize::MAX,
        touched_groups: usize::MAX,
    });
    assert!(!policy.options(options).contiguous_births);
    policy.observe(BirthShape {
        births: usize::MAX,
        touched_groups: usize::MAX / 16,
    });
    assert!(policy.options(options).contiguous_births);
    // Private forced modes are test controls, not stateful production modes.
    let forced_linked = MergeOptions {
        contiguous_births: false,
        adaptive_births: false,
        ..options
    };
    let forced_contiguous = MergeOptions {
        contiguous_births: true,
        adaptive_births: false,
        ..options
    };
    policy.observe(BirthShape {
        births: 1,
        touched_groups: 1,
    });
    assert!(!policy.options(forced_linked).contiguous_births);
    assert!(policy.options(forced_contiguous).contiguous_births);
}

#[test]
fn complete_birth_shape_counts_vector_linked_zero_and_pruned_births() {
    fn fill<const CONTIGUOUS: bool>(scratch: &mut MergeScratch) {
        scratch.left::<CONTIGUOUS>(5, 2, 1, true).unwrap();
        scratch.left::<CONTIGUOUS>(5, 3, 0, true).unwrap();
        scratch.left::<CONTIGUOUS>(5, 4, 1, true).unwrap();
        scratch.right::<CONTIGUOUS>(7, 8, 5, 0, true).unwrap();
        scratch.left::<CONTIGUOUS>(9, 6, 1, false).unwrap();
    }
    let execution = Execution::new(1).unwrap();
    let arena = AllocationArena::new(1, 8);
    execution.pool.install(|| {
        for contiguous in [false, true] {
            let mut scratch =
                MergeScratch::new(10, [IdDirectory::default(), IdDirectory::default()]);
            let rule = MergeRule {
                pair: (0, 1),
                replacement: 2,
            };
            // Prior producer's nodes remain allocated, but do not enter B.
            scratch.left::<false>(6, 1, 1, true).unwrap();
            scratch.flush_rule(&rule, 0);
            let before = scratch.remaining_nodes;
            if contiguous {
                fill::<true>(&mut scratch);
            } else {
                fill::<false>(&mut scratch);
            }
            assert_eq!(
                scratch.birth_shape_since(before),
                BirthShape {
                    births: 4,
                    touched_groups: 4
                }
            );
            if contiguous {
                // Chain length2 alone would undercount the promoted group.
                assert_eq!(scratch.birth_vectors.len(), 1);
                assert_eq!(scratch.birth_vectors[0], [2, 3, 4]);
            }
            let mut births = Vec::new();
            if contiguous {
                scratch.flush_rule_with_births::<true>(
                    &rule,
                    0,
                    100,
                    &arena,
                    &execution,
                    &mut births,
                )
            } else {
                scratch.flush_rule_with_births::<false>(
                    &rule,
                    0,
                    100,
                    &arena,
                    &execution,
                    &mut births,
                )
            }
            .unwrap();
            assert!(births.is_empty(), "all sampled births were pruned");
            assert_eq!(scratch.left.touched_len() + scratch.right.touched_len(), 0);
            assert_eq!(before - scratch.remaining_nodes, 4);
        }
    });
}

#[test]
fn selected_rule_reuse_clears_shared_endpoints_before_domain_growth() {
    let mut selected = SelectedRuleIndex::default();
    selected.reset(
        &[
            MergeRule {
                pair: (1, 2),
                replacement: 4,
            },
            MergeRule {
                pair: (1, 3),
                replacement: 5,
            },
        ],
        6,
    );
    assert_eq!(selected.heads[1], MULTIPLE);
    assert_eq!(selected.multiple.len(), 2);
    selected.reset(
        &[MergeRule {
            pair: (3, 6),
            replacement: 7,
        }],
        8,
    );
    assert_eq!(selected.heads[1], EMPTY);
    assert_eq!(selected.tails[2], EMPTY);
    assert_eq!(selected.tails[3], EMPTY);
    assert!(selected.multiple.is_empty());
    assert_eq!(selected.heads[3], pair_key((6, 7)));
    assert_eq!(selected.tails[6], pair_key((3, 7)));
    selected.reset(&[], 8);
    assert_eq!(selected.heads[3], EMPTY);
    assert_eq!(selected.tails[6], EMPTY);
}

#[test]
fn complete_birth_vectors_preserve_counts_positions_buckets_and_tiny_fallback() {
    fn fill<const CONTIGUOUS: bool>(
        scratch: &mut MergeScratch,
        positions: &[u64],
        left_neighbor: u32,
        right_born: u32,
    ) -> Result<()> {
        for (i, &position) in positions.iter().enumerate() {
            let weight = (i % 5) as u64;
            scratch.left::<CONTIGUOUS>(left_neighbor, position, weight, true)?;
            scratch.right::<CONTIGUOUS>(5, right_born, position, weight, true)?;
            // Zero mass does not suppress coordinate collection/promotion.
            scratch.left::<CONTIGUOUS>(8, position, 0, true)?;
        }
        scratch.right::<CONTIGUOUS>(7, 7, 0, 11, false)
    }
    let workers = 1;
    let execution = Execution::new(workers).unwrap();
    let arena = AllocationArena::new(workers, 4096);
    execution.pool.install(|| {
        for count in [0, 1, 2, 3, 129] {
            for (left_neighbor, right_born) in [(4, 6), (3, 3)] {
                let positions: Vec<_> = (0..count)
                    .map(|i| u64::from(u32::MAX) - count as u64 + (i / 2) as u64)
                    .collect();
                let mass: u64 = (0..count).map(|i| (i % 5) as u64).sum();
                for floor in [1, mass + 1] {
                    let rule = MergeRule {
                        pair: (1, 2),
                        replacement: 3,
                    };
                    let mut expected_births = Vec::new();
                    if count != 0 && mass >= floor {
                        expected_births.push((
                            pair_key((left_neighbor, 3)),
                            mass,
                            positions.clone(),
                        ));
                        expected_births.push((pair_key((3, right_born)), mass, positions.clone()));
                    }
                    let mut expected_removals = Vec::new();
                    if mass != 0 {
                        expected_removals.push((
                            pair_key((left_neighbor, 1)),
                            pair_key((left_neighbor, 3)),
                            mass,
                            mass,
                            4,
                        ));
                        expected_removals.push((pair_key((2, 5)), pair_key((3, 5)), mass, 0, 5));
                    }
                    expected_removals.push((pair_key((2, 7)), pair_key((3, 7)), 11, 0, 5));
                    for contiguous in [false, true] {
                        execution
                            .with_merge_scratch(10, |scratch| {
                                if contiguous {
                                    fill::<true>(scratch, &positions, left_neighbor, right_born)?;
                                } else {
                                    fill::<false>(scratch, &positions, left_neighbor, right_born)?;
                                }
                                let mut births = Vec::new();
                                if contiguous {
                                    scratch.flush_rule_with_births::<true>(
                                        &rule,
                                        2,
                                        floor,
                                        &arena,
                                        &execution,
                                        &mut births,
                                    )
                                } else {
                                    scratch.flush_rule_with_births::<false>(
                                        &rule,
                                        2,
                                        floor,
                                        &arena,
                                        &execution,
                                        &mut births,
                                    )
                                }?;
                                assert!(scratch.birth_vectors.is_empty());
                                let actual_births: Vec<_> = births
                                    .iter()
                                    .map(|birth| {
                                        (
                                            birth.key,
                                            birth.weight,
                                            birth.positions.iter().collect::<Vec<_>>(),
                                        )
                                    })
                                    .collect();
                                assert_eq!(actual_births, expected_births);
                                let chunk = scratch.take_chunk();
                                let actual_removals: Vec<_> = chunk
                                    .changes
                                    .iter()
                                    .map(|event| {
                                        assert!(event.positions.is_empty());
                                        (
                                            event.removed_key,
                                            event.born_key,
                                            event.removed_weight,
                                            event.born_weight,
                                            event.bucket,
                                        )
                                    })
                                    .collect();
                                assert_eq!(actual_removals, expected_removals);
                                // Drain resets these entries before another rule.
                                for neighbor in [left_neighbor, 8] {
                                    let group = scratch.left.touch(neighbor);
                                    assert_eq!(
                                        (group.removed, group.born, group.positions.len()),
                                        (0, 0, 0)
                                    );
                                    assert!(group.vector.is_none());
                                }
                                Ok(())
                            })
                            .unwrap();
                    }
                }
            }
        }
    });
}

#[test]
fn linked_fallback_keeps_full_width_coordinates_and_overflow_order() {
    let mut scratch = MergeScratch::new(8, Default::default());
    let positions = [0, 1 << 32, 1 << 63, u64::MAX];
    for &position in &positions {
        scratch.left::<false>(4, position, 0, true).unwrap();
    }
    assert!(scratch.left.touch(4).vector.is_none());
    assert_eq!(scratch.birth_vectors.capacity(), 0);
    let rule = MergeRule {
        pair: (1, 2),
        replacement: 3,
    };
    scratch.flush_rule(&rule, 1);
    let chunk = scratch.take_chunk();
    assert_eq!(chunk.changes.len(), 1);
    let event = &chunk.changes[0];
    assert_eq!(
        (event.removed_key, event.born_key, event.bucket),
        (pair_key((4, 1)), pair_key((4, 3)), 2)
    );
    assert_eq!(
        chunk.chains.reversed(event.positions).collect::<Vec<_>>(),
        positions.into_iter().rev().collect::<Vec<_>>()
    );

    fn check_overflow<const CONTIGUOUS: bool>() {
        let mut chains = PositionChains::new();
        let mut vectors = Vec::new();
        let mut group = NeighborChanges::default();
        let mut remaining = PositionChains::MAX_NODES;
        for position in 0..3 {
            MergeScratch::birth::<CONTIGUOUS>(
                &mut group,
                &mut chains,
                &mut vectors,
                &mut remaining,
                position,
                0,
            )
            .unwrap();
        }
        group.born = u64::MAX;
        let before = remaining;
        assert!(
            MergeScratch::birth::<CONTIGUOUS>(
                &mut group,
                &mut chains,
                &mut vectors,
                &mut remaining,
                4,
                1
            )
            .is_err()
        );
        assert_eq!(group.born, u64::MAX);
        let len = group.vector.map_or(group.positions.len(), |index| {
            vectors[index.get() as usize - 1].len()
        });
        assert_eq!(len, 3);
        assert_eq!(remaining, before);
    }
    check_overflow::<false>();
    check_overflow::<true>();
}

#[test]
fn completed_births_isolate_directions_and_retire_pruned_values() {
    for workers in [1, 4] {
        let execution = Execution::new(workers).unwrap();
        let arena = AllocationArena::new(workers, 256);
        execution.pool.install(|| {
            execution
                .with_merge_scratch(12, |scratch| {
                    let rule = MergeRule {
                        pair: (1, 2),
                        replacement: 3,
                    };
                    for rank in 0..3 {
                        let left: Vec<_> = (0..3).map(|i| (100 * rank + i) as u64).collect();
                        let right: Vec<_> = left.iter().map(|&position| position + 10).collect();
                        for (&left, &right) in left.iter().zip(&right) {
                            scratch.left::<true>(4, left, 1, true)?;
                            scratch.right::<true>(5, 4, right, 1, true)?;
                            scratch.left::<true>(8, left, 0, true)?;
                        }
                        let mut births = Vec::new();
                        scratch.flush_rule_with_births::<true>(
                            &rule,
                            rank,
                            1,
                            &arena,
                            &execution,
                            &mut births,
                        )?;
                        let actual: std::collections::BTreeMap<_, _> = births
                            .iter()
                            .map(|birth| {
                                (
                                    birth.key,
                                    (birth.weight, birth.positions.iter().collect::<Vec<_>>()),
                                )
                            })
                            .collect();
                        assert_eq!(actual.len(), 2);
                        assert_eq!(actual[&pair_key((4, 3))], (3, left));
                        assert_eq!(actual[&pair_key((3, 4))], (3, right));
                        assert!(
                            scratch.birth_vectors.is_empty(),
                            "the pruned zero vector also retires"
                        );
                    }
                    // A later partial/wide producer shares this scratch, but must
                    // neither inspect prior header slots nor truncate coordinates.
                    let positions = [0, 1 << 32, 1 << 63, u64::MAX];
                    for &position in &positions {
                        scratch.left::<false>(9, position, 0, true)?;
                    }
                    assert!(scratch.left.touch(9).vector.is_none());
                    scratch.flush_rule(&rule, 3);
                    assert!(scratch.birth_vectors.is_empty());
                    let chunk = scratch.take_chunk();
                    let partial = chunk
                        .changes
                        .iter()
                        .find(|event| event.born_key == pair_key((9, 3)))
                        .unwrap();
                    assert_eq!(
                        chunk.chains.reversed(partial.positions).collect::<Vec<_>>(),
                        positions.into_iter().rev().collect::<Vec<_>>()
                    );
                    Ok(())
                })
                .unwrap();
        });
    }
}

#[test]
fn linked_complete_drain_after_vector_rule_keeps_full_width_sources() {
    let execution = Execution::new(4).unwrap();
    let arena = AllocationArena::new(4, 256);
    execution.pool.install(|| {
        execution
            .with_merge_scratch(12, |scratch| {
                let rule = MergeRule {
                    pair: (1, 2),
                    replacement: 3,
                };
                for position in [0, 1, 2] {
                    scratch.left::<true>(4, position, 1, true)?;
                }
                let mut births = Vec::new();
                scratch.flush_rule_with_births::<true>(
                    &rule,
                    0,
                    1,
                    &arena,
                    &execution,
                    &mut births,
                )?;
                let header_capacity = scratch.birth_vectors.capacity();
                assert!(header_capacity > 0);
                assert!(scratch.birth_vectors.is_empty());
                births.clear();
                let prior_changes = scratch.changes.len();
                let positions = [
                    u64::from(u32::MAX) - 1,
                    u64::from(u32::MAX) + 3,
                    u64::MAX - 3,
                ];
                for position in positions {
                    scratch.left::<false>(4, position, 1, true)?;
                    scratch.right::<false>(5, 6, position, 1, true)?;
                }
                scratch.flush_rule_with_births::<false>(
                    &rule,
                    1,
                    1,
                    &arena,
                    &execution,
                    &mut births,
                )?;
                assert!(scratch.birth_vectors.is_empty());
                assert_eq!(scratch.birth_vectors.capacity(), header_capacity);
                let actual: Vec<_> = births
                    .iter()
                    .map(|birth| {
                        (
                            birth.key,
                            birth.weight,
                            birth.positions.iter().collect::<Vec<_>>(),
                        )
                    })
                    .collect();
                assert_eq!(
                    actual,
                    [
                        (pair_key((4, 3)), 3, positions.to_vec()),
                        (pair_key((3, 6)), 3, positions.to_vec()),
                    ]
                );
                let removals: Vec<_> = scratch.changes[prior_changes..]
                    .iter()
                    .map(|event| {
                        assert!(event.positions.is_empty());
                        (event.removed_key, event.removed_weight, event.bucket)
                    })
                    .collect();
                assert_eq!(
                    removals,
                    [(pair_key((4, 1)), 3, 2), (pair_key((2, 5)), 3, 3)]
                );
                assert_eq!(scratch.left.touched_len() + scratch.right.touched_len(), 0);
                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn encoder_error_mid_drain_keeps_unconsumed_vectors_owned_until_scratch_cleanup() {
    let execution = Execution::new(4).unwrap();
    let arena = AllocationArena::new(4, 256);
    execution.pool.install(|| {
        let result: Result<()> = execution.with_merge_scratch(12, |scratch| {
            for position in [0, 1, 2] {
                scratch.left::<true>(4, position, 1, true)?;
            }
            for position in [0, 2, 1] {
                scratch.right::<true>(5, 6, position, 1, true)?;
            }
            for position in [10, 11, 12] {
                scratch.right::<true>(7, 8, position, 1, true)?;
            }
            assert_eq!(scratch.birth_vectors.len(), 3);
            let rule = MergeRule {
                pair: (1, 2),
                replacement: 3,
            };
            let mut births = Vec::new();
            let error = scratch
                .flush_rule_with_births::<true>(&rule, 0, 1, &arena, &execution, &mut births)
                .expect_err("unsorted promoted source is rejected");
            // Left publication happened before the bad right group. No
            // rollback is promised; the remaining right payload is still owned.
            assert_eq!(births.len(), 1);
            assert_eq!(births[0].positions.iter().collect::<Vec<_>>(), [0, 1, 2]);
            assert!(scratch.birth_vectors[0].is_empty());
            assert!(scratch.birth_vectors[1].is_empty());
            assert_eq!(scratch.birth_vectors[2], [10, 11, 12]);
            Err(error)
        });
        assert!(result.is_err());
        execution
            .with_merge_scratch(12, |scratch| {
                assert!(scratch.birth_vectors.is_empty());
                assert_eq!(scratch.birth_vectors.capacity(), 0);
                for neighbor in [4, 5, 6, 7, 8] {
                    for group in [scratch.left.touch(neighbor), scratch.right.touch(neighbor)] {
                        assert_eq!(
                            (group.removed, group.born, group.positions.len()),
                            (0, 0, 0)
                        );
                        assert!(group.vector.is_none());
                    }
                }
                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn promoted_birth_error_releases_values_and_reuses_clean_directories() {
    let execution = Execution::new(4).unwrap();
    execution.pool.install(|| {
        let check_clean = || {
            execution
                .with_merge_scratch(8, |scratch| {
                    for neighbor in [4, 5, 6] {
                        for group in [scratch.left.touch(neighbor), scratch.right.touch(neighbor)] {
                            assert_eq!(
                                (group.removed, group.born, group.positions.len()),
                                (0, 0, 0)
                            );
                            assert!(group.vector.is_none());
                        }
                    }
                    Ok(())
                })
                .unwrap();
        };
        let result: Result<()> = execution.with_merge_scratch(8, |scratch| {
            for position in 0..3 {
                scratch.left::<true>(4, position, 1, true)?;
                scratch.right::<true>(5, 6, position, 1, true)?;
            }
            assert!(scratch.left.touch(4).vector.is_some());
            scratch.left.touch(4).removed = u64::MAX;
            // Earlier updates remain in the failed attempt until scratch is
            // discarded. Cleanup releases values; it does not roll them back.
            scratch.left::<true>(4, 4, 1, true)
        });
        assert!(result.is_err());
        check_clean();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execution.with_merge_scratch(8, |scratch| -> Result<()> {
                for position in 0..3 {
                    scratch.left::<true>(4, position, 1, true)?;
                    scratch.right::<true>(5, 6, position, 1, true)?;
                }
                panic!("unwind after birth promotion");
            })
        }));
        assert!(panic.is_err());
        check_clean();
    });
}
