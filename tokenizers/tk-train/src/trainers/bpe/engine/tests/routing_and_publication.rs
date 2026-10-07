//! Original event order, owner routing, and parallel rule/position order.
use super::*;

#[test]
fn routing_keeps_stable_births_and_original_count_actions() {
    use super::storage::{PositionChain, PositionChains};
    use merge::{ChangeAction, EventChunk, MergeEvents, PairChanges};
    use pair_index::{pair_key, shard_for};

    let chunks = (0..7_u32)
        .map(|producer| {
            let mut chains = PositionChains::new();
            let changes = (0..97_u32)
                .map(|index| {
                    let removed_key = pair_key((u32::MAX - producer, index));
                    let born_key = if index % 5 == 0 {
                        removed_key
                    } else {
                        pair_key((index, u32::MAX - producer))
                    };
                    let mut positions = PositionChain::default();
                    if index % 3 != 0 {
                        chains
                            .push(
                                &mut positions,
                                (u64::from(producer) << 32) + u64::from(index),
                            )
                            .unwrap();
                    }
                    PairChanges {
                        removed_key,
                        born_key,
                        removed_weight: u64::from(index % 4 != 0),
                        born_weight: 0,
                        positions,
                        bucket: (producer * 41 + index * 79) % 512,
                    }
                })
                .collect();
            EventChunk { chains, changes }
        })
        .collect();
    let events = MergeEvents {
        buckets: 512,
        chunks,
    };
    for workers in [1, 4, 16, 64] {
        let routes = events.route(workers);
        for (owner, route) in routes.iter().enumerate() {
            let mut expected_actions = Vec::new();
            for (producer, chunk) in events.chunks.iter().enumerate() {
                for (index, change) in chunk.changes.iter().enumerate() {
                    let removal = change.removed_weight != 0
                        && shard_for(change.removed_key, workers) == owner;
                    let birth = !change.positions.is_empty()
                        && shard_for(change.born_key, workers) == owner;
                    if removal || birth {
                        expected_actions.push((producer, index, removal, birth));
                    }
                }
            }
            let actual_actions: Vec<_> = route
                .changes
                .iter()
                .map(|reference| {
                    (
                        reference.chunk,
                        reference.index(),
                        matches!(
                            reference.action(),
                            ChangeAction::Remove | ChangeAction::Both
                        ),
                        matches!(reference.action(), ChangeAction::Birth | ChangeAction::Both),
                    )
                })
                .collect();
            assert_eq!(
                actual_actions, expected_actions,
                "workers={workers}, owner={owner}"
            );

            // Enumerate buckets explicitly as the oracle. Inside each bucket,
            // the original producer/record order is the required stable order.
            let mut expected_births = Vec::new();
            for bucket in 0..512 {
                for (producer, chunk) in events.chunks.iter().enumerate() {
                    for (index, change) in chunk.changes.iter().enumerate() {
                        if change.bucket == bucket
                            && !change.positions.is_empty()
                            && shard_for(change.born_key, workers) == owner
                        {
                            expected_births.push((producer, index));
                        }
                    }
                }
            }
            let actual_births: Vec<_> = route
                .births
                .iter()
                .map(|&index| {
                    let reference = &route.changes[index];
                    (reference.chunk, reference.index())
                })
                .collect();
            assert_eq!(
                actual_births, expected_births,
                "workers={workers}, owner={owner}"
            );
        }
    }
    assert!(
        MergeEvents {
            buckets: 0,
            chunks: Vec::new()
        }
        .route(4)
        .iter()
        .all(|route| route.changes.is_empty() && route.births.is_empty())
    );
}

#[test]
fn reusable_routes_clear_old_actions_and_regroup_for_new_bucket_domain() {
    use super::storage::{PositionChain, PositionChains};
    use merge::{ChangeAction, EventChunk, MergeEvents, OwnerRoute, PairChanges};
    use pair_index::{ShardRouter, pair_key};

    const WORKERS: usize = 4;
    const OWNER: usize = 2;
    let router = ShardRouter::new(WORKERS);
    let keys: Vec<_> = (0..100_u32)
        .map(|id| pair_key((id, id + 101)))
        .filter(|&key| router.owner(key) == OWNER)
        .take(3)
        .collect();
    assert_eq!(keys.len(), 3);
    assert!(keys.iter().all(|&key| router.owner(key) == OWNER));

    let make_round = |buckets, rows: &[&[(u32, bool)]], key, start| {
        let chunks = rows
            .iter()
            .enumerate()
            .map(|(chunk, row)| {
                let mut chains = PositionChains::new();
                let changes = row
                    .iter()
                    .enumerate()
                    .map(|(index, &(bucket, both))| {
                        let mut positions = PositionChain::default();
                        chains
                            .push(&mut positions, start + (chunk * 10 + index) as u64)
                            .unwrap();
                        PairChanges {
                            removed_key: key,
                            born_key: key,
                            removed_weight: u64::from(both),
                            born_weight: u64::from(both),
                            positions,
                            bucket,
                        }
                    })
                    .collect();
                EventChunk { chains, changes }
            })
            .collect();
        MergeEvents { buckets, chunks }
    };
    let mut rounds = [
        (
            make_round(
                8,
                &[
                    &[(7, false), (0, true), (7, false)],
                    &[(2, true), (0, false)],
                ],
                keys[0],
                100,
            ),
            vec![
                (0, 1, 0, true),
                (1, 1, 0, false),
                (1, 0, 2, true),
                (0, 0, 7, false),
                (0, 2, 7, false),
            ],
        ),
        (
            make_round(
                2,
                &[&[(1, false), (0, true)], &[(1, true), (0, false)]],
                keys[1],
                200,
            ),
            vec![
                (0, 1, 0, true),
                (1, 1, 0, false),
                (0, 0, 1, false),
                (1, 0, 1, true),
            ],
        ),
        (
            make_round(
                11,
                &[
                    &[(10, true), (3, false), (10, false)],
                    &[(0, false), (3, true), (10, false)],
                ],
                keys[2],
                300,
            ),
            vec![
                (1, 0, 0, false),
                (0, 1, 3, false),
                (1, 1, 3, true),
                (0, 0, 10, true),
                (0, 2, 10, false),
                (1, 2, 10, false),
            ],
        ),
    ];
    // This action must disappear at the next dispatch, independently of births.
    rounds[0].0.chunks[0].changes.push(PairChanges {
        removed_key: keys[0],
        born_key: keys[0],
        removed_weight: 1,
        born_weight: 0,
        positions: PositionChain::default(),
        bucket: 7,
    });
    let mut routes = (0..WORKERS)
        .map(|_| OwnerRoute::default())
        .collect::<Vec<_>>();
    let mut first_capacities = None;
    for (round, (events, expected_births)) in rounds.iter().enumerate() {
        events.dispatch_into(&mut routes, router);
        assert!(routes.iter().enumerate().all(|(owner, route)| {
            if owner == OWNER {
                route.births.len() == expected_births.len()
            } else {
                route.changes.is_empty() && route.births.is_empty()
            }
        }));
        // Every round must enter group_births beyond its <2-birth shortcut.
        assert!(routes[OWNER].births.len() >= 2);
        for route in &mut routes {
            route.group_births(events);
        }

        let describe = |index: usize| {
            let reference = &routes[OWNER].changes[index];
            let change = &events.chunks[reference.chunk].changes[reference.index()];
            let action = match reference.action() {
                ChangeAction::Birth => "birth",
                ChangeAction::Both => "both",
                ChangeAction::Remove => "remove",
            };
            (
                reference.chunk,
                reference.index(),
                change.bucket,
                action,
                change.removed_key,
                change.born_key,
                events.chunks[reference.chunk]
                    .chains
                    .reversed(change.positions)
                    .collect::<Vec<_>>(),
            )
        };
        let expected = |&(chunk, index, bucket, both): &(usize, usize, u32, bool)| {
            (
                chunk,
                index,
                bucket,
                if both { "both" } else { "birth" },
                keys[round],
                keys[round],
                vec![(round as u64 + 1) * 100 + (chunk * 10 + index) as u64],
            )
        };
        let actual_births: Vec<_> = routes[OWNER]
            .births
            .iter()
            .map(|&index| describe(index))
            .collect();
        assert_eq!(
            actual_births,
            expected_births.iter().map(expected).collect::<Vec<_>>(),
            "round={round}: stable buckets must reference this round's events"
        );
        // Grouping reorders births only. Checked count actions keep the original
        // producer/record sequence, including each Both action.
        let mut original: Vec<_> = expected_births.iter().map(expected).collect();
        if round == 0 {
            original.push((0, 3, 7, "remove", keys[0], keys[0], Vec::new()));
        }
        original.sort_unstable_by_key(|&(chunk, index, ..)| (chunk, index));
        let actual_actions: Vec<_> = (0..routes[OWNER].changes.len()).map(describe).collect();
        assert_eq!(actual_actions, original);

        let capacities: Vec<_> = routes
            .iter()
            .map(|route| (route.changes.capacity(), route.births.capacity()))
            .collect();
        if let Some(first) = &first_capacities {
            assert!(capacities.iter().zip(first).all(
                |(current, first): (&(usize, usize), &(usize, usize))| current.0 >= first.0
                    && current.1 >= first.1
            ));
        } else {
            first_capacities = Some(capacities);
        }
    }
    let capacities: Vec<_> = routes
        .iter()
        .map(|route| (route.changes.capacity(), route.births.capacity()))
        .collect();
    let empty = MergeEvents {
        buckets: 0,
        chunks: Vec::new(),
    };
    empty.dispatch_into(&mut routes, router);
    for (route, capacity) in routes.iter_mut().zip(capacities) {
        route.group_births(&empty);
        route.assert_cleared();
        assert_eq!(
            (route.changes.capacity(), route.births.capacity()),
            capacity
        );
    }
}

#[test]
fn routed_batches_preserve_rule_and_position_order_with_many_workers() {
    let mut words = counts(&[("aaaaaaa", 5), ("abcabc", 3), ("baab", 0), ("", 1)]);
    for index in 0..128_u32 {
        let word: String = (0..3)
            .map(|offset| char::from_u32(0x4000 + index * 3 + offset).unwrap())
            .collect();
        words.insert(word.into(), 2 + u64::from(index % 3));
    }
    let trainer = BpeTrainer::builder()
        .vocab_size(700)
        .min_frequency(2)
        .show_progress(false)
        .build();
    check_with_workers(&trainer, &words, &[1, 4, 16, 64]);

    let mut aliases = trainer;
    aliases.continuing_subword_prefix = Some("a".into());
    aliases.end_of_word_suffix = Some("a".into());
    aliases.max_token_length = Some(5);
    check_with_workers(
        &aliases,
        &counts(&[("aaaaaaa", 5), ("abcabc", 3), ("baab", 0)]),
        &[1, 16, 64],
    );
}
