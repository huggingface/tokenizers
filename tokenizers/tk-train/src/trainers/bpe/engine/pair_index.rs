//! Frequency interpretation, candidate snapshots, and birth cohort ownership.
//! Fresh domains can retire low counts permanently. Reusable identities preserve
//! a signed ledger and a separate position owner for each published birth cohort.
mod commit;
use super::storage::SortedPositions;
use super::{IdentityPolicy, initial_pairs::InitialPairTable};
use ahash::AHashMap;
use dary_heap::OctonaryHeap;
use rayon::prelude::*;
use std::{cmp::Ordering, collections::VecDeque};
use tk_encode::{Result, models::bpe::Pair};

#[inline]
pub(super) fn pair_key(pair: Pair) -> u64 {
    (u64::from(pair.0) << 32) | u64::from(pair.1)
}
#[inline]
pub(super) fn key_pair(key: u64) -> Pair {
    ((key >> 32) as u32, key as u32)
}
#[inline]
pub(super) fn shard_for(key: u64, shards: usize) -> usize {
    // This choice changes ownership only; pair priority is independent of it.
    let mixed = ((key ^ (key >> 32)).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 32) as usize;
    if shards.is_power_of_two() {
        mixed & (shards - 1)
    } else {
        mixed % shards
    }
}
/// Exact original ownership with division precomputed once for non-power-of-two
/// counts. The mixed key is u32: reciprocal rounding can undershoot by at most
/// one quotient unit, repaired by one subtraction. Ownership never changes.
#[derive(Clone, Copy)]
pub(super) struct ShardRouter {
    shards: usize,
    reciprocal: Option<(u64, u64)>,
}
impl ShardRouter {
    pub(super) fn new(shards: usize) -> Self {
        assert!(shards != 0);
        let reciprocal = if !shards.is_power_of_two() && u32::try_from(shards).is_ok() {
            Some(((1_u64 << 32) / shards as u64, shards as u64))
        } else {
            None
        };
        Self { shards, reciprocal }
    }
    pub(super) fn shards(self) -> usize {
        self.shards
    }
    #[inline]
    pub(super) fn owner(self, key: u64) -> usize {
        let mixed = ((key ^ (key >> 32)).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 32) as u32;
        self.mixed_owner(mixed)
    }
    #[inline]
    fn mixed_owner(self, mixed: u32) -> usize {
        if self.shards.is_power_of_two() {
            mixed as usize & (self.shards - 1)
        } else if let Some((reciprocal, divisor)) = self.reciprocal {
            let mixed = u64::from(mixed);
            let quotient = (mixed * reciprocal) >> 32;
            let remainder = mixed - quotient * divisor;
            (if remainder >= divisor {
                remainder - divisor
            } else {
                remainder
            }) as usize
        } else {
            mixed as usize % self.shards
        }
    }
}

/// Count and occurrence-list ownership used by initialization and fresh keys.
/// `ledger_count_bits` is a nonnegative weighted count in these states. Reuse
/// initialization moves counts into a separate signed ledger and transfers
/// positions to independently owned cohorts. Lists may contain stale positions.
pub(super) struct PairState<'arena> {
    pub(super) ledger_count_bits: u64,
    pub(super) positions: SortedPositions<'arena>,
}
/// Cached selection priority: larger count, then smaller pair-ID key, wins.
/// Fresh snapshots are decreasing-count upper bounds, certified by `best` before
/// removal. Reuse snapshots hold the shared signed ledger's `u64` bit pattern and
/// are lazily repaired at the queue frontier; they are not fresh upper bounds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PairPriority {
    pub(super) key: u64,
    pub(super) priority_count: u64,
}
impl Ord for PairPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority_count
            .cmp(&other.priority_count)
            .then_with(|| other.key.cmp(&self.key))
    }
}
impl PartialOrd for PairPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
/// An occurrence-list owner paired with a possibly stale priority snapshot.
/// Weighted count differs from list length; preparation validates old positions
/// against the live corpus. Reuse may retain multiple cohorts for one pair, whose
/// priorities snapshot the shared ledger rather than individual cohort mass.
pub(super) struct MergeCandidate<'arena> {
    pub(super) priority: PairPriority,
    pub(super) positions: SortedPositions<'arena>,
}
impl Eq for MergeCandidate<'_> {}
impl PartialEq for MergeCandidate<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}
impl Ord for MergeCandidate<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
    }
}
impl PartialOrd for MergeCandidate<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct PairShard<'arena> {
    // Fresh states own position lists. Reusable IDs publish independent cohorts, so
    // their count table stores only numeric ledger bits, including zero/negative
    // values. Exactly one table is populated after initialization.
    states: AHashMap<u64, PairState<'arena>>,
    ledger: AHashMap<u64, u64>,
    priorities: OctonaryHeap<PairPriority>,
    prefix: VecDeque<PairPriority>,
}
enum Selection<'arena> {
    Fresh {
        leaders: OctonaryHeap<(PairPriority, usize)>,
    },
    Cohorts {
        candidates: OctonaryHeap<MergeCandidate<'arena>>,
    },
}
pub(super) struct PairIndex<'arena> {
    shards: Vec<PairShard<'arena>>,
    policy: IdentityPolicy,
    minimum_frequency: u64,
    selection: Selection<'arena>,
    routes: Vec<super::merge::OwnerRoute>,
    prepared_births: Vec<Vec<super::merge::CompletedBirth<'arena>>>,
}

impl PairShard<'_> {
    fn subtract_fresh(&mut self, key: u64, amount: u64, floor: u64) -> Result<()> {
        if let Some(state) = self.states.get_mut(&key) {
            state.ledger_count_bits = state
                .ledger_count_bits
                .checked_sub(amount)
                .ok_or("BPE fresh removal exceeds the current count")?;
            if state.ledger_count_bits < floor {
                self.states.remove(&key);
            }
        }
        Ok(())
    }
}

impl PairShard<'_> {
    fn upper(&self) -> Option<PairPriority> {
        self.prefix
            .front()
            .copied()
            .or_else(|| self.priorities.peek().copied())
    }
    fn exact(&mut self, floor: u64) -> Option<PairPriority> {
        if self.prefix.is_empty() {
            self.prepare_prefix(floor);
        }
        self.prefix.front().copied()
    }
    fn heap_exact(&mut self, floor: u64) -> Option<PairPriority> {
        loop {
            let top = self.priorities.peek().copied()?;
            let Some(state) = self.states.get(&top.key) else {
                self.priorities.pop();
                continue;
            };
            let count = state.ledger_count_bits;
            if count < floor {
                self.states.remove(&top.key);
                self.priorities.pop();
                continue;
            }
            if top.priority_count != count {
                // Counts of an existing fresh key only decrease. Correcting its
                // upper bound can reveal another winner; compare again afterward.
                *self
                    .priorities
                    .peek_mut()
                    .expect("the observed heap head exists") = PairPriority {
                    key: top.key,
                    priority_count: count,
                };
                continue;
            }
            return Some(top);
        }
    }
    fn consume(&mut self) {
        if self.prefix.pop_front().is_none() {
            self.priorities.pop();
        }
    }
    fn restore_prefix(&mut self) {
        for candidate in self.prefix.drain(..) {
            self.priorities.push(candidate);
        }
    }
    fn prepare_prefix(&mut self, floor: u64) {
        for _ in self.prefix.len()..4 {
            let Some(candidate) = self.heap_exact(floor) else {
                break;
            };
            self.priorities.pop();
            self.prefix.push_back(candidate);
        }
    }
}
impl<'arena> PairIndex<'arena> {
    pub(super) fn from_initial_pairs(
        initial: InitialPairTable<'arena>,
        policy: IdentityPolicy,
        minimum_frequency: u64,
    ) -> Result<Self> {
        if policy == IdentityPolicy::AllowActiveReuse
            && (initial.weighted_mass > i64::MAX as u128
                || initial.maximum_word_weight > i64::MAX as u64)
        {
            return Err("BPE identity-reuse weighted edge mass or word weight exceeds i64".into());
        }
        let outputs: Vec<_> = initial
            .shards
            .into_par_iter()
            .map(|mut states| {
                let mut candidates = Vec::new();
                let mut ledger = AHashMap::new();
                let priorities = if policy == IdentityPolicy::FirstActivationOnly {
                    states
                        .iter()
                        .map(|(&key, state)| PairPriority {
                            key,
                            priority_count: state.ledger_count_bits,
                        })
                        .collect()
                } else {
                    ledger =
                        AHashMap::with_capacity_and_hasher(states.len(), states.hasher().clone());
                    for (key, state) in std::mem::take(&mut states) {
                        ledger.insert(key, state.ledger_count_bits);
                        if state.ledger_count_bits > 0 {
                            candidates.push(MergeCandidate {
                                priority: PairPriority {
                                    key,
                                    priority_count: state.ledger_count_bits,
                                },
                                positions: state.positions,
                            });
                        }
                    }
                    OctonaryHeap::new()
                };
                let mut shard = PairShard {
                    states,
                    ledger,
                    priorities,
                    prefix: VecDeque::with_capacity(4),
                };
                if policy == IdentityPolicy::FirstActivationOnly {
                    shard.prepare_prefix(minimum_frequency.max(1));
                }
                (shard, candidates)
            })
            .collect();
        let (shards, candidates): (Vec<_>, Vec<_>) = outputs.into_iter().unzip();
        let candidates = candidates.into_iter().flatten().collect::<Vec<_>>().into();
        let selection = match policy {
            IdentityPolicy::FirstActivationOnly => Selection::Fresh {
                leaders: OctonaryHeap::new(),
            },
            IdentityPolicy::AllowActiveReuse => Selection::Cohorts { candidates },
        };
        Ok(Self {
            shards,
            policy,
            minimum_frequency,
            selection,
            routes: Vec::new(),
            prepared_births: Vec::new(),
        })
    }

    /// Build the fresh owner frontier for a selection phase.
    /// Count updates wait until `end_selection` returns cached prefixes to heaps.
    pub(super) fn begin_selection(&mut self) {
        if let Selection::Fresh { leaders } = &mut self.selection {
            let mut heads = std::mem::take(leaders).into_vec();
            heads.clear();
            for (shard, state) in self.shards.iter().enumerate() {
                if let Some(priority) = state.upper() {
                    heads.push((priority, shard));
                }
            }
            *leaders = heads.into();
        }
    }
    /// Certify the current selection frontier, repairing stale priorities.
    /// Fresh selection compares decreasing upper bounds across owners. Reuse
    /// repairs only the head cohort against its shared ledger and retains the
    /// existing unsigned ordering of signed ledger bits, including negative values.
    /// Call `take_best` before changing selection/count state to consume this winner.
    pub(super) fn best(&mut self) -> Option<PairPriority> {
        match &mut self.selection {
            Selection::Fresh { leaders } => {
                best_first_activation(&mut self.shards, leaders, self.minimum_frequency.max(1))
            }
            Selection::Cohorts { candidates } => {
                best_active_reuse(&self.shards, candidates, self.minimum_frequency)
            }
        }
    }
    /// Consume the winner most recently certified by `best`.
    /// The caller must not change selection state between certification and removal.
    /// Fresh mode transfers the pair state's positions. Reuse removes one cohort
    /// while leaving its shared ledger and other cohorts intact.
    /// The returned list retains its `'arena` storage lifetime after this borrow ends.
    pub(super) fn take_best(&mut self) -> MergeCandidate<'arena> {
        match &mut self.selection {
            Selection::Fresh { leaders } => {
                let (priority, shard) = *leaders
                    .peek()
                    .expect("selection certified the global winner");
                self.shards[shard].consume();
                if let Some(next) = self.shards[shard].upper() {
                    *leaders.peek_mut().expect("the observed leader exists") = (next, shard);
                } else {
                    leaders.pop();
                }
                let state = self.shards[shard]
                    .states
                    .remove(&priority.key)
                    .expect("the certified pair has a state");
                MergeCandidate {
                    priority,
                    positions: state.positions,
                }
            }
            Selection::Cohorts { candidates } => candidates
                .pop()
                .expect("selection certified the cohort winner"),
        }
    }
    /// Return unconsumed fresh prefixes to heaps before commit changes counts.
    /// Keeping a cached prefix through commit would bypass stale-count repair.
    pub(super) fn end_selection(&mut self) {
        if self.policy == IdentityPolicy::FirstActivationOnly {
            for shard in &mut self.shards {
                shard.restore_prefix();
            }
        }
    }
    #[cfg(test)]
    pub(super) fn prepare_prefixes(&mut self) {
        if self.policy == IdentityPolicy::FirstActivationOnly {
            let floor = self.minimum_frequency.max(1);
            // Callers install this operation in the training pool.
            use rayon::prelude::*;
            self.shards
                .par_iter_mut()
                .for_each(|shard| shard.prepare_prefix(floor));
        }
    }
}

/// Certify a fresh winner by comparing decreasing count upper bounds across owners.
/// Existing fresh keys only lose mass; removed keys cannot revive in this attempt.
fn best_first_activation(
    shards: &mut [PairShard<'_>],
    leaders: &mut OctonaryHeap<(PairPriority, usize)>,
    floor: u64,
) -> Option<PairPriority> {
    loop {
        let (upper, shard) = leaders.peek().copied()?;
        let exact = shards[shard].exact(floor);
        if exact == Some(upper) {
            return exact;
        }
        if let Some(priority) = exact {
            *leaders.peek_mut().expect("the observed leader exists") = (priority, shard);
        } else {
            leaders.pop();
        }
    }
}

/// Repair only the head cohort using the pair's shared ledger snapshot.
/// Preserve the reference queue's unsigned ordering of signed bits, including
/// negative values. This selection condition differs from positive-ledger birth
/// publication and provides no decreasing upper-bound proof.
fn best_active_reuse(
    shards: &[PairShard<'_>],
    candidates: &mut OctonaryHeap<MergeCandidate<'_>>,
    minimum_frequency: u64,
) -> Option<PairPriority> {
    loop {
        let top = candidates.peek()?;
        let key = top.priority.key;
        let count = shards[shard_for(key, shards.len())].ledger[&key];
        if top.priority.priority_count == count {
            return (count != 0 && count >= minimum_frequency).then_some(top.priority);
        }
        let mut top = candidates.pop().expect("the observed candidate exists");
        top.priority.priority_count = count;
        candidates.push(top);
    }
}

#[cfg(test)]
mod tests {
    use super::super::storage::{AllocationArena, PositionChain, PositionChains};
    use super::super::{
        execution::Execution,
        merge::{CompletedBirth, EventChunk, MergeEvents, PairChanges},
    };
    use super::*;
    fn initial<'arena>(
        items: &[(Pair, u64, u64)],
        workers: usize,
        arena: &'arena AllocationArena,
    ) -> InitialPairTable<'arena> {
        let mut shards: Vec<_> = (0..workers).map(|_| AHashMap::new()).collect();
        let mut scratch = super::super::storage::PositionEncodingScratch::default();
        for &(pair, count, position) in items {
            let key = pair_key(pair);
            let owner = shard_for(key, workers);
            let lease = arena.lease(owner);
            shards[owner].insert(
                key,
                PairState {
                    ledger_count_bits: count,
                    positions: SortedPositions::from_sorted(&[position], &mut scratch, &lease)
                        .unwrap(),
                },
            );
        }
        InitialPairTable {
            shards,
            weighted_mass: items.iter().map(|item| u128::from(item.1)).sum(),
            maximum_word_weight: items.iter().map(|item| item.1).max().unwrap_or(0),
        }
    }

    #[test]
    fn completed_and_partial_births_publish_once_after_ordered_removals() {
        for workers in [1, 4] {
            let execution = Execution::new(workers).unwrap();
            let arena = AllocationArena::new(workers, 64);
            execution.pool.install(|| {
                let old = (0, 1);
                let complete = (2, 3);
                let partial = (4, 5);
                let mut index = PairIndex::from_initial_pairs(
                    initial(&[(old, 3, 1)], workers, &arena),
                    IdentityPolicy::FirstActivationOnly,
                    2,
                )
                .unwrap();
                // Commit follows end_selection, which returns cached fresh
                // priorities to their heaps before count changes invalidate them.
                index.begin_selection();
                index.end_selection();
                let positions = {
                    let lease = arena.lease(execution.current_worker());
                    let mut scratch = super::super::storage::PositionEncodingScratch::default();
                    SortedPositions::from_sorted(&[9, 12], &mut scratch, &lease).unwrap()
                };
                let mut events = MergeEvents {
                    buckets: 2,
                    chunks: Vec::new(),
                };
                // Each fragment is below the floor; together they must be kept.
                for position in [2, 7] {
                    let mut chains = PositionChains::new();
                    let mut positions = PositionChain::default();
                    chains.push(&mut positions, position).unwrap();
                    events.chunks.push(EventChunk {
                        chains,
                        changes: vec![PairChanges {
                            removed_key: pair_key(old),
                            born_key: pair_key(partial),
                            removed_weight: 1,
                            born_weight: 1,
                            positions,
                            bucket: 1,
                        }],
                    });
                }
                index
                    .commit_merges_with_prepared(
                        &events,
                        6,
                        &execution,
                        &arena,
                        vec![CompletedBirth {
                            key: pair_key(complete),
                            weight: 5,
                            positions,
                        }],
                    )
                    .unwrap();
                drop(events);
                index.begin_selection();
                for (pair, count, expected_positions) in
                    [(complete, 5, [9, 12]), (partial, 2, [2, 7])]
                {
                    assert_eq!(
                        index.best().unwrap(),
                        PairPriority {
                            key: pair_key(pair),
                            priority_count: count,
                        }
                    );
                    assert_eq!(
                        index.take_best().positions.iter().collect::<Vec<_>>(),
                        expected_positions,
                    );
                }
                assert!(index.best().is_none(), "no retired key or duplicate birth");
            });
        }
    }

    #[test]
    fn reuse_both_checks_removal_before_birth_without_netting() {
        let execution = Execution::new(1).unwrap();
        let arena = AllocationArena::new(1, 16);
        execution.pool.install(|| {
            let pair = (1, 2);
            let mut index = PairIndex::from_initial_pairs(
                initial(&[(pair, i64::MAX as u64, 1)], 1, &arena),
                IdentityPolicy::AllowActiveReuse,
                1,
            )
            .unwrap();
            let mut chains = PositionChains::new();
            let mut positions = PositionChain::default();
            chains.push(&mut positions, 5).unwrap();
            let events = MergeEvents {
                buckets: 2,
                chunks: vec![EventChunk {
                    chains,
                    changes: vec![PairChanges {
                        removed_key: pair_key(pair),
                        born_key: pair_key(pair),
                        removed_weight: 1,
                        born_weight: 1,
                        positions,
                        bucket: 1,
                    }],
                }],
            };
            // Addition first would overflow, although removal first fits.
            index.commit_merges(&events, 3, &execution, &arena).unwrap();
            assert_eq!(index.shards[0].ledger[&pair_key(pair)], i64::MAX as u64);
            index.shards[0]
                .ledger
                .insert(pair_key(pair), i64::MIN as u64);
            // A zero net update still fails at the intermediate subtraction.
            let error = index
                .commit_merges(&events, 3, &execution, &arena)
                .unwrap_err();
            assert_eq!(
                error.to_string(),
                "BPE identity-reuse count subtraction exceeds i64"
            );
            for route in &index.routes {
                route.assert_cleared();
            }
            assert!(index.prepared_births.iter().all(Vec::is_empty));
        });
    }

    #[test]
    fn fresh_count_error_clears_routed_and_prepared_births_after_join() {
        let execution = Execution::new(1).unwrap();
        let arena = AllocationArena::new(1, 16);
        execution.pool.install(|| {
            let old = (0, 1);
            let partial = (2, 3);
            let complete = (4, 5);
            let mut index = PairIndex::from_initial_pairs(
                initial(&[(old, 4, 1)], 1, &arena),
                IdentityPolicy::FirstActivationOnly,
                1,
            )
            .unwrap();
            index.begin_selection();
            index.end_selection();
            let positions = {
                let lease = arena.lease(execution.current_worker());
                let mut scratch = super::super::storage::PositionEncodingScratch::default();
                SortedPositions::from_sorted(&[9, 12], &mut scratch, &lease).unwrap()
            };
            let mut events = MergeEvents {
                buckets: 2,
                chunks: Vec::new(),
            };
            // Both births route to the sole owner and fill grouping scratch.
            // The second removal fails after the first has changed the count.
            for (position, removed_weight) in [(2, 1), (7, 4)] {
                let mut chains = PositionChains::new();
                let mut positions = PositionChain::default();
                chains.push(&mut positions, position).unwrap();
                events.chunks.push(EventChunk {
                    chains,
                    changes: vec![PairChanges {
                        removed_key: pair_key(old),
                        born_key: pair_key(partial),
                        removed_weight,
                        born_weight: 1,
                        positions,
                        bucket: 1,
                    }],
                });
            }
            let fixture_route = events.route(1);
            assert_eq!(fixture_route[0].births.len(), 2);
            // Prepared births are legal only under the fresh policy, and this
            // complete key is absent from the routed partial births.
            assert_ne!(complete, partial);
            let error = index
                .commit_merges_with_prepared(
                    &events,
                    6,
                    &execution,
                    &arena,
                    vec![CompletedBirth {
                        key: pair_key(complete),
                        weight: 5,
                        positions,
                    }],
                )
                .unwrap_err();
            assert_eq!(
                error.to_string(),
                "BPE fresh removal exceeds the current count"
            );
            assert_eq!(index.routes.len(), 1);
            for route in &index.routes {
                route.assert_cleared();
            }
            assert_eq!(index.prepared_births.len(), 1);
            assert!(index.prepared_births[0].capacity() >= 1);
            assert!(index.prepared_births.iter().all(Vec::is_empty));
            // Cleanup discards buffered entries, not earlier count mutations.
            // The failed attempt is discarded; selection must not resume.
            assert_eq!(index.shards[0].states[&pair_key(old)].ledger_count_bits, 3);
        });
    }

    #[test]
    fn interrupted_frontier_corrects_stale_counts_and_keeps_birth_ties() {
        let execution = Execution::new(2).unwrap();
        let arena = AllocationArena::new(2, 6);
        execution.pool.install(|| {
            let items = [
                ((0, 1), 9, 1),
                ((1, 2), 8, 2),
                ((2, 3), 8, 3),
                ((3, 4), 5, 4),
                ((5, 6), 8, 5),
                ((6, 7), 7, 6),
            ];
            let mut index = PairIndex::from_initial_pairs(
                initial(&items, 2, &arena),
                IdentityPolicy::FirstActivationOnly,
                3,
            )
            .unwrap();
            index.prepare_prefixes();
            index.begin_selection();
            assert_eq!(key_pair(index.best().unwrap().key), (0, 1));
            index.take_best();
            index.end_selection();
            let mut chains = PositionChains::new();
            let mut positions = PositionChain::default();
            chains.push(&mut positions, 7).unwrap();
            let mut changes: Vec<_> = [((1, 2), 6), ((2, 3), 4), ((3, 4), 5)]
                .into_iter()
                .map(|(pair, weight)| PairChanges {
                    removed_key: pair_key(pair),
                    born_key: pair_key((4, 8)),
                    removed_weight: weight,
                    born_weight: 0,
                    positions: PositionChain::default(),
                    bucket: 1,
                })
                .collect();
            changes.push(PairChanges {
                removed_key: pair_key((8, 9)),
                born_key: pair_key((4, 8)),
                removed_weight: 0,
                born_weight: 8,
                positions,

                bucket: 1,
            });
            let events = MergeEvents {
                buckets: 2,
                chunks: vec![EventChunk { chains, changes }],
            };
            index
                .commit_merges(&events, 10, &execution, &arena)
                .unwrap();
            drop(events);
            index.prepare_prefixes();
            // Independent priority list: descend by frequency, ascend by full pair.
            let mut expected = vec![((2, 3), 4), ((5, 6), 8), ((6, 7), 7), ((4, 8), 8)];
            expected.sort_by(|(left, lc), (right, rc)| rc.cmp(lc).then_with(|| left.cmp(right)));
            for (pair, count) in expected {
                index.begin_selection();
                let winner = index.best().unwrap();
                assert_eq!((key_pair(winner.key), winner.priority_count), (pair, count));
                index.take_best();
                index.end_selection();
            }
            index.begin_selection();
            assert!(index.best().is_none());
            index.end_selection();
        });
    }
    #[test]
    fn cohort_negative_count_repairs_only_when_its_snapshot_reaches_the_head() {
        let execution = Execution::new(2).unwrap();
        let arena = AllocationArena::new(2, 3);
        execution.pool.install(|| {
            let low = (0, 1);
            let high = (1..)
                .map(|id| (id, id + 1))
                .find(|&pair| shard_for(pair_key(pair), 2) != shard_for(pair_key(low), 2))
                .unwrap();
            assert_ne!(shard_for(pair_key(low), 2), shard_for(pair_key(high), 2));
            let mut index = PairIndex::from_initial_pairs(
                initial(&[(low, 1, 1), (high, 2, 2)], 2, &arena),
                IdentityPolicy::AllowActiveReuse,
                1,
            )
            .unwrap();
            let events = MergeEvents {
                buckets: 2,
                chunks: vec![EventChunk {
                    chains: PositionChains::new(),
                    changes: vec![PairChanges {
                        removed_key: pair_key(low),
                        born_key: pair_key((3, 4)),
                        removed_weight: 2,
                        born_weight: 0,
                        positions: PositionChain::default(),
                        bucket: 1,
                    }],
                }],
            };
            index.commit_merges(&events, 5, &execution, &arena).unwrap();
            drop(events);
            // Mainline repairs only the global heap head. Eagerly repairing each
            // shard would raise the negative count first and change this order.
            assert_eq!(key_pair(index.best().unwrap().key), high);
            index.take_best();
            let winner = index.best().unwrap();
            assert_eq!(
                (key_pair(winner.key), winner.priority_count),
                (low, (-1_i64) as u64)
            );
        });
    }
    #[test]
    fn cohort_birth_publication_keeps_low_positive_cohorts_and_skips_negative_counts() {
        let execution = Execution::new(1).unwrap();
        let arena = AllocationArena::new(1, 16);
        execution.pool.install(|| {
            let pair = (1, 2);
            let mut index = PairIndex::from_initial_pairs(
                initial(&[(pair, 0, 1), ((3, 4), 10, 2)], 1, &arena),
                IdentityPolicy::AllowActiveReuse,
                3,
            )
            .unwrap();
            for (position, weight) in [(5, 1), (7, 2)] {
                let mut chains = PositionChains::new();
                let mut positions = PositionChain::default();
                chains.push(&mut positions, position).unwrap();
                let events = MergeEvents {
                    buckets: 2,
                    chunks: vec![EventChunk {
                        chains,
                        changes: vec![PairChanges {
                            removed_key: pair_key((8, 9)),
                            born_key: pair_key(pair),
                            removed_weight: 0,
                            born_weight: weight,
                            positions,
                            bucket: 1,
                        }],
                    }],
                };
                index
                    .commit_merges(&events, 10, &execution, &arena)
                    .unwrap();
                drop(events);
            }
            assert_eq!(key_pair(index.best().unwrap().key), (3, 4));
            index.take_best();
            for position in [7, 5] {
                assert_eq!(index.best().unwrap().priority_count, 3);
                assert_eq!(
                    index.take_best().positions.iter().collect::<Vec<_>>(),
                    [position]
                );
            }
            assert!(index.best().is_none());
            let mut index = PairIndex::from_initial_pairs(
                initial(&[(pair, 0, 1)], 1, &arena),
                IdentityPolicy::AllowActiveReuse,
                1,
            )
            .unwrap();
            let mut chains = PositionChains::new();
            let mut positions = PositionChain::default();
            chains.push(&mut positions, 5).unwrap();
            let events = MergeEvents {
                buckets: 2,
                chunks: vec![EventChunk {
                    chains,
                    changes: vec![PairChanges {
                        removed_key: pair_key(pair),
                        born_key: pair_key(pair),
                        removed_weight: 2,
                        born_weight: 1,
                        positions,
                        bucket: 1,
                    }],
                }],
            };
            index.commit_merges(&events, 3, &execution, &arena).unwrap();
            drop(events);
            assert!(index.best().is_none());
        });
    }
    #[test]
    fn fresh_queue_preserves_survivors_after_retiring_other_keys() {
        let execution = Execution::new(1).unwrap();
        let arena = AllocationArena::new(1, 64);
        execution.pool.install(|| {
            let items: Vec<_> = (0..64)
                .map(|id| ((id, 30_000), 20, u64::from(id) + 1))
                .collect();
            let mut index = PairIndex::from_initial_pairs(
                initial(&items, 1, &arena),
                IdentityPolicy::FirstActivationOnly,
                2,
            )
            .unwrap();
            index.end_selection();
            let changes = (0..64)
                .filter(|id| id % 10 != 0)
                .map(|id| PairChanges {
                    removed_key: pair_key((id, 30_000)),
                    born_key: pair_key((30_001, 30_002)),
                    removed_weight: 20,
                    born_weight: 0,
                    positions: PositionChain::default(),
                    bucket: 1,
                })
                .collect();
            let events = MergeEvents {
                buckets: 2,
                chunks: vec![EventChunk {
                    chains: PositionChains::new(),
                    changes,
                }],
            };
            index
                .commit_merges(&events, 30_003, &execution, &arena)
                .unwrap();
            drop(events);
            for id in (0..64).step_by(10) {
                index.begin_selection();
                let next = index.best().unwrap();
                assert_eq!(
                    (key_pair(next.key), next.priority_count),
                    ((id, 30_000), 20)
                );
                index.take_best();
                index.end_selection();
            }
            index.begin_selection();
            assert!(index.best().is_none());
            index.end_selection();
        });
    }
}

#[cfg(test)]
mod router_tests {
    use super::{ShardRouter, shard_for};
    #[test]
    fn reciprocal_router_is_exact_for_full_mixed_range_boundaries() {
        for shards in [1, 2, 3, 5, 6, 7, 8, 12, 24, 64, u32::MAX as usize] {
            let router = ShardRouter::new(shards);
            for mixed in [0, 1, 2, 5, 6, 7, 65535, 65536, u32::MAX - 1, u32::MAX] {
                assert_eq!(router.mixed_owner(mixed), mixed as usize % shards);
            }
            let mut key = 17_u64;
            for _ in 0..100000 {
                key = key
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                assert_eq!(router.owner(key), shard_for(key, shards));
            }
        }
    }
}
