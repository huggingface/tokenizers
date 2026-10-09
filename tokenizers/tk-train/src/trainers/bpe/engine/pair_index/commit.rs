//! Owner-local commit after corpus writes join.
//! Stable birth grouping precedes ordered counts, completed-birth publication,
//! and reduction/encoding of routed births. All owners join before event release.
use super::super::{
    execution::Execution,
    merge::{ChangeAction as Action, CompletedBirth, EventChunk, MergeEvents, OwnerRoute},
    storage::AllocationArena,
};
use super::*;
impl<'arena> PairIndex<'arena> {
    /// Route aggregates to their count owners, then encode complete birth cohorts.
    /// Position nodes remain borrowed until every owner has joined.
    #[cfg(test)]
    pub(in super::super) fn commit_merges(
        &mut self,
        events: &MergeEvents,
        identities: usize,
        execution: &Execution,
        arena: &'arena AllocationArena,
    ) -> Result<()> {
        self.commit_merges_with_prepared(events, identities, execution, arena, Vec::new())
    }
    /// Commit the applied batch in one joined owner phase.
    /// Selection must have ended, restoring fresh prefixes, and writes must have
    /// joined. Run this phase inside the training pool.
    /// Completed births are fresh-only and are moved without decoding; their keys
    /// must not also appear as routed births. Events own chains borrowed by owner
    /// encoders and must remain alive until return. Count/encoding errors can leave
    /// shards partly changed and corpus writes already applied; discard the attempt.
    pub(in super::super) fn commit_merges_with_prepared(
        &mut self,
        events: &MergeEvents,
        identities: usize,
        execution: &Execution,
        arena: &'arena AllocationArena,
        births: Vec<CompletedBirth<'arena>>,
    ) -> Result<()> {
        let policy = self.policy;
        let floor = self.minimum_frequency.max(1);
        let router = ShardRouter::new(self.shards.len());
        events.dispatch_into(&mut self.routes, router);
        debug_assert!(births.is_empty() || policy == IdentityPolicy::FirstActivationOnly);
        // Serial metadata routing moves completed births, with no regrouping or
        // codec operation. Owners publish within the existing commit phase.
        self.prepared_births
            .resize_with(self.shards.len(), Vec::new);
        for births in &mut self.prepared_births {
            births.clear();
        }
        for birth in births {
            self.prepared_births[router.owner(birth.key)].push(birth);
        }

        let result = self
            .shards
            .par_iter_mut()
            .zip(self.prepared_births.par_iter_mut())
            .zip(self.routes.par_iter_mut())
            // Owners without changes keep their counts and valid priorities.
            // Leave their lazy queue refill to selection and avoid scheduling
            // empty codec/directory work, at every corpus and vocabulary scale.
            .filter(|((_, prepared), route)| !route.changes.is_empty() || !prepared.is_empty())
            .map(
                |((shard, prepared), route)| -> Result<Vec<MergeCandidate<'arena>>> {
                    // PERF: Stable grouping by rule/direction lets each bucket
                    // reuse one neighbor directory instead of per-pair hash tables.
                    // It stays before worker leases and ordered count actions.
                    route.group_births(events);
                    shard.apply_ordered_counts(route, events, policy, floor)?;
                    shard.publish_completed_births(prepared, policy);
                    shard.reduce_encode_and_publish_births(
                        route, events, identities, execution, arena, policy, floor,
                    )
                },
            )
            .collect::<Result<Vec<_>>>();

        // Routes contain numeric references only. Drop them after all owner jobs
        // have joined, including on error, while retaining their vector capacity.
        for route in &mut self.routes {
            route.clear();
        }
        for births in &mut self.prepared_births {
            births.clear();
        }
        let candidates = result?;

        if let Selection::Cohorts { candidates: queue } = &mut self.selection {
            for births in candidates {
                queue.extend(births);
            }
        }
        Ok(())
    }
}

impl<'arena> PairShard<'arena> {
    /// Apply routed count actions in original order, removing before adding for
    /// `Both`. Fresh subtraction retires keys below `floor`; reuse updates the
    /// checked signed ledger even when a key has no published cohort.
    /// An error leaves earlier actions applied: the caller must discard the attempt.
    fn apply_ordered_counts(
        &mut self,
        route: &OwnerRoute,
        events: &MergeEvents,
        policy: IdentityPolicy,
        floor: u64,
    ) -> Result<()> {
        match policy {
            IdentityPolicy::FirstActivationOnly => {
                self.apply_first_activation_counts(route, events, floor)
            }
            IdentityPolicy::AllowActiveReuse => self.apply_active_reuse_counts(route, events),
        }
    }

    /// Fresh keys cannot revive. Retire low counts after each checked removal.
    fn apply_first_activation_counts(
        &mut self,
        route: &OwnerRoute,
        events: &MergeEvents,
        floor: u64,
    ) -> Result<()> {
        for reference in &route.changes {
            if matches!(reference.action(), Action::Remove | Action::Both) {
                let change = &events.chunks[reference.chunk].changes[reference.index()];
                self.subtract_fresh(change.removed_key, change.removed_weight, floor)?;
            }
        }
        Ok(())
    }

    /// Reuse keys share a signed ledger across independently owned cohorts.
    /// Check every intermediate value, with removal before birth for `Both`.
    fn apply_active_reuse_counts(
        &mut self,
        route: &OwnerRoute,
        events: &MergeEvents,
    ) -> Result<()> {
        for reference in &route.changes {
            let change = &events.chunks[reference.chunk].changes[reference.index()];
            if matches!(reference.action(), Action::Remove | Action::Both) {
                let count = self.ledger.entry(change.removed_key).or_default();
                let amount = i64::try_from(change.removed_weight)
                    .map_err(|_| "BPE identity-reuse removal exceeds i64")?;
                *count = (*count as i64)
                    .checked_sub(amount)
                    .ok_or("BPE identity-reuse count subtraction exceeds i64")?
                    as u64;
            }
            if matches!(reference.action(), Action::Birth | Action::Both) {
                let count = self.ledger.entry(change.born_key).or_default();
                let amount = i64::try_from(change.born_weight)
                    .map_err(|_| "BPE identity-reuse birth exceeds i64")?;
                *count = (*count as i64)
                    .checked_add(amount)
                    .ok_or("BPE identity-reuse count addition exceeds i64")?
                    as u64;
            }
        }
        Ok(())
    }

    /// Publish sole-producer fresh births after ordered counts and before routed
    /// birth reduction. Move the encoded lists directly into index state; each
    /// key must be new here and absent from the remaining routed birth chains.
    fn publish_completed_births(
        &mut self,
        births: &mut Vec<CompletedBirth<'arena>>,
        policy: IdentityPolicy,
    ) {
        debug_assert!(births.is_empty() || policy == IdentityPolicy::FirstActivationOnly);
        for birth in births.drain(..) {
            self.insert_fresh(birth.key, birth.weight, birth.positions);
        }
    }

    /// Publish the count and positions together with their selectable priority.
    fn insert_fresh(&mut self, key: u64, count: u64, positions: SortedPositions<'arena>) {
        debug_assert!(
            !self.states.contains_key(&key),
            "fresh birth has one producer rule"
        );
        self.states.insert(
            key,
            PairState {
                ledger_count_bits: count,
                positions,
            },
        );
        self.priorities.push(PairPriority {
            key,
            priority_count: count,
        });
    }

    /// Reduce each stably grouped bucket, then encode and publish one complete key
    /// at a time. Ordered counts and completed-birth publication must have finished.
    /// Fresh counts are pruned only after all fragments are summed; reuse retains
    /// positive-ledger cohorts even below the selection floor. Fresh prefix refill
    /// also runs for owners without routed births.
    ///
    /// Worker leases cover sequential work only. Event chains remain borrowed until
    /// return, within the existing joined owner phase; no per-key position copy or
    /// extra parallel phase is introduced. Errors require discarding the attempt.
    #[allow(clippy::too_many_arguments)]
    fn reduce_encode_and_publish_births(
        &mut self,
        route: &OwnerRoute,
        events: &MergeEvents,
        identities: usize,
        execution: &Execution,
        arena: &'arena AllocationArena,
        policy: IdentityPolicy,
        floor: u64,
    ) -> Result<Vec<MergeCandidate<'arena>>> {
        if route.births.is_empty() {
            if policy == IdentityPolicy::FirstActivationOnly {
                self.prepare_prefix(floor);
            }
            return Ok(Vec::new());
        }
        let worker = execution.current_worker();
        let lease = arena.lease(worker);
        let mut scratch = execution.encoding(worker);
        let mut candidates = Vec::new();
        execution.with_accumulator::<BirthGroup, _>(identities, 0, |neighbors| {
            // PERF: One owner-level fragment allocation serves every key
            // and rule/direction bucket. Per-key vectors would allocate for
            // each key receiving positions from more than one producer.
            let mut fragments = Vec::<Fragment<'_>>::new();
            let mut remaining = route.births.as_slice();
            while let Some(&first_index) = remaining.first() {
                let first_ref = &route.changes[first_index];
                let first = &events.chunks[first_ref.chunk].changes[first_ref.index()];
                let bucket = first.bucket;
                let end = remaining.partition_point(|&index| {
                    let reference = &route.changes[index];
                    events.chunks[reference.chunk].changes[reference.index()].bucket == bucket
                });
                let (births, next) = remaining.split_at(end);
                remaining = next;
                let left = bucket & 1 == 0;
                let pair = key_pair(first.born_key);
                let replacement = if left { pair.1 } else { pair.0 };
                for &reference_index in births {
                    let reference = &route.changes[reference_index];
                    let chunk = &events.chunks[reference.chunk];
                    let index = reference.index();
                    let change = &chunk.changes[index];
                    let pair = key_pair(change.born_key);
                    let neighbor = if left { pair.0 } else { pair.1 };
                    let group = neighbors.touch(neighbor);
                    group.weight = group
                        .weight
                        .checked_add(change.born_weight)
                        .ok_or("BPE birth frequency exceeds u64")?;
                    group.occurrences = group
                        .occurrences
                        .checked_add(change.positions.len())
                        .ok_or("BPE birth position count exceeds resident bounds")?;
                    fragments.push(Fragment {
                        chunk,
                        index,
                        next: group.head,
                    });
                    group.head = fragments.len() - 1;
                }
                for (neighbor, group) in neighbors.drain() {
                    let key = pair_key(if left {
                        (neighbor, replacement)
                    } else {
                        (replacement, neighbor)
                    });
                    // PERF: Fresh keys cannot revive. Reduce all producers
                    // and reject low counts before touching the global map;
                    // inserting then deleting them causes avoidable growth
                    // and tombstone churn. Signed ledgers already record
                    // ordered changes and retain every positive birth cohort,
                    // including counts below the selection floor.
                    let count = if policy == IdentityPolicy::FirstActivationOnly {
                        group.weight
                    } else {
                        self.ledger[&key]
                    };
                    if if policy == IdentityPolicy::FirstActivationOnly {
                        count < floor
                    } else {
                        (count as i64) <= 0
                    } {
                        continue;
                    }
                    let mut head = group.head;
                    // Fresh jobs supply spatially disjoint runs. Identity-reuse
                    // AA births may combine interleaved left/right chains;
                    // the common encoder merges those actual overlaps.
                    let fragments = &fragments;
                    let sources = std::iter::from_fn(move || {
                        if head == usize::MAX {
                            return None;
                        }
                        let fragment = &fragments[head];
                        head = fragment.next;
                        Some(fragment)
                    })
                    .map(|fragment| {
                        (
                            &fragment.chunk.chains,
                            fragment.chunk.changes[fragment.index].positions,
                        )
                    });
                    let positions = if policy == IdentityPolicy::FirstActivationOnly {
                        // Fresh buckets own disjoint, spatially ordered jobs.
                        // The count is already complete. Encode their reverse
                        // traversal without rereading each source's endpoints.

                        SortedPositions::from_reversed_iter(
                            group.occurrences,
                            sources.flat_map(|(owner, chain)| owner.reversed(chain)),
                            &mut scratch,
                            &lease,
                        )?
                    } else {
                        SortedPositions::from_reversed_chains(sources, &mut scratch, &lease)?
                    };
                    debug_assert_eq!(positions.len(), group.occurrences);

                    if policy == IdentityPolicy::FirstActivationOnly {
                        self.insert_fresh(key, count, positions);
                    } else {
                        candidates.push(MergeCandidate {
                            priority: PairPriority {
                                key,
                                priority_count: count,
                            },
                            positions,
                        });
                    }
                }
                fragments.clear();
            }

            if policy == IdentityPolicy::FirstActivationOnly {
                // PERF: Refill while this owner is already running. A
                // separate pool phase would schedule the same owners again.
                self.prepare_prefix(floor);
            }
            Ok(candidates)
        })
    }
}

/// Complete mass and a reverse-linked fragment list for one neighbor in a bucket.
struct BirthGroup {
    weight: u64,
    head: usize,
    occurrences: usize,
}
impl Default for BirthGroup {
    fn default() -> Self {
        Self {
            weight: 0,
            head: usize::MAX,
            occurrences: 0,
        }
    }
}
struct Fragment<'events> {
    chunk: &'events EventChunk,
    index: usize,
    next: usize,
}
