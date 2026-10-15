//! Read-only preparation: select a path, build writes and account neighbor events.
mod aa;
mod cohort;
mod ordinary;
use super::super::storage::{
    AllocationArena, IdAccumulator, IdDirectory, PositionBuffer, PositionChain, PositionChains,
    SortedPositions,
};
use super::super::{
    IdentityPolicy, WORD_SEPARATOR_ID, aa_parity,
    corpus::{Corpus, PairMatch, PairMatcher, SlotStorage, WordWeightCursor},
    execution::Execution,
    pair_index::{MergeCandidate, pair_key},
};
use super::{
    CompletedBirth, EventChunk, MergeEvents, MergeRule, PairChanges, PreparedJob, PreparedMerges,
    WritePlan,
};
use ahash::AHashMap;
use std::num::NonZeroU32;
use tk_encode::{Result, models::bpe::Pair};
/// One job's writes, buffered neighbor events, and already encoded fresh births.
struct PreparedOutput<'arena> {
    job: PreparedJob,
    chunks: Vec<EventChunk>,
    completed_births: Vec<CompletedBirth<'arena>>,
    birth_shape: BirthShape,
    #[cfg(test)]
    birth_paths: BirthPaths,
}
#[derive(Default)]
struct NeighborChanges {
    removed: u64,
    born: u64,
    // Linked groups retain all coordinates here. After promotion this keeps
    // only the first two seed nodes; `vector` names the full coordinate list.
    positions: PositionChain,
    // A scratch-local header index occupies the linked layout's trailing pad
    // on 64-bit targets: removed8 + born8 + chain12 + index4 = 32B,
    // equal to the old aligned layout. Some 32-bit layouts grow 28B to 32B;
    // correctness is portable, but this equality is not a universal ABI claim.
    // Tiny/ineligible groups own no Vec header or payload.
    vector: Option<NonZeroU32>,
}
#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(std::mem::size_of::<NeighborChanges>() == 32);
    assert!(std::mem::size_of::<(u32, NeighborChanges)>() == 40);
};
const _: () = assert!(PositionChains::MAX_NODES < u32::MAX as usize);
pub(in super::super) struct MergeScratch {
    left: IdAccumulator<NeighborChanges>,
    right: IdAccumulator<NeighborChanges>,
    chains: PositionChains,
    changes: Vec<PairChanges>,
    remaining_nodes: usize,
    // Only complete compact producers allocate headers. Indices are unique
    // across both directions until their whole drain ends. Empty header capacity
    // may serve the next rule; coordinate payloads retire at each group's drain.
    birth_vectors: Vec<Vec<u32>>,
}
impl MergeScratch {
    pub(in super::super) fn new(token_id_count: usize, directories: [IdDirectory; 2]) -> Self {
        let [left, right] = directories;
        Self {
            left: IdAccumulator::with_directory(token_id_count, left),
            right: IdAccumulator::with_directory(token_id_count, right),
            chains: PositionChains::new(),
            changes: Vec::new(),
            remaining_nodes: PositionChains::MAX_NODES,
            birth_vectors: Vec::new(),
        }
    }
    /// Sample only a complete eligible task, after collection and before drain.
    fn birth_shape_since(&self, nodes_before: usize) -> BirthShape {
        BirthShape {
            births: nodes_before - self.remaining_nodes,
            touched_groups: self
                .left
                .touched_len()
                .saturating_add(self.right.touched_len()),
        }
    }
    pub(in super::super) fn into_directories(self) -> [IdDirectory; 2] {
        [self.left.into_directory(), self.right.into_directory()]
    }
    // PERF: Rules in one job share their node allocation. Drain only the
    // neighbor directories between rules; handing off nodes here would create
    // a separately growing allocation for every rule and prevent buffer reuse.
    fn flush_rule(&mut self, rule: &MergeRule, rank: usize) {
        debug_assert!(self.birth_vectors.is_empty());
        self.changes.extend(
            Self::neighbor_events(&mut self.left, &mut self.right, rule, rank).map(
                |(event, vector)| {
                    debug_assert!(vector.is_none(), "buffered jobs keep linked births");
                    event
                },
            ),
        );
    }
    // One direction/bucket rule for buffered events and complete producers.
    fn neighbor_events(
        left: &mut IdAccumulator<NeighborChanges>,
        right: &mut IdAccumulator<NeighborChanges>,
        rule: &MergeRule,
        rank: usize,
    ) -> impl Iterator<Item = (PairChanges, Option<NonZeroU32>)> {
        let left = left.drain().map(move |(neighbor, change)| {
            (
                PairChanges {
                    removed_key: pair_key((neighbor, rule.pair.0)),
                    born_key: pair_key((neighbor, rule.replacement)),
                    removed_weight: change.removed,
                    born_weight: change.born,
                    positions: change.positions,

                    bucket: (rank * 2) as u32,
                },
                change.vector,
            )
        });
        let right = right.drain().map(move |(neighbor, change)| {
            (
                PairChanges {
                    removed_key: pair_key((rule.pair.1, neighbor)),
                    born_key: pair_key((rule.replacement, neighbor)),
                    removed_weight: change.removed,
                    born_weight: change.born,
                    positions: change.positions,

                    bucket: (rank * 2 + usize::from(neighbor != rule.replacement)) as u32,
                },
                change.vector,
            )
        });
        left.chain(right)
    }
    /// Drain a complete producer's neighbors, pruning before direct encoding.
    /// The caller must cover the whole rule without a partial node-budget flush.
    /// Removal events survive; completed births are published only through the
    /// returned records, with no second event scan or position re-encoding.
    #[allow(clippy::too_many_arguments)]
    fn flush_rule_with_births<'arena, const CONTIGUOUS: bool>(
        &mut self,
        rule: &MergeRule,
        rank: usize,
        floor: u64,
        arena: &'arena AllocationArena,
        execution: &Execution,
        births: &mut Vec<CompletedBirth<'arena>>,
    ) -> Result<()> {
        let worker = execution.current_worker();
        let lease = arena.lease(worker);
        let chains = &self.chains;
        let changes = &mut self.changes;
        let vectors = &mut self.birth_vectors;
        // These are the original neighbor-directory drains, not a second pass
        // over emitted events. Each birth already has its complete mass/chain.
        let mut emit = |(mut event, index): (PairChanges, Option<NonZeroU32>)| -> Result<()> {
            // Taking a vector also retires pruned groups. The linked kernel
            // folds this branch away because its collection never creates indices.
            let source = if CONTIGUOUS {
                index.map(|index| std::mem::take(&mut vectors[index.get() as usize - 1]))
            } else {
                debug_assert!(index.is_none(), "linked collection publishes no pool index");
                None
            };
            let count = source.as_ref().map_or(event.positions.len(), Vec::len);
            let positions = if count != 0 && event.born_weight >= floor {
                Some(match &source {
                    None => SortedPositions::from_reversed_iter_direct(
                        count,
                        chains.reversed(event.positions),
                        &lease,
                    )?,
                    Some(positions) => SortedPositions::from_reversed_iter_direct(
                        count,
                        positions.iter().rev().map(|&p| u64::from(p)),
                        &lease,
                    )?,
                })
            } else {
                None
            };
            if let Some(positions) = positions {
                births.push(CompletedBirth {
                    key: event.born_key,
                    weight: event.born_weight,
                    positions,
                });
            }
            // No birth event is emitted for the completed producer. The count
            // owner's original removal actions retain their keys and weights.
            if event.removed_weight != 0 {
                event.positions = PositionChain::default();
                changes.push(event);
            }
            Ok(())
        };
        Self::neighbor_events(&mut self.left, &mut self.right, rule, rank)
            .try_for_each(&mut emit)?;
        // No group in either direction retains an index now. On error, scratch
        // owns and drops all remaining payloads with the discarded attempt.
        if CONTIGUOUS {
            vectors.clear();
        } else {
            debug_assert!(
                vectors.is_empty(),
                "linked complete jobs own no pool payloads"
            );
        }
        Ok(())
    }
    fn take_chunk(&mut self) -> EventChunk {
        debug_assert!(self.birth_vectors.is_empty());
        self.remaining_nodes = PositionChains::MAX_NODES;
        EventChunk {
            chains: std::mem::take(&mut self.chains),
            changes: std::mem::take(&mut self.changes),
        }
    }
    fn remove(group: &mut NeighborChanges, weight: u64) -> Result<()> {
        group.removed = group
            .removed
            .checked_add(weight)
            .ok_or("BPE neighbor removal mass exceeds u64")?;
        Ok(())
    }
    // PERF: This update runs for every newborn boundary. Expose its small
    // successful path to the caller so count and chain updates share local
    // values; overflow error construction must not keep it out of line.
    #[inline(always)]
    fn birth<const CONTIGUOUS: bool>(
        group: &mut NeighborChanges,
        chains: &mut PositionChains,
        vectors: &mut Vec<Vec<u32>>,
        remaining_nodes: &mut usize,
        position: u64,
        weight: u64,
    ) -> Result<()> {
        group.born = group
            .born
            .checked_add(weight)
            .ok_or("BPE neighbor birth mass exceeds u64")?;
        if CONTIGUOUS {
            debug_assert!(position <= u64::from(u32::MAX));
            if let Some(index) = group.vector {
                vectors[index.get() as usize - 1].push(position as u32);
            } else if group.positions.len() == 2 {
                // Reuse the existing tiny chain as the small-buffer fallback.
                // Prior positions also belong to this admitted resident corpus.
                let first = chains.first(group.positions).unwrap();
                let second = chains.last(group.positions).unwrap();
                debug_assert!(first <= u64::from(u32::MAX) && second <= u64::from(u32::MAX));
                let mut positions = Vec::with_capacity(4);
                positions.push(first as u32);
                positions.push(second as u32);
                positions.push(position as u32);
                let index = NonZeroU32::new(
                    u32::try_from(vectors.len() + 1)
                        .expect("logical node budget bounds header indices"),
                )
                .expect("header indices start at one");
                vectors.push(positions);
                group.vector = Some(index);
            } else {
                chains.push(&mut group.positions, position)?;
            }
        } else {
            // Compile-time linked path: no pool index probe for partial/AA/reuse.
            chains.push(&mut group.positions, position)?;
        }
        *remaining_nodes -= 1;
        Ok(())
    }
    fn left<const CONTIGUOUS: bool>(
        &mut self,
        neighbor: u32,
        position: u64,
        weight: u64,
        birth: bool,
    ) -> Result<()> {
        let group = self.left.touch(neighbor);
        Self::remove(group, weight)?;
        if birth {
            Self::birth::<CONTIGUOUS>(
                group,
                &mut self.chains,
                &mut self.birth_vectors,
                &mut self.remaining_nodes,
                position,
                weight,
            )?;
        }
        Ok(())
    }
    fn right<const CONTIGUOUS: bool>(
        &mut self,
        removed: u32,
        born: u32,
        position: u64,
        weight: u64,
        birth: bool,
    ) -> Result<()> {
        let group = self.right.touch(removed);
        Self::remove(group, weight)?;
        if birth {
            // A cohort keeps the same neighbor on removal and birth. Preserve
            // checked removal-before-birth order while sharing its lookup.
            let group = if removed == born {
                group
            } else {
                self.right.touch(born)
            };
            Self::birth::<CONTIGUOUS>(
                group,
                &mut self.chains,
                &mut self.birth_vectors,
                &mut self.remaining_nodes,
                position,
                weight,
            )?;
        }
        Ok(())
    }
}
// PERF: Prefetch a bounded distance ahead to overlap scattered endpoint loads
// without retaining a fully decoded position list.
const PREFETCH_DISTANCE: usize = 16;
const EMPTY: u64 = u64::MAX;
const MULTIPLE: u64 = u64::MAX - 1;
/// Dense directories recognize selected neighbors without a hash lookup in the
/// common case. Shared heads or tails use the complete-key map.
#[derive(Default)]
pub(in super::super) struct SelectedRuleIndex {
    heads: Vec<u64>,
    tails: Vec<u64>,
    multiple: AHashMap<u64, u32>,
    pairs: Vec<Pair>,
}
impl SelectedRuleIndex {
    fn reset(&mut self, rules: &[MergeRule], token_id_count: usize) {
        for (head, tail) in self.pairs.drain(..) {
            self.heads[head as usize] = EMPTY;
            self.tails[tail as usize] = EMPTY;
        }
        self.heads.resize(token_id_count, EMPTY);
        self.tails.resize(token_id_count, EMPTY);
        self.multiple.clear();
        for rule in rules {
            self.pairs.push(rule.pair);
            let head = &mut self.heads[rule.pair.0 as usize];
            *head = if *head == EMPTY {
                pair_key((rule.pair.1, rule.replacement))
            } else {
                MULTIPLE
            };
            let tail = &mut self.tails[rule.pair.1 as usize];
            *tail = if *tail == EMPTY {
                pair_key((rule.pair.0, rule.replacement))
            } else {
                MULTIPLE
            };
        }
        for rule in rules {
            if self.heads[rule.pair.0 as usize] == MULTIPLE
                || self.tails[rule.pair.1 as usize] == MULTIPLE
            {
                self.multiple.insert(pair_key(rule.pair), rule.replacement);
            }
        }
    }
    fn left_selected<S: SlotStorage>(&self, corpus: &Corpus<S>, before: u64, prior: u32) -> bool {
        let tail = self.tails[prior as usize];
        if tail == EMPTY {
            return false;
        }
        let previous = corpus.token(before - 1);
        if tail == MULTIPLE {
            self.multiple.contains_key(&pair_key((previous, prior)))
        } else {
            (tail >> 32) as u32 == previous
        }
    }
    fn final_next<S: SlotStorage>(&self, corpus: &Corpus<S>, after: u64, next: u32) -> u32 {
        let head = self.heads[next as usize];
        if head == EMPTY {
            return next;
        }
        let following = corpus.token(after + corpus.span_by_id(next));
        if head == MULTIPLE {
            self.multiple
                .get(&pair_key((next, following)))
                .copied()
                .unwrap_or(next)
        } else if (head >> 32) as u32 == following {
            head as u32
        } else {
            next
        }
    }
}
enum SelectedNeighbors<'rules> {
    Rules(&'rules SelectedRuleIndex),
    Adjacent {
        previous: Option<u64>,
        following: Option<u64>,
    },
}
struct RulePreparation<'prep, S: SlotStorage> {
    corpus: &'prep Corpus<S>,
    scratch: &'prep mut MergeScratch,
    rule: &'prep MergeRule,
    rank: usize,
    matcher: PairMatcher<'prep, S>,
    /// Strict admission gate for newborn neighbors, in retained-symbol slots.
    /// Initial candidates and the selected merge itself have no extra length gate.
    birth_span_limit: u64,
    chunks: Vec<EventChunk>,
    positions: PositionBuffer,
    // PERF: Weighted words are contiguous and often share weights. Cache the
    // current interval across sorted occurrence visits instead of searching
    // immutable boundaries for every rewrite. The cursor also handles resets.
    weights: WordWeightCursor<'prep>,
}
impl<'prep, S: SlotStorage> RulePreparation<'prep, S> {
    fn new(
        corpus: &'prep Corpus<S>,
        scratch: &'prep mut MergeScratch,
        rule: &'prep MergeRule,
        rank: usize,
        birth_span_limit: u64,
    ) -> Self {
        Self {
            corpus,
            scratch,
            rule,
            rank,
            matcher: corpus.matcher(rule.pair),
            birth_span_limit,
            chunks: Vec::new(),
            positions: PositionBuffer::default(),
            weights: corpus.weight_cursor(),
        }
    }
    fn room<const CONTIGUOUS: bool>(&mut self) {
        if self.scratch.remaining_nodes < 2 {
            assert!(
                !CONTIGUOUS,
                "complete producer's node budget forbids a partial flush"
            );
            self.scratch.flush_rule(self.rule, self.rank);
            self.chunks.push(self.scratch.take_chunk());
        }
    }
    fn fresh<const CONTIGUOUS: bool>(
        &mut self,
        matched: PairMatch,
        neighbors: SelectedNeighbors<'_>,
    ) -> Result<()> {
        self.room::<CONTIGUOUS>();
        let weight = self.weights.weight(matched.left_start);
        let prior = self.corpus.token(matched.left_start - 1);
        if prior != WORD_SEPARATOR_ID {
            let prior_span = self.corpus.span_by_id(prior);
            let before = matched.left_start - prior_span;
            let left_selected = match neighbors {
                SelectedNeighbors::Adjacent { previous, .. } => {
                    previous.is_some_and(|start| start + matched.merged_span == matched.left_start)
                }
                SelectedNeighbors::Rules(selected) => {
                    selected.left_selected(self.corpus, before, prior)
                }
            };
            if !left_selected {
                self.scratch.left::<CONTIGUOUS>(
                    prior,
                    before,
                    weight,
                    prior_span + matched.merged_span < self.birth_span_limit,
                )?;
            }
        }
        let next = self.corpus.token(matched.next_start);
        if next != WORD_SEPARATOR_ID {
            let final_next = match neighbors {
                SelectedNeighbors::Adjacent { following, .. } => {
                    if following == Some(matched.next_start) {
                        self.rule.replacement
                    } else {
                        next
                    }
                }
                SelectedNeighbors::Rules(selected) => {
                    selected.final_next(self.corpus, matched.next_start, next)
                }
            };
            self.scratch.right::<CONTIGUOUS>(
                next,
                final_next,
                matched.left_start,
                weight,
                matched.merged_span + self.corpus.span_by_id(final_next) < self.birth_span_limit,
            )?;
        }
        Ok(())
    }
    fn finish_with_births<'arena, const CONTIGUOUS: bool>(
        self,
        arena: &'arena AllocationArena,
        execution: &Execution,
        floor: u64,
        births: &mut Vec<CompletedBirth<'arena>>,
    ) -> Result<(WritePlan, Vec<EventChunk>)> {
        debug_assert!(
            self.chunks.is_empty(),
            "allocation node budget excludes partial flush"
        );
        self.scratch.flush_rule_with_births::<CONTIGUOUS>(
            self.rule, self.rank, floor, arena, execution, births,
        )?;
        Ok((
            WritePlan {
                rule: *self.rule,
                positions: self.positions,
            },
            self.chunks,
        ))
    }
    fn finish(self) -> (WritePlan, Vec<EventChunk>) {
        self.scratch.flush_rule(self.rule, self.rank);
        (
            WritePlan {
                rule: *self.rule,
                positions: self.positions,
            },
            self.chunks,
        )
    }
}
#[derive(Clone, Copy, Debug)]
pub(in super::super) struct MergeOptions {
    pub(in super::super) single_producer_fast: bool,
    pub(in super::super) contiguous_births: bool,
    #[cfg(test)]
    pub(in super::super) adaptive_births: bool,
}
impl Default for MergeOptions {
    fn default() -> Self {
        Self {
            single_producer_fast: true,
            contiguous_births: true,
            #[cfg(test)]
            adaptive_births: true,
        }
    }
}
/// Actual shape of eligible complete ordinary producers, before floor pruning.
/// Summed only after reader jobs join; it never participates in model semantics.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(in super::super) struct BirthShape {
    births: usize,
    touched_groups: usize,
}
impl BirthShape {
    fn add(&mut self, other: Self) {
        self.births = self.births.saturating_add(other.births);
        self.touched_groups = self.touched_groups.saturating_add(other.touched_groups);
    }
}
// Test-only evidence of actual task dispatch and promotion, sampled once before
// each complete drain. Production shape feedback needs neither these fields nor
// a per-occurrence counter.
#[cfg(test)]
#[derive(Clone, Copy, Default, Debug)]
pub(in super::super) struct BirthPaths {
    pub(in super::super) eligible_tasks: usize,
    pub(in super::super) contiguous_tasks: usize,
    pub(in super::super) promoted_groups: usize,
}
#[cfg(test)]
impl BirthPaths {
    fn add(&mut self, other: Self) {
        self.eligible_tasks += other.eligible_tasks;
        self.contiguous_tasks += other.contiguous_tasks;
        self.promoted_groups += other.promoted_groups;
    }
}
/// Attempt-local feedback selects one const kernel for the next batch.
/// History predicts performance only: the next batch need not share its shape.
/// The cutoff 16 is an empirical candidate, not a proof of CPU or HWM benefit.
#[derive(Debug)]
pub(in super::super) struct ContiguousBirthPolicy {
    enabled: bool,
}
impl Default for ContiguousBirthPolicy {
    fn default() -> Self {
        Self { enabled: true }
    }
}
impl ContiguousBirthPolicy {
    pub(in super::super) fn options(&self, options: MergeOptions) -> MergeOptions {
        #[cfg(test)]
        if !options.adaptive_births {
            return options;
        }
        MergeOptions {
            contiguous_births: options.contiguous_births && self.enabled,
            ..options
        }
    }
    #[allow(clippy::manual_checked_ops)]
    pub(in super::super) fn observe(&mut self, shape: BirthShape) {
        if shape.touched_groups != 0 {
            // C_all includes removal-only groups, so C_all >= C_nonempty and
            // B/C_all conservatively understates mean nonempty chain length.
            // Integer division is equivalent to B >= 16*C without overflowing
            // the product. Saturating totals are deterministic across workers;
            // B saturation understates the ratio, C saturation rejects at 16.
            self.enabled = shape.births / shape.touched_groups >= 16;
        }
        // No eligible producer: retain the mode, rather than infer a zero mean.
    }
}
/// Prepare all jobs against one stable corpus and join every reader before return.
/// Rules and candidates must be nonempty and correspond in accepted rank order.
/// Complete fresh producers return encoded births; partial, AA, and reuse jobs
/// return event chains for owner reduction. An error returns no applicable plan.
#[allow(clippy::too_many_arguments)]
pub(in super::super) fn prepare_merges_with_births<'arena, S: SlotStorage>(
    corpus: &Corpus<S>,
    rules: &[MergeRule],
    candidates: &[MergeCandidate<'_>],
    policy: IdentityPolicy,
    token_id_count: usize,
    birth_span_limit: usize,
    execution: &Execution,
    arena: &'arena AllocationArena,
    floor: u64,
    options: MergeOptions,
) -> Result<(PreparedMerges, Vec<CompletedBirth<'arena>>)> {
    let floor = floor.max(1);
    let outputs =
        if policy == IdentityPolicy::AllowActiveReuse || rules[0].pair.0 == rules[0].pair.1 {
            let prepare = if policy == IdentityPolicy::AllowActiveReuse {
                cohort::prepare::<S>
            } else {
                aa::prepare::<S>
            };
            prepare(
                corpus,
                &rules[0],
                &candidates[0],
                token_id_count,
                birth_span_limit as u64,
                execution,
            )?
            .into_iter()
            .map(|(job, chunks)| PreparedOutput {
                job,
                chunks,
                completed_births: Vec::new(),
                birth_shape: BirthShape::default(),
                #[cfg(test)]
                birth_paths: BirthPaths::default(),
            })
            .collect()
        } else {
            ordinary::prepare(
                corpus,
                rules,
                candidates,
                token_id_count,
                birth_span_limit,
                execution,
                arena,
                floor,
                options,
            )?
        };
    let mut jobs = Vec::new();
    let mut chunks = Vec::new();
    let mut births = Vec::new();
    let mut birth_shape = BirthShape::default();
    #[cfg(test)]
    let mut birth_paths = BirthPaths::default();
    for output in outputs {
        birth_shape.add(output.birth_shape);
        #[cfg(test)]
        birth_paths.add(output.birth_paths);
        births.extend(output.completed_births);
        jobs.push(output.job);
        chunks.extend(output.chunks);
    }
    Ok((
        PreparedMerges {
            jobs,
            birth_shape,
            #[cfg(test)]
            birth_paths,
            events: MergeEvents {
                chunks,
                buckets: rules.len() * 2,
            },
        },
        births,
    ))
}
#[cfg(test)]
mod tests;
