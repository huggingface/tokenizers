//! Ordinary batches: partition positions and prepare complete or partial producers.
use super::*;
use rayon::prelude::*;
struct PositionTask {
    rank: usize,
    begin: usize,
    end: usize,
    /// Covers the entire candidate-position list for this rule, not the batch.
    whole_rule: bool,
    /// Complete producer whose entire job fits the conservative node budget.
    encode_births_directly: bool,
}
impl PositionTask {
    fn new(rank: usize, begin: usize, end: usize, total: usize) -> Self {
        Self {
            rank,
            begin,
            end,
            whole_rule: begin == 0 && end == total,
            encode_births_directly: false,
        }
    }
}
// Each matched source position emits at most one birth per direction. Thus
// 2*sum(raw task positions) bounds every logical node in this whole job,
// including stale positions that emit none. Before any remaining match there
// are at least two budget units, so an admitted complete task cannot flush.
fn job_node_budget_fits(tasks: &[PositionTask], capacity: usize) -> bool {
    tasks
        .iter()
        .try_fold(0usize, |total, task| {
            total.checked_add(task.end - task.begin)
        })
        .and_then(|total| total.checked_mul(2))
        .is_some_and(|nodes| nodes <= capacity)
}
fn position_jobs(
    candidates: &[MergeCandidate<'_>],
    workers: usize,
    fast_enabled: bool,
) -> Vec<Vec<PositionTask>> {
    let mut jobs = position_jobs_whole(candidates, workers);
    if fast_enabled {
        let capacity = PositionChains::MAX_NODES;
        for job in &mut jobs {
            if job_node_budget_fits(job, capacity) {
                for task in job {
                    task.encode_births_directly = task.whole_rule;
                }
            }
        }
    }
    jobs
}
fn position_jobs_whole(
    candidates: &[MergeCandidate<'_>],
    workers: usize,
) -> Vec<Vec<PositionTask>> {
    let total: usize = candidates
        .iter()
        .map(|candidate| candidate.positions.len())
        .sum();
    let chunk = total.div_ceil(workers).clamp(1, 1 << 26);
    let tail_budget = chunk.div_ceil(128).min(64);
    // Keep ordinary pairs intact and expose more ready jobs to the pool.
    // Small pairs share a job; large pairs retain the original maximum range.
    let grain = total
        .div_ceil(workers.saturating_mul(4))
        .max(4096)
        .min(chunk);
    let mut jobs = Vec::<Vec<PositionTask>>::new();
    let mut filled = chunk;
    for (rank, candidate) in candidates.iter().enumerate() {
        let count = candidate.positions.len();
        if count == 0 {
            continue;
        }
        if count <= chunk {
            if filled >= grain || filled + count > grain {
                jobs.push(Vec::new());
                filled = 0;
            }
            jobs.last_mut()
                .expect("the current job exists")
                .push(PositionTask::new(rank, 0, count, count));
            filled += count;
            continue;
        }
        // A large candidate gets spatially ordered ranges, without increasing
        // its producer count merely to reach the smaller scheduling grain.
        let mut begin = 0;
        while begin < count {
            let remaining = count - begin;
            let mut take = chunk.min(remaining);
            if remaining - take <= tail_budget {
                take = remaining;
            }
            jobs.push(vec![PositionTask::new(
                rank,
                begin,
                begin + take,
                candidate.positions.len(),
            )]);
            begin += take;
        }
        filled = chunk;
    }
    jobs
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare<'arena, S: SlotStorage>(
    corpus: &Corpus<S>,
    rules: &[MergeRule],
    candidates: &[MergeCandidate<'_>],
    token_id_count: usize,
    birth_span_limit: usize,
    execution: &Execution,
    arena: &'arena AllocationArena,
    floor: u64,
    options: MergeOptions,
) -> Result<Vec<PreparedOutput<'arena>>> {
    let jobs = position_jobs(
        candidates,
        execution.workers(),
        options.single_producer_fast,
    );
    let mut selected = execution.selected_rules();
    selected.reset(rules, token_id_count);

    jobs.into_par_iter()
        .map(|tasks| -> Result<_> {
            execution.with_merge_scratch(token_id_count, |scratch| {
                let mut outputs = Vec::new();
                let mut births = Vec::new();
                let mut birth_shape = BirthShape::default();
                #[cfg(test)]
                let mut birth_paths = BirthPaths::default();
                for task in tasks {
                    // Geometry, not alphabet or input kind, admits compact
                    // coordinates: max index < corpus.len() <= u32::MAX.
                    let eligible = task.encode_births_directly && corpus.len() <= u32::MAX as usize;
                    let plan = RulePreparation::new(
                        corpus,
                        scratch,
                        &rules[task.rank],
                        task.rank,
                        birth_span_limit as u64,
                    );
                    let positions = &candidates[task.rank].positions;
                    // Dispatch collection, shape sampling, and complete drain
                    // together. A linked producer cannot reach a Vec-aware drain.
                    let (output, shape) = if eligible && options.contiguous_births {
                        prepare_task::<true, S>(
                            plan,
                            positions,
                            &task,
                            &selected,
                            arena,
                            execution,
                            floor,
                            &mut births,
                            #[cfg(test)]
                            &mut birth_paths,
                        )?
                    } else {
                        prepare_task::<false, S>(
                            plan,
                            positions,
                            &task,
                            &selected,
                            arena,
                            execution,
                            floor,
                            &mut births,
                            #[cfg(test)]
                            &mut birth_paths,
                        )?
                    };
                    outputs.push(output);
                    birth_shape.add(shape);
                }
                if let Some((_, chunks)) = outputs.last_mut() {
                    chunks.push(scratch.take_chunk());
                }

                let (writes, chunks): (Vec<_>, Vec<_>) = outputs.into_iter().unzip();
                let chunks = chunks.into_iter().flatten().collect::<Vec<_>>();

                Ok(PreparedOutput {
                    job: PreparedJob {
                        writes,
                        word_region: None,
                    },
                    chunks,
                    completed_births: births,
                    birth_shape,
                    #[cfg(test)]
                    birth_paths,
                })
            })
        })
        .collect::<Result<Vec<_>>>()
}

// The task's const choice spans collection and drain, including wide complete
// producers and linked adaptive fallback. AA/reuse keep their buffered path.
#[allow(clippy::too_many_arguments)]
fn prepare_task<'arena, const CONTIGUOUS: bool, S: SlotStorage>(
    mut plan: RulePreparation<'_, S>,
    positions: &SortedPositions<'_>,
    task: &PositionTask,
    selected: &SelectedRuleIndex,
    arena: &'arena AllocationArena,
    execution: &Execution,
    floor: u64,
    births: &mut Vec<CompletedBirth<'arena>>,
    #[cfg(test)] birth_paths: &mut BirthPaths,
) -> Result<((WritePlan, Vec<EventChunk>), BirthShape)> {
    let eligible = task.encode_births_directly && plan.corpus.len() <= u32::MAX as usize;
    debug_assert!(!CONTIGUOUS || eligible);
    let nodes_before = plan.scratch.remaining_nodes;
    prepare_positions::<CONTIGUOUS, S>(&mut plan, positions, task, selected)?;
    let shape = if eligible {
        // Complete/noflush gives exact B: every actual birth decrements the
        // budget once. Capture C before drains empty both directories. Linked
        // fallback also samples, so the next batch can re-enable contiguous.
        plan.scratch.birth_shape_since(nodes_before)
    } else {
        BirthShape::default()
    };
    #[cfg(test)]
    if eligible {
        birth_paths.eligible_tasks += 1;
        birth_paths.contiguous_tasks += usize::from(CONTIGUOUS);
        // The pool has one header per actually promoted group and is cleared
        // after both direction drains. Read before finish consumes its payloads.
        birth_paths.promoted_groups += plan.scratch.birth_vectors.len();
    }
    let output = if task.encode_births_directly {
        plan.finish_with_births::<CONTIGUOUS>(arena, execution, floor, births)?
    } else {
        debug_assert!(!CONTIGUOUS);
        plan.finish()
    };
    Ok((output, shape))
}

// Dispatch once per task; the linked monomorph carries no pool-index branch.
fn prepare_positions<const CONTIGUOUS: bool, S: SlotStorage>(
    plan: &mut RulePreparation<'_, S>,
    positions: &SortedPositions<'_>,
    task: &PositionTask,
    selected: &SelectedRuleIndex,
) -> Result<()> {
    let mut cursor = positions.cursor(task.begin..task.end);
    // PERF: Interleave decoding and consumption through a small ring. Each
    // refill prefetches one ring ahead without retaining a full decoded tile.
    let mut ring = [0; PREFETCH_DISTANCE];
    let mut active = cursor.decode_into(&mut ring);
    for &position in &ring[..active] {
        plan.corpus.prefetch(position);
    }
    let mut head = 0;
    while active != 0 {
        let position = ring[head];
        if let Some(next) = cursor.next() {
            ring[head] = next;
            plan.corpus.prefetch(next);
        } else {
            active -= 1;
        }
        head = (head + 1) % PREFETCH_DISTANCE;
        if let Some(matched) = plan.matcher.get(position) {
            plan.fresh::<CONTIGUOUS>(matched, SelectedNeighbors::Rules(selected))?;
            plan.positions.push_position(position);
        }
    }
    Ok(())
}
#[cfg(test)]
mod producer_budget_tests {
    use super::*;
    #[test]
    fn node_budget_includes_partial_tasks_and_accepts_exact_capacity() {
        let tasks = [
            PositionTask::new(0, 0, 4, 4),
            PositionTask::new(1, 3, 8, 12),
        ];
        assert!(tasks[0].whole_rule);
        assert!(!tasks[1].whole_rule);
        assert!(job_node_budget_fits(&tasks, 18));
        assert!(!job_node_budget_fits(&tasks, 17));
        assert!(!job_node_budget_fits(
            &[PositionTask::new(0, 0, usize::MAX, usize::MAX)],
            usize::MAX
        ));
        assert!(!job_node_budget_fits(
            &[
                PositionTask::new(0, 0, usize::MAX, usize::MAX),
                PositionTask::new(1, 0, 1, 1)
            ],
            usize::MAX
        ));
    }
}
