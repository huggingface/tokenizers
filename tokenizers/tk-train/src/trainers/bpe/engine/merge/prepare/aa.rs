//! Select non-overlapping AA occurrences across chunks before preparing writes.
use super::*;
use rayon::prelude::*;
pub(super) fn prepare<S: SlotStorage>(
    corpus: &Corpus<S>,
    rule: &MergeRule,
    candidate: &MergeCandidate<'_>,
    token_id_count: usize,
    birth_span_limit: u64,
    execution: &Execution,
) -> Result<Vec<(PreparedJob, Vec<EventChunk>)>> {
    let chunk = candidate
        .positions
        .len()
        .div_ceil(execution.workers())
        .clamp(1, 1 << 26);
    let ranges: Vec<_> = (0..candidate.positions.len())
        .step_by(chunk)
        .map(|begin| begin..(begin + chunk).min(candidate.positions.len()))
        .collect();
    let matcher = corpus.matcher(rule.pair);
    let valid: Vec<_> = ranges
        .into_par_iter()
        .map(|range| {
            let mut positions = PositionBuffer::default();
            for position in candidate.positions.cursor(range) {
                if matcher.get(position).is_some() {
                    positions.push_position(position);
                }
            }
            positions
        })
        .collect();
    let span = corpus.span_by_id(rule.pair.0);
    let summaries: Vec<_> = valid
        .iter()
        .map(|positions| {
            aa_parity::summarize_by(positions.len(), |index| positions.position(index), span)
        })
        .collect();
    let parities = aa_parity::incoming_parities(&summaries, span);
    let chosen: Vec<_> = valid
        .into_par_iter()
        .zip(parities)
        .map(|(positions, parity)| {
            let mut chosen = PositionBuffer::default();
            aa_parity::for_each_selected(positions.positions(), span, parity, |position| {
                chosen.push_position(position)
            });
            chosen
        })
        .collect();
    let mut previous = Vec::with_capacity(chosen.len());
    let mut last = None;
    for positions in &chosen {
        previous.push(last);
        if !positions.is_empty() {
            last = Some(positions.position(positions.len() - 1));
        }
    }
    let mut following = vec![None; chosen.len()];
    let mut first = None;
    for (index, positions) in chosen.iter().enumerate().rev() {
        following[index] = first;
        if !positions.is_empty() {
            first = Some(positions.position(0));
        }
    }
    chosen
        .into_par_iter()
        .enumerate()
        .map(|(index, positions)| -> Result<_> {
            execution.with_merge_scratch(token_id_count, |scratch| {
                let mut plan = RulePreparation::new(corpus, scratch, rule, 0, birth_span_limit);
                for (offset, position) in positions.positions().enumerate() {
                    let previous = if offset == 0 {
                        previous[index]
                    } else {
                        Some(positions.position(offset - 1))
                    };
                    let following = if offset + 1 == positions.len() {
                        following[index]
                    } else {
                        Some(positions.position(offset + 1))
                    };
                    let matched = matcher.geometry(position);
                    plan.fresh::<false>(
                        matched,
                        SelectedNeighbors::Adjacent {
                            previous,
                            following,
                        },
                    )?;
                }
                plan.positions = positions;
                let (write, mut chunks) = plan.finish();
                chunks.push(scratch.take_chunk());

                Ok((
                    PreparedJob {
                        writes: vec![write],
                        word_region: None,
                    },
                    chunks,
                ))
            })
        })
        .collect()
}
