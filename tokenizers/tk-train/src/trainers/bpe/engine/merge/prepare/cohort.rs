//! Identity-reuse cohorts preserve intermediate word boundaries and birth semantics.
use super::*;
use rayon::prelude::*;
enum CohortSource {
    WordIndices(Vec<usize>),
    Positions(std::ops::Range<usize>),
}
struct CohortTask {
    region: std::ops::Range<u64>,
    source: CohortSource,
}
pub(super) fn prepare<S: SlotStorage>(
    corpus: &Corpus<S>,
    rule: &MergeRule,
    candidate: &MergeCandidate<'_>,
    token_id_count: usize,
    birth_span_limit: u64,
    execution: &Execution,
) -> Result<Vec<(PreparedJob, Vec<EventChunk>)>> {
    let tasks = if corpus.needs_word_scan() {
        word_tasks(corpus, &candidate.positions, execution.workers())
    } else {
        position_tasks(corpus, &candidate.positions, execution.workers())
    };
    tasks
        .into_par_iter()
        .map(|task| -> Result<_> {
            execution.with_merge_scratch(token_id_count, |scratch| {
                let mut plan = RulePreparation::new(corpus, scratch, rule, 0, birth_span_limit);
                match task.source {
                    CohortSource::WordIndices(words) => plan.scan_words(words)?,
                    CohortSource::Positions(range) => {
                        plan.scan_positions(candidate.positions.cursor(range), task.region.start)?;
                    }
                }
                let (write, mut chunks) = plan.finish();
                chunks.push(scratch.take_chunk());

                Ok((
                    PreparedJob {
                        writes: vec![write],
                        word_region: Some(task.region),
                    },
                    chunks,
                ))
            })
        })
        .collect()
}

// Task regions contain complete words, so jobs never share logical neighbors.
fn word_tasks<S: SlotStorage>(
    corpus: &Corpus<S>,
    positions: &SortedPositions<'_>,
    workers: usize,
) -> Vec<CohortTask> {
    let count = positions.len();
    let chunk = count.div_ceil(workers).max(1);
    // Ordered cohorts make word IDs nondecreasing. Each range caches its
    // current word, then adjacent dedup joins range boundaries as well.
    let parts: Vec<Vec<usize>> = (0..count)
        .step_by(chunk)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|begin| {
            let mut words = Vec::new();
            let mut current = None;
            for position in positions.cursor(begin..(begin + chunk).min(count)) {
                let word = match current {
                    Some((word, end)) if position < end => word,
                    _ => corpus.word_containing(position),
                };
                current = Some((word, corpus.word_end(word)));
                if words.last() != Some(&word) {
                    words.push(word);
                }
            }
            words
        })
        .collect();
    let mut words: Vec<_> = parts.into_iter().flatten().collect();
    words.dedup();
    let chunk = words.len().div_ceil(workers).max(1);
    words
        .chunks(chunk)
        .enumerate()
        .map(|(index, part)| {
            let start = if index == 0 {
                0
            } else {
                corpus.word_start(part[0])
            };
            let end = words
                .get((index + 1) * chunk)
                .map_or(corpus.len() as u64, |&word| corpus.word_start(word));
            CohortTask {
                region: start..end,
                source: CohortSource::WordIndices(part.to_vec()),
            }
        })
        .collect::<Vec<_>>()
}

fn position_tasks<S: SlotStorage>(
    corpus: &Corpus<S>,
    positions: &SortedPositions<'_>,
    workers: usize,
) -> Vec<CohortTask> {
    let count = positions.len();
    let chunk = count.div_ceil(workers).max(1);
    // Before alias reuse or a length gate, only cut boundaries need a word
    // lookup. Move each cut to the first occurrence in that complete word.
    let mut cuts = vec![(0, 0)];
    for desired in (chunk..count).step_by(chunk) {
        let position = positions
            .cursor(desired..desired + 1)
            .next()
            .expect("the cut index belongs to the cohort");
        let pivot = corpus.word_start(corpus.word_containing(position));
        let begin = positions.lower_bound(pivot);
        if begin > cuts.last().expect("the first cut exists").0 {
            cuts.push((begin, pivot));
        }
    }
    cuts.push((count, corpus.len() as u64));
    cuts.windows(2)
        .map(|cuts| CohortTask {
            region: cuts[0].1..cuts[1].1,
            source: CohortSource::Positions(cuts[0].0..cuts[1].0),
        })
        .collect()
}

/// One logical token after earlier matches in this word, before endpoint writes.
#[derive(Clone, Copy)]
struct LogicalToken {
    id: u32,
    start: u64,
    span: u64,
}

impl<S: SlotStorage> RulePreparation<'_, S> {
    fn scan_words(&mut self, words: Vec<usize>) -> Result<()> {
        for word in words {
            let mut previous = None;
            let mut position = self.corpus.word_start(word);
            while self.corpus.token(position) != WORD_SEPARATOR_ID {
                if let Some(matched) = self.matcher.get(position) {
                    previous = Some(self.record_cohort_match(matched, previous)?);
                    position = matched.next_start;
                } else {
                    let span = self.corpus.span(position);
                    previous = Some(LogicalToken {
                        id: self.corpus.token(position),
                        start: position,
                        span,
                    });
                    position += span;
                }
            }
        }
        Ok(())
    }
    fn scan_positions(
        &mut self,
        positions: impl Iterator<Item = u64>,
        region_start: u64,
    ) -> Result<()> {
        let mut previous = None;
        let mut after = region_start;
        for position in positions {
            if position < after {
                continue;
            }
            if let Some(matched) = self.matcher.get(position) {
                if position != after {
                    let id = self.corpus.token(position - 1);
                    previous = (id != WORD_SEPARATOR_ID).then(|| {
                        let span = self.corpus.span(position - 1);
                        LogicalToken {
                            id,
                            start: position - span,
                            span,
                        }
                    });
                }
                previous = Some(self.record_cohort_match(matched, previous)?);
                after = matched.next_start;
            }
        }
        Ok(())
    }

    /// Record a reuse match against the logical preceding token; preserve
    /// intermediate boundaries and removal-before-birth neighbor accounting.
    fn record_cohort_match(
        &mut self,
        matched: PairMatch,
        previous: Option<LogicalToken>,
    ) -> Result<LogicalToken> {
        self.room::<false>();
        let weight = self.weights.weight(matched.left_start);
        if let Some(previous) = previous {
            self.scratch.left::<false>(
                previous.id,
                previous.start,
                weight,
                previous.span + matched.merged_span < self.birth_span_limit,
            )?;
        }
        let next = self.corpus.token(matched.next_start);
        if next != WORD_SEPARATOR_ID {
            self.scratch.right::<false>(
                next,
                next,
                matched.left_start,
                weight,
                matched.merged_span + self.corpus.span(matched.next_start) < self.birth_span_limit,
            )?;
        }
        self.positions.push_position(matched.left_start);
        Ok(LogicalToken {
            id: self.rule.replacement,
            start: matched.left_start,
            span: matched.merged_span,
        })
    }
}
