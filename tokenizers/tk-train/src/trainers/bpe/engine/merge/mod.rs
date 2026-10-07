//! Merge contracts and application of prepared writes after all readers join.
mod events;
mod prepare;
use super::{
    corpus::{Corpus, SlotStorage},
    storage::{PositionBuffer, SortedPositions},
};
pub(super) use events::{ChangeAction, EventChunk, MergeEvents, OwnerRoute, PairChanges};
#[cfg(test)]
pub(super) use prepare::BirthPaths;
pub(super) use prepare::{
    ContiguousBirthPolicy, MergeOptions, MergeScratch, SelectedRuleIndex,
    prepare_merges_with_births,
};
use rayon::prelude::*;
use tk_encode::models::bpe::Pair;
/// One accepted rule in sequential priority order.
#[derive(Clone, Copy)]
pub(super) struct MergeRule {
    pub(super) pair: Pair,
    pub(super) replacement: u32,
}
struct WritePlan {
    rule: MergeRule,
    positions: PositionBuffer,
}
struct PreparedJob {
    writes: Vec<WritePlan>,
    word_region: Option<std::ops::Range<u64>>,
}
/// A complete fresh birth encoded by its sole producer during preparation.
///
/// `weight` includes every contributing occurrence for `key`. The producer
/// covers the rule's full candidate-position list and its job fits the node budget.
/// Positions borrow the training arena, not the worker lease used to encode them.
/// Commit consumes this record once; its birth is absent from routed events.
/// Signed reuse ledgers and historical cohorts remain separate index state.
pub(super) struct CompletedBirth<'arena> {
    pub(super) key: u64,
    pub(super) weight: u64,
    pub(super) positions: SortedPositions<'arena>,
}

/// Owned writes and neighbor events produced while reading a stable corpus.
///
/// Preparation establishes disjoint endpoint spans, or complete whole-word jobs
/// when occurrence spans are materialized. The coordinator must apply this plan
/// to the same corpus snapshot; the type does not identify its instance/version.
pub(super) struct PreparedMerges {
    pub(super) birth_shape: prepare::BirthShape,
    #[cfg(test)]
    pub(super) birth_paths: BirthPaths,
    jobs: Vec<PreparedJob>,
    events: MergeEvents,
}
impl PreparedMerges {
    /// Consume the plan, apply parallel writes, and return events after join.
    ///
    /// The corpus must still be the snapshot used during preparation. The mutable
    /// borrow excludes safe concurrent access until all writes join, and consuming
    /// `self` prevents applying this plan value twice. There is no rollback.
    pub(super) fn apply<S: SlotStorage>(self, corpus: &mut Corpus<S>) -> MergeEvents {
        if corpus.has_occurrence_spans() {
            let regions: Vec<_> = self
                .jobs
                .iter()
                .map(|job| {
                    job.word_region
                        .clone()
                        .expect("occurrence spans require complete whole-word jobs")
                })
                .collect();
            let writers = corpus
                .word_writers(&regions)
                .expect("the occurrence plane is materialized");
            self.jobs
                .par_iter()
                .zip(writers.into_par_iter())
                .for_each(|(job, mut writer)| {
                    for write in &job.writes {
                        write
                            .positions
                            .positions()
                            .for_each(|position| writer.merge(position, write.rule.replacement));
                    }
                });
        } else {
            self.jobs.par_iter().for_each(|job| {
                for write in &job.writes {
                    let matcher = corpus.matcher(write.rule.pair);
                    write.positions.positions().for_each(|position| {
                        // SAFETY: preparation selected disjoint endpoint spans.
                        // apply holds the mutable corpus borrow until pool join;
                        // geometry reads immutable ID spans, never token IDs.
                        unsafe {
                            corpus.write_endpoints(
                                matcher.geometry(position),
                                write.rule.replacement,
                            );
                        }
                    });
                }
            });
        }
        self.events
    }
}
