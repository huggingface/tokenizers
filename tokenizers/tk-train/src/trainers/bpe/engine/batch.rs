//! Reusable selection workspace and the rules that certify one safe merge batch.
use super::{IdentityPolicy, corpus, merge, pair_index, vocabulary};
use crate::trainers::bpe::BpeTrainer;
use ahash::AHashSet;
use tk_encode::Result;

pub(super) enum BatchSelection {
    Ready,
    Finished,
    RestartForReuse,
}

/// Reused workspace owning accepted rules and their selected position readers.
/// The coordinator releases candidates after preparation joins and before commit.
/// Multiple rules may share a head or a tail. Crossed head/tail overlap is
/// forbidden; the first incompatible candidate ends the prefix without skipping.
/// AA and reserved-ID rules run alone, and active reuse accepts one rule at a time.
#[derive(Default)]
pub(super) struct RuleBatch<'arena> {
    pub(super) rules: Vec<merge::MergeRule>,
    pub(super) candidates: Vec<pair_index::MergeCandidate<'arena>>,
    heads: AHashSet<u32>,
    tails: AHashSet<u32>,
}
impl<'arena> RuleBatch<'arena> {
    /// Select a priority-preserving prefix, stopping at the first incompatible rule.
    /// Candidates must be empty on entry. Accepted rules consume index candidates,
    /// resolve vocabulary identities, and prepare corpus span metadata before writes.
    /// `RestartForReuse` requires abandoning this attempt: earlier accepted rules
    /// may already have changed vocabulary/span metadata. Rebuild from original
    /// words with the retained alphabet; neither restart nor errors roll back state.
    /// Fresh batching relies on decreasing old-boundary counts and increasing new
    /// IDs; see DESIGN.md's priority proof. Reserved IDs lack that tie-break bound.
    #[cfg_attr(test, allow(clippy::too_many_arguments))]
    pub(super) fn select<S: corpus::SlotStorage>(
        &mut self,
        trainer: &BpeTrainer,
        vocabulary: &mut vocabulary::Vocabulary,
        corpus: &mut corpus::Corpus<S>,
        index: &mut pair_index::PairIndex<'arena>,
        policy: IdentityPolicy,
        #[cfg(test)] trace: &mut Vec<(tk_encode::models::bpe::Pair, u64, u32)>,
    ) -> Result<BatchSelection> {
        let cap = if policy == IdentityPolicy::FirstActivationOnly {
            256.min(trainer.vocab_size - vocabulary.len())
        } else {
            1
        };
        debug_assert!(self.candidates.is_empty());
        self.rules.clear();
        self.heads.clear();
        self.tails.clear();
        index.begin_selection();
        while self.rules.len() < cap {
            let Some(priority) = index.best() else {
                break;
            };
            let pair = pair_index::key_pair(priority.key);
            if !self.rules.is_empty()
                && (pair.0 == pair.1
                    || self.tails.contains(&pair.0)
                    || self.heads.contains(&pair.1))
            {
                break;
            }
            let token = vocabulary.merge_token(pair);
            if policy == IdentityPolicy::FirstActivationOnly && vocabulary.reuses_active_id(&token)
            {
                // Fresh pruning and fused batches omit intermediate cohorts.
                // Switching this index in place would lose observable births.
                // Input words remain unchanged: rebuild all cohorts instead,
                // before consuming this candidate or writing its batch.

                return Ok(BatchSelection::RestartForReuse);
            }
            let reserved = token.existing_id.is_some();
            // A reserved ID can precede the old witness in a birth tie.
            if reserved && !self.rules.is_empty() {
                break;
            }
            let candidate = index.take_best();
            let identity = vocabulary.resolve_merge(token)?;
            corpus.prepare_spans(pair, identity.id, identity.reused_active_id);
            self.rules.push(merge::MergeRule {
                pair,
                replacement: identity.id,
            });
            self.candidates.push(candidate);
            #[cfg(test)]
            trace.push((pair, priority.priority_count, identity.id));
            if reserved || pair.0 == pair.1 || self.rules.len() == cap {
                break;
            }
            // Only a following rule needs these conflict checks. A
            // single-rule round never allocates the two hash tables.
            self.heads.insert(pair.0);
            self.tails.insert(pair.1);
        }
        index.end_selection();
        Ok(if self.rules.is_empty() {
            BatchSelection::Finished
        } else {
            BatchSelection::Ready
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        execution::Execution, initial_pairs::InitialPairTable, storage::AllocationArena,
    };
    use super::*;
    use crate::progress::TrainingProgress;
    use crate::trainers::bpe::word_counts::WordCountsView;

    #[test]
    fn shared_endpoints_stop_at_first_crossed_rule() {
        let trainer = BpeTrainer::builder()
            .vocab_size(40)
            .min_frequency(1)
            .show_progress(false)
            .build();
        let words = [("ab", 10), ("ac", 9), ("dc", 8), ("bx", 7), ("ef", 6)]
            .into_iter()
            .map(|(word, count)| (compact_str::CompactString::from(word), count))
            .collect();
        for workers in [1, 4] {
            let execution = Execution::new(workers).unwrap();
            let progress = TrainingProgress::new(false, trainer.progress_format).unwrap();
            execution.pool.install(|| {
                let mut retained = None;
                let mut vocabulary = vocabulary::Vocabulary::initialize(
                    &trainer,
                    WordCountsView::from_map(&words),
                    workers,
                    &progress,
                    &mut retained,
                )
                .unwrap();
                let plan = corpus::CorpusPlan::build(
                    WordCountsView::from_map(&words),
                    &mut vocabulary,
                    IdentityPolicy::FirstActivationOnly,
                    false,
                    &progress,
                )
                .unwrap();
                let arena = AllocationArena::new(workers, plan.initial_edges());
                let initial =
                    InitialPairTable::build(&plan, 1, &execution, &arena, &progress).unwrap();
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
                    1,
                )
                .unwrap();
                let mut batch = RuleBatch::default();
                let mut trace = Vec::new();
                assert!(matches!(
                    batch
                        .select(
                            &trainer,
                            &mut vocabulary,
                            &mut corpus,
                            &mut index,
                            IdentityPolicy::FirstActivationOnly,
                            &mut trace,
                        )
                        .unwrap(),
                    BatchSelection::Ready
                ));
                // Alphabet IDs: a=0, b=1, c=2, d=3, e=4, f=5, x=6.
                // ab/ac share a head; ac/dc share a tail. bx is the first crossed
                // rule, so the compatible but lower-priority ef cannot be skipped to.
                assert_eq!(trace, [((0, 1), 10, 7), ((0, 2), 9, 8), ((3, 2), 8, 9)]);
                assert_eq!(batch.candidates.len(), 3);
                index.begin_selection();
                assert_eq!(pair_index::key_pair(index.best().unwrap().key), (1, 6));
                index.end_selection();
            });
        }
    }
}
