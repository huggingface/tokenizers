//! One BPE coordinator over shared vocabulary, corpus, and occurrence storage.
//! The visible round is select, prepare, release candidates, apply, commit, then
//! release events. Preparation readers and corpus writers join before the next
//! phase. Errors discard the attempt; commit may fail after writes/partial counts.
//! Active-ID reuse restarts from original words with the already selected alphabet.
use crate::trainers::bpe::word_counts::WordCountsView;
mod aa_parity;
mod batch;
mod corpus;
mod execution;
mod initial_pairs;
mod merge;
mod pair_index;
mod storage;
mod vocabulary;
use super::BpeTrainer;
use crate::progress::TrainingProgress;
#[cfg(test)]
use ahash::AHashMap;
use batch::{BatchSelection, RuleBatch};
#[cfg(test)]
use compact_str::CompactString;
use storage::AllocationArena;
use tk_encode::{
    Result,
    models::bpe::{Merges, Vocab},
    vocab::bucket_added_vocabulary::AddedToken,
};
const WORD_SEPARATOR_ID: u32 = u32::MAX;
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(test, derive(Debug))]
enum IdentityPolicy {
    /// New IDs and first activations of reserved IDs; existing keys cannot revive.
    FirstActivationOnly,
    /// Active identities may revive keys; rebuild with a signed ledger and cohorts.
    AllowActiveReuse,
}
#[cfg(test)]
#[derive(Debug)]
enum BirthObservation {
    Attempt(IdentityPolicy),
    Round {
        enabled_before: bool,
        enabled_after: bool,
        paths: merge::BirthPaths,
    },
}
#[cfg(test)]
type BirthObserver<'a> = Option<&'a mut (dyn FnMut(BirthObservation) + Send)>;
type ModelParts = (Vocab, Merges, Vec<AddedToken>);
enum AttemptOutcome<T = ModelParts> {
    Complete(T),
    RestartForReuse,
}
pub(super) fn train(
    trainer: &BpeTrainer,
    word_counts: WordCountsView<'_>,
    workers: usize,
    #[cfg(test)] observe: Option<&mut (dyn FnMut(tk_encode::models::bpe::Pair, u64, u32) + Send)>,
) -> Result<ModelParts> {
    train_with_merge_options(
        trainer,
        word_counts,
        workers,
        merge::MergeOptions::default(),
        #[cfg(test)]
        observe,
        #[cfg(test)]
        None,
    )
}
fn train_with_merge_options(
    trainer: &BpeTrainer,
    word_counts: WordCountsView<'_>,
    workers: usize,
    merge_options: merge::MergeOptions,
    #[cfg(test)] mut observe: Option<
        &mut (dyn FnMut(tk_encode::models::bpe::Pair, u64, u32) + Send),
    >,
    #[cfg(test)] mut birth_observe: BirthObserver<'_>,
) -> Result<ModelParts> {
    let execution = execution::Execution::new(workers)?;
    execution.pool.install(|| {
        let progress = TrainingProgress::new(trainer.show_progress, trainer.progress_format)?;
        // Every attempt begins with first activations. A nonempty affix does
        // not by itself require one-rule cohort execution. Stop before accepting
        // the first reused active ID, then rebuild with the same coordinator.
        let mut training = Training {
            trainer,
            execution: &execution,
            merge_options,
            progress: &progress,
            policy: IdentityPolicy::FirstActivationOnly,
            retained_alphabet: None,
            #[cfg(test)]
            trace: Vec::new(),
        };
        loop {
            #[cfg(test)]
            if let Some(observer) = birth_observe.as_mut() {
                observer(BirthObservation::Attempt(training.policy));
            }
            #[cfg(test)]
            training.trace.clear();
            match training.attempt(
                word_counts,
                #[cfg(test)]
                &mut birth_observe,
            )? {
                AttemptOutcome::Complete(parts) => {
                    // Publish only the successfully completed attempt's trace;
                    // traces from abandoned attempts were discarded.
                    #[cfg(test)]
                    if let Some(observer) = observe.as_mut() {
                        for (pair, count, id) in training.trace {
                            observer(pair, count, id);
                        }
                    }
                    return Ok(parts);
                }
                AttemptOutcome::RestartForReuse => {
                    // All position lists and their arena were dropped by the attempt.
                    // Retain neither speculative values nor encoding allocations.
                    execution.release_scratch();
                    training.policy = IdentityPolicy::AllowActiveReuse;
                }
            }
        }
    })
}
/// Shared state for a complete training call, including an identity-reuse restart.
/// Slot layouts use the same round algorithm and retain this call's alphabet.
struct Training<'a> {
    trainer: &'a BpeTrainer,
    execution: &'a execution::Execution,
    merge_options: merge::MergeOptions,
    progress: &'a TrainingProgress,
    policy: IdentityPolicy,
    retained_alphabet: Option<Vec<char>>,
    #[cfg(test)]
    trace: Vec<(tk_encode::models::bpe::Pair, u64, u32)>,
}
impl Training<'_> {
    fn attempt(
        &mut self,
        word_counts: WordCountsView<'_>,
        #[cfg(test)] birth_observe: &mut BirthObserver<'_>,
    ) -> Result<AttemptOutcome> {
        let trainer = self.trainer;
        let execution = self.execution;
        let progress = self.progress;
        let policy = self.policy;
        let workers = execution.workers();
        let mut vocabulary = vocabulary::Vocabulary::initialize(
            trainer,
            word_counts,
            workers,
            progress,
            &mut self.retained_alphabet,
        )?;
        let prepared_corpus = corpus::CorpusPlan::build(
            word_counts,
            &mut vocabulary,
            policy,
            trainer.max_token_length.is_some(),
            progress,
        )?;
        if vocabulary.len() >= trainer.vocab_size && prepared_corpus.initial_counts_fit_u64() {
            drop(prepared_corpus);
            execution.release_scratch();
            progress.stage("Compute merges", trainer.vocab_size);
            return Ok(complete_model(trainer, vocabulary, Vec::new()));
        }
        match corpus::slot_bits(trainer.vocab_size.max(vocabulary.len())) {
            16 => self.run::<corpus::U16Slots>(
                vocabulary,
                prepared_corpus,
                #[cfg(test)]
                birth_observe,
            ),
            24 => self.run::<corpus::PackedU24Slots>(
                vocabulary,
                prepared_corpus,
                #[cfg(test)]
                birth_observe,
            ),
            _ => self.run::<corpus::U32Slots>(
                vocabulary,
                prepared_corpus,
                #[cfg(test)]
                birth_observe,
            ),
        }
    }

    fn run<S: corpus::SlotStorage>(
        &mut self,
        mut vocabulary: vocabulary::Vocabulary,
        prepared_corpus: corpus::CorpusPlan<'_>,
        #[cfg(test)] birth_observe: &mut BirthObserver<'_>,
    ) -> Result<AttemptOutcome> {
        let trainer = self.trainer;
        let execution = self.execution;
        let progress = self.progress;
        let policy = self.policy;
        let workers = execution.workers();

        let arena = AllocationArena::new(workers, prepared_corpus.initial_edges());
        let initial = initial_pairs::InitialPairTable::build(
            &prepared_corpus,
            if policy == IdentityPolicy::FirstActivationOnly {
                trainer.min_frequency.max(1)
            } else {
                0
            },
            execution,
            &arena,
            progress,
        )?;
        if vocabulary.len() >= trainer.vocab_size {
            // The total-mass proof was inconclusive. Initial construction has now
            // retained every checked per-key and signed-policy validation.
            drop(initial);
            drop(prepared_corpus);
            drop(arena);
            execution.release_scratch();

            progress.stage("Compute merges", trainer.vocab_size);
            return Ok(complete_model(trainer, vocabulary, Vec::new()));
        }
        // In fresh mode every successful rule consumes at least one physical edge;
        // each appended ID belongs to one such rule. Thus final ID count is bounded
        // by min(target, initial IDs + initial physical edges), not weighted mass.
        // Reuse can select historical cohorts without that progress proof, so this
        // remains only a capacity hint: actual domains always grow beyond it.
        let expected_ids = expected_id_domain(
            trainer.vocab_size,
            vocabulary.len(),
            prepared_corpus.initial_edges(),
        );
        execution.expect_id_domain(expected_ids);
        let mut corpus = prepared_corpus.materialize::<S>(workers, policy, progress)?;
        let mut index =
            pair_index::PairIndex::from_initial_pairs(initial, policy, trainer.min_frequency)?;

        let merges = match self.merge_loop(
            &mut vocabulary,
            &mut corpus,
            &mut index,
            &arena,
            #[cfg(test)]
            birth_observe,
        )? {
            AttemptOutcome::Complete(merges) => merges,
            AttemptOutcome::RestartForReuse => return Ok(AttemptOutcome::RestartForReuse),
        };
        // Training state does not participate in model output. Release position
        // owners before their arena, and free the corpus and scratch before
        // constructing the public vocabulary and merge strings.
        drop(index);
        drop(corpus);
        drop(arena);
        execution.release_scratch();

        Ok(complete_model(trainer, vocabulary, merges))
    }
    /// Complete joined rounds, or request a fresh attempt for active-ID reuse.
    /// Batch candidates and adaptive history belong only to this attempt.
    fn merge_loop<'arena, S: corpus::SlotStorage>(
        &mut self,
        vocabulary: &mut vocabulary::Vocabulary,
        corpus: &mut corpus::Corpus<S>,
        index: &mut pair_index::PairIndex<'arena>,
        arena: &'arena AllocationArena,
        #[cfg(test)] birth_observe: &mut BirthObserver<'_>,
    ) -> Result<AttemptOutcome<Vec<tk_encode::models::bpe::Pair>>> {
        let trainer = self.trainer;
        let execution = self.execution;
        let progress = self.progress;
        let policy = self.policy;
        let merge_options = self.merge_options;
        #[cfg(test)]
        let trace = &mut self.trace;
        let mut merges = Vec::new();
        // PERF: Reuse bounded selection workspace across all rounds. Clearing
        // candidates releases their position lists before commit without reallocating
        // the vector; rule and conflict storage never exceeds the batch limit.
        let mut batch = RuleBatch::default();
        // A fresh state for each attempt, including a rebuild for active-ID reuse.
        // Explore the first eligible batch; only successful joined work/commit can
        // influence the next one. No worker scheduling history survives a restart.
        let mut contiguous_births = merge::ContiguousBirthPolicy::default();
        let work = progress.stage("Compute merges", trainer.vocab_size);
        while vocabulary.len() < trainer.vocab_size {
            match batch.select(
                trainer,
                vocabulary,
                corpus,
                index,
                policy,
                #[cfg(test)]
                trace,
            )? {
                BatchSelection::Ready => {}
                BatchSelection::Finished => break,
                BatchSelection::RestartForReuse => return Ok(AttemptOutcome::RestartForReuse),
            }
            merges.extend(batch.rules.iter().map(|rule| rule.pair));
            #[cfg(test)]
            let enabled_before = contiguous_births.options(merge_options).contiguous_births;
            let (prepared, prepared_births) = merge::prepare_merges_with_births(
                corpus,
                &batch.rules,
                &batch.candidates,
                policy,
                vocabulary.len(),
                trainer.max_token_length.unwrap_or(usize::MAX),
                execution,
                arena,
                trainer.min_frequency.max(1),
                contiguous_births.options(merge_options),
            )?;

            // PERF: Preparation owns all writes and birth events. Selected
            // position lists have no remaining reader; release them before allocating
            // the next generation during commit.
            batch.candidates.clear();
            let birth_shape = prepared.birth_shape;
            #[cfg(test)]
            let birth_paths = prepared.birth_paths;
            let events = prepared.apply(corpus);

            index.commit_merges_with_prepared(
                &events,
                vocabulary.len(),
                execution,
                arena,
                prepared_births,
            )?;
            contiguous_births.observe(birth_shape);
            #[cfg(test)]
            if let Some(observer) = birth_observe.as_mut() {
                observer(BirthObservation::Round {
                    enabled_before,
                    enabled_after: contiguous_births.options(merge_options).contiguous_births,
                    paths: birth_paths,
                });
            }

            drop(events);
            work.learned(merges.len());
        }
        Ok(AttemptOutcome::Complete(merges))
    }
}
fn expected_id_domain(target: usize, initial_ids: usize, physical_edges: usize) -> usize {
    target
        .min(initial_ids.saturating_add(physical_edges))
        .min(u32::MAX as usize)
}

fn complete_model(
    trainer: &BpeTrainer,
    vocabulary: vocabulary::Vocabulary,
    merges: Vec<tk_encode::models::bpe::Pair>,
) -> AttemptOutcome {
    let (vocab, merges) = vocabulary.into_model_parts(merges);
    AttemptOutcome::Complete((vocab, merges, trainer.special_tokens.clone()))
}

#[cfg(test)]
mod tests;
