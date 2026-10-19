//! Proof-oriented engine tests; fixtures and oracle comparison helpers are shared.
use super::*;
use tk_encode::models::bpe::Pair;
fn counts(items: &[(&str, u64)]) -> AHashMap<CompactString, u64> {
    items
        .iter()
        .map(|&(word, count)| (word.into(), count))
        .collect()
}
pub(super) fn check(trainer: &BpeTrainer, words: &AHashMap<CompactString, u64>) {
    check_with_workers(trainer, words, &[1, 4]);
}
fn check_with_workers(
    trainer: &BpeTrainer,
    words: &AHashMap<CompactString, u64>,
    workers: &[usize],
) {
    let mut expected_trace = Vec::new();
    let expected = trainer
        .do_train_observed(words, |pair, count, id| {
            expected_trace.push((pair, count, id))
        })
        .unwrap();
    for &workers in workers {
        let mut trace = Vec::<(Pair, u64, u32)>::new();
        let actual = train_with_merge_options(
            trainer,
            WordCountsView::from_map(words),
            workers,
            merge::MergeOptions::default(),
            Some(&mut |pair, count, id| trace.push((pair, count, id))),
            None,
        )
        .unwrap();
        assert_eq!(
            trace, expected_trace,
            "workers={workers}, prefix={:?}, suffix={:?}, limit={:?}",
            trainer.continuing_subword_prefix, trainer.end_of_word_suffix, trainer.max_token_length
        );
        assert_eq!(actual, expected, "workers={workers}");
    }
}

mod identity_reuse;
mod initial_corpus;
mod producer_fast_path;
mod public_contract;
mod routing_and_publication;
mod semantic_parity;
