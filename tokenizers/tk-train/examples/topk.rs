//! Times `WordLevelTrainer::train` as the number of distinct words grows.
//!
//! The change under test replaces "order every distinct word, then take `vocab_size`" with
//! "partition at `vocab_size`, order only that". The win is a function of n/k, so n is swept and k
//! is held at the default 30_000. Real corpora sit at the high end: millions of distinct word forms
//! against a 30k vocabulary.
//!
//! `feed` is outside the timer -- it builds the count map, which this change does not touch. Only
//! `train` is timed, which is the ordering.
//!
//! Every word is fed once, so every count is equal and the comparator always falls through to its
//! word tie-break. That is the long tail of a real corpus (most word forms occur once) and it is
//! also the case where ordering costs the most, so read these as the favourable end.
//!
//! Statistic is the minimum over the passes: a pass that got descheduled can only be slower.
//!
//! Usage: topk [n_distinct,...] [passes]

use std::time::Instant;

use tk_encode::models::wordlevel::WordLevel;
use tk_train::Trainer;
use tk_train::trainers::wordlevel::WordLevelTrainer;

const VOCAB_SIZE: usize = 30_000;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let sizes: Vec<usize> = args
        .get(1)
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![50_000, 200_000, 1_000_000, 4_000_000]);
    let passes: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    // How many distinct counts to spread the words over. 1 means every count is equal, so the
    // comparator always falls through to its word tie-break -- the costly end. Higher values let
    // most comparisons settle on the integer count instead.
    let spread: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    for n in sizes {
        let words: Vec<String> = (0..n).map(|i| format!("w{i:08}")).collect();

        let mut best = f64::MAX;
        let mut vocab = 0usize;
        for _ in 0..passes {
            let mut trainer = WordLevelTrainer::builder()
                .show_progress(false)
                .vocab_size(VOCAB_SIZE)
                .build()
                .unwrap();
            // Word i is fed 1 + (i % spread) times, so counts run over `spread` distinct values.
            let fed = words
                .iter()
                .enumerate()
                .flat_map(|(i, w)| std::iter::repeat_n(w.clone(), 1 + (i % spread)));
            trainer.feed(fed, |s| Ok(vec![s.to_string()])).unwrap();

            let mut model = WordLevel::default();
            let t = Instant::now();
            trainer.train(&mut model).unwrap();
            let secs = t.elapsed().as_secs_f64();

            vocab = model.vocab.len();
            best = best.min(secs);
        }
        println!(
            "{{\"n_distinct\":{n},\"vocab\":{vocab},\"secs_min\":{:.4},\"words_per_sec\":{:.0}}}",
            best,
            n as f64 / best,
        );
    }
}
