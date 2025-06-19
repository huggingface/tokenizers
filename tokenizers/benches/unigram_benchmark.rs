#[macro_use]
extern crate criterion;

mod common;

use common::iter_bench_train;

use criterion::{Criterion, Throughput};
use tokenizers::models::unigram::{Unigram, UnigramTrainerBuilder};
use tokenizers::models::TrainerWrapper;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer;

// pub fn bench_train(c: &mut Criterion) {
//     let trainer = UnigramTrainer::builder()
//         .show_progress(false)
//         .unk_token(Some("<UNK>".into()))
//         .build()
//         .unwrap();
//
//     let mut model = Unigram::default();
//
//     let content = read_to_string("data/big.txt").unwrap();
//     c.bench_function("Unigram Train vocabulary (medium)", |b| {
//         b.iter_custom(|iters| {
//             let mut duration = Duration::new(0, 0);
//             for _i in 0..iters {
//                 let sentences = sentences.clone();
//                 let start = Instant::now();
//                 trainer.do_train(sentences, &mut model).unwrap();
//                 duration = duration.checked_add(start.elapsed()).unwrap();
//             }
//             duration
//         })
//     });
// }
fn bench_train(c: &mut Criterion) {
    let mut trainer: TrainerWrapper = UnigramTrainerBuilder::default()
        .show_progress(false)
        .build()
        .unwrap()
        .into();
    let mut tokenizer = Tokenizer::new(Unigram::default()).into_inner();
    tokenizer.with_pre_tokenizer(Some(Whitespace {}));
    let mut group = c.benchmark_group("unigram-train-large");
    let data = std::fs::read_to_string("data/big.txt").unwrap();
    group.throughput(Throughput::Bytes(data.len() as u64));
    group.bench_function("BPE Train vocabulary (big)", |b| {
        b.iter_custom(|iters| {
            iter_bench_train(
                iters,
                &mut tokenizer,
                &mut trainer,
                vec!["data/big.txt".to_string()],
            )
        })
    });
}

criterion_group! {
    name = benches_train;
    config = Criterion::default().sample_size(10);
    targets = bench_train
}

criterion_main!(benches_train);
