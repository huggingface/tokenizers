#[macro_use]
extern crate criterion;

use criterion::{black_box, BatchSize, Criterion};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::{Duration, Instant};
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::tokenizer::{AddedToken, EncodeInput, Tokenizer};

static SINGLE_INPUT: &str = "Our model, called GPT-2 (a successor to GPT), was trained \
simply to predict the next word in 40GB of Internet text. Due to our concerns about \
malicious applications of the technology, we are not releasing the trained model. \
As an experiment in responsible disclosure, we are instead releasing a much smaller \
model for researchers to experiment with, as well as a technical paper.\n\
\n\
GPT-2 is a large transformer-based language model with 1.5 billion parameters, trained on a dataset \
of 8 million web pages. GPT-2 is trained with a simple objective: predict the next word, given all \
of the previous words within some text. The diversity of the dataset causes this simple goal \
to contain naturally occurring demonstrations of many tasks across diverse domains. \
GPT-2 is a direct scale-up of GPT, with more than 10X the parameters and trained on \
more than 10X the amount of data.\n\
\n\
GPT-2 displays a broad set of capabilities, including the ability to generate conditional \
synthetic text samples of unprecedented quality, where we prime the model with an input \
and have it generate a lengthy continuation. In addition, GPT-2 outperforms other language \
models trained on specific domains (like Wikipedia, news, or books) without needing to \
use these domain-specific training datasets. On language tasks like question answering, \
reading comprehension, summarization, and translation, GPT-2 begins to learn these tasks \
from the raw text, using no task-specific training data. While scores on these \
downstream tasks are far from state-of-the-art, they suggest that the tasks can \
benefit from unsupervised techniques, given sufficient (unlabeled) data and compute.";

fn create_gpt2_tokenizer(bpe: &BPE) -> Tokenizer {
    let mut tokenizer = Tokenizer::new(Box::new(bpe.clone()));
    tokenizer.with_pre_tokenizer(Box::new(ByteLevel::new(true)));
    tokenizer.with_decoder(Box::new(ByteLevel::new(false)));
    tokenizer.add_tokens(&[
        AddedToken {
            content: String::from("ing"),
            single_word: false,
        },
        AddedToken {
            content: String::from("[ENT]"),
            single_word: true,
        },
    ]);
    tokenizer
}

fn bench_gpt2_encode(c: &mut Criterion) {
    let bpe = BPE::from_files("benches/gpt2-vocab.json", "benches/gpt2-merges.txt").unwrap();

    c.bench_function("BPE GPT2 encode", |b| {
        b.iter_batched(
            || {
                (
                    create_gpt2_tokenizer(&bpe),
                    EncodeInput::Single(SINGLE_INPUT.into()),
                )
            },
            |(tokenizer, input)| tokenizer.encode(input),
            BatchSize::LargeInput,
        )
    });
}

fn bench_gpt2_encode_batch(c: &mut Criterion) {
    let bpe = BPE::from_files("benches/gpt2-vocab.json", "benches/gpt2-merges.txt").unwrap();
    let lines: Vec<EncodeInput> = BufReader::new(File::open(Path::new("benches/big.txt")).unwrap())
        .lines()
        .map(|l| EncodeInput::Single(l.unwrap()))
        .collect();

    c.bench_function("BPE GPT2 encode batch", |b| {
        b.iter_custom(|iter| {
            // HACK: This is very time-consuming, so instead of doing it `iter` times,
            // we just do it once and multiply the resulting time by `iter` to
            // pretend that we did it a bunch of times and each was exactly the same.
            let tokenizer = create_gpt2_tokenizer(&bpe);
            let lines = lines.clone();
            let start = Instant::now();
            black_box(tokenizer.encode_batch(lines).unwrap());
            start.elapsed().checked_mul(iter as u32).unwrap()
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = bench_gpt2_encode
}
criterion_group! {
    name = batch_benches;
    config = Criterion::default().sample_size(10).warm_up_time(Duration::from_secs(10));
    targets = bench_gpt2_encode_batch
}
criterion_main!(benches, batch_benches);
