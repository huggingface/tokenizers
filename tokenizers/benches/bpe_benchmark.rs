#[macro_use]
extern crate criterion;

use criterion::{black_box, BatchSize, Criterion};
use std::fs::File;
use std::io::{self, BufRead, BufReader};
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

static BATCH_SIZE: usize = 1_000;
static BIG_BATCH_SIZE: usize = 10_000;

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

fn line_to_input(line: io::Result<String>) -> EncodeInput {
    EncodeInput::Single(line.unwrap())
}

fn bench_gpt2(c: &mut Criterion) {
    let bpe = BPE::from_files("benches/gpt2-vocab.json", "benches/gpt2-merges.txt").unwrap();

    // Benchmarks encoding a single input from a fresh tokenizer.
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

    // Benchmarks encoding many inputs on a single tokenizer.
    c.bench_function("BPE GPT2 encode many", |b| {
        b.iter_custom(|iters| {
            let tokenizer = create_gpt2_tokenizer(&bpe);
            let mut lines = BufReader::new(File::open(Path::new("benches/big.txt")).unwrap())
                .lines()
                .map(line_to_input);
            let mut duration = Duration::new(0, 0);
            for _i in 0..iters {
                let input = match lines.next() {
                    Some(line) => line,
                    None => {
                        // Reset the lines iterator.
                        lines = BufReader::new(File::open(Path::new("benches/big.txt")).unwrap())
                            .lines()
                            .map(line_to_input);
                        lines.next().unwrap()
                    }
                };
                let start = Instant::now();
                let _ = black_box(tokenizer.encode(input));
                duration = duration.checked_add(start.elapsed()).unwrap();
            }
            duration
        })
    });

    let mut big_batch: Vec<EncodeInput> = Vec::with_capacity(BIG_BATCH_SIZE);
    let mut batches: Vec<Vec<EncodeInput>> = vec![vec![]];
    for line in BufReader::new(File::open(Path::new("benches/big.txt")).unwrap())
        .lines()
        .map(line_to_input)
    {
        if big_batch.len() < BIG_BATCH_SIZE {
            big_batch.push(line.clone());
        }
        if batches.last().unwrap().len() >= BATCH_SIZE {
            batches.push(vec![]);
        }
        batches.last_mut().unwrap().push(line);
    }

    // Benchmarks encoding a single big batch on a new tokenizer.
    c.bench_function("BPE GPT2 encode batch", |b| {
        b.iter_batched(
            || (create_gpt2_tokenizer(&bpe), big_batch.clone()),
            |(tokenizer, input)| tokenizer.encode_batch(input),
            BatchSize::LargeInput,
        )
    });

    // Benchmarks encoding a many smaller batches on a single tokenizer.
    c.bench_function("BPE GPT2 encode batch many", |b| {
        b.iter_custom(|iters| {
            let tokenizer = create_gpt2_tokenizer(&bpe);
            let mut duration = Duration::new(0, 0);
            let mut batch_index: usize = 0;
            for _i in 0..iters {
                if batch_index >= batches.len() {
                    batch_index = 0;
                }
                let batch = batches[batch_index].clone();
                let start = Instant::now();
                let _ = black_box(tokenizer.encode_batch(batch));
                duration = duration.checked_add(start.elapsed()).unwrap();
            }
            duration
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20).warm_up_time(Duration::from_secs(5));
    targets = bench_gpt2
}
criterion_main!(benches);
