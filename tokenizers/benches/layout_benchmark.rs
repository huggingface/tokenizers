#[macro_use]
extern crate criterion;

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::{Duration, Instant};

use criterion::black_box;
use criterion::Criterion;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::{EncodeInput, Encoding, PostProcessor, Tokenizer};

/// Simple TemplateProcessing
fn create_processor() -> TemplateProcessing {
    TemplateProcessing::builder()
        .try_single("[CLS]:0 $A:0 [SEP]:0")
        .unwrap()
        .try_pair("[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1")
        .unwrap()
        .special_tokens(vec![("[CLS]", 0), ("[SEP]", 1)])
        .build()
        .unwrap()
}

pub fn bench_layout(c: &mut Criterion) {
    let processor = create_processor();
    let tokenizer = Tokenizer::from_file("data/albert-base-v1-tokenizer.json").unwrap();
    let mut encodeds: Vec<Encoding> = vec![];
    for line in BufReader::new(File::open(Path::new("data/big.txt")).unwrap()).lines() {
        let line: EncodeInput = line.unwrap().into();

        let encoded: Encoding = tokenizer.encode(line, false).unwrap();
        encodeds.push(encoded);
    }

    c.bench_function("TemplateProcessing single encode", |b| {
        b.iter_custom(|iters| {
            let mut duration = Duration::new(0, 0);
            for i in 0..iters as usize {
                let encoded_index = i % encodeds.len();
                let encoded: Encoding = encodeds[encoded_index].clone();

                let start = Instant::now();
                let _ = black_box(processor.process(encoded, None, false));
                duration = duration.checked_add(start.elapsed()).unwrap();
            }
            duration
        })
    });
    c.bench_function("TemplateProcessing pair encode", |b| {
        b.iter_custom(|iters| {
            let mut duration = Duration::new(0, 0);
            for i in 0..iters as usize {
                let encoded_index = i % encodeds.len();
                let encoded: Encoding = encodeds[encoded_index].clone();

                let encoded_index2 = (i + 1) % encodeds.len();
                let pair: Encoding = encodeds[encoded_index2].clone();

                let start = Instant::now();
                let _ = black_box(processor.process(encoded, Some(pair), false));
                duration = duration.checked_add(start.elapsed()).unwrap();
            }
            duration
        })
    });
}

criterion_group! {
    name = layout_benches;
    config = Criterion::default().sample_size(20);
    targets = bench_layout
}

criterion_main!(layout_benches);
