//! Encode throughput by input shape.
//!
//! Batches of sequences or of pairs, short lines or long documents, with and without special
//! tokens. Every row of a length encodes the same bytes, so the rows compare directly: `short`
//! against `long` is the per-sequence cost, and `add_special_tokens=true` against `false`, or
//! `batch` against `pair-batch`, is mostly the post-processor. The stage breakdown is
//! `encode.rs`; the model matrix and the cross-engine numbers are huggingface/tokbench.
//!
//! Two fixtures whose templates do different work. `llama-3` puts one token in front of a
//! sequence, so the single-sequence path shifts every id to make room, and gives the second half
//! of a pair type id 1. `roberta` wraps a sequence in `<s>`/`</s>`, separates a pair with two
//! `</s>`, and tags nothing, so no type ids are built at all.
//!
//! Serial: `cargo bench -p tk-serialize` builds without `parallelism`, so a batch is the calling
//! thread encoding its items one after the other with one scratch. With the feature on, a batch
//! over 8 KiB goes to the thread pool and its rows measure the pool as well.
//!
//! Short windows on purpose: an iteration is the whole corpus, tens of milliseconds, so one second
//! of warm-up and two of measurement already give criterion fifty samples. A laptop drifts by a few
//! percent between two runs of the same code, and a longer window does not cure that. When every
//! row moves together, the machine moved, not the code. Alternate the two builds and compare
//! medians.

use std::hint::black_box;
use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
};
use tk_encode::pipeline::PipelineTokenizer;

/// The `datasets.map(batched=True)` default, which is how most batches reach `encode`.
const BATCH_SIZE: usize = 1_000;
/// About a thousand tokens: a typical document or context chunk.
const DOC_BYTES: usize = 4 * 1024;

const FIXTURES: [(&str, &str); 2] = [
    ("llama3", "../data/llama-3-tokenizer.json"),
    ("roberta", "../data/roberta.json"),
];

/// Read a real config from `../data`. They are all still version `1.0`, so run the upgrade pass
/// first -- the reader only accepts canonical `2.0`.
fn load(path: &str) -> PipelineTokenizer {
    let canonical = tk_convert::canonicalize_file(path).unwrap();
    tk_serialize::from_json(&canonical).unwrap()
}

/// Consecutive lines joined into documents of at least [`DOC_BYTES`]. The tail that does not
/// fill one is dropped.
fn documents(lines: &[&str]) -> Vec<String> {
    let mut docs = Vec::new();
    let mut doc = String::new();
    for line in lines {
        doc.push_str(line);
        doc.push('\n');
        if doc.len() >= DOC_BYTES {
            docs.push(std::mem::take(&mut doc));
        }
    }
    docs
}

/// One length's inputs, in both shapes a row takes. An even number of items, so the pairs cover
/// exactly the same bytes as the singles.
struct Shapes<'a> {
    bytes: usize,
    items: Vec<&'a str>,
    pairs: Vec<(&'a str, &'a str)>,
}

impl<'a> Shapes<'a> {
    fn new(mut items: Vec<&'a str>) -> Self {
        items.truncate(items.len() & !1);
        let pairs = items.chunks_exact(2).map(|p| (p[0], p[1])).collect();
        Self {
            bytes: items.iter().map(|item| item.len()).sum(),
            items,
            pairs,
        }
    }
}

fn rows(
    group: &mut BenchmarkGroup<'_, WallTime>,
    tokenizer: &PipelineTokenizer,
    length: &str,
    shapes: &Shapes,
) {
    group.throughput(Throughput::Bytes(shapes.bytes as u64));
    for specials in [true, false] {
        let id = |shape: &str| format!("{length}/{shape}/add_special_tokens={specials}");
        group.bench_function(id("batch"), |b| {
            b.iter(|| {
                for batch in shapes.items.chunks(BATCH_SIZE) {
                    black_box(tokenizer.encode(batch, specials).wait().unwrap());
                }
            })
        });
        group.bench_function(id("pair-batch"), |b| {
            b.iter(|| {
                for batch in shapes.pairs.chunks(BATCH_SIZE) {
                    black_box(tokenizer.encode(batch, specials).wait().unwrap());
                }
            })
        });
    }
}

pub fn encode_shapes(c: &mut Criterion) {
    let data = std::fs::read_to_string("../data/big.txt").unwrap();
    // A blank line encodes to special tokens alone, which is its own regime.
    let lines: Vec<&str> = data
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    let docs = documents(&lines);
    let short = Shapes::new(lines);
    let long = Shapes::new(docs.iter().map(String::as_str).collect());

    for (name, path) in FIXTURES {
        let tokenizer = load(path);
        let mut group = c.benchmark_group(format!("encode-shapes/{name}"));
        group.sampling_mode(SamplingMode::Flat);
        group.sample_size(50);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(2));
        rows(&mut group, &tokenizer, "short", &short);
        rows(&mut group, &tokenizer, "long", &long);
        group.finish();
    }
}

criterion_group!(encode_shapes_benches, encode_shapes);
criterion_main!(encode_shapes_benches);
