#[macro_use]
extern crate criterion;

use criterion::{Criterion, Throughput};
use std::fs::File;
use std::io::{BufRead, BufReader};
use tokenizers::{
    pre_tokenizers::{PreTokenizer, PreTokenizerWrapper},
    PreTokenizedString, Tokenizer,
};

#[cfg(feature = "pcre2")]
const BACKEND: &str = "pcre2";
#[cfg(all(feature = "onig", not(feature = "pcre2")))]
const BACKEND: &str = "onig";
#[cfg(all(
    feature = "fancy-regex",
    not(any(feature = "onig", feature = "pcre2"))
))]
const BACKEND: &str = "fancy-regex";

fn load_inputs(path: &str, limit: usize) -> Vec<String> {
    BufReader::new(File::open(path).expect("read sample file"))
        .lines()
        .take(limit)
        .map(|l| l.expect("line"))
        .collect()
}

fn bench_split(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    name: &str,
    pretok: PreTokenizerWrapper,
    inputs: &[String],
) {
    let total_bytes: u64 = inputs.iter().map(|s| s.len() as u64).sum();
    group.throughput(Throughput::Bytes(total_bytes));

    group.bench_function(name.to_string(), |b| {
        let pretok = pretok.clone();
        b.iter(|| {
            for text in inputs {
                let mut pts = PreTokenizedString::from(text.as_str());
                pretok.pre_tokenize(&mut pts).expect("pre-tokenize");
            }
        });
    });
}

fn regex_split(c: &mut Criterion) {
    // Keep the run quick but representative.
    let inputs = load_inputs("data/big.txt", 2_000);
    let mut group = c.benchmark_group(format!("regex-split-{BACKEND}"));

    let gpt2 = Tokenizer::from_file("data/tokenizer.json").expect("load gpt2 tokenizer");
    let gpt2_pretok = gpt2
        .get_pre_tokenizer()
        .expect("gpt2 pretokenizer")
        .clone();
    bench_split(&mut group, "bytelevel-gpt2", gpt2_pretok, &inputs);

    let llama = Tokenizer::from_file("data/llama-3-tokenizer.json").expect("load llama3 tokenizer");
    let llama_pretok = llama
        .get_pre_tokenizer()
        .expect("llama3 pretokenizer")
        .clone();
    bench_split(&mut group, "llama3-split", llama_pretok, &inputs);

    group.finish();
}

criterion_group!(regex_split_group, regex_split);
criterion_main!(regex_split_group);
