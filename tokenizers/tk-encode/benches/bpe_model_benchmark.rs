//! Here I want to benchmark various ways we can run BPE merge.
//!
//! Four axes, so a cell is `{model}-{corpus}` / `{engine}/cache={on|off}/par={on|off}`:
//!   * engine      -- the legacy `Tokenizer` (old merge) vs the `PipelineTokenizer` (current)
//!   * cache       -- `resize_cache(0)` turns the word cache off on both engines
//!   * parallelism -- `set_parallelism`
//!   * model x corpus
//!
//! Both engines run the full encode (normalize + split + merge), so the comparison includes
//! pre-tokenization. The legacy side materializes a `String` and offsets per token while the
//! pipeline side emits ids only, which flatters the pipeline by whatever that allocation costs.
//!
//! Corpora beyond english/japanese live in `../data/corpora` (see `CORPORA`); missing files are
//! skipped, as are models that are neither in `../data` nor reachable on the hub.

#[macro_use]
extern crate criterion;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput};
use tk_encode::Tokenizer;
use tk_encode::models::ModelWrapper;
use tk_encode::pipeline::PipelineTokenizer;
use tk_encode::utils::parallelism::set_parallelism;

/// Local `tokenizer.json`s.
const TOKENIZERS: &[(&str, &str)] = &[
    ("gpt2", "../data/gpt2.json"),
    ("llama-3", "../data/llama-3-tokenizer.json"),
    ("deepseek", "../data/deepseek-v4.json"),
];

/// Tried on the hub when absent from `../data` -- needs the `http` feature, and gemma is gated, so
/// this silently contributes nothing unless the repo is already in the local hub cache.
const HUB_TOKENIZERS: &[(&str, &str)] = &[("gemma", "google/gemma-2-2b-it")];

const CORPORA: &[(&str, &str)] = &[
    ("english", "../data/big.txt"),
    ("japanese", "../data/unigram_wagahaiwa_nekodearu.txt"),
    ("code", "../data/corpora/code.txt"),
    ("dense", "../data/corpora/dense.txt"),
    ("greek", "../data/corpora/greek.txt"),
    ("russian", "../data/corpora/russian.txt"),
    ("korean", "../data/corpora/korean.txt"),
    ("arabic", "../data/corpora/arabic.txt"),
    ("hindi", "../data/corpora/hindi.txt"),
    ("thai", "../data/corpora/thai.txt"),
    ("chinese", "../data/corpora/chinese.txt"),
];

/// One chunk size: the axes above already multiply out, and 10 kB documents sit in the middle of
/// the range the old four-size sweep covered.
const CHUNK_BYTES: usize = 10 * 1024;

/// Cap per corpus so every language contributes comparable work.
const CORPUS_BYTES: usize = 1_200_000;

fn make_chunks(lines: &[&str], target_bytes: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur = String::new();
    for line in lines {
        if !cur.is_empty() {
            cur.push('\n');
        }
        cur.push_str(line);
        if cur.len() >= target_bytes {
            chunks.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

fn load(name: &str, path: &str) -> Option<Tokenizer> {
    if let Ok(tok) = Tokenizer::from_file(path) {
        return Some(tok);
    }
    #[cfg(feature = "http")]
    if let Ok(tok) = Tokenizer::from_pretrained(path, None) {
        return Some(tok);
    }
    eprintln!("bpe bench: skip {name} -- {path} not loadable");
    None
}

/// Fresh tokenizer with the word cache in the requested state, plus the pipeline built from it.
fn pair(name: &str, path: &str, cache: bool) -> Option<(Tokenizer, PipelineTokenizer)> {
    let mut oracle = load(name, path)?;
    if !cache {
        if let ModelWrapper::BPE(bpe) = oracle.get_model_mut() {
            bpe.resize_cache(0);
        }
    }
    let pipeline = match PipelineTokenizer::try_from(&oracle) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("bpe bench: skip {name} -- pipeline: {e}");
            return None;
        }
    };
    Some((oracle, pipeline))
}

fn bench_pipeline(c: &mut Criterion) {
    let models: Vec<(&str, &str)> = TOKENIZERS
        .iter()
        .chain(HUB_TOKENIZERS.iter())
        .copied()
        .collect();

    for (tok_name, tok_path) in models {
        if !matches!(
            load(tok_name, tok_path).as_ref().map(|t| t.get_model()),
            Some(ModelWrapper::BPE(_))
        ) {
            eprintln!("bpe bench: skip {tok_name} -- not a BPE model");
            continue;
        }

        for cache in [true, false] {
            let Some((oracle, pipeline)) = pair(tok_name, tok_path, cache) else {
                continue;
            };
            let cache_tag = if cache { "on" } else { "off" };

            for (corpus, path) in CORPORA {
                let Ok(text) = std::fs::read_to_string(path) else {
                    continue;
                };
                let mut end = CORPUS_BYTES.min(text.len());
                while end > 0 && !text.is_char_boundary(end) {
                    end -= 1;
                }
                let lines: Vec<&str> = text[..end]
                    .lines()
                    .filter(|l| !l.trim().is_empty())
                    .collect();
                let chunks = make_chunks(&lines, CHUNK_BYTES);
                let total_bytes: u64 = chunks.iter().map(|s| s.len() as u64).sum();
                if total_bytes == 0 {
                    continue;
                }

                let mut group = c.benchmark_group(format!("{tok_name}-{corpus}"));
                group.throughput(Throughput::Bytes(total_bytes));
                for par in [true, false] {
                    set_parallelism(par);
                    let par_tag = if par { "on" } else { "off" };
                    group.bench_with_input(
                        BenchmarkId::new(format!("legacy/cache={cache_tag}/par={par_tag}"), "10kB"),
                        &chunks,
                        |b, chunks| {
                            b.iter(|| {
                                for chunk in chunks {
                                    black_box(oracle.encode(chunk.as_str(), false).unwrap());
                                }
                            })
                        },
                    );
                    group.bench_with_input(
                        BenchmarkId::new(
                            format!("pipeline/cache={cache_tag}/par={par_tag}"),
                            "10kB",
                        ),
                        &chunks,
                        |b, chunks| {
                            b.iter(|| {
                                for chunk in chunks {
                                    black_box(pipeline.encode(chunk, false).unwrap());
                                }
                            })
                        },
                    );
                }
                group.finish();
            }
        }
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(3))
        .warm_up_time(std::time::Duration::from_millis(500));
    targets = bench_pipeline
}
criterion_main!(benches);
