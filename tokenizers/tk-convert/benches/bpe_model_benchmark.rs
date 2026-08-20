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
//! `{model}-{corpus}-merge` isolates the model stage instead: one pre-token at a time, taken from
//! the model's own pre-tokenizer. Each side gets the form its own design expects -- with ByteLevel
//! the legacy model reads the remapped string its pre-tokenizer produces, while the current model
//! reads the original slice at the same offsets and folds that remap into conversion.
//!
//! Corpora beyond english/japanese live in `../data/corpora` (see `CORPORA`); missing files are
//! skipped, as are models that are neither in `../data` nor reachable on the hub.

#[macro_use]
extern crate criterion;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput};
use tk_convert::Tokenizer;
use tk_convert::models::ModelWrapper;
use tk_encode::pipeline::{Model as PipelineModelTrait, PipelineModel, PipelineTokenizer};
use tk_encode::tokenizer::{
    Model as LegacyModelTrait, NormalizedString, Normalizer, OffsetReferential, OffsetType,
    PreTokenizedString, PreTokenizer,
};
use tk_encode::utils::parallelism::set_parallelism;

/// Local `tokenizer.json`s.
const TOKENIZERS: &[(&str, &str)] = &[
    ("gpt2", "../data/gpt2.json"),
    ("llama-3", "../data/llama-3-tokenizer.json"),
    ("deepseek", "../data/deepseek-v4.json"),
    ("llama-2", "../data/llama-2.json"),
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
    if !cache && let ModelWrapper::BPE(bpe) = oracle.get_model_mut() {
        bpe.resize_cache(0);
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

/// Both forms of one real pre-tokenization, i.e. what each engine's model is actually handed.
/// `.0` is the model's own pre-tokenizer output -- with ByteLevel that string has already been
/// remapped bytes->unicode, which is the form the legacy model looks up. `.1` is the original slice
/// at the same offsets, which is what the current model takes, because it does that remap itself.
fn model_inputs(oracle: &Tokenizer, text: &str) -> Vec<(String, String)> {
    // the model's own normalizer runs first: llama-2 rewrites every space to U+2581, and without it
    // nothing would be found in the vocab and both engines would just measure byte fallback
    let mut normalized = NormalizedString::from(text);
    if let Some(normalizer) = oracle.get_normalizer()
        && normalizer.normalize(&mut normalized).is_err()
    {
        return vec![];
    }
    let normalized = normalized.get().to_string();

    // sentencepiece-style models (llama-2) declare no pre-tokenizer, so the model is handed whole
    // sequences. Feed it documents rather than the entire corpus as one pre-token.
    let Some(pre_tokenizer) = oracle.get_pre_tokenizer() else {
        return normalized
            .as_bytes()
            .chunks(CHUNK_BYTES)
            .scan(0usize, |start, _| {
                let from = *start;
                if from >= normalized.len() {
                    return None;
                }
                let mut to = (from + CHUNK_BYTES).min(normalized.len());
                while to < normalized.len() && !normalized.is_char_boundary(to) {
                    to += 1;
                }
                *start = to;
                Some(normalized[from..to].to_string())
            })
            .map(|chunk| (chunk.clone(), chunk))
            .collect();
    };
    let mut pre_tokenized = PreTokenizedString::from(normalized.as_str());
    if pre_tokenizer.pre_tokenize(&mut pre_tokenized).is_err() {
        return vec![];
    }
    pre_tokenized
        .get_splits(OffsetReferential::Original, OffsetType::Byte)
        .into_iter()
        .filter(|(piece, offsets, _)| !piece.is_empty() && offsets.1 > offsets.0)
        .map(|(piece, offsets, _)| {
            (
                piece.to_string(),
                normalized[offsets.0..offsets.1].to_string(),
            )
        })
        .collect()
}

/// The model stage alone: legacy `BPE::tokenize` against the current
/// `PipelineBPE::tokenize_pipeline`, one pre-token at a time, straight from the model's own
/// pre-tokenizer. Caches off on both sides, so this is conversion + merge and nothing else.
fn bench_merge_stage(c: &mut Criterion) {
    for (tok_name, tok_path) in TOKENIZERS.iter().chain(HUB_TOKENIZERS.iter()).copied() {
        let Some((oracle, pipeline)) = pair(tok_name, tok_path, false) else {
            continue;
        };
        let ModelWrapper::BPE(legacy) = oracle.get_model() else {
            continue;
        };
        let PipelineModel::BPE(current) = pipeline.get_model() else {
            continue;
        };

        for (corpus, path) in CORPORA {
            let Ok(text) = std::fs::read_to_string(path) else {
                continue;
            };
            let mut end = CORPUS_BYTES.min(text.len());
            while end > 0 && !text.is_char_boundary(end) {
                end -= 1;
            }
            let inputs = model_inputs(&oracle, &text[..end]);
            let total_bytes: u64 = inputs.iter().map(|(_, raw)| raw.len() as u64).sum();
            if total_bytes == 0 {
                continue;
            }

            let mut group = c.benchmark_group(format!("{tok_name}-{corpus}-merge"));
            group.throughput(Throughput::Bytes(total_bytes));
            group.bench_with_input(
                BenchmarkId::new("legacy", "pretoken"),
                &inputs,
                |b, inputs| {
                    b.iter(|| {
                        for (pretokenized, _) in inputs {
                            black_box(legacy.tokenize(black_box(pretokenized.as_str())).unwrap());
                        }
                    })
                },
            );
            group.bench_with_input(
                BenchmarkId::new("pipeline", "pretoken"),
                &inputs,
                |b, inputs| {
                    let mut scratch = current.init_scratch();
                    let mut output = Vec::new();
                    b.iter(|| {
                        for (_, raw) in inputs {
                            output.clear();
                            current
                                .tokenize_pipeline(
                                    black_box(raw.as_str()),
                                    &mut scratch,
                                    &mut output,
                                )
                                .unwrap();
                            black_box(output.as_slice());
                        }
                    })
                },
            );
            group.finish();
        }
    }
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
                                    black_box(pipeline.encode(chunk, false).wait().unwrap());
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
    targets = bench_merge_stage, bench_pipeline
}
criterion_main!(benches);
