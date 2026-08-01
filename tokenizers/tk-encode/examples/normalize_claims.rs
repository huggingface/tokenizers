//! What the space rewrite costs when written, and what the zero-copy path wins back.
//!
//! For each SentencePiece-shaped model this runs the `encode_generic::<STAGE>` ablation
//! ladder (the same methodology as `fixture_bench`) over long and short inputs, then
//! decomposes the fused normalizer's pass into its three parts: counting the spaces,
//! allocating the rewrite `String`, and writing it. Models taking the zero-copy path
//! (see `ZeroCopyMetaspace`) get an interleaved A/B against the written rewrite, with
//! ids compared over the whole corpus first. A last probe measures what the second
//! special-token scan adds once a tokenizer holds `normalized` added tokens.

use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use atomsplit::literal::Literal;
use tk_encode::pipeline::{Model, PipelineTokenizer};
use tk_encode::{AddedToken, NormalizerWrapper, Tokenizer};

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");
const REPS: usize = 9;
const CHUNK_BYTES: usize = 10 * 1024;
const MAX_BYTES: usize = 512 * 1024;

/// (config file, prepend for the decomposition replica; `None` skips it because the
/// model runs the `drop_whitespace` path, which is a different rewrite)
const MODELS: &[(&str, Option<bool>)] = &[
    ("llama-2.json", Some(true)),
    ("gemma-4.json", Some(false)),
    ("t5-base.json", None),
    ("albert-base-v1-tokenizer.json", None),
];

const FIXTURES: &[&str] = &[
    "fixtures/lang/eng_Latn.txt",
    "fixtures/lang/cmn_Hani.txt",
    "fixtures/modalities/code_mixed.txt",
];

/// The words `fixture_bench` injects as `normalized:true` added tokens, so the probe
/// exercises the same second-scan state the comparative benchmark runs under.
const NORMALIZED_WORDS: &[&str] = &["widgetron", "flibberjast", "zorptastic", "quibblenaut"];

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn timed(mut run: impl FnMut()) -> f64 {
    run(); // warm-up
    let mut samples = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let start = Instant::now();
        run();
        samples.push(start.elapsed().as_secs_f64());
    }
    median(samples)
}

fn stage_secs<const STAGE: u8>(pipeline: &PipelineTokenizer, chunks: &[String]) -> f64 {
    let mut out = Vec::new();
    let mut pre_tokens = Vec::new();
    let mut scratch = pipeline.get_model().init_scratch();
    timed(|| {
        for chunk in chunks {
            out.clear();
            let _ = pipeline.encode_generic::<STAGE>(
                chunk,
                true,
                &mut pre_tokens,
                &mut scratch,
                &mut out,
            );
            black_box(&out);
            black_box(&pre_tokens);
        }
    })
}

/// Whole lines accumulated up to `chunk_bytes`, `MAX_BYTES` in total. `chunk_bytes = 0`
/// keeps each line its own chunk (the short-input regime).
fn chunks_of(text: &str, chunk_bytes: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut total = 0usize;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        current.push_str(line);
        if current.len() >= chunk_bytes {
            total += current.len();
            chunks.push(std::mem::take(&mut current));
            if total >= MAX_BYTES {
                return chunks;
            }
        } else {
            current.push(' ');
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

struct Ladder {
    added: f64,
    norm: f64,
    split: f64,
    model: f64,
    post: f64,
    total_mbs: f64,
}

fn ladder(pipeline: &PipelineTokenizer, chunks: &[String], bytes: usize) -> Ladder {
    let t_frame = stage_secs::<{ PipelineTokenizer::STAGE_FRAME }>(pipeline, chunks);
    let t_norm = stage_secs::<{ PipelineTokenizer::STAGE_NORMALIZE }>(pipeline, chunks);
    let t_split = stage_secs::<{ PipelineTokenizer::STAGE_SPLIT }>(pipeline, chunks);
    let t_model = stage_secs::<{ PipelineTokenizer::STAGE_MODEL }>(pipeline, chunks);
    let t_post = stage_secs::<{ PipelineTokenizer::STAGE_POSTPROCESS }>(pipeline, chunks);
    let nspb = |secs: f64| secs * 1e9 / bytes as f64;
    Ladder {
        added: nspb(t_frame.max(0.0)),
        norm: nspb((t_norm - t_frame).max(0.0)),
        split: nspb((t_split - t_norm).max(0.0)),
        model: nspb((t_model - t_split).max(0.0)),
        post: nspb((t_post - t_model).max(0.0)),
        total_mbs: bytes as f64 / t_post / 1e6,
    }
}

/// The fused normalizer's pass, split into its three costs over the same chunks:
/// the space count, the `String` allocation, and the full rewrite (count + alloc +
/// write). All three follow `MetaspaceNormalizer::normalize`'s non-`drop_whitespace`
/// arm byte for byte, so `rewrite` should land on the ladder's norm marginal.
fn decompose(chunks: &[String], bytes: usize, prepend: bool) -> (f64, f64, f64) {
    let space = Literal::new(b" ").unwrap();
    let delimiter = "\u{2581}";
    let counts: Vec<usize> = chunks
        .iter()
        .map(|c| space.count_matches(c.as_bytes()))
        .collect();
    let nspb = |secs: f64| secs * 1e9 / bytes as f64;

    let count_only = timed(|| {
        for chunk in chunks {
            black_box(space.count_matches(chunk.as_bytes()));
        }
    });
    let alloc_only = timed(|| {
        for (chunk, &count) in chunks.iter().zip(&counts) {
            let s = String::with_capacity(chunk.len() + 2 * count + if prepend { 3 } else { 0 });
            black_box(&s);
        }
    });
    let rewrite = timed(|| {
        for chunk in chunks {
            let count = space.count_matches(chunk.as_bytes());
            if !prepend && count == 0 {
                black_box(chunk.as_str());
                continue;
            }
            let mut rewritten =
                String::with_capacity(chunk.len() + 2 * count + if prepend { 3 } else { 0 });
            if prepend {
                rewritten.push_str(delimiter);
            }
            let mut prev = 0;
            space.for_each_match(chunk.as_bytes(), |start| {
                rewritten.push_str(&chunk[prev..start]);
                rewritten.push_str(delimiter);
                prev = start + 1;
            });
            rewritten.push_str(&chunk[prev..]);
            black_box(&rewritten);
        }
    });
    (nspb(count_only), nspb(alloc_only), nspb(rewrite))
}

fn main() {
    let fixtures: Vec<(String, String)> = FIXTURES
        .iter()
        .map(|rel| {
            let path = Path::new(DATA).join(rel);
            let name = path.file_stem().unwrap().to_str().unwrap().to_string();
            (name, std::fs::read_to_string(&path).unwrap())
        })
        .collect();

    for &(file, replica_prepend) in MODELS {
        let path = Path::new(DATA).join(file);
        let mut tok = match Tokenizer::from_file(&path) {
            Ok(t) => t,
            Err(e) => {
                println!("== {file}: load failed: {e}");
                continue;
            }
        };
        let mut pipeline = match PipelineTokenizer::try_from(&tok) {
            Ok(p) => p,
            Err(e) => {
                println!("== {file}: no pipeline: {e}");
                continue;
            }
        };
        // The ladder times the written stages; the zero-copy path gets its own A/B below.
        pipeline.disable_zero_copy();
        let pretok = format!("{:?}", pipeline.get_pre_tokenizer());
        let pretok = pretok.split(['(', ' ']).next().unwrap_or("?");
        println!("== {file} (pre-tokenizer: {pretok})");

        for (name, text) in &fixtures {
            for (regime, chunk_bytes) in [("10kB", CHUNK_BYTES), ("line", 0)] {
                let chunks = chunks_of(text, chunk_bytes);
                let bytes: usize = chunks.iter().map(String::len).sum();
                let l = ladder(&pipeline, &chunks, bytes);
                let total = l.added + l.norm + l.split + l.model + l.post;
                println!(
                    "  {name:<12} {regime:<4} ({:>5} chunks, {:>4} kB): added {:.3} | norm {:.3} | split {:.3} | model {:.3} | post {:.3} ns/B  e2e {:.0} MB/s  norm = {:.1}% of encode",
                    chunks.len(),
                    bytes / 1024,
                    l.added,
                    l.norm,
                    l.split,
                    l.model,
                    l.post,
                    l.total_mbs,
                    100.0 * l.norm / total.max(1e-9),
                );
                if let Some(prepend) = replica_prepend {
                    let (count, alloc, rewrite) = decompose(&chunks, bytes, prepend);
                    println!(
                        "  {:12} {regime:<4}  norm decomposed: count {count:.3} + alloc {alloc:.3} + write {:.3} = replica {rewrite:.3} (ladder said {:.3})",
                        "",
                        (rewrite - count - alloc).max(0.0),
                        l.norm,
                    );
                }
            }
        }

        // Zero-copy A/B: the same binary and corpus, the two paths interleaved rep by rep so
        // frequency drift hits both equally. Ids are compared over the whole corpus first.
        if replica_prepend.is_some() {
            let zero_copy = PipelineTokenizer::try_from(&tok).unwrap();
            assert!(
                zero_copy.has_zero_copy(),
                "{file}: the zero-copy path should fire"
            );
            let mut written = PipelineTokenizer::try_from(&tok).unwrap();
            written.disable_zero_copy();
            for (name, text) in &fixtures {
                for (regime, chunk_bytes) in [("10kB", CHUNK_BYTES), ("line", 0)] {
                    let chunks = chunks_of(text, chunk_bytes);
                    let bytes: usize = chunks.iter().map(String::len).sum();
                    for chunk in &chunks {
                        let ids = |p: &PipelineTokenizer| -> Vec<u32> {
                            p.encode(chunk, true)
                                .unwrap()
                                .iter()
                                .map(|t| t.id)
                                .collect()
                        };
                        assert_eq!(ids(&zero_copy), ids(&written), "{file} {name}: ids diverge");
                    }
                    let pass = |p: &PipelineTokenizer| {
                        let mut n = 0usize;
                        for chunk in &chunks {
                            n += p.encode(chunk, true).unwrap().len();
                        }
                        black_box(n);
                    };
                    pass(&zero_copy); // warm-up
                    pass(&written);
                    let mut zc = Vec::with_capacity(REPS);
                    let mut wr = Vec::with_capacity(REPS);
                    for _ in 0..REPS {
                        let t = Instant::now();
                        pass(&zero_copy);
                        zc.push(t.elapsed().as_secs_f64());
                        let t = Instant::now();
                        pass(&written);
                        wr.push(t.elapsed().as_secs_f64());
                    }
                    let (zc, wr) = (median(zc), median(wr));
                    println!(
                        "  A/B {name:<12} {regime:<4}: zero-copy {:.0} MB/s vs written {:.0} MB/s  ({:+.1}%)",
                        bytes as f64 / zc / 1e6,
                        bytes as f64 / wr / 1e6,
                        100.0 * (wr / zc - 1.0),
                    );
                }
            }
        }

        // t5 and albert declare more normalizers than the space rewrite (t5 a Precompiled
        // charsmap, albert five steps and then one). The design only removes the rewrite,
        // so its share is measured by stripping the declared normalizer: what is left in
        // the norm marginal comes from the Metaspace pre-tokenizer's rewriting half.
        if replica_prepend.is_none() {
            let mut stripped = Tokenizer::from_file(&path).unwrap();
            let _ = stripped.with_normalizer(None::<NormalizerWrapper>);
            if let Ok(metaspace_only) = PipelineTokenizer::try_from(&stripped) {
                let (name, text) = &fixtures[0];
                let chunks = chunks_of(text, CHUNK_BYTES);
                let bytes: usize = chunks.iter().map(String::len).sum();
                let l = ladder(&metaspace_only, &chunks, bytes);
                println!(
                    "  declared normalizer stripped, {name} 10kB: norm {:.3} ns/B is the metaspace share",
                    l.norm,
                );
            }
        }

        // The second scan runs over every normalized chunk, but on an empty normalized
        // vocabulary `Buckets::match_bytes` returns before touching the text. Injecting
        // normalized added tokens (as `fixture_bench` does for every model) makes it a
        // real pass; the frame marginal shows what the design's build-time gate saves.
        let injected: Vec<AddedToken> = NORMALIZED_WORDS
            .iter()
            .map(|w| AddedToken::from(*w, false).normalized(true))
            .collect();
        let _ = tok.add_tokens(injected);
        if let Ok(with_normalized) = PipelineTokenizer::try_from(&tok) {
            let (name, text) = &fixtures[0];
            let chunks = chunks_of(text, CHUNK_BYTES);
            let bytes: usize = chunks.iter().map(String::len).sum();
            let before = ladder(&pipeline, &chunks, bytes);
            let after = ladder(&with_normalized, &chunks, bytes);
            println!(
                "  2nd-scan probe on {name} 10kB: added {:.3} -> {:.3} ns/B with {} normalized added tokens",
                before.added,
                after.added,
                NORMALIZED_WORDS.len(),
            );
        }
        println!();
    }
}
