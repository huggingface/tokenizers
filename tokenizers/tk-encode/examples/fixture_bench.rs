//! Comparative throughput of the reference `Tokenizer` vs the experimental
//! `PipelineTokenizer`, for every model in `examples/bench_models.json` across
//! every corpus in `data/fixtures/` (languages + modalities), on ~10 kB inputs
//! — the regime where per-input overhead is amortized (see `pipeline_benchmark.rs`
//! for the size sweep).
//!
//! `PipelineTokenizer` is a work in progress: it only builds for tokenizers whose
//! pre-tokenizer is Bert / Whitespace / None. Models it can't build (byte-level
//! BPE, SentencePiece/Unigram, …) are reported as `supported: false` with their
//! pipeline shape, rather than benched — the CI grid renders those as roadmap cards.
//!
//! Emits a JSON array (one object per model) on stdout, consumed by
//! `.github/scripts/render_pipeline_bench.py` in CI.

use std::convert::TryFrom;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::{json, Value};
use tk_encode::pipeline::PipelineTokenizer;
use tk_encode::{ModelWrapper, Tokenizer};

const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");
const MANIFEST: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/bench_models.json");
const CHUNK_BYTES: usize = 10 * 1024;
const MAX_CHUNKS: usize = 100;
const REPS: usize = 5;

fn make_chunks(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur = String::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        if !cur.is_empty() {
            cur.push('\n');
        }
        cur.push_str(line);
        if cur.len() >= CHUNK_BYTES {
            chunks.push(std::mem::take(&mut cur));
            if chunks.len() == MAX_CHUNKS {
                return chunks;
            }
        }
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

fn median_secs(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn time_pass(encode: &dyn Fn(&str) -> usize, chunks: &[String]) -> f64 {
    let start = Instant::now();
    let mut n = 0usize;
    for chunk in chunks {
        n += encode(chunk);
    }
    black_box(n);
    start.elapsed().as_secs_f64()
}

/// Median wall-time (seconds) of one pass over `chunks` through the shared encode core
/// `PipelineTokenizer::encode_generic::<STAGE>`. `STAGE` is a const generic, so each
/// level is a branchless specialization with the later stages compiled out — timing
/// successive levels and subtracting gives each stage's marginal cost (the ablation
/// ladder), no profiler and no per-segment instrumentation.
///
/// Both caller-owned buffers are reused across chunks and `black_box`'d each iteration:
/// `output` anchors the special-scan/normalize/model work, `pre_tokens` anchors the
/// split stage, so under fat LTO no dead partial stage gets optimized away. The
/// `black_box` lives here, in the bench — never in the library.
fn stage_secs<const STAGE: u8>(pipeline: &PipelineTokenizer, chunks: &[String]) -> f64 {
    let mut out = Vec::new();
    let mut pre_tokens = Vec::new();
    let mut run = || {
        for chunk in chunks {
            out.clear();
            let _ = pipeline.encode_generic::<STAGE>(chunk, &mut out, &mut pre_tokens);
            black_box(&out);
            black_box(&pre_tokens);
        }
    };
    run(); // warm-up
    let mut samples = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let start = Instant::now();
        run();
        samples.push(start.elapsed().as_secs_f64());
    }
    median_secs(samples)
}

fn fixture_files() -> Vec<(String, PathBuf)> {
    let mut files = Vec::new();
    for group in ["lang", "modalities"] {
        let dir = Path::new(DATA_DIR).join("fixtures").join(group);
        let mut entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("{}: {e} — run `make fixtures` first", dir.display()))
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|x| x == "txt"))
            .collect();
        entries.sort();
        for path in entries {
            files.push((group.to_string(), path));
        }
    }
    files
}

/// Local path to a manifest entry's config: `data/<file>`, else `data/<name>.json`.
/// Both come from the test-data dataset (see the Makefile `bench-models` target).
fn model_path(entry: &Value) -> PathBuf {
    let name = entry["name"].as_str().unwrap();
    let file = entry
        .get("file")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| format!("{name}.json"));
    Path::new(DATA_DIR).join(file)
}

fn model_kind(tok: &Tokenizer) -> &'static str {
    match tok.get_model() {
        ModelWrapper::BPE(_) => "BPE",
        ModelWrapper::WordPiece(_) => "WordPiece",
        ModelWrapper::WordLevel(_) => "WordLevel",
        ModelWrapper::Unigram(_) => "Unigram",
    }
}

/// A short pre-tokenizer descriptor read from the raw json — its `type`, and for
/// a Sequence the (de-duplicated) inner types, e.g. `Split+ByteLevel`.
fn pretok_label(path: &Path) -> String {
    let v: Value = std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(Value::Null);
    let pt = &v["pre_tokenizer"];
    match pt["type"].as_str() {
        None => "None".to_string(),
        Some("Sequence") => {
            let mut inner: Vec<&str> = pt["pretokenizers"]
                .as_array()
                .map(|a| a.iter().filter_map(|x| x["type"].as_str()).collect())
                .unwrap_or_default();
            inner.dedup();
            inner.join("+")
        }
        Some("BertPreTokenizer") => "Bert".to_string(),
        Some(other) => other.to_string(),
    }
}

fn bench_model(
    tok: &Tokenizer,
    pipeline: &PipelineTokenizer,
    files: &[(String, PathBuf)],
) -> Vec<Value> {
    let legacy_enc = |s: &str| tok.encode(s, false).unwrap().len();
    let pipeline_enc = |s: &str| pipeline.encode(s, false).unwrap().len();

    let mut rows = Vec::new();
    for (group, path) in files {
        let name = path.file_stem().unwrap().to_str().unwrap().to_string();
        let text = std::fs::read_to_string(path).unwrap();
        let chunks = make_chunks(&text);
        let bytes: usize = chunks.iter().map(String::len).sum();

        let ids_match = chunks.iter().take(3).all(|c| {
            let expected = tok.encode(c.as_str(), false).unwrap();
            let got: Vec<u32> = pipeline
                .encode(c, false)
                .unwrap()
                .iter()
                .map(|t| t.id)
                .collect();
            expected.get_ids() == got
        });

        // interleave impls so frequency/thermal drift hits both equally
        time_pass(&legacy_enc, &chunks);
        time_pass(&pipeline_enc, &chunks);
        let (mut legacy_s, mut pipeline_s) = (Vec::new(), Vec::new());
        for _ in 0..REPS {
            legacy_s.push(time_pass(&legacy_enc, &chunks));
            pipeline_s.push(time_pass(&pipeline_enc, &chunks));
        }
        let (l, p) = (median_secs(legacy_s), median_secs(pipeline_s));
        let (legacy_mbps, pipeline_mbps) = (bytes as f64 / l / 1e6, bytes as f64 / p / 1e6);
        eprintln!("  {name}: legacy {legacy_mbps:.1} MB/s, pipeline {pipeline_mbps:.1} MB/s");

        // Staged decomposition of the pipeline's own encode via the ablation ladder:
        // time each cumulative stage level, then subtract to isolate each stage's cost
        // (e.g. model = t_model - t_split). Levels are the named STAGE_* consts on
        // PipelineTokenizer rather than bare 0/1/2/3.
        let t_frame = stage_secs::<{ PipelineTokenizer::STAGE_FRAME }>(pipeline, &chunks);
        let t_norm = stage_secs::<{ PipelineTokenizer::STAGE_NORMALIZE }>(pipeline, &chunks);
        let t_split = stage_secs::<{ PipelineTokenizer::STAGE_SPLIT }>(pipeline, &chunks);
        let t_model = stage_secs::<{ PipelineTokenizer::STAGE_MODEL }>(pipeline, &chunks);
        // Two distinct "split" costs are separated here: `added_split` is the
        // added/special-token scan (the SpecialSegmentIterator over the AddedVocabulary,
        // captured by the FRAME level), `pre_tokenize` is the pre-tokenizer split. All
        // four stages sum exactly to `total`.
        let nspb = |secs: f64| secs * 1e9 / bytes as f64;
        let (ns_added, ns_norm, ns_split, ns_model) = (
            nspb(t_frame.max(0.0)),
            nspb((t_norm - t_frame).max(0.0)),
            nspb((t_split - t_norm).max(0.0)),
            nspb((t_model - t_split).max(0.0)),
        );
        eprintln!(
            "    stages ns/byte: added-split {ns_added:.2}, norm {ns_norm:.2}, pre-split {ns_split:.2}, model {ns_model:.2}"
        );

        rows.push(json!({
            "fixture": name,
            "group": group,
            "bytes": bytes,
            "chunks": chunks.len(),
            "legacy_mbps": legacy_mbps,
            "pipeline_mbps": pipeline_mbps,
            "speedup": l / p,
            "ids_match": ids_match,
            // pipeline-only stage decomposition (ns/byte); the four stages sum to total.
            "stage_ns_per_byte": {
                "added_split": ns_added,
                "normalize": ns_norm,
                "pre_tokenize": ns_split,
                "model": ns_model,
                "total": nspb(t_model),
            },
        }));
    }
    rows
}

fn main() {
    let manifest: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(MANIFEST).unwrap()).unwrap();
    let files = fixture_files();

    let mut out: Vec<Value> = Vec::new();
    for entry in &manifest {
        let name = entry["name"].as_str().unwrap().to_string();
        let repo = entry.get("repo").and_then(Value::as_str).unwrap_or("");
        let path = model_path(entry);
        eprintln!("== {name} ({repo}) ==");

        let tok = match Tokenizer::from_file(&path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  load failed: {e}");
                out.push(json!({
                    "model": name, "repo": repo, "shape": "?",
                    "supported": false, "reason": format!("load error: {e}"), "results": [],
                }));
                continue;
            }
        };
        let shape = format!("{} · {}", model_kind(&tok), pretok_label(&path));

        match PipelineTokenizer::try_from(&tok) {
            // A model can satisfy the pipeline's *build* constraints yet still fail at
            // *encode* time — e.g. a Sequence containing ByteLevel, which rewrites bytes
            // and has no range-based impl. Probe once and downgrade to "unsupported"
            // (with the reason) instead of panicking partway through the bench.
            Ok(pipeline) => match pipeline.encode("The quick brown fox jumps 123.", false) {
                Ok(_) => {
                    let rows = bench_model(&tok, &pipeline, &files);
                    out.push(json!({
                        "model": name, "repo": repo, "shape": shape,
                        "supported": true, "results": rows,
                    }));
                }
                Err(e) => {
                    eprintln!("  builds but can't encode yet ({shape}): {e}");
                    out.push(json!({
                        "model": name, "repo": repo, "shape": shape,
                        "supported": false, "reason": format!("{e}"), "results": [],
                    }));
                }
            },
            Err(_) => {
                eprintln!("  unsupported by PipelineTokenizer ({shape})");
                out.push(json!({
                    "model": name, "repo": repo, "shape": shape,
                    "supported": false, "results": [],
                }));
            }
        }
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&Value::Array(out)).unwrap()
    );
}
