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
use tk_encode::pipeline::{PipelineTokenizer, StageNanos};
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

/// One instrumented pass over `chunks`, accumulating per-stage nanoseconds via the
/// pipeline's staged-timing path (normalize / pre-tokenize / model, plus the whole-encode
/// total). This is the "measure the time taken adding staged" decomposition — no external
/// profiler, just one timed region per stage.
fn stage_pass(pipeline: &PipelineTokenizer, chunks: &[String]) -> StageNanos {
    let mut t = StageNanos::default();
    let mut n = 0usize;
    for chunk in chunks {
        n += pipeline.encode_timed(chunk, &mut t).unwrap();
    }
    black_box(n);
    t
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

        // Staged decomposition of the pipeline's own encode: where does its time go?
        // Take the median-by-total instrumented run, reported as ns/byte per stage.
        stage_pass(pipeline, &chunks); // warm-up
        let mut staged: Vec<StageNanos> =
            (0..REPS).map(|_| stage_pass(pipeline, &chunks)).collect();
        staged.sort_by_key(|s| s.total);
        let st = staged[staged.len() / 2];
        let other = st.total.saturating_sub(st.normalize + st.pre_tokenize + st.model);
        let nspb = |ns: u128| ns as f64 / bytes as f64;
        eprintln!(
            "    stages ns/byte: norm {:.2}, split {:.2}, model {:.2}, other {:.2}",
            nspb(st.normalize),
            nspb(st.pre_tokenize),
            nspb(st.model),
            nspb(other),
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
            // pipeline-only stage decomposition (ns/byte); stages + other ≈ total.
            "stage_ns_per_byte": {
                "normalize": nspb(st.normalize),
                "pre_tokenize": nspb(st.pre_tokenize),
                "model": nspb(st.model),
                "other": nspb(other),
                "total": nspb(st.total),
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
