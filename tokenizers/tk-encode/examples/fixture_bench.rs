//! Comparative benchmark of the experimental `PipelineTokenizer` against the
//! latest *released* `tokenizers` crate (the bar to beat — the in-tree legacy
//! `Tokenizer` is on its way out, so the release is the reference), for every
//! model in `examples/bench_models.json` across every corpus in `data/fixtures/`.
//!
//! Per fixture it measures throughput on ~10 kB inputs (the regime where
//! per-input overhead is amortized — see `pipeline_benchmark.rs` for the size
//! sweep). Per model it also measures resident-set deltas by re-spawning itself
//! as `--memory <impl> <model.json>` children — one implementation per process,
//! so allocator page reuse across implementations can't blur the attribution.
//!
//! The in-tree `Tokenizer` is *not* benched: it only builds the pipeline and
//! serves as the id-correctness oracle (`ids_match`, which CI fails on).
//! `ids_match_baseline` — pipeline vs the released crate — is report-only,
//! since a branch may intentionally fix encode behavior. Models the pipeline
//! can't build yet are reported as `supported: false` with their pipeline shape
//! rather than benched — the CI grid renders those as roadmap cards. Each
//! manifest entry carries a `desc`: a one-line label of the workload archetype
//! the model exercises, passed through to the report.
//!
//! Emits one JSON object (`{baseline, models}`) on stdout, consumed by
//! `.github/scripts/render_pipeline_bench.py` in CI. For local iteration the
//! data dir, manifest and rep count can be overridden with the
//! `FIXTURE_BENCH_DATA` / `FIXTURE_BENCH_MANIFEST` / `FIXTURE_BENCH_REPS` env vars.

use std::convert::TryFrom;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use serde_json::{json, Value};
use tk_encode::pipeline::PipelineTokenizer;
use tk_encode::{AddedToken, ModelWrapper, Tokenizer};
use tokenizers_release::{AddedToken as BaselineAddedToken, Tokenizer as BaselineTokenizer};

const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");
const MANIFEST: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/bench_models.json");
// Keep in sync with the `tokenizers-release` pin in Cargo.toml.
const BASELINE_VERSION: &str = "0.23.1";
const CHUNK_BYTES: usize = 10 * 1024;
const MAX_CHUNKS: usize = 100;
const REPS: usize = 5;
const PROBE: &str = "The quick brown fox jumps 123.";
// Chunks per fixture for the memory children's encode pass: enough to warm the
// lazy structures and grow the output buffers, small enough to stay cheap.
const MEM_CHUNKS_PER_FIXTURE: usize = 2;

// Added tokens injected into every loaded tokenizer so the `added_*` fixtures exercise
// the added-token split for whichever model is benched — no bespoke tokenizer config.
// `ADDED_SPECIAL` (normalized:false) is matched on the raw pass; `ADDED_NORMALIZED`
// (normalized:true) on the normalized pass. The strings are distinctive markers that do
// not occur in the language/modality corpora, so they leave those results untouched. The
// `added_*` fixtures are built from exactly these strings (see the dataset FIXTURES.md).
const ADDED_SPECIAL: &[&str] = &["<|xs0|>", "<|xs1|>", "<|xs2|>", "<|xs3|>", "<|xs4|>"];
const ADDED_NORMALIZED: &[&str] = &[
    "widgetron",
    "flibberjast",
    "zorptastic",
    "quibblenaut",
    "snorlaxian",
    "blorptronic",
    "wuzzlefang",
    "crungledorf",
];

fn data_dir() -> PathBuf {
    std::env::var_os("FIXTURE_BENCH_DATA").map_or_else(|| PathBuf::from(DATA_DIR), PathBuf::from)
}

fn manifest_path() -> PathBuf {
    std::env::var_os("FIXTURE_BENCH_MANIFEST")
        .map_or_else(|| PathBuf::from(MANIFEST), PathBuf::from)
}

fn reps() -> usize {
    std::env::var("FIXTURE_BENCH_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(REPS)
}

/// Inject the benchmark's added tokens into `tok` before the pipeline is built from it,
/// so both the oracle and the pipeline see them (and stay id-for-id identical).
fn inject_added_tokens(tok: &mut Tokenizer) {
    let special = ADDED_SPECIAL
        .iter()
        .map(|s| AddedToken::from(*s, true).normalized(false));
    let normalized = ADDED_NORMALIZED
        .iter()
        .map(|s| AddedToken::from(*s, false).normalized(true));
    let _ = tok.add_special_tokens(special);
    let _ = tok.add_tokens(normalized);
}

/// Same injection through the released crate's `AddedToken`.
fn inject_added_tokens_baseline(tok: &mut BaselineTokenizer) {
    let special = ADDED_SPECIAL
        .iter()
        .map(|s| BaselineAddedToken::from(*s, true).normalized(false));
    let normalized = ADDED_NORMALIZED
        .iter()
        .map(|s| BaselineAddedToken::from(*s, false).normalized(true));
    let _ = tok.add_special_tokens(special);
    let _ = tok.add_tokens(normalized);
}

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
    let mut samples = Vec::with_capacity(reps());
    for _ in 0..reps() {
        let start = Instant::now();
        run();
        samples.push(start.elapsed().as_secs_f64());
    }
    median_secs(samples)
}

fn fixture_files() -> Vec<(String, PathBuf)> {
    let mut files = Vec::new();
    for group in ["lang", "modalities"] {
        let dir = data_dir().join("fixtures").join(group);
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
    data_dir().join(file)
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

/// Resident set size in bytes. Exact on Linux (`/proc`); on macOS a `ps`
/// fallback good enough for local iteration — CI runs on Linux.
fn rss_now() -> Option<i64> {
    if cfg!(target_os = "linux") {
        proc_status_bytes("VmRSS:")
    } else {
        let out = Command::new("ps")
            .args(["-o", "rss=", "-p"])
            .arg(std::process::id().to_string())
            .output()
            .ok()?;
        let kb: i64 = String::from_utf8(out.stdout).ok()?.trim().parse().ok()?;
        Some(kb * 1024)
    }
}

/// Peak resident set (VmHWM) in bytes — Linux only, `None` elsewhere.
fn rss_peak() -> Option<i64> {
    proc_status_bytes("VmHWM:")
}

fn proc_status_bytes(key: &str) -> Option<i64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let kb: i64 = status
        .lines()
        .find(|l| l.starts_with(key))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()?;
    Some(kb * 1024)
}

/// `--memory <impl> <model.json>` child entry: load one implementation, encode a
/// capped pass over the fixtures, print `{load_bytes, encode_bytes, peak_bytes}`.
/// One implementation per process so the deltas attribute cleanly.
fn memory_child(which: &str, model: &Path) {
    let mut chunks: Vec<String> = Vec::new();
    for (_, path) in &fixture_files() {
        let text = std::fs::read_to_string(path).unwrap();
        chunks.extend(make_chunks(&text).into_iter().take(MEM_CHUNKS_PER_FIXTURE));
    }

    let rss0 = rss_now().unwrap_or(0);
    let mut n = 0usize;
    let (after_load, after_encode) = match which {
        "baseline" => {
            let mut tok = BaselineTokenizer::from_file(model).unwrap();
            inject_added_tokens_baseline(&mut tok);
            let after_load = rss_now().unwrap_or(0);
            for c in &chunks {
                n += tok.encode(c.as_str(), false).unwrap().len();
            }
            (after_load, rss_now().unwrap_or(0))
        }
        "pipeline" => {
            let mut tok = Tokenizer::from_file(model).unwrap();
            inject_added_tokens(&mut tok);
            let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
            // Count only the pipeline's own structures; the transient source
            // Tokenizer still shows in peak_bytes, which is honest — building a
            // pipeline currently requires one.
            drop(tok);
            let after_load = rss_now().unwrap_or(0);
            for c in &chunks {
                n += pipeline.encode(c, false).unwrap().len();
            }
            (after_load, rss_now().unwrap_or(0))
        }
        other => panic!("unknown impl {:?}", other),
    };
    black_box(n);

    println!(
        "{}",
        json!({
            "load_bytes": after_load - rss0,
            "encode_bytes": after_encode - after_load,
            "peak_bytes": rss_peak().map(|p| p - rss0),
        })
    );
}

/// Re-run this binary once per available implementation to get per-impl memory
/// numbers that a shared address space couldn't provide.
fn measure_memory(model: &Path, baseline_ok: bool) -> Value {
    let exe = std::env::current_exe().unwrap();
    let mut out = serde_json::Map::new();
    for (key, ok) in [("baseline", baseline_ok), ("pipeline", true)] {
        if !ok {
            out.insert(key.into(), Value::Null);
            continue;
        }
        let res = Command::new(&exe)
            .arg("--memory")
            .arg(key)
            .arg(model)
            .output()
            .expect("failed to spawn memory child");
        if !res.status.success() {
            eprintln!(
                "  memory child {key} failed: {}",
                String::from_utf8_lossy(&res.stderr)
            );
            out.insert(key.into(), Value::Null);
            continue;
        }
        let parsed: Value = serde_json::from_slice(&res.stdout).unwrap_or(Value::Null);
        out.insert(key.into(), parsed);
    }
    Value::Object(out)
}

type EncodeFn<'a> = Box<dyn Fn(&str) -> usize + 'a>;

fn bench_model(
    baseline: Option<&BaselineTokenizer>,
    oracle: &Tokenizer,
    pipeline: &PipelineTokenizer,
    files: &[(String, PathBuf)],
) -> Vec<Value> {
    let pipe_enc = |s: &str| pipeline.encode(s, false).unwrap().len();
    let base_enc: Option<EncodeFn> =
        baseline.map(|b| Box::new(move |s: &str| b.encode(s, false).unwrap().len()) as EncodeFn);

    let mut rows = Vec::new();
    for (group, path) in files {
        let name = path.file_stem().unwrap().to_str().unwrap().to_string();
        let text = std::fs::read_to_string(path).unwrap();
        let chunks = make_chunks(&text);
        let bytes: usize = chunks.iter().map(String::len).sum();

        let pipe_ids = |c: &String| -> Vec<u32> {
            pipeline
                .encode(c, false)
                .unwrap()
                .iter()
                .map(|t| t.id)
                .collect()
        };
        // The correctness gate CI fails on: pipeline vs this tree's Tokenizer.
        let ids_match = chunks
            .iter()
            .take(3)
            .all(|c| oracle.encode(c.as_str(), false).unwrap().get_ids() == pipe_ids(c));
        // Report-only: pipeline vs the released crate (a branch may fix encode bugs).
        let ids_match_baseline = baseline.map(|b| {
            chunks
                .iter()
                .take(3)
                .all(|c| b.encode(c.as_str(), false).unwrap().get_ids() == pipe_ids(c))
        });

        // interleave both impls so frequency/thermal drift hits them equally
        if let Some(be) = &base_enc {
            time_pass(be, &chunks);
        }
        time_pass(&pipe_enc, &chunks);
        let (mut base_s, mut pipe_s) = (Vec::new(), Vec::new());
        for _ in 0..reps() {
            if let Some(be) = &base_enc {
                base_s.push(time_pass(be, &chunks));
            }
            pipe_s.push(time_pass(&pipe_enc, &chunks));
        }
        let mbps = |secs: f64| bytes as f64 / secs / 1e6;
        let base_mbps = (!base_s.is_empty()).then(|| mbps(median_secs(base_s)));
        let pipe_mbps = mbps(median_secs(pipe_s));

        let fmt = |v: Option<f64>| v.map_or("—".into(), |v| format!("{v:.1}"));
        eprintln!(
            "  {name}: baseline {} MB/s, pipeline {pipe_mbps:.1} MB/s",
            fmt(base_mbps)
        );

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
            "mbps": { "baseline": base_mbps, "pipeline": pipe_mbps },
            "ids_match": ids_match,
            "ids_match_baseline": ids_match_baseline,
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
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--memory") {
        memory_child(&args[2], Path::new(&args[3]));
        return;
    }

    let manifest: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(manifest_path()).unwrap()).unwrap();
    let files = fixture_files();

    let mut models: Vec<Value> = Vec::new();
    for entry in &manifest {
        let name = entry["name"].as_str().unwrap().to_string();
        let repo = entry.get("repo").and_then(Value::as_str).unwrap_or("");
        let desc = entry.get("desc").and_then(Value::as_str).unwrap_or("");
        let path = model_path(entry);
        eprintln!("== {name} ({repo}) ==");

        let mut tok = match Tokenizer::from_file(&path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  load failed: {e}");
                models.push(json!({
                    "model": name, "repo": repo, "desc": desc, "shape": "?",
                    "supported": false, "reason": format!("load error: {e}"),
                    "results": [], "memory": Value::Null,
                }));
                continue;
            }
        };
        // Give every model the benchmark's added tokens so the `added_*` fixtures have
        // something to match, before the pipeline is derived from `tok`.
        inject_added_tokens(&mut tok);
        let shape = format!("{} · {}", model_kind(&tok), pretok_label(&path));

        // A model can satisfy the pipeline's *build* constraints yet still fail at
        // *encode* time — e.g. a Sequence containing ByteLevel, which rewrites bytes
        // and has no range-based impl. Probe once and downgrade to "unsupported"
        // (with the reason) instead of panicking partway through the bench.
        let pipeline = match PipelineTokenizer::try_from(&tok) {
            Ok(p) => match p.encode(PROBE, false) {
                Ok(_) => p,
                Err(e) => {
                    eprintln!("  pipeline builds but can't encode yet ({shape}): {e}");
                    models.push(json!({
                        "model": name, "repo": repo, "desc": desc, "shape": shape,
                        "supported": false, "reason": format!("{e}"),
                        "results": [], "memory": Value::Null,
                    }));
                    continue;
                }
            },
            Err(_) => {
                eprintln!("  unsupported by PipelineTokenizer ({shape})");
                models.push(json!({
                    "model": name, "repo": repo, "desc": desc, "shape": shape,
                    "supported": false, "results": [], "memory": Value::Null,
                }));
                continue;
            }
        };

        // The released crate may not load (or encode) a config that needs
        // features newer than the release — bench without it rather than fail.
        let baseline = match BaselineTokenizer::from_file(&path) {
            Ok(mut b) => {
                inject_added_tokens_baseline(&mut b);
                match b.encode(PROBE, false) {
                    Ok(_) => Some(b),
                    Err(e) => {
                        eprintln!("  baseline v{BASELINE_VERSION} loads but can't encode: {e}");
                        None
                    }
                }
            }
            Err(e) => {
                eprintln!("  baseline v{BASELINE_VERSION} can't load this config: {e}");
                None
            }
        };

        let rows = bench_model(baseline.as_ref(), &tok, &pipeline, &files);
        let memory = measure_memory(&path, baseline.is_some());

        models.push(json!({
            "model": name, "repo": repo, "desc": desc, "shape": shape,
            "supported": true,
            "results": rows, "memory": memory,
        }));
    }

    let out = json!({
        "baseline": { "crate": "tokenizers", "version": BASELINE_VERSION },
        "models": models,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
