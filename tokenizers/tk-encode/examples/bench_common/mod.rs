//! Inputs and timing shared by the two benchmark binaries.
//!
//! Both walk the same work list — every model in `examples/bench_models.json` against every corpus
//! in `data/fixtures/` — but measure different things, so they are separate binaries built with
//! different features: `fixture_bench` compares the pipeline against the released crate, while
//! `pretok_engines` compares our split against real regex engines. Keeping the corpus loading, the
//! model list and the timing here means a number from one is directly comparable to a number from
//! the other.
//!
//! Both binaries are built by CI and take `--shard <i> <n>`, which selects the i-th of `n` slices of
//! the model list so the work fans out over parallel runners.

// Each binary uses a subset of this module; the rest is not dead code, just unused over there.
#![allow(dead_code)]

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::Value;

pub const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");
pub const MANIFEST: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/bench_models.json");
/// Input size per encode call. ~10 kB is large enough that per-call overhead is amortized, so the
/// number reflects steady-state throughput (`benches/pipeline_benchmark.rs` sweeps the sizes).
pub const CHUNK_BYTES: usize = 10 * 1024;
pub const MAX_CHUNKS: usize = 100;
/// Timed passes per measurement; the median is reported.
pub const REPS: usize = 5;

/// One corpus, already cut into encode-sized chunks.
pub struct Fixture {
    pub group: &'static str,
    pub name: String,
    pub chunks: Vec<String>,
    pub bytes: usize,
}

/// Cuts `text` on line boundaries into chunks of at least [`CHUNK_BYTES`], so a chunk never splits a
/// line and every model sees the same inputs.
pub fn make_chunks(text: &str) -> Vec<String> {
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

/// The sorted `.txt` fixtures under `data/fixtures/{lang,modalities}`, tagged with group.
pub fn fixture_paths() -> Vec<(&'static str, PathBuf)> {
    let mut out = Vec::new();
    for group in ["lang", "modalities"] {
        let dir = Path::new(DATA_DIR).join("fixtures").join(group);
        let mut paths: Vec<_> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("{}: {e} — run `make fixtures` first", dir.display()))
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|x| x == "txt"))
            .collect();
        paths.sort();
        out.extend(paths.into_iter().map(|p| (group, p)));
    }
    out
}

/// Every corpus in `data/fixtures/{lang,modalities}`, read and chunked once.
pub fn load_fixtures() -> Vec<Fixture> {
    fixture_paths()
        .into_iter()
        .map(|(group, path)| {
            let name = path.file_stem().unwrap().to_str().unwrap().to_string();
            let chunks = make_chunks(&std::fs::read_to_string(&path).unwrap());
            let bytes = chunks.iter().map(String::len).sum();
            Fixture {
                group,
                name,
                chunks,
                bytes,
            }
        })
        .collect()
}

/// The models this process is responsible for, read from [`MANIFEST`].
///
/// `args` is the raw command line. Without `--shard i n` the slice is the whole manifest.
pub fn shard(args: &[String]) -> Vec<Value> {
    let (i, n): (usize, usize) = match (args.get(1).map(String::as_str), args.get(2), args.get(3)) {
        (Some("--shard"), Some(i), Some(n)) => {
            (i.parse().unwrap(), n.parse::<usize>().unwrap().max(1))
        }
        _ => (0, 1),
    };
    let full: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(MANIFEST).unwrap()).unwrap();
    let (lo, hi) = (i * full.len() / n, (i + 1) * full.len() / n);
    eprintln!("shard {i}/{n}: models {lo}..{hi} of {}", full.len());
    full[lo.min(full.len())..hi.min(full.len())].to_vec()
}

// ── timing ──────────────────────────────────────────────────────────────────

pub fn median_secs(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

/// Median ns/byte of a warmed-up `run` over `len` bytes. `run` returns a value that is `black_box`'d
/// so the work cannot be optimized away; the `black_box` belongs here, in the benchmark, never in the
/// library.
pub fn timed_ns(len: usize, mut run: impl FnMut() -> usize) -> f64 {
    if len == 0 {
        return 0.0;
    }
    run(); // warm-up
    let mut s = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        black_box(run());
        s.push(t.elapsed().as_secs_f64());
    }
    median_secs(s) * 1e9 / len as f64
}

// ── the model list ──────────────────────────────────────────────────────────

/// Path to a manifest entry's `tokenizer.json`, defaulting to `<name>.json`.
pub fn model_path(entry: &Value) -> PathBuf {
    let name = entry["name"].as_str().unwrap();
    let file = entry
        .get("file")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| format!("{name}.json"));
    Path::new(DATA_DIR).join(file)
}

fn split_regex(p: &Value) -> Option<String> {
    (p["type"].as_str() == Some("Split"))
        .then(|| p["pattern"]["Regex"].as_str().map(str::to_string))
        .flatten()
}

/// The ordered Split regexes a model's pre-tokenizer applies (deepseek → 3; a lone `Split` → 1; a
/// byte-map `ByteLevel` with no Split → GPT-2's implicit regex, the canonical spec in atomsplit).
/// Empty → the model has no regex to compare a regex engine against (Bert, Metaspace,
/// WhitespaceSplit, a literal-string `Split`, …).
pub fn pretok_regexes(path: &Path) -> Vec<String> {
    let v: Value = std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(Value::Null);
    let pt = &v["pre_tokenizer"];
    match pt["type"].as_str() {
        Some("Split") => split_regex(pt).into_iter().collect(),
        Some("ByteLevel") => vec![atomsplit::regexes::GPT2.to_string()],
        Some("Sequence") => {
            let arr = pt["pretokenizers"].as_array().cloned().unwrap_or_default();
            let res: Vec<String> = arr.iter().filter_map(split_regex).collect();
            if !res.is_empty() {
                res
            } else if arr.iter().any(|p| p["type"] == "ByteLevel") {
                vec![atomsplit::regexes::GPT2.to_string()]
            } else {
                vec![]
            }
        }
        _ => vec![],
    }
}
