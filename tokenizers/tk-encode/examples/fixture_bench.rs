//! Comparative benchmark of the experimental `PipelineTokenizer` against two
//! references — the latest *released* `tokenizers` crate (the bar to beat — the
//! in-tree legacy `Tokenizer` is on its way out, so the release is the
//! reference) and sebpop's performance branch (`sebpop/tokenizers#upstream`) —
//! for every model in `examples/bench_models.json` across every corpus in
//! `data/fixtures/`.
//!
//! Per fixture it measures single-thread throughput on ~10 kB inputs (the regime
//! where per-input overhead is amortized — see `pipeline_benchmark.rs` for the
//! size sweep). Per model it also (a) runs a **multi-thread throughput sweep** —
//! pipeline vs the release at 1/2/4/8/device-max threads over the whole corpus, so
//! parallel scaling is visible — and (b) measures resident-set deltas by re-spawning
//! itself as `--memory <impl> <model.json>` children — one implementation per
//! process, so allocator page reuse across implementations can't blur the attribution.
//!
//! The in-tree `Tokenizer` is *not* benched: it only builds the pipeline and
//! serves as the id-correctness oracle (`ids_match`, which CI fails on).
//! `ids_match_baseline` — pipeline vs the released crate — is report-only,
//! since a branch may intentionally fix encode behavior. Models the pipeline
//! can't build yet are reported with empty `results` (plus the failure
//! `reason`) and their pipeline shape rather than benched — the CI grid
//! renders those as roadmap cards. Each manifest entry carries a `desc`: a
//! one-line label of the workload archetype the model exercises, passed
//! through to the report.
//!
//! Emits one JSON object (`{baseline, sebpop, models}`) on stdout, consumed by
//! `.github/scripts/render_pipeline_bench.py` in CI.

use std::convert::TryFrom;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use logos::Logos;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rayon::ThreadPoolBuilder;
use serde_json::{json, Value};
use tk_encode::pipeline::{Model, PipelineTokenizer};
use tk_encode::{AddedToken, ModelWrapper, Tokenizer};
use tokenizers_release::{AddedToken as BaselineAddedToken, Tokenizer as BaselineTokenizer};
use tokenizers_sebpop::{AddedToken as SebpopAddedToken, Tokenizer as SebpopTokenizer};

const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");
const MANIFEST: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/bench_models.json");
// Keep in sync with the `tokenizers-release` pin in Cargo.toml.
const BASELINE_VERSION: &str = "0.23.1";
// Keep in sync with the `tokenizers-sebpop` git dependency in Cargo.toml.
const SEBPOP_REF: &str = "sebpop/upstream";
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

/// Same injection through the sebpop branch's `AddedToken` (its pre-split API
/// takes slices, not iterators).
fn inject_added_tokens_sebpop(tok: &mut SebpopTokenizer) {
    let special: Vec<SebpopAddedToken> = ADDED_SPECIAL
        .iter()
        .map(|s| SebpopAddedToken::from(*s, true).normalized(false))
        .collect();
    let normalized: Vec<SebpopAddedToken> = ADDED_NORMALIZED
        .iter()
        .map(|s| SebpopAddedToken::from(*s, false).normalized(true))
        .collect();
    let _ = tok.add_special_tokens(&special);
    let _ = tok.add_tokens(&normalized);
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

/// Thread counts for the scaling sweep: 1 (the single-thread anchor) + 2/4/8 + the device max,
/// deduped and capped at max (so an 8-core box reports `[1,2,4,8]`, a 6-core `[1,2,4,6]`).
fn thread_counts() -> Vec<usize> {
    let max = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut c: Vec<usize> = IntoIterator::into_iter([1usize, 2, 4, 8, max])
        .filter(|&n| n <= max)
        .collect();
    c.sort_unstable();
    c.dedup();
    c
}

/// Median MB/s of encoding `chunks` across `n` threads in a *private* rayon pool (so the sweep can't
/// perturb — or be perturbed by — the global pool). One `encode` call per chunk; the sum is `black_box`'d
/// so the work can't be elided.
fn par_mbps(
    encode: impl Fn(&str) -> usize + Sync,
    chunks: &[String],
    bytes: usize,
    n: usize,
) -> f64 {
    let pool = ThreadPoolBuilder::new().num_threads(n).build().unwrap();
    let run = || pool.install(|| chunks.par_iter().map(|c| encode(c.as_str())).sum::<usize>());
    black_box(run()); // warm the pool + lazy structures
    let mut samples = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        black_box(run());
        samples.push(t.elapsed().as_secs_f64());
    }
    bytes as f64 / median_secs(samples) / 1e6
}

/// Multi-thread throughput sweep for one model — pipeline vs the released crate vs sebpop's branch at
/// 1/2/4/8/max threads over the whole fixture corpus (thread-spawn/scheduling overhead amortized). All
/// impls encode the same chunk list through a fresh pool per count; interleaved so thermal drift hits
/// them equally.
fn bench_threads(
    baseline: Option<&BaselineTokenizer>,
    sebpop: Option<&SebpopTokenizer>,
    pipeline: &PipelineTokenizer,
    chunks: &[String],
) -> Value {
    let bytes: usize = chunks.iter().map(String::len).sum();
    let counts = thread_counts();
    let (mut pipe, mut base, mut seb) = (Vec::new(), Vec::new(), Vec::new());
    for &n in &counts {
        let b = baseline.map(|b| par_mbps(|s| b.encode(s, false).unwrap().len(), chunks, bytes, n));
        let sp = sebpop.map(|t| par_mbps(|s| t.encode(s, false).unwrap().len(), chunks, bytes, n));
        let p = par_mbps(
            |s| pipeline.encode(s, false).unwrap().len(),
            chunks,
            bytes,
            n,
        );
        eprintln!(
            "    {n} thread(s): pipeline {p:.1} MB/s{}{}",
            b.map_or(String::new(), |v| format!(", baseline {v:.1} MB/s")),
            sp.map_or(String::new(), |v| format!(", sebpop {v:.1} MB/s"))
        );
        pipe.push(p);
        base.push(b);
        seb.push(sp);
    }
    json!({ "counts": counts, "pipeline_mbps": pipe, "baseline_mbps": base, "sebpop_mbps": seb })
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
    let mut scratch = pipeline.get_model().init_scratch();
    let mut run = || {
        for chunk in chunks {
            out.clear();
            let _ =
                pipeline.encode_generic::<STAGE>(chunk, &mut pre_tokens, &mut scratch, &mut out);
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
        "sebpop" => {
            let mut tok = SebpopTokenizer::from_file(model).unwrap();
            inject_added_tokens_sebpop(&mut tok);
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
fn measure_memory(model: &Path, baseline_ok: bool, sebpop_ok: bool) -> Value {
    let exe = std::env::current_exe().unwrap();
    let mut out = serde_json::Map::new();
    for (key, ok) in [
        ("baseline", baseline_ok),
        ("sebpop", sebpop_ok),
        ("pipeline", true),
    ] {
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

// ── pre-tokenize comparison: classify (SIMD vs scalar) + fsm, vs an onig regex reference ──
// The pipeline's `pre_tokenize` stage is `classify (SIMD) + fsm`. These numbers let the report show it
// beating a real regex engine both WITH and WITHOUT SIMD: `classify`/`classify_scalar` over the same
// bytes isolate the classify half (scalar-pipeline = pre_tokenize + (cls_scalar − cls_simd)), and `onig`
// times the model's own pre-tokenizer regex(es) — the split a regex-based tokenizer actually pays for.

/// GPT-2's implicit pre-tokenize regex (a byte-map `ByteLevel` splits on this before mapping bytes) —
/// the canonical spec in atomsplit.
use atomsplit::regexes::GPT2 as GPT2_REGEX;

fn split_regex(p: &Value) -> Option<String> {
    (p["type"].as_str() == Some("Split"))
        .then(|| p["pattern"]["Regex"].as_str().map(str::to_string))
        .flatten()
}

/// The ordered Split regexes a model's pre-tokenizer applies (deepseek → 3; a lone `Split` → 1; a
/// byte-map `ByteLevel` with no Split → GPT-2's implicit regex). Empty → no regex reference (Bert,
/// Metaspace, WhitespaceSplit, …) → onig reported null for that model.
fn pretok_regexes(path: &Path) -> Vec<String> {
    let v: Value = std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(Value::Null);
    let pt = &v["pre_tokenizer"];
    match pt["type"].as_str() {
        Some("Split") => split_regex(pt).into_iter().collect(),
        Some("ByteLevel") => vec![GPT2_REGEX.to_string()],
        Some("Sequence") => {
            let arr = pt["pretokenizers"].as_array().cloned().unwrap_or_default();
            let res: Vec<String> = arr.iter().filter_map(split_regex).collect();
            if !res.is_empty() {
                res
            } else if arr.iter().any(|p| p["type"] == "ByteLevel") {
                vec![GPT2_REGEX.to_string()]
            } else {
                vec![]
            }
        }
        _ => vec![],
    }
}

/// Median ns/byte of a warmed-up `run` over `len` bytes (`run` returns a value that's `black_box`'d so
/// the work isn't optimized away).
fn timed_ns(len: usize, mut run: impl FnMut() -> usize) -> f64 {
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

/// Median ns/byte to classify `bytes` once via the SIMD or scalar path.
fn classify_ns(bytes: &[u8], scalar: bool) -> f64 {
    let mut tags = vec![0u8; bytes.len()];
    timed_ns(bytes.len(), || {
        if scalar {
            atomsplit::classify::classify_scalar(bytes, &mut tags);
        } else {
            atomsplit::classify::classify(bytes, &mut tags);
        }
        tags[bytes.len() / 2] as usize
    })
}

/// ns/byte for the composed Isolated split chain under onig — each regex splits the previous pieces
/// (gaps + matches), exactly how the reference tokenizer applies a `Sequence` of Splits. `None` when the
/// model has no regex pre-tokenizer, or onig rejects a pattern.
fn onig_reference_ns(text: &str, regexes: &[String]) -> Option<f64> {
    if regexes.is_empty() || text.is_empty() {
        return None;
    }
    let res: Vec<onig::Regex> = regexes
        .iter()
        .map(|r| onig::Regex::new(r))
        .collect::<Result<_, _>>()
        .ok()?;
    Some(timed_ns(text.len(), || {
        let mut pieces = vec![(0usize, text.len())];
        for re in &res {
            let mut next = Vec::with_capacity(pieces.len() * 2);
            for (s, e) in pieces.drain(..) {
                let sub = &text[s..e];
                let mut prev = 0usize;
                for (ms, me) in re.find_iter(sub) {
                    if ms > prev {
                        next.push((s + prev, s + ms));
                    }
                    next.push((s + ms, s + me));
                    prev = me;
                }
                if prev < sub.len() {
                    next.push((s + prev, e));
                }
            }
            pieces = next;
        }
        pieces.len()
    }))
}

/// Same composed Isolated split chain, but under the pure-Rust `fancy-regex` engine — a second
/// reference. `find_iter` yields `Result<Match, _>`; a match error aborts that regex's pass (rare,
/// backtrack-limit) and the piece is left un-split. `None` if no regex pre-tokenizer or a bad pattern.
fn fancy_reference_ns(text: &str, regexes: &[String]) -> Option<f64> {
    if regexes.is_empty() || text.is_empty() {
        return None;
    }
    let res: Vec<fancy_regex::Regex> = regexes
        .iter()
        .map(|r| fancy_regex::Regex::new(r))
        .collect::<Result<_, _>>()
        .ok()?;
    Some(timed_ns(text.len(), || {
        let mut pieces = vec![(0usize, text.len())];
        for re in &res {
            let mut next = Vec::with_capacity(pieces.len() * 2);
            for (s, e) in pieces.drain(..) {
                let sub = &text[s..e];
                let mut prev = 0usize;
                for m in re.find_iter(sub) {
                    let Ok(m) = m else { break };
                    let (ms, me) = (m.start(), m.end());
                    if ms > prev {
                        next.push((s + prev, s + ms));
                    }
                    next.push((s + ms, s + me));
                    prev = me;
                }
                if prev < sub.len() {
                    next.push((s + prev, e));
                }
            }
            pieces = next;
        }
        pieces.len()
    }))
}

/// Same composed chain under PCRE2 (C) — the third reference. Built with `utf(true).ucp(true)` so
/// `\p{L}`/`\p{N}`/`\s` are Unicode-aware and byte offsets land on char boundaries, matching onig and
/// fancy-regex, and **JIT-compiled** (`jit_if_available`) so PCRE2 is benched at its best. Operates on
/// `&[u8]`; `find_iter` yields `Result<Match, _>` (a match error breaks the pass). `None` if no regex
/// pre-tokenizer, or PCRE2 rejects/fails a pattern.
fn pcre2_reference_ns(text: &str, regexes: &[String]) -> Option<f64> {
    if regexes.is_empty() || text.is_empty() {
        return None;
    }
    let res: Vec<pcre2::bytes::Regex> = regexes
        .iter()
        .map(|r| {
            pcre2::bytes::RegexBuilder::new()
                .utf(true)
                .ucp(true)
                .jit_if_available(true) // PCRE2's JIT is where its speed is — bench it at its best.
                .build(r)
        })
        .collect::<Result<_, _>>()
        .ok()?;
    let bytes = text.as_bytes();
    Some(timed_ns(text.len(), || {
        let mut pieces = vec![(0usize, text.len())];
        for re in &res {
            let mut next = Vec::with_capacity(pieces.len() * 2);
            for (s, e) in pieces.drain(..) {
                let sub = &bytes[s..e];
                let mut prev = 0usize;
                for m in re.find_iter(sub) {
                    let Ok(m) = m else { break };
                    let (ms, me) = (m.start(), m.end());
                    if ms > prev {
                        next.push((s + prev, s + ms));
                    }
                    next.push((s + ms, s + me));
                    prev = me;
                }
                if prev < sub.len() {
                    next.push((s + prev, e));
                }
            }
            pieces = next;
        }
        pieces.len()
    }))
}

// logos DFA lexers approximating the GPT splits (no look-ahead / case-insensitive → boundaries differ
// slightly; a raw-throughput reference like fancy, not a byte-exact oracle). Only families logos can
// express get a number; deepseek / variants / non-regex pretoks report null.
#[derive(Logos)]
enum LGpt2 {
    #[regex(r"'s|'t|'re|'ve|'m|'ll|'d")]
    Contraction,
    #[regex(r" ?\p{L}+")]
    Word,
    #[regex(r" ?\p{N}+")]
    Num,
    #[regex(r" ?[^\s\p{L}\p{N}]+")]
    Other,
    #[regex(r"\s+")]
    Space,
}
#[derive(Logos)]
enum LCl100k {
    #[regex(r"'s|'t|'re|'ve|'m|'ll|'d", priority = 5)]
    Contraction,
    #[regex(r"[^\r\n\p{L}\p{N}]?\p{L}+", priority = 4)]
    Word,
    #[regex(r"\p{N}\p{N}?\p{N}?")]
    Num,
    #[regex(r" ?[^\s\p{L}\p{N}]+[\r\n]*", priority = 2)]
    Other,
    #[regex(r"\s+")]
    Space,
}
#[derive(Logos)]
enum LO200k {
    #[regex(r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+('s|'t|'re|'ve|'m|'ll|'d)?", priority = 6)]
    LettersA,
    #[regex(r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*('s|'t|'re|'ve|'m|'ll|'d)?", priority = 5)]
    LettersB,
    #[regex(r"\p{N}\p{N}?\p{N}?")]
    Num,
    #[regex(r" ?[^\s\p{L}\p{N}]+[\r\n/]*", priority = 2)]
    Other,
    #[regex(r"\s+")]
    Space,
}

fn lex_count<'s, T: Logos<'s, Source = str>>(s: &'s str) -> usize
where
    T::Extras: Default,
{
    let mut lex = T::lexer(s);
    let mut n = 0;
    while lex.next().is_some() {
        n += 1;
    }
    n
}

/// logos throughput (ns/byte) when the model's pre-tokenizer is a single regex logos can express (matched
/// against the canonical gpt2/cl100k/o200k specs); `None` otherwise (deepseek, cl100k-variants, non-regex).
fn logos_reference_ns(regexes: &[String], text: &str) -> Option<f64> {
    if text.is_empty() || regexes.len() != 1 {
        return None;
    }
    let r = regexes[0].as_str();
    let f: fn(&str) -> usize = if r == atomsplit::regexes::GPT2 {
        |s| lex_count::<LGpt2>(s)
    } else if r == atomsplit::regexes::CL100K {
        |s| lex_count::<LCl100k>(s)
    } else if r == atomsplit::regexes::O200K {
        |s| lex_count::<LO200k>(s)
    } else {
        return None;
    };
    Some(timed_ns(text.len(), || f(text)))
}

fn bench_model(
    baseline: Option<&BaselineTokenizer>,
    sebpop: Option<&SebpopTokenizer>,
    oracle: &Tokenizer,
    pipeline: &PipelineTokenizer,
    files: &[(String, PathBuf)],
    model_json: &Path,
) -> Vec<Value> {
    let pipe_enc = |s: &str| pipeline.encode(s, false).unwrap().len();
    // per-model: the reference regex(es) onig will time on each fixture (empty for non-regex pretoks).
    let regexes = pretok_regexes(model_json);
    let base_enc = baseline.map(|b| move |s: &str| b.encode(s, false).unwrap().len());
    let seb_enc = sebpop.map(|t| move |s: &str| t.encode(s, false).unwrap().len());

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
        // Report-only: pipeline vs sebpop's branch.
        let ids_match_sebpop = sebpop.map(|t| {
            chunks
                .iter()
                .take(3)
                .all(|c| t.encode(c.as_str(), false).unwrap().get_ids() == pipe_ids(c))
        });

        // interleave all impls so frequency/thermal drift hits them equally
        if let Some(be) = &base_enc {
            time_pass(be, &chunks);
        }
        if let Some(se) = &seb_enc {
            time_pass(se, &chunks);
        }
        time_pass(&pipe_enc, &chunks);
        let (mut base_s, mut seb_s, mut pipe_s) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..REPS {
            if let Some(be) = &base_enc {
                base_s.push(time_pass(be, &chunks));
            }
            if let Some(se) = &seb_enc {
                seb_s.push(time_pass(se, &chunks));
            }
            pipe_s.push(time_pass(&pipe_enc, &chunks));
        }
        let mbps = |secs: f64| bytes as f64 / secs / 1e6;
        let base_mbps = (!base_s.is_empty()).then(|| mbps(median_secs(base_s)));
        let seb_mbps = (!seb_s.is_empty()).then(|| mbps(median_secs(seb_s)));
        let pipe_mbps = mbps(median_secs(pipe_s));

        let fmt = |v: Option<f64>| v.map_or("—".into(), |v| format!("{v:.1}"));
        eprintln!(
            "  {name}: baseline {} MB/s, sebpop {} MB/s, pipeline {pipe_mbps:.1} MB/s",
            fmt(base_mbps),
            fmt(seb_mbps)
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

        // pre_tokenize (= classify SIMD + fsm) vs classify-scalar and vs the onig regex reference, over
        // the same corpus. Report shows the split beating a regex engine with AND without SIMD.
        let corpus: String = chunks.concat();
        let cls_simd = classify_ns(corpus.as_bytes(), false);
        let cls_scalar = classify_ns(corpus.as_bytes(), true);
        let onig_ns = onig_reference_ns(&corpus, &regexes);
        let fancy_ns = fancy_reference_ns(&corpus, &regexes);
        let pcre2_ns = pcre2_reference_ns(&corpus, &regexes);
        let logos_ns = logos_reference_ns(&regexes, &corpus);
        if [onig_ns, fancy_ns, pcre2_ns, logos_ns]
            .iter()
            .any(Option::is_some)
        {
            // fsm is the scalar jump-table in both pipes; SIMD/scalar is the classify pass only.
            let scalar_pipe = ns_split + (cls_scalar - cls_simd).max(0.0);
            let vs = |r: Option<f64>| {
                r.map_or("—".into(), |v| {
                    format!(
                        "{:.1}×/{:.1}×",
                        v / ns_split.max(1e-9),
                        v / scalar_pipe.max(1e-9)
                    )
                })
            };
            eprintln!(
                "    pre-tok: SIMD-cls {ns_split:.2} / scalar-cls {scalar_pipe:.2} ns/B · vs onig {} · vs fancy {} · vs pcre2 {} · vs logos {}",
                vs(onig_ns),
                vs(fancy_ns),
                vs(pcre2_ns),
                vs(logos_ns)
            );
        }

        rows.push(json!({
            "fixture": name,
            "group": group,
            "bytes": bytes,
            "chunks": chunks.len(),
            "mbps": { "baseline": base_mbps, "sebpop": seb_mbps, "pipeline": pipe_mbps },
            "ids_match": ids_match,
            "ids_match_baseline": ids_match_baseline,
            "ids_match_sebpop": ids_match_sebpop,
            "stage_ns_per_byte": {
                "added_split": ns_added,
                "normalize": ns_norm,
                "pre_tokenize": ns_split,
                "model": ns_model,
                "total": nspb(t_model),
            },
            // classify SIMD vs scalar + the onig / fancy-regex references for the pre-tokenize regex
            // (null when the model has no regex pre-tokenizer). simd-pipe = pre_tokenize; scalar-pipe =
            // pre_tokenize + (cls_scalar − cls_simd); the renderer derives the ×-vs-engine ratios.
            "pretok_vs_regex": {
                "cls_simd": cls_simd,
                "cls_scalar": cls_scalar,
                "onig": onig_ns,
                "fancy": fancy_ns,
                "pcre2": pcre2_ns,
                "logos": logos_ns,
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

    // Optional model sharding for CI matrix fan-out: `--shard <i> <n>` benches only the i-th of `n`
    // contiguous manifest chunks, so the models split across parallel isolated runners and the partial
    // JSONs are concatenated downstream. Without it, `(0, 1)` = the whole manifest, unchanged.
    let (shard, nshards): (usize, usize) =
        match (args.get(1).map(String::as_str), args.get(2), args.get(3)) {
            (Some("--shard"), Some(i), Some(n)) => {
                (i.parse().unwrap(), n.parse::<usize>().unwrap().max(1))
            }
            _ => (0, 1),
        };
    let full: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(MANIFEST).unwrap()).unwrap();
    let (lo, hi) = (
        shard * full.len() / nshards,
        (shard + 1) * full.len() / nshards,
    );
    let manifest = &full[lo.min(full.len())..hi.min(full.len())];
    eprintln!(
        "shard {shard}/{nshards}: models {lo}..{hi} of {}",
        full.len()
    );
    let files = fixture_files();
    // The whole corpus, concatenated once: the multi-thread sweep runs over all fixtures so
    // thread-spawn/scheduling overhead is amortized and the scaling curve is stable.
    let all_chunks: Vec<String> = files
        .iter()
        .flat_map(|(_, p)| make_chunks(&std::fs::read_to_string(p).unwrap()))
        .collect();

    let mut models: Vec<Value> = Vec::new();
    for entry in manifest {
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
                    "reason": format!("load error: {e}"),
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
                        "reason": format!("{e}"),
                        "results": [], "memory": Value::Null,
                    }));
                    continue;
                }
            },
            Err(_) => {
                eprintln!("  unsupported by PipelineTokenizer ({shape})");
                models.push(json!({
                    "model": name, "repo": repo, "desc": desc, "shape": shape,
                    "results": [], "memory": Value::Null,
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

        // Same fallback for sebpop's branch (based on an older tree, so it may
        // predate configs the release handles).
        let sebpop = match SebpopTokenizer::from_file(&path) {
            Ok(mut t) => {
                inject_added_tokens_sebpop(&mut t);
                match t.encode(PROBE, false) {
                    Ok(_) => Some(t),
                    Err(e) => {
                        eprintln!("  {SEBPOP_REF} loads but can't encode: {e}");
                        None
                    }
                }
            }
            Err(e) => {
                eprintln!("  {SEBPOP_REF} can't load this config: {e}");
                None
            }
        };

        let rows = bench_model(
            baseline.as_ref(),
            sebpop.as_ref(),
            &tok,
            &pipeline,
            &files,
            &path,
        );
        let memory = measure_memory(&path, baseline.is_some(), sebpop.is_some());
        let threads = bench_threads(baseline.as_ref(), sebpop.as_ref(), &pipeline, &all_chunks);

        models.push(json!({
            "model": name, "repo": repo, "desc": desc, "shape": shape,
            "results": rows, "memory": memory, "threads": threads,
        }));
    }

    let out = json!({
        "baseline": { "crate": "tokenizers", "version": BASELINE_VERSION },
        "sebpop": { "crate": "tokenizers", "ref": SEBPOP_REF },
        "models": models,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
