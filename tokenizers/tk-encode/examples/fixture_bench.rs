//! Comparative benchmark of the experimental `PipelineTokenizer` against the
//! latest *released* `tokenizers` crate (the bar to beat — the in-tree legacy
//! `Tokenizer` is on its way out, so the release is the reference), for every
//! model in `examples/bench_models.json` across every corpus in `data/fixtures/`.
//! The baseline is always driven through `encode_fast` — its offset-free path —
//! because the pipeline's `encode` computes no offsets either; timing the
//! baseline's offset-tracking `encode` would flatter the pipeline.
//!
//! Four isolated phases per model:
//!
//! 1. **Throughput** — single-thread warm MB/s per fixture on ~10 kB inputs
//!    (the regime where per-input overhead is amortized — see
//!    `pipeline_benchmark.rs` for the size sweep). Every fixture starts from a
//!    cold cache on both sides: the pipeline is rebuilt (fresh scratch pool →
//!    fresh BPE word cache) and the baseline is cloned (the released BPE's
//!    `Clone` starts with an empty cache). The warm-up pass then fills each
//!    cache from that corpus alone — the state a plain `.encode()` loop over
//!    the corpus reaches — so per-fixture numbers don't depend on which
//!    fixtures ran before them. Both sides encode with `add_special_tokens`
//!    on, so the headline includes the post-process stage the ladder charges.
//! 2. **Stage breakdown** — the `encode_generic::<STAGE>` ablation ladder plus
//!    the pre-tokenize-vs-regex-engine references, on fresh caller-owned
//!    scratches, fully separate from the phase-1 timings.
//! 3. **Scaling & memory** — a multi-thread throughput sweep (1/2/4/8/max) over
//!    the whole corpus on fresh instances, and resident-set deltas measured by
//!    re-spawning this binary as `--memory <impl> <model.json>` children — one
//!    implementation per process, so allocator page reuse can't blur the
//!    attribution.
//! 4. **Decode** — the inverse direction, anchored on the RELEASED crate (never
//!    the in-tree legacy `Tokenizer`, which is being removed — oracles must not
//!    depend on it). The release's `encode_fast` produces the id stream; pipeline
//!    and released `decode` then consume the SAME ids (single-thread throughput +
//!    the 1/2/4/8/max sweep + a decode-pass RSS delta), gated by `text_match`
//!    (pipeline decode == released decode). While `PipelineTokenizer::decode` is a
//!    loud stub the pipeline decode series is `null` (rendered "pending"); the
//!    released baseline is measured regardless. No released baseline → no decode
//!    oracle → the phase is skipped for that model.
//!
//! Correctness is judged against the released crate, never the in-tree
//! `Tokenizer` (which is being removed and only *builds* the pipeline here):
//! `ids_match` (encode ids) and `text_match` (decode text) both compare against
//! the release and both fail CI. They are `null` when the release can't load the
//! model — no reference, so no gate. Models the pipeline can't build (or encode)
//! yet are reported with empty `results` (plus the failure `reason`) and their
//! pipeline shape rather than benched — the CI grid renders those as roadmap
//! cards. Each manifest entry carries a `desc`: a one-line label of the workload
//! archetype the model exercises, passed through to the report.
//!
//! Emits one JSON object (`{baseline, models}`) on stdout, consumed by
//! `.github/scripts/render_pipeline_bench.py` in CI.

use std::convert::TryFrom;
use std::hint::black_box;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

mod bench_common;

use rayon::ThreadPoolBuilder;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde_json::{Value, json};
use tk_encode::pipeline::{Model, PipelineTokenizer};
use tk_encode::{AddedToken, ModelWrapper, Tokenizer};
use tokenizers_release::{AddedToken as BaselineAddedToken, Tokenizer as BaselineTokenizer};

use bench_common::{
    CHUNK_BYTES, Fixture, REPS, fixture_paths, load_fixtures, make_chunks, median_secs, model_path,
    shard,
};

// Keep in sync with the `tokenizers-release` pin in Cargo.toml.
const BASELINE_VERSION: &str = "0.23.1";
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

// ── fixtures ────────────────────────────────────────────────────────────────

// ── timing helpers ──────────────────────────────────────────────────────────

fn time_pass(encode: &dyn Fn(&str) -> usize, chunks: &[String]) -> f64 {
    let start = Instant::now();
    let mut n = 0usize;
    for chunk in chunks {
        n += encode(chunk);
    }
    black_box(n);
    start.elapsed().as_secs_f64()
}

// ── phase 1: throughput ─────────────────────────────────────────────────────

/// Warm single-thread throughput + the id gates for one fixture.
///
/// Both implementations start this fixture cold: a rebuilt pipeline (fresh
/// scratch pool → fresh BPE word cache) and a cloned baseline (the released
/// BPE's `Clone` starts with an empty cache). The id checks and the warm-up
/// pass then fill the caches from this corpus alone — what a plain `.encode()`
/// loop over it reaches — and the REPS passes measure that warm steady state.
/// The two impls are interleaved so frequency/thermal drift hits them equally.
fn bench_throughput(
    baseline: Option<&BaselineTokenizer>,
    oracle: &Tokenizer,
    f: &Fixture,
) -> Value {
    let pipeline = PipelineTokenizer::try_from(oracle).expect("probed at model load");
    let base = baseline.cloned();

    let pipe_enc = |s: &str| pipeline.encode(s, true).unwrap().len();
    let base_enc = base
        .as_ref()
        .map(|b| move |s: &str| b.encode_fast(s, true).unwrap().len());

    let pipe_ids = |c: &String, add_special_tokens: bool| -> Vec<u32> {
        pipeline
            .encode(c, add_special_tokens)
            .unwrap()
            .iter()
            .map(|t| t.id)
            .collect()
    };
    // The correctness gate CI fails on: pipeline ids == the released crate's ids,
    // for both `add_special_tokens` values (`true` exercises the post-process
    // stage). The in-tree `Tokenizer` only *builds* the pipeline here — never the
    // reference, since it is on its way out. `None` when the release can't load
    // this model (no reference to compare against).
    let ids_match = base.as_ref().map(|b| {
        [false, true].into_iter().all(|add_special_tokens| {
            f.chunks.iter().take(3).all(|c| {
                b.encode_fast(c.as_str(), add_special_tokens)
                    .unwrap()
                    .get_ids()
                    == pipe_ids(c, add_special_tokens)
            })
        })
    });

    if let Some(be) = &base_enc {
        time_pass(be, &f.chunks); // warm-up
    }
    time_pass(&pipe_enc, &f.chunks); // warm-up
    let (mut base_s, mut pipe_s) = (Vec::new(), Vec::new());
    for _ in 0..REPS {
        if let Some(be) = &base_enc {
            base_s.push(time_pass(be, &f.chunks));
        }
        pipe_s.push(time_pass(&pipe_enc, &f.chunks));
    }
    let mbps = |secs: f64| f.bytes as f64 / secs / 1e6;
    let base_mbps = (!base_s.is_empty()).then(|| mbps(median_secs(base_s)));
    let pipe_mbps = mbps(median_secs(pipe_s));

    eprintln!(
        "  {}: baseline {} MB/s, pipeline {pipe_mbps:.1} MB/s",
        f.name,
        base_mbps.map_or("—".into(), |v| format!("{v:.1}"))
    );

    json!({
        "fixture": f.name,
        "group": f.group,
        "mbps": { "baseline": base_mbps, "pipeline": pipe_mbps },
        "ids_match": ids_match,
    })
}

// ── phase 2: stage breakdown + pre-tokenize references ─────────────────────

/// Median wall-time (seconds) of one pass over `chunks` through the shared encode core
/// `PipelineTokenizer::encode_generic::<STAGE>`. `STAGE` is a const generic, so each
/// level is a branchless specialization with the later stages compiled out — timing
/// successive levels and subtracting gives each stage's marginal cost (the ablation
/// ladder), no profiler and no per-segment instrumentation.
///
/// The scratch is created fresh here (never taken from the pipeline's pool), so the
/// stage numbers are warmed on this fixture alone and can't perturb — or be flattered
/// by — the phase-1 cache state. Both caller-owned buffers are reused across chunks
/// and `black_box`'d each iteration: `output` anchors the special-scan/normalize/model
/// work, `pre_tokens` anchors the split stage, so under fat LTO no dead partial stage
/// gets optimized away. The `black_box` lives here, in the bench — never in the library.
fn stage_secs<const STAGE: u8>(pipeline: &PipelineTokenizer, chunks: &[String]) -> f64 {
    let mut out = Vec::new();
    let mut pre_tokens = Vec::new();
    let mut scratch = pipeline.get_model().init_scratch();
    let mut run = || {
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

/// The `stage_ns_per_byte` object of one fixture's report row. How our split compares to a regex
/// engine is measured by the `pretok_engines` binary instead, which is built with those engines.
fn bench_stages(pipeline: &PipelineTokenizer, f: &Fixture) -> Value {
    let t_frame = stage_secs::<{ PipelineTokenizer::STAGE_FRAME }>(pipeline, &f.chunks);
    let t_norm = stage_secs::<{ PipelineTokenizer::STAGE_NORMALIZE }>(pipeline, &f.chunks);
    let t_split = stage_secs::<{ PipelineTokenizer::STAGE_SPLIT }>(pipeline, &f.chunks);
    let t_model = stage_secs::<{ PipelineTokenizer::STAGE_MODEL }>(pipeline, &f.chunks);
    let t_post = stage_secs::<{ PipelineTokenizer::STAGE_POSTPROCESS }>(pipeline, &f.chunks);
    // Two distinct "split" costs: `added_split` is the added/special-token scan (the
    // SpecialSegmentIterator over the AddedVocabulary, captured by the FRAME level),
    // `pre_tokenize` is the pre-tokenizer split, `post` the special-token id-frame
    // splice. All five stages sum exactly to `total`.
    let nspb = |secs: f64| secs * 1e9 / f.bytes as f64;
    let (ns_added, ns_norm, ns_split, ns_model, ns_post) = (
        nspb(t_frame.max(0.0)),
        nspb((t_norm - t_frame).max(0.0)),
        nspb((t_split - t_norm).max(0.0)),
        nspb((t_model - t_split).max(0.0)),
        nspb((t_post - t_model).max(0.0)),
    );
    eprintln!(
        "  {} stages ns/byte: added-split {ns_added:.2}, norm {ns_norm:.2}, pre-split {ns_split:.2}, model {ns_model:.2}, post {ns_post:.2}",
        f.name
    );

    json!({
        "added_split": ns_added,
        "normalize": ns_norm,
        "pre_tokenize": ns_split,
        "model": ns_model,
        "post": ns_post,
        "total": nspb(t_post),
    })
}

// ── phase 3: multi-thread scaling + memory ──────────────────────────────────

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

/// Multi-thread throughput sweep for one model — pipeline vs the released crate at 1/2/4/8/max threads
/// over the whole fixture corpus (thread-spawn/scheduling overhead amortized). Both impls encode the same
/// chunk list through a fresh pool per count; interleaved so thermal drift hits them equally. Per-thread
/// caches here fill from the whole mixed stream — the normal `.encode()` regime for a parallel workload.
fn bench_threads(
    baseline: Option<&BaselineTokenizer>,
    pipeline: &PipelineTokenizer,
    chunks: &[String],
) -> Value {
    let bytes: usize = chunks.iter().map(String::len).sum();
    let counts = thread_counts();
    let (mut pipe, mut base) = (Vec::new(), Vec::new());
    for &n in &counts {
        let b =
            baseline.map(|b| par_mbps(|s| b.encode_fast(s, true).unwrap().len(), chunks, bytes, n));
        let p = par_mbps(
            |s| pipeline.encode(s, true).unwrap().len(),
            chunks,
            bytes,
            n,
        );
        eprintln!(
            "    {n} thread(s): pipeline {p:.1} MB/s{}",
            b.map_or(String::new(), |v| format!(", baseline {v:.1} MB/s"))
        );
        pipe.push(p);
        base.push(b);
    }
    json!({ "counts": counts, "pipeline_mbps": pipe, "baseline_mbps": base })
}

// ── decode: throughput + scaling ─────────────────────────────────────────────
// Mirror of the encode phases, over the inverse direction, anchored on the
// RELEASED crate — never the in-tree legacy `Tokenizer` (which is being removed;
// oracles must not depend on it). The id stream is produced by the release's
// `encode_fast`, and both decoders — pipeline and released baseline — consume
// those SAME ids, so decode is a clean apples-to-apples judged against the
// release. No released baseline → no decode oracle, so the whole phase is null.
// `pipeline_ok` is the once-probed "can the pipeline decode yet" flag: while
// `PipelineTokenizer::decode` is a loud stub it is false, so the pipeline series
// is `null` (rendered "pending") and only the baseline bar is drawn.

/// Encode every chunk with the released crate into its id stream (untimed input;
/// specials included, so decode sees the frame tokens a real stream carries).
fn ids_of(baseline: &BaselineTokenizer, chunks: &[String]) -> Vec<Vec<u32>> {
    chunks
        .iter()
        .map(|c| {
            baseline
                .encode_fast(c.as_str(), true)
                .unwrap()
                .get_ids()
                .to_vec()
        })
        .collect()
}

/// Warm single-thread decode throughput + the `text_match` gate for one fixture.
/// `text_match` = pipeline decode == released decode (the gate CI fails on). All
/// null when there is no released baseline to judge against.
fn bench_decode(
    baseline: Option<&BaselineTokenizer>,
    pipeline: &PipelineTokenizer,
    f: &Fixture,
    pipeline_ok: bool,
) -> Value {
    let null = json!({
        "decode_mbps": { "baseline": Value::Null, "pipeline": Value::Null },
        "text_match": Value::Null,
    });
    let Some(baseline) = baseline else {
        return null;
    };
    let ids = ids_of(baseline, &f.chunks);
    // Throughput basis: bytes of text decode emits, measured once via the release.
    let dec_bytes: usize = ids
        .iter()
        .map(|i| baseline.decode(i, false).unwrap().len())
        .sum();
    if dec_bytes == 0 {
        return null;
    }

    let one_pass = |dec: &dyn Fn(&[u32]) -> usize| -> f64 {
        let start = Instant::now();
        let mut n = 0usize;
        for i in &ids {
            n += dec(i);
        }
        black_box(n);
        start.elapsed().as_secs_f64()
    };
    let mbps = |secs: f64| dec_bytes as f64 / secs / 1e6;

    // Correctness gate (first 3 chunks): pipeline decode == released decode.
    let text_match = pipeline_ok.then(|| {
        ids.iter()
            .take(3)
            .all(|i| pipeline.decode(i, false).unwrap() == baseline.decode(i, false).unwrap())
    });

    // Interleaved warm-up + REPS so thermal drift hits both equally.
    one_pass(&|i| baseline.decode(i, false).unwrap().len());
    if pipeline_ok {
        one_pass(&|i| pipeline.decode(i, false).unwrap().len());
    }
    let (mut base_s, mut pipe_s) = (Vec::new(), Vec::new());
    for _ in 0..REPS {
        base_s.push(one_pass(&|i| baseline.decode(i, false).unwrap().len()));
        if pipeline_ok {
            pipe_s.push(one_pass(&|i| pipeline.decode(i, false).unwrap().len()));
        }
    }
    let base_mbps = mbps(median_secs(base_s));
    let pipe_mbps = (!pipe_s.is_empty()).then(|| mbps(median_secs(pipe_s)));

    eprintln!(
        "  {} decode: baseline {base_mbps:.1} MB/s, pipeline {}",
        f.name,
        pipe_mbps.map_or("pending".into(), |v: f64| format!("{v:.1} MB/s")),
    );

    json!({
        "decode_mbps": { "baseline": base_mbps, "pipeline": pipe_mbps },
        "text_match": text_match,
    })
}

/// Median MB/s of decoding `ids` across `n` threads in a private rayon pool.
fn par_decode_mbps(
    decode: impl Fn(&[u32]) -> usize + Sync,
    ids: &[Vec<u32>],
    bytes: usize,
    n: usize,
) -> f64 {
    let pool = ThreadPoolBuilder::new().num_threads(n).build().unwrap();
    let run = || pool.install(|| ids.par_iter().map(|i| decode(i.as_slice())).sum::<usize>());
    black_box(run());
    let mut samples = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let t = Instant::now();
        black_box(run());
        samples.push(t.elapsed().as_secs_f64());
    }
    bytes as f64 / median_secs(samples) / 1e6
}

/// Multi-thread decode throughput sweep — pipeline vs the released crate at
/// 1/2/4/8/max threads over the whole fixture corpus's id stream (release-produced).
fn bench_decode_threads(
    baseline: Option<&BaselineTokenizer>,
    pipeline: &PipelineTokenizer,
    all_chunks: &[String],
    pipeline_ok: bool,
) -> Value {
    let Some(baseline) = baseline else {
        return json!({ "counts": [], "pipeline_mbps": [], "baseline_mbps": [] });
    };
    let ids = ids_of(baseline, all_chunks);
    let bytes: usize = ids
        .iter()
        .map(|i| baseline.decode(i, false).unwrap().len())
        .sum();
    let counts = thread_counts();
    let (mut pipe, mut base) = (Vec::new(), Vec::new());
    for &n in &counts {
        let b = par_decode_mbps(|i| baseline.decode(i, false).unwrap().len(), &ids, bytes, n);
        let p = pipeline_ok
            .then(|| par_decode_mbps(|i| pipeline.decode(i, false).unwrap().len(), &ids, bytes, n));
        eprintln!(
            "    decode {n} thread(s): pipeline {}, baseline {b:.1} MB/s",
            p.map_or("pending".into(), |v: f64| format!("{v:.1} MB/s")),
        );
        pipe.push(p);
        base.push(b);
    }
    json!({ "counts": counts, "pipeline_mbps": pipe, "baseline_mbps": base })
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

/// A small text sample for the memory child: the first `MEM_CHUNKS_PER_FIXTURE`
/// chunks of each fixture, read as a bounded prefix instead of slurping the whole
/// (~90 MB) corpus. Loading it all would spike the transient allocation well above
/// the tokenizer itself, polluting `rss0` and `VmHWM` — corrupting the very numbers
/// this child measures. The prefix contains those first chunks, so the encode pass
/// is byte-for-byte what a full load would have produced.
fn memory_sample() -> Vec<String> {
    use std::io::Read;
    let cap = (CHUNK_BYTES * (MEM_CHUNKS_PER_FIXTURE + 1)) as u64;
    let mut chunks = Vec::new();
    for (_, path) in fixture_paths() {
        let mut buf = Vec::new();
        std::fs::File::open(&path)
            .unwrap()
            .take(cap)
            .read_to_end(&mut buf)
            .unwrap();
        let valid = std::str::from_utf8(&buf).map_or_else(|e| e.valid_up_to(), |_| buf.len());
        let text = std::str::from_utf8(&buf[..valid]).unwrap();
        chunks.extend(make_chunks(text).into_iter().take(MEM_CHUNKS_PER_FIXTURE));
    }
    chunks
}

/// `--memory <impl> <model.json>` child entry: load one implementation, encode a
/// capped pass over the fixtures, then decode that pass's ids, printing
/// `{load_bytes, encode_bytes, decode_bytes, peak_bytes}`. One implementation per
/// process so the deltas attribute cleanly. `decode_bytes` is `null` when the
/// implementation can't decode yet (the pipeline's loud stub).
fn memory_child(which: &str, model: &Path) {
    let chunks = memory_sample();

    let rss0 = rss_now().unwrap_or(0);
    let mut n = 0usize;
    let mut ids: Vec<Vec<u32>> = Vec::new();
    let (after_load, after_encode, decode_bytes) = match which {
        "baseline" => {
            let mut tok = BaselineTokenizer::from_file(model).unwrap();
            inject_added_tokens_baseline(&mut tok);
            let after_load = rss_now().unwrap_or(0);
            for c in &chunks {
                let enc = tok.encode_fast(c.as_str(), true).unwrap();
                n += enc.len();
                ids.push(enc.get_ids().to_vec());
            }
            let after_encode = rss_now().unwrap_or(0);
            for i in &ids {
                n += tok.decode(i, false).map(|s| s.len()).unwrap_or(0);
            }
            let decode_bytes = rss_now().unwrap_or(0) - after_encode;
            (after_load, after_encode, Some(decode_bytes))
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
                let enc = pipeline.encode(c, true).unwrap();
                n += enc.len();
                ids.push(enc.iter().map(|t| t.id).collect());
            }
            let after_encode = rss_now().unwrap_or(0);
            // Decode is a loud stub today → skip the pass and report null, so the
            // decode memory bar reads "pending" rather than a bogus 0.
            let decode_bytes = if ids
                .first()
                .is_some_and(|i| pipeline.decode(i, false).is_ok())
            {
                for i in &ids {
                    n += pipeline.decode(i, false).map(|s| s.len()).unwrap_or(0);
                }
                Some(rss_now().unwrap_or(0) - after_encode)
            } else {
                None
            };
            (after_load, after_encode, decode_bytes)
        }
        other => panic!("unknown impl {:?}", other),
    };
    black_box(n);

    println!(
        "{}",
        json!({
            "load_bytes": after_load - rss0,
            "encode_bytes": after_encode - after_load,
            "decode_bytes": decode_bytes,
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

// ── model manifest helpers ──────────────────────────────────────────────────

/// Local path to a manifest entry's config: `data/<file>`, else `data/<name>.json`.
/// Both come from the test-data dataset (see the Makefile `bench-models` target).
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--memory") {
        memory_child(&args[2], Path::new(&args[3]));
        return;
    }

    let manifest = shard(&args);
    let fixtures = load_fixtures();
    // The whole corpus, flattened once: the multi-thread sweep runs over all fixtures so
    // thread-spawn/scheduling overhead is amortized and the scaling curve is stable.
    let all_chunks: Vec<String> = fixtures.iter().flat_map(|f| f.chunks.clone()).collect();

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
                    "model": name, "desc": desc, "shape": "?",
                    "reason": format!("load error: {e}"),
                    "results": [], "memory": Value::Null,
                }));
                continue;
            }
        };
        // Give every model the benchmark's added tokens so the `added_*` fixtures have
        // something to match, before any pipeline is derived from `tok`.
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
                        "model": name, "desc": desc, "shape": shape,
                        "reason": format!("{e}"),
                        "results": [], "memory": Value::Null,
                    }));
                    continue;
                }
            },
            Err(_) => {
                eprintln!("  unsupported by PipelineTokenizer ({shape})");
                models.push(json!({
                    "model": name, "desc": desc, "shape": shape,
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
                match b.encode_fast(PROBE, false) {
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

        let mut rows: Vec<Value> = fixtures
            .iter()
            .map(|f| bench_throughput(baseline.as_ref(), &tok, f))
            .collect();
        for (row, f) in rows.iter_mut().zip(&fixtures) {
            let stages = bench_stages(&pipeline, f);
            row.as_object_mut()
                .unwrap()
                .insert("stage_ns_per_byte".into(), stages);
        }
        // Decode: probe once whether the pipeline can decode yet (a loud stub
        // today → the pipeline decode series is `null`, rendered "pending"). The
        // probe uses a bare id slice so it never touches the legacy encode path.
        let (decode_ok, decode_reason) = match pipeline.decode(&[0], false) {
            Ok(_) => (true, None),
            Err(e) => {
                eprintln!("  pipeline decode pending ({shape}): {e}");
                (false, Some(format!("{e}")))
            }
        };
        for (row, f) in rows.iter_mut().zip(&fixtures) {
            let dec = bench_decode(baseline.as_ref(), &pipeline, f, decode_ok);
            let row = row.as_object_mut().unwrap();
            for (k, v) in dec.as_object().unwrap() {
                row.insert(k.clone(), v.clone());
            }
        }
        let decode_threads =
            bench_decode_threads(baseline.as_ref(), &pipeline, &all_chunks, decode_ok);

        let memory = measure_memory(&path, baseline.is_some());
        let threads = bench_threads(baseline.as_ref(), &pipeline, &all_chunks);

        models.push(json!({
            "model": name, "desc": desc, "shape": shape,
            "results": rows, "memory": memory, "threads": threads,
            "decode_threads": decode_threads, "decode_reason": decode_reason,
        }));
    }

    let out = json!({
        "baseline": { "crate": "tokenizers", "version": BASELINE_VERSION },
        "models": models,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
