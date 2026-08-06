//! Comparative benchmark of the experimental `PipelineTokenizer` against the
//! latest *released* `tokenizers` crate (the bar to beat; the in-tree legacy
//! `Tokenizer` is on its way out, so the release is the reference), for every
//! model in `examples/bench_models.json` across every corpus in `data/fixtures/`.
//! The baseline is always driven through `encode_fast`, its offset-free path,
//! because the pipeline's `encode` computes no offsets either; timing the
//! baseline's offset-tracking `encode` would flatter the pipeline.
//!
//! # Measurement regime: whole corpus, cold caches
//!
//! Every fixture is benched **in full**: the file is cut into ~10 kB chunks at
//! line boundaries (concatenating the chunks reproduces the file byte for byte)
//! and one pass encodes every chunk once. ~10 kB is the regime where per-call
//! overhead is amortized; see `pipeline_benchmark.rs` for the size sweep.
//!
//! Every timed pass starts **cold**: the pipeline is rebuilt and the baseline
//! re-cloned before the pass, and thrown away after it. The released BPE and
//! Unigram keep a per-instance word cache and their `Clone` resets it (checked
//! against the released sources), so no pass re-encodes text its cache has
//! already seen; the pipeline's BPE and Unigram engines keep theirs in a
//! scratch pool on the pipeline instance, and the rebuild resets that pool the
//! same way. Within a pass, caches fill from the corpus stream, which is
//! exactly what a plain `.encode()` loop over that much fresh text reaches.
//! Re-encoding the same corpus against an already-warm cache (what this bench
//! used to measure) is not a workload either implementation serves, and it
//! flattered whichever side caches more.
//!
//! # Four isolated phases per model
//!
//! 1. **Throughput**: single-thread MB/s per fixture, the median of `REPS` cold
//!    passes, both implementations alternating so frequency drift hits them
//!    equally. Both sides encode with `add_special_tokens` on, so the headline
//!    includes the post-process cost.
//! 2. **Input-size response**: the same comparison at chunk sizes from ~256 B
//!    (chat messages, dominated by per-call overhead) to ~256 kB (whole
//!    documents), over one fixed sample mixing every fixture, so the report
//!    shows how much of the ~10 kB headline speedup survives at either end.
//! 3. **Scaling & memory**: a multi-thread throughput sweep (1/2/4/8/max) per
//!    fixture group (`lang`, `modalities`), cold instance per pass, and
//!    resident-set deltas measured by re-spawning this binary as
//!    `--memory <impl> <model.json>` children; one implementation per process,
//!    so allocator page reuse can't blur the attribution. The children also
//!    report exact allocation counts, read from the counting global allocator
//!    ([`CountingAlloc`]) around the same load and per-fixture encode brackets.
//! 4. **Instructions**: instructions retired (`Ir`) per fixture and
//!    implementation, from `--instructions <impl> <model.json>` children run
//!    under callgrind, one cold pass over the whole corpus. Skipped without
//!    valgrind (local macOS runs); see the next section for why it exists.
//!
//! # Why the deterministic lanes (instructions, allocations)
//!
//! Wall time on shared CI runners carries a noise floor of several percent, so
//! the timing phases cannot honestly resolve a 1-3% regression. Instruction and
//! allocation counts can: both are properties of the executed code path, not of
//! the machine they ran on. The same commit reproduces the same numbers, and
//! the report can flag changes far below the timing noise: exactly for
//! allocation counts, and within ~1% for instructions, which shift slightly
//! with the C library or CPU dispatch (the `env` stamp in the output is there
//! so the report can tell an environment change from a code change). Neither
//! lane sees parallelism (callgrind serializes threads), and fewer instructions
//! is not always faster (a vectorized rewrite can retire fewer and lose on
//! memory stalls), so the timing phases stay the ground truth for *time*; these
//! lanes are the tripwire for *work*.
//!
//! Decode is not benched: `PipelineTokenizer::decode` is a loud stub, so there
//! is nothing to compare yet. When it lands, decode gets its own phase judged
//! the same way (release-produced ids, both decoders consuming the same stream).
//!
//! Correctness is judged against the released crate, never the in-tree
//! `Tokenizer` (which is being removed and only *builds* the pipeline here):
//! `ids_match` compares the pipeline's encode ids against the release and fails
//! CI. It is `null` when the release can't load the
//! model (no reference, so no gate). Models the pipeline can't build (or encode)
//! yet are reported with empty `results` (plus the failure `reason`) and their
//! pipeline shape rather than benched; the CI grid renders those as roadmap
//! cards. Each manifest entry carries a `desc`: a one-line label of the workload
//! archetype the model exercises, passed through to the report.
//!
//! Emits one JSON object (`{baseline, env, models}`) on stdout, consumed by
//! `.github/scripts/render_pipeline_bench.py` in CI.

use std::alloc::{GlobalAlloc, Layout, System};
use std::convert::TryFrom;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering::Relaxed};
use std::time::Instant;

use rayon::ThreadPoolBuilder;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde_json::{Value, json};
use tk_encode::pipeline::PipelineTokenizer;
use tk_encode::{AddedToken, ModelWrapper, Tokenizer};
use tokenizers_release::{AddedToken as BaselineAddedToken, Tokenizer as BaselineTokenizer};

const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");
const MANIFEST: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/bench_models.json");
// Keep in sync with the `tokenizers-release` pin in Cargo.toml.
const BASELINE_VERSION: &str = "0.23.1";
const CHUNK_BYTES: usize = 10 * 1024;
// Timed passes per number. Every pass runs on a cold instance, so the median is
// over independent cold runs, not over re-encodes of an already-seen corpus.
const REPS: usize = 5;
const PROBE: &str = "The quick brown fox jumps 123.";
// The fixture groups under `data/fixtures/`, also the granularity of the
// multi-thread sweeps and of the report's thread-scaling charts.
const GROUPS: [&str; 2] = ["lang", "modalities"];
// Chunk sizes for the input-size sweep: ~256 B chat messages up to ~256 kB
// documents. The ~10 kB headline regime sits inside the range, so the curve
// shows how much of the headline speedup survives at either end.
const SIZE_SWEEP: &[usize] = &[256, 1024, 4096, 16 * 1024, 64 * 1024, 256 * 1024];
// Total text the size sweep re-chunks, spread evenly across fixtures: enough for
// tens of thousands of calls at the smallest size, small enough that six sizes
// cost about one extra phase-1 pass.
const SIZE_SAMPLE_BYTES: usize = 8 * 1024 * 1024;

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

// ── counting allocator ──────────────────────────────────────────────────────

static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE_BYTES: AtomicI64 = AtomicI64::new(0);
static PEAK_LIVE_BYTES: AtomicI64 = AtomicI64::new(0);

/// The binary's global allocator: [`System`] plus four counters. Encoding a
/// fixed corpus single-threaded allocates a deterministic number of times, so
/// the counts diff exactly across commits where wall time can only be fenced
/// with noise margins; the `--memory` children snapshot them per fixture.
/// The counters are relaxed atomics, two adds per heap call: far below timing
/// noise for the wall-time phases running in the same binary.
struct CountingAlloc;

fn count_alloc(size: usize) {
    ALLOC_COUNT.fetch_add(1, Relaxed);
    ALLOC_BYTES.fetch_add(size as u64, Relaxed);
    let live = LIVE_BYTES.fetch_add(size as i64, Relaxed) + size as i64;
    PEAK_LIVE_BYTES.fetch_max(live, Relaxed);
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_alloc(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_alloc(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    // A grow counts as one allocation of the new size; the old block leaves the
    // live count. Both sizes are momentarily live, matching an alloc-copy-free.
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count_alloc(new_size);
        LIVE_BYTES.fetch_sub(layout.size() as i64, Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size() as i64, Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// One reading of the allocation counters; subtract two to attribute a phase.
#[derive(Clone, Copy)]
struct AllocSnap {
    count: u64,
    bytes: u64,
}

fn alloc_snap() -> AllocSnap {
    AllocSnap {
        count: ALLOC_COUNT.load(Relaxed),
        bytes: ALLOC_BYTES.load(Relaxed),
    }
}

impl AllocSnap {
    fn delta_json(&self) -> Value {
        let now = alloc_snap();
        json!({ "count": now.count - self.count, "bytes": now.bytes - self.bytes })
    }
}

/// Reset the live-bytes peak to the current level, so the reported peak covers
/// only what happens afterwards: the allocator-level mirror of [`reset_rss_peak`].
fn reset_peak_live() -> i64 {
    let live = LIVE_BYTES.load(Relaxed);
    PEAK_LIVE_BYTES.store(live, Relaxed);
    live
}

// ── fixtures ────────────────────────────────────────────────────────────────

struct Fixture {
    group: &'static str,
    name: String,
    chunks: Vec<String>,
    bytes: usize,
}

/// Cut `text` into ~10 kB chunks at line boundaries. Lossless: concatenating the
/// chunks reproduces `text` byte for byte, so the benched input is exactly the
/// corpus, blank lines and trailing newlines included.
fn make_chunks(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut cur = String::new();
    for line in text.split_inclusive('\n') {
        cur.push_str(line);
        if cur.len() >= CHUNK_BYTES {
            chunks.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

/// The sorted `.txt` fixtures under `data/fixtures/{lang,modalities}`, tagged with group.
fn fixture_paths() -> Vec<(&'static str, PathBuf)> {
    let mut out = Vec::new();
    for group in GROUPS {
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

/// Every corpus in `data/fixtures/{lang,modalities}`, read and chunked once, in full.
fn load_fixtures() -> Vec<Fixture> {
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

// ── timing helpers ──────────────────────────────────────────────────────────

fn median_secs(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn time_pass<S: AsRef<str>>(encode: &dyn Fn(&str) -> usize, chunks: &[S]) -> f64 {
    let start = Instant::now();
    let mut n = 0usize;
    for chunk in chunks {
        n += encode(chunk.as_ref());
    }
    black_box(n);
    start.elapsed().as_secs_f64()
}

// ── phase 1: throughput ─────────────────────────────────────────────────────

/// Cold single-thread throughput + the id gate for one fixture.
///
/// Each of the `REPS` timed passes runs on an instance that has never seen the
/// corpus: the baseline is re-cloned (the released models' `Clone` starts with an
/// empty cache) and the pipeline rebuilt from `oracle`, both outside the timed
/// region. The two implementations alternate so frequency and thermal drift hit
/// them equally. The id gate runs first, on instances of its own, so its encodes
/// cannot warm anything a timed pass sees.
fn bench_throughput(
    baseline: Option<&BaselineTokenizer>,
    oracle: &Tokenizer,
    f: &Fixture,
) -> Value {
    // The correctness gate CI fails on: pipeline ids == the released crate's ids,
    // for both `add_special_tokens` values (`true` exercises the post-process
    // stage). The in-tree `Tokenizer` only *builds* the pipeline here — never the
    // reference, since it is on its way out. `None` when the release can't load
    // this model (no reference to compare against).
    let gate = PipelineTokenizer::try_from(oracle).expect("probed at model load");
    let pipe_ids = |c: &String, add_special_tokens: bool| -> Vec<u32> {
        gate.encode(c, add_special_tokens)
            .wait()
            .unwrap()
            .iter()
            .flatten()
            .map(|t| t.id)
            .collect()
    };
    let ids_match = baseline.map(|b| {
        [false, true].into_iter().all(|add_special_tokens| {
            f.chunks.iter().take(3).all(|c| {
                b.encode_fast(c.as_str(), add_special_tokens)
                    .unwrap()
                    .get_ids()
                    == pipe_ids(c, add_special_tokens)
            })
        })
    });

    let (mut base_s, mut pipe_s) = (Vec::new(), Vec::new());
    for _ in 0..REPS {
        if let Some(b) = baseline {
            let cold = b.clone();
            base_s.push(time_pass(
                &|s| cold.encode_fast(s, true).unwrap().len(),
                &f.chunks,
            ));
        }
        let cold = PipelineTokenizer::try_from(oracle).expect("probed at model load");
        pipe_s.push(time_pass(
            &|s| cold.encode(s, true).wait().unwrap().first().unwrap().len(),
            &f.chunks,
        ));
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

// ── phase 2: input-size response ────────────────────────────────────────────

/// A corpus sample for the input-size sweep: up to an equal share of every
/// fixture (the whole fixture when it is smaller), so the mix matches the suite
/// without paying six full-corpus passes. The share is drawn as chunks spread
/// evenly across the whole fixture, never its head alone, so the sample is real
/// varied text from end to end: no fixture's opening boilerplate is overweighted
/// and nothing is repeated.
fn size_sample(fixtures: &[Fixture]) -> String {
    let budget = SIZE_SAMPLE_BYTES / fixtures.len().max(1);
    let mut sample = String::new();
    for f in fixtures {
        let step = (f.bytes / budget.max(1)).max(1);
        let mut taken = 0;
        for c in f.chunks.iter().step_by(step) {
            if taken >= budget {
                break;
            }
            let mut e = (budget - taken).min(c.len());
            while e < c.len() && !c.is_char_boundary(e) {
                e += 1;
            }
            sample.push_str(&c[..e]);
            taken += e;
        }
    }
    sample
}

/// Cut `text` into `size`-byte chunks, each cut moved forward to the next char
/// boundary. The line-boundary chunker can't serve the small sizes: one long
/// line would blow a 256-byte target.
fn sized_chunks(text: &str, size: usize) -> Vec<&str> {
    let mut chunks = Vec::with_capacity(text.len() / size + 1);
    let mut s = 0;
    while s < text.len() {
        let mut e = (s + size).min(text.len());
        while e < text.len() && !text.is_char_boundary(e) {
            e += 1;
        }
        chunks.push(&text[s..e]);
        s = e;
    }
    chunks
}

fn size_label(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else {
        format!("{} kB", bytes / 1024)
    }
}

/// Throughput of both implementations at every `SIZE_SWEEP` chunk size, over the
/// same `sample`, single thread. Cold instance per pass and interleaved reps,
/// exactly like phase 1; only the chunk size varies, so the curve isolates how
/// per-call overhead and amortization move the headline comparison.
fn bench_sizes(baseline: Option<&BaselineTokenizer>, oracle: &Tokenizer, sample: &str) -> Value {
    let (mut pipe, mut base) = (Vec::new(), Vec::new());
    for &size in SIZE_SWEEP {
        let chunks = sized_chunks(sample, size);
        let (mut base_s, mut pipe_s) = (Vec::new(), Vec::new());
        for _ in 0..REPS {
            if let Some(b) = baseline {
                let cold = b.clone();
                base_s.push(time_pass(
                    &|s| cold.encode_fast(s, true).unwrap().len(),
                    &chunks,
                ));
            }
            let cold = PipelineTokenizer::try_from(oracle).expect("probed at model load");
            pipe_s.push(time_pass(
                &|s| cold.encode(s, true).wait().unwrap().first().unwrap().len(),
                &chunks,
            ));
        }
        let mbps = |secs: f64| sample.len() as f64 / secs / 1e6;
        let b = (!base_s.is_empty()).then(|| mbps(median_secs(base_s)));
        let p = mbps(median_secs(pipe_s));
        eprintln!(
            "    {}: pipeline {p:.1} MB/s{}",
            size_label(size),
            b.map_or(String::new(), |v| format!(", baseline {v:.1} MB/s"))
        );
        pipe.push(p);
        base.push(b);
    }
    json!({ "bytes": SIZE_SWEEP, "pipeline_mbps": pipe, "baseline_mbps": base })
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

/// Median MB/s of encoding `chunks` across `n` threads in a *private* rayon pool
/// (so the sweep can't perturb — or be perturbed by — the global pool). Each timed
/// pass encodes through a cold instance built by `fresh`, outside the timed
/// region; one throwaway pass on its own instance first forces the pool to spawn
/// its threads. One `encode` call per chunk; the sum is `black_box`'d so the work
/// can't be elided.
fn par_mbps<E: Fn(&str) -> usize + Sync>(
    fresh: impl Fn() -> E,
    chunks: &[&String],
    bytes: usize,
    n: usize,
) -> f64 {
    let pool = ThreadPoolBuilder::new().num_threads(n).build().unwrap();
    let pass = |enc: &E| pool.install(|| chunks.par_iter().map(|c| enc(c.as_str())).sum::<usize>());
    black_box(pass(&fresh()));
    let mut samples = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let cold = fresh();
        let t = Instant::now();
        black_box(pass(&cold));
        samples.push(t.elapsed().as_secs_f64());
    }
    bytes as f64 / median_secs(samples) / 1e6
}

/// Multi-thread throughput sweep over one fixture group's corpus: pipeline vs the
/// released crate at 1/2/4/8/max threads. Every timed pass gets a cold instance,
/// same as the single-thread phase; within a pass the per-thread caches fill from
/// the mixed stream, which is what a parallel `.encode()` run over fresh text
/// reaches. The two implementations alternate per thread count so thermal drift
/// hits them equally.
fn bench_threads(
    baseline: Option<&BaselineTokenizer>,
    oracle: &Tokenizer,
    chunks: &[&String],
) -> Value {
    let bytes: usize = chunks.iter().map(|c| c.len()).sum();
    let counts = thread_counts();
    let (mut pipe, mut base) = (Vec::new(), Vec::new());
    for &n in &counts {
        let b = baseline.map(|b| {
            par_mbps(
                || {
                    let cold = b.clone();
                    move |s: &str| cold.encode_fast(s, true).unwrap().len()
                },
                chunks,
                bytes,
                n,
            )
        });
        let p = par_mbps(
            || {
                let cold = PipelineTokenizer::try_from(oracle).expect("probed at model load");
                move |s: &str| cold.encode(s, true).wait().unwrap().first().unwrap().len()
            },
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

/// Reset the peak-RSS watermark to the current RSS, so `rss_peak` reflects only
/// what happens afterwards. Writing `5` to `/proc/self/clear_refs` is the
/// documented reset (see proc(5)); a silent no-op elsewhere, matching `rss_peak`
/// being Linux-only.
fn reset_rss_peak() {
    let _ = std::fs::write("/proc/self/clear_refs", "5");
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

/// The `--memory` children's encode pass: every chunk of every fixture, once,
/// with the allocation counters snapshotted per fixture. The returned rows are
/// exact and deterministic (the corpus is fixed and the pass single-threaded),
/// so the report can flag *any* change against the base branch instead of
/// fencing with noise margins. `input_bytes` rides along so the renderer can
/// normalize to allocations per MB of input.
fn encode_counting_allocs(encode: &dyn Fn(&str) -> usize, fixtures: &[Fixture]) -> Vec<Value> {
    let mut rows = Vec::with_capacity(fixtures.len());
    let mut n = 0usize;
    for f in fixtures {
        let before = alloc_snap();
        for c in &f.chunks {
            n += encode(c);
        }
        let mut row = before.delta_json();
        row["fixture"] = json!(f.name);
        row["input_bytes"] = json!(f.bytes);
        rows.push(row);
    }
    black_box(n);
    rows
}

/// `--memory <impl> <model.json>` child entry: load one implementation and
/// encode the whole fixture corpus, printing
/// `{load_bytes, encode_bytes, peak_bytes, allocs}`. One implementation per
/// process so the deltas attribute cleanly. `allocs` carries the exact
/// allocator traffic from [`CountingAlloc`]: the load phase, per-fixture encode
/// rows ([`encode_counting_allocs`]), and the peak of live heap bytes.
///
/// Two rules keep the RSS deltas attributable:
/// - The corpus is loaded (and the peak watermark reset) before the first
///   measurement, so the fixture text cancels out of every delta.
/// - Nothing is freed between measurements. Freed memory tends to stay resident
///   in the allocator, where the next allocation grows into it instead of into
///   new pages, smearing one delta into the next. That is why the pipeline's
///   source `Tokenizer` stays alive to the end (its size is bracketed out by
///   measuring around the pipeline build; it still shows in `peak_bytes`, which
///   is honest — building a pipeline currently requires one), and why the encode
///   pass drops each encoding instead of keeping every id: `encode_bytes` is the
///   tokenizer's own growth over the corpus, not the size of the output ids.
fn memory_child(which: &str, model: &Path) {
    let fixtures = load_fixtures();
    reset_rss_peak();
    let live0 = reset_peak_live();
    let rss0 = rss_now().unwrap_or(0);

    let (load_bytes, encode_bytes, load_allocs, encode_allocs) = match which {
        "baseline" => {
            let before_load = alloc_snap();
            let mut tok = BaselineTokenizer::from_file(model).unwrap();
            inject_added_tokens_baseline(&mut tok);
            let load_allocs = before_load.delta_json();
            let after_load = rss_now().unwrap_or(0);
            let rows =
                encode_counting_allocs(&|s| tok.encode_fast(s, true).unwrap().len(), &fixtures);
            let after_encode = rss_now().unwrap_or(0);
            (
                after_load - rss0,
                after_encode - after_load,
                load_allocs,
                rows,
            )
        }
        "pipeline" => {
            let mut tok = Tokenizer::from_file(model).unwrap();
            inject_added_tokens(&mut tok);
            let before_build = rss_now().unwrap_or(0);
            let before_build_allocs = alloc_snap();
            let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
            let load_allocs = before_build_allocs.delta_json();
            let after_build = rss_now().unwrap_or(0);
            let rows = encode_counting_allocs(
                &|s| {
                    pipeline
                        .encode(s, true)
                        .wait()
                        .unwrap()
                        .first()
                        .unwrap()
                        .len()
                },
                &fixtures,
            );
            let after_encode = rss_now().unwrap_or(0);
            (
                after_build - before_build,
                after_encode - after_build,
                load_allocs,
                rows,
            )
        }
        other => panic!("unknown impl {:?}", other),
    };

    println!(
        "{}",
        json!({
            "load_bytes": load_bytes,
            "encode_bytes": encode_bytes,
            "peak_bytes": rss_peak().map(|p| p - rss0),
            "allocs": {
                "load": load_allocs,
                "encode": encode_allocs,
                "peak_live_bytes": PEAK_LIVE_BYTES.load(Relaxed) - live0,
            },
        })
    );
}

// ── instruction lane (callgrind) ────────────────────────────────────────────

// Explicit callgrind *client requests*: zero the counters entering a fixture,
// dump them (tagged with the fixture name) leaving it, so each fixture yields
// one dump holding exactly its encode cost. Callgrind's own function-boundary
// triggers (`--zero-before`/`--dump-after`) lose track of the marker under fat
// LTO and fire once per run instead of once per call; requests compiled into
// the child cannot be confused that way. They are no-ops outside valgrind.
// Linux-only: crabgrind needs the valgrind headers at build time (the CI build
// job installs them); elsewhere the lane never runs, so these are stubs.
#[cfg(target_os = "linux")]
fn cg_zero_stats() {
    crabgrind::callgrind::zero_stats();
}
#[cfg(target_os = "linux")]
fn cg_dump_stats(reason: &str) {
    let reason = std::ffi::CString::new(reason).unwrap();
    crabgrind::callgrind::dump_stats(reason.as_c_str());
}
#[cfg(not(target_os = "linux"))]
fn cg_zero_stats() {}
#[cfg(not(target_os = "linux"))]
fn cg_dump_stats(_reason: &str) {}

/// The instruction lane's unit of measurement: one cold pass over one fixture,
/// bracketed by the zero and dump requests. The dump carries the fixture name,
/// which [`measure_instructions`] checks when it attributes the parts.
fn instrumented_pass(encode: &dyn Fn(&str) -> usize, f: &Fixture) {
    cg_zero_stats();
    let mut n = 0usize;
    for c in &f.chunks {
        n += encode(c);
    }
    black_box(n);
    cg_dump_stats(&f.name);
}

/// `--instructions <impl> <model.json>` child entry: one implementation, one
/// cold instance per fixture (the phase-1 regime), each fixture's corpus encoded
/// once inside [`instrumented_pass`]. Run under callgrind by
/// [`measure_instructions`]; on its own it is just a slow no-op.
fn instructions_child(which: &str, model: &Path) {
    let fixtures = load_fixtures();
    match which {
        "baseline" => {
            let mut tok = BaselineTokenizer::from_file(model).unwrap();
            inject_added_tokens_baseline(&mut tok);
            for f in &fixtures {
                let cold = tok.clone();
                instrumented_pass(&|s| cold.encode_fast(s, true).unwrap().len(), f);
            }
        }
        "pipeline" => {
            let mut tok = Tokenizer::from_file(model).unwrap();
            inject_added_tokens(&mut tok);
            for f in &fixtures {
                let cold = PipelineTokenizer::try_from(&tok).unwrap();
                instrumented_pass(
                    &|s| cold.encode(s, true).wait().unwrap().first().unwrap().len(),
                    f,
                );
            }
        }
        other => panic!("unknown impl {:?}", other),
    }
}

fn valgrind_version() -> Option<String> {
    let out = Command::new("valgrind").arg("--version").output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Instructions retired (`Ir`) from one callgrind dump part. The part file
/// carries an `events:` header naming the counted columns and a `summary:` line
/// with their totals; `desc: Trigger:` says what caused the dump and must name
/// `fixture` (the dump request's reason string), which pins each part to its
/// fixture instead of trusting file order alone. The program-termination dump
/// (bare file, no fixture in its trigger) never parses as a fixture.
fn parse_ir(path: &Path, fixture: &str) -> Result<u64, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let field = |key: &str| {
        text.lines()
            .find_map(|l| l.strip_prefix(key))
            .map(str::trim)
            .ok_or_else(|| format!("{}: no `{key}` line", path.display()))
    };
    let trigger = field("desc: Trigger:")?;
    if !trigger.contains(fixture) {
        return Err(format!(
            "{}: trigger {trigger:?} is not fixture {fixture:?}",
            path.display()
        ));
    }
    let ir_col = field("events:")?
        .split_whitespace()
        .position(|e| e == "Ir")
        .ok_or_else(|| format!("{}: no Ir event", path.display()))?;
    field("summary:")?
        .split_whitespace()
        .nth(ir_col)
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| format!("{}: bad summary", path.display()))
}

/// Instructions retired per fixture, per implementation, by re-running this
/// binary as `--instructions` children under callgrind. `Null` without valgrind
/// (macOS local runs); a per-impl `null` when its child or a dump fails.
///
/// One child per implementation over the whole corpus: callgrind's simulated
/// instruction counts are deterministic where wall time is not, so a single
/// pass resolves changes far below the timing phases' noise floor, and the
/// per-fixture attribution comes from the [`instrumented_pass`] dump requests
/// rather than one process per fixture. Expects exactly one dump per fixture,
/// in fixture order and carrying the fixture's name; any mismatch (a missed
/// request, a crashed pass) fails that implementation loudly rather than
/// mis-attributing dumps.
fn measure_instructions(model: &Path, baseline_ok: bool, fixtures: &[Fixture]) -> Value {
    if valgrind_version().is_none() {
        eprintln!("  instruction lane: valgrind not found, skipped");
        return Value::Null;
    }
    let exe = std::env::current_exe().unwrap();
    let mut out = serde_json::Map::new();
    for (key, ok) in [("baseline", baseline_ok), ("pipeline", true)] {
        if !ok {
            out.insert(key.into(), Value::Null);
            continue;
        }
        let base = std::env::temp_dir().join(format!("cg-{}-{key}.out", std::process::id()));
        let part = |i: usize| base.with_extension(format!("out.{i}"));
        // Clear leftovers from a crashed earlier run: a stale part would trip the
        // more-dumps-than-fixtures check below.
        for i in 1.. {
            if std::fs::remove_file(part(i)).is_err() {
                break;
            }
        }
        let res = Command::new("valgrind")
            .arg("--tool=callgrind")
            .arg(format!("--callgrind-out-file={}", base.display()))
            .arg(&exe)
            .arg("--instructions")
            .arg(key)
            .arg(model)
            .output()
            .expect("failed to spawn valgrind");
        let irs: Result<Vec<u64>, String> = if !res.status.success() {
            Err(format!(
                "child failed: {}",
                String::from_utf8_lossy(&res.stderr)
            ))
        } else if part(fixtures.len() + 1).exists() {
            Err("more dumps than fixtures, refusing to attribute".to_string())
        } else {
            fixtures
                .iter()
                .enumerate()
                .map(|(i, f)| parse_ir(&part(i + 1), &f.name))
                .collect()
        };
        for i in 1.. {
            if std::fs::remove_file(part(i)).is_err() {
                break;
            }
        }
        let _ = std::fs::remove_file(&base);
        match irs {
            Ok(irs) => {
                let per_fixture: serde_json::Map<String, Value> = fixtures
                    .iter()
                    .zip(&irs)
                    .map(|(f, ir)| (f.name.clone(), json!(ir)))
                    .collect();
                out.insert(key.into(), Value::Object(per_fixture));
            }
            Err(e) => {
                eprintln!("  instruction lane {key} failed: {e}");
                out.insert(key.into(), Value::Null);
            }
        }
    }
    Value::Object(out)
}

// ── environment stamp ───────────────────────────────────────────────────────

/// The measuring machine's identity, embedded in the output so the report can
/// tell an environment change from a code regression: instruction counts are
/// deterministic *per environment*, and a runner-image or CPU-model change
/// shifts them without any code changing.
fn env_stamp() -> Value {
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|v| v.trim().to_string())
        })
        .or_else(|| {
            let out = Command::new("sysctl")
                .args(["-n", "machdep.cpu.brand_string"])
                .output()
                .ok()?;
            out.status
                .success()
                .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
        });
    let glibc = Command::new("getconf")
        .arg("GNU_LIBC_VERSION")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty());
    json!({ "cpu": cpu, "glibc": glibc, "valgrind": valgrind_version() })
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
/// Both come from the test-data dataset (see the Makefile `models` target).
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--memory") {
        memory_child(&args[2], Path::new(&args[3]));
        return;
    }
    if args.get(1).map(String::as_str) == Some("--instructions") {
        instructions_child(&args[2], Path::new(&args[3]));
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
    let fixtures = load_fixtures();
    // The corpus per fixture group, flattened once: the multi-thread sweeps run
    // over a whole group so thread-spawn/scheduling overhead is amortized and the
    // scaling curve is stable, and per group so the report can show how scaling
    // differs between natural language and the code/math/added-token workloads.
    let group_chunks: Vec<(&str, Vec<&String>)> = GROUPS
        .iter()
        .map(|g| {
            let chunks = fixtures
                .iter()
                .filter(|f| f.group == *g)
                .flat_map(|f| f.chunks.iter())
                .collect();
            (*g, chunks)
        })
        .collect();
    let sample = size_sample(&fixtures);

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
        match PipelineTokenizer::try_from(&tok) {
            Ok(p) => match p.encode(PROBE, false).wait() {
                Ok(_) => {}
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
        }

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

        // Per-fixture instruction counts join the throughput rows: `ir` raw and
        // `ir_per_byte` normalized, mirroring the `mbps` shape.
        let instructions = measure_instructions(&path, baseline.is_some(), &fixtures);
        if let Some(by_impl) = instructions.as_object() {
            for (row, f) in rows.iter_mut().zip(&fixtures) {
                let ir = |impl_: &str| {
                    by_impl
                        .get(impl_)
                        .and_then(|v| v.get(&f.name))
                        .and_then(Value::as_u64)
                };
                let (b, p) = (ir("baseline"), ir("pipeline"));
                if b.is_none() && p.is_none() {
                    continue;
                }
                let per_byte = |ir: Option<u64>| ir.map(|n| n as f64 / f.bytes as f64);
                row["ir"] = json!({ "baseline": b, "pipeline": p });
                row["ir_per_byte"] = json!({ "baseline": per_byte(b), "pipeline": per_byte(p) });
                eprintln!(
                    "  {} Ir/B: baseline {}, pipeline {}",
                    f.name,
                    per_byte(b).map_or("—".into(), |v| format!("{v:.1}")),
                    per_byte(p).map_or("—".into(), |v| format!("{v:.1}")),
                );
            }
        }

        eprintln!("  input-size sweep:");
        let input_sizes = bench_sizes(baseline.as_ref(), &tok, &sample);

        let memory = measure_memory(&path, baseline.is_some());
        let threads = Value::Object(
            group_chunks
                .iter()
                .map(|(g, chunks)| {
                    eprintln!("  encode thread sweep ({g}):");
                    (
                        g.to_string(),
                        bench_threads(baseline.as_ref(), &tok, chunks),
                    )
                })
                .collect(),
        );

        models.push(json!({
            "model": name, "desc": desc, "shape": shape,
            "results": rows, "memory": memory, "threads": threads,
            "input_sizes": input_sizes,
        }));
    }

    let out = json!({
        "baseline": { "crate": "tokenizers", "version": BASELINE_VERSION },
        "env": env_stamp(),
        "models": models,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
