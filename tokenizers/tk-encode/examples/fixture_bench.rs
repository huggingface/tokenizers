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
//! 3. **Scaling & memory**: a multi-thread throughput sweep (2 and 4 threads)
//!    per fixture group (`lang`, `modalities`), cold instance per pass, and
//!    resident-set deltas measured by re-spawning this binary as
//!    `--memory <impl> <model.json>` children; one implementation per process,
//!    so allocator page reuse can't blur the attribution. The children also
//!    report exact allocation counts, read from the counting global allocator
//!    ([`CountingAlloc`]) around the same load and per-fixture brackets, for the
//!    encode pass and phase 4's decode pass alike.
//! 4. **Decode**: the inverse direction, over an id stream the *release* mints
//!    (see below), so both decoders consume exactly the same ids and decode is
//!    judged on decode alone even where the pipeline's encode diverges —
//!    single-thread MB/s per fixture, the same 2/4-thread sweep per group, and
//!    a decode leg in the memory children, all gated by `text_match`. No
//!    released baseline means no id stream, so the whole phase is `null`.
//!
//! # Why the allocation lane
//!
//! Wall time on shared CI runners carries a noise floor of several percent, so
//! the timing phases cannot honestly resolve a 1-3% regression. Allocation
//! counts can: they are a property of the executed code path, not of the
//! machine it ran on, so the same commit reproduces the same numbers exactly
//! and the report can flag changes far below the timing noise. The lane sees
//! no parallelism (the counting pass is single-threaded), and fewer
//! allocations is not always faster, so the timing phases stay the ground
//! truth for *time*; the lane is the tripwire for *work*. (An instruction-count
//! lane under callgrind used to sit next to it, measuring the same thing with
//! more coverage, but a callgrind pass over the corpus cost 10+ minutes per
//! model where this lane is free.)
//!
//! The counting is confined to the `--memory` children ([`COUNTING`]) because it
//! is not free under threads: the counters are shared atomics, and with them on
//! the sweeps reported the released crate *losing* throughput from 2 to 4 threads.
//! Measured on gpt2, baseline MB/s at 2 → 4 threads, counting off vs on:
//! encode 9.5 → 17.6 vs 4.8 → 3.1, decode 37.5 → 64.8 vs 13.1 → 7.5. The
//! pipeline's own numbers moved by noise (219 → 222 at 2 threads), because the
//! contention scales with how often an implementation allocates and the release
//! allocates ~3000× more per decode — so the lane was silently distorting exactly
//! the comparison the sweeps exist to draw. Single-thread numbers were unaffected;
//! uncontended atomics are cheap, it is the cross-core traffic that bites.
//!
//! # Cached baseline numbers (`--baseline-from`)
//!
//! The released crate is pinned, so its numbers only move when the machine
//! changes, yet timing it dominates a full run: it encodes at a few MB/s where
//! the pipeline does tens. `--baseline-from <merged bench JSON>` skips every
//! baseline timing lane and copies the release's numbers (phase-1 MB/s, size
//! sweep, thread sweep, memory, decode) from that file, a previous run of this
//! bench that did measure them; CI caches one per bench-input hash. The release
//! is still loaded — it is the oracle both gates compare against, and the only
//! thing that may mint the decode phase's ids — and one **canary**
//! fixture ([`CANARY_FIXTURE`]) re-measures baseline throughput in the phase-1
//! regime. Each model's output pairs the canary's measured and cached MB/s so
//! the report can tell whether the cached numbers still describe this machine.
//! Allocation counts are exact, so the copied ones need no canary.
//!
//! `--model <name>` benches a single manifest entry; CI fans out one job per
//! model and concatenates the partial JSONs in manifest order.
//!
//! # The decode sample ([`DECODE_SAMPLE_BYTES`])
//!
//! Decode replays a capped prefix of each fixture, not the whole corpus.
//! Minting the ids runs at *release encode* speed — the slowest thing in the
//! bench, and exactly what `--baseline-from` exists to skip — while decoding
//! them is two orders of magnitude faster, so a full-corpus stream would cost
//! far more to prepare than to measure. Throughput is reported over the *input*
//! bytes those ids came from: the same denominator the encode phases use, so
//! decode and encode MB/s are directly comparable, and it costs no extra pass
//! to obtain (measuring the decoded bytes instead would need a whole released
//! decode pass that cached-baseline mode has no other reason to run).
//!
//! Decode passes reuse one instance per implementation rather than re-instancing
//! per rep like the encode phases: neither decoder caches anything (both are
//! `&self` lookups), so a cold instance measures the same thing and the rebuild
//! would cost more than the pass. `PipelineWordPiece` keeps no id → token map,
//! so `PipelineTokenizer::decode` still fails loudly there — those models report
//! a `null` pipeline decode series (rendered "pending") while the release's is
//! measured regardless.
//!
//! Correctness is judged against the released crate, never the in-tree
//! `Tokenizer` (which is being removed and only *builds* the pipeline here):
//! `ids_match` (encode ids) and `text_match` (decode text) both compare against
//! the release and fail CI. They are `null` when the release can't load the
//! model (no reference, so no gate). Models the pipeline can't build (or encode)
//! yet are reported with empty `results` (plus the failure `reason`) and their
//! pipeline shape rather than benched; the CI grid renders those as roadmap
//! cards. Each manifest entry carries a `desc`: a one-line label of the workload
//! archetype the model exercises, passed through to the report.
//!
//! Emits one JSON object (`{baseline, env, models}`) on stdout, consumed by
//! `.github/scripts/render_pipeline_bench.py` in CI.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::convert::TryFrom;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering::Relaxed};
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
const REPS: usize = 3;
const PROBE: &str = "The quick brown fox jumps 123.";
// The fixture groups under `data/fixtures/`, also the granularity of the
// multi-thread sweeps and of the report's thread-scaling charts.
const GROUPS: [&str; 2] = ["lang", "modalities"];
// Thread counts for the scaling sweep. Two points show the scaling shape;
// phase 1 already anchors single-thread, and every extra count costs `REPS`
// cold passes over a whole fixture group per implementation.
const THREAD_COUNTS: [usize; 2] = [2, 4];
// The fixture the cached-baseline canary re-measures (see the module docs).
// Present in every fixture set and representative of the headline workload.
const CANARY_FIXTURE: &str = "eng_Latn";
// Chunk sizes for the input-size sweep: ~256 B chat messages up to ~256 kB
// documents. The ~10 kB headline regime sits inside the range, so the curve
// shows how much of the headline speedup survives at either end.
const SIZE_SWEEP: &[usize] = &[256, 1024, 4096, 16 * 1024, 64 * 1024, 256 * 1024];
// Total text the size sweep re-chunks, spread evenly across fixtures: enough for
// tens of thousands of calls at the smallest size, small enough that six sizes
// cost about one extra phase-1 pass.
const SIZE_SAMPLE_BYTES: usize = 8 * 1024 * 1024;
// How much of each fixture the decode phase replays (see the module docs). The
// cost driver is minting the ids at release encode speed, so this trades decode
// pass length against that: enough that a multi-thread pass over a whole fixture
// group stays well clear of pool-scheduling noise, and small enough that the mint
// stays a fraction of one phase-1 pass.
const DECODE_SAMPLE_BYTES: usize = 2 * 1024 * 1024;

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

/// Set at the top of `main` in a `--memory` child, never in the parent. The
/// counters are four shared atomic read-modify-writes per heap call, which two or
/// more threads hammering the allocator serialize on hard enough to *invert* a
/// scaling curve — the implementation that allocates more per byte loses the
/// most, which is exactly the comparison the thread sweeps exist to make. So the
/// process that measures wall time does not count, and the process that counts
/// does not measure wall time. Reading one uncontended relaxed bool per call is
/// what the parent pays instead.
///
/// The handful of allocations made before the flag is set (runtime startup, argv)
/// are freed while it is set, so `LIVE_BYTES` carries a few KB of negative drift
/// against a peak of ~100 MB. It is the same drift on every run of a given model,
/// so the counts still diff exactly across commits.
static COUNTING: AtomicBool = AtomicBool::new(false);

/// The binary's global allocator: [`System`] plus four counters, live only in the
/// `--memory` children (see [`COUNTING`]). Encoding a fixed corpus
/// single-threaded allocates a deterministic number of times, so the counts diff
/// exactly across commits where wall time can only be fenced with noise margins;
/// the children snapshot them per fixture.
struct CountingAlloc;

fn count_alloc(size: usize) {
    if !COUNTING.load(Relaxed) {
        return;
    }
    ALLOC_COUNT.fetch_add(1, Relaxed);
    ALLOC_BYTES.fetch_add(size as u64, Relaxed);
    let live = LIVE_BYTES.fetch_add(size as i64, Relaxed) + size as i64;
    PEAK_LIVE_BYTES.fetch_max(live, Relaxed);
}

fn uncount_alloc(size: usize) {
    if COUNTING.load(Relaxed) {
        LIVE_BYTES.fetch_sub(size as i64, Relaxed);
    }
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
        uncount_alloc(layout.size());
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        uncount_alloc(layout.size());
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

/// The prefix of `f`'s chunks the decode phase replays — whole chunks totalling
/// at most [`DECODE_SAMPLE_BYTES`]. A single chunk over the cap (one very long
/// line) is still taken, so the sample is never empty for a non-empty fixture.
fn decode_chunks(f: &Fixture) -> &[String] {
    let mut total = 0;
    let n = f
        .chunks
        .iter()
        .position(|c| {
            total += c.len();
            total > DECODE_SAMPLE_BYTES
        })
        .unwrap_or(f.chunks.len());
    &f.chunks[..n.max(1).min(f.chunks.len())]
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
///
/// With `time_baseline` false (cached-baseline mode) the id gate still runs but
/// the baseline is never timed; `main` copies its MB/s from the cache.
fn bench_throughput(
    baseline: Option<&BaselineTokenizer>,
    oracle: &Tokenizer,
    f: &Fixture,
    time_baseline: bool,
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
        if let Some(b) = baseline.filter(|_| time_baseline) {
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
/// per-call overhead and amortization move the headline comparison. With
/// `time_baseline` false the baseline series is left empty for `main` to fill
/// from the cache.
fn bench_sizes(
    baseline: Option<&BaselineTokenizer>,
    oracle: &Tokenizer,
    sample: &str,
    time_baseline: bool,
) -> Value {
    let (mut pipe, mut base) = (Vec::new(), Vec::new());
    for &size in SIZE_SWEEP {
        let chunks = sized_chunks(sample, size);
        let (mut base_s, mut pipe_s) = (Vec::new(), Vec::new());
        for _ in 0..REPS {
            if let Some(b) = baseline.filter(|_| time_baseline) {
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

/// Median MB/s of running `items` across `n` threads in a *private* rayon pool
/// (so the sweep can't perturb — or be perturbed by — the global pool). Each timed
/// pass works through a fresh instance built by `fresh`, outside the timed
/// region; one throwaway pass on its own instance first forces the pool to spawn
/// its threads. One call per item; the sum is `black_box`'d so the work can't be
/// elided. Both directions share this: the encode sweep passes text chunks and a
/// `fresh` that rebuilds a cold tokenizer, the decode sweep id slices and a
/// `fresh` that hands back the one stateless instance.
fn par_mbps<T: Sync, E: Fn(&T) -> usize + Sync>(
    fresh: impl Fn() -> E,
    items: &[T],
    bytes: usize,
    n: usize,
) -> f64 {
    let pool = ThreadPoolBuilder::new().num_threads(n).build().unwrap();
    let pass = |run: &E| pool.install(|| items.par_iter().map(run).sum::<usize>());
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
/// released crate at each `THREAD_COUNTS` entry. Every timed pass gets a cold
/// instance, same as the single-thread phase; within a pass the per-thread caches
/// fill from the mixed stream, which is what a parallel `.encode()` run over fresh
/// text reaches. The two implementations alternate per thread count so thermal
/// drift hits them equally. With `time_baseline` false the baseline series is
/// left empty for `main` to fill from the cache.
fn bench_threads(
    baseline: Option<&BaselineTokenizer>,
    oracle: &Tokenizer,
    chunks: &[&String],
    time_baseline: bool,
) -> Value {
    let bytes: usize = chunks.iter().map(|c| c.len()).sum();
    let counts = THREAD_COUNTS;
    let (mut pipe, mut base) = (Vec::new(), Vec::new());
    for &n in &counts {
        let b = baseline.filter(|_| time_baseline).map(|b| {
            par_mbps(
                || {
                    let cold = b.clone();
                    move |s: &&String| cold.encode_fast(s.as_str(), true).unwrap().len()
                },
                chunks,
                bytes,
                n,
            )
        });
        let p = par_mbps(
            || {
                let cold = PipelineTokenizer::try_from(oracle).expect("probed at model load");
                move |s: &&String| {
                    cold.encode(s.as_str(), true)
                        .wait()
                        .unwrap()
                        .first()
                        .unwrap()
                        .len()
                }
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

// ── phase 4: decode ─────────────────────────────────────────────────────────

/// One fixture's decode workload: the ids to replay and the input bytes they were
/// minted from, which is also the throughput denominator (see the module docs).
struct DecodeSample {
    ids: Vec<Vec<u32>>,
    bytes: usize,
}

/// Mint the decode phase's id stream with the *release*, one entry per fixture in
/// `fixtures` order. Both decoders then consume these same ids, so the phase
/// measures decode alone — a pipeline whose encode diverges still gets a fair
/// decode number (and `ids_match` fails the run separately). `add_special_tokens`
/// is on, matching phase 1, so decode sees the frame tokens a real stream carries.
fn decode_samples(baseline: &BaselineTokenizer, fixtures: &[Fixture]) -> Vec<DecodeSample> {
    fixtures
        .iter()
        .map(|f| {
            let chunks = decode_chunks(f);
            DecodeSample {
                ids: chunks
                    .iter()
                    .map(|c| {
                        baseline
                            .encode_fast(c.as_str(), true)
                            .unwrap()
                            .get_ids()
                            .to_vec()
                    })
                    .collect(),
                bytes: chunks.iter().map(String::len).sum(),
            }
        })
        .collect()
}

/// Single-thread decode throughput + the `text_match` gate for one fixture.
///
/// `text_match` = pipeline decode == released decode over the sample's first
/// chunks: the gate CI fails on. It is `null` while `pipeline_ok` is false (the
/// pipeline can't decode this model at all), which is also when the pipeline
/// series stays `null` and renders as "pending". With `time_baseline` false the
/// baseline is never timed; `main` copies its MB/s from the cache.
fn bench_decode(
    baseline: &BaselineTokenizer,
    pipeline: &PipelineTokenizer,
    f: &Fixture,
    sample: &DecodeSample,
    pipeline_ok: bool,
    time_baseline: bool,
) -> Value {
    let text_match = pipeline_ok.then(|| {
        sample
            .ids
            .iter()
            .take(3)
            .all(|i| pipeline.decode(i, false).unwrap() == baseline.decode(i, false).unwrap())
    });

    let one_pass = |dec: &dyn Fn(&[u32]) -> usize| -> f64 {
        let start = Instant::now();
        let mut n = 0usize;
        for i in &sample.ids {
            n += dec(i);
        }
        black_box(n);
        start.elapsed().as_secs_f64()
    };
    let base_pass = |i: &[u32]| baseline.decode(i, false).unwrap().len();
    let pipe_pass = |i: &[u32]| pipeline.decode(i, false).unwrap().len();

    // One throwaway pass each, then the reps alternate, so thermal drift hits
    // both equally — the same interleaving phase 1 uses.
    if time_baseline {
        one_pass(&base_pass);
    }
    if pipeline_ok {
        one_pass(&pipe_pass);
    }
    let (mut base_s, mut pipe_s) = (Vec::new(), Vec::new());
    for _ in 0..REPS {
        if time_baseline {
            base_s.push(one_pass(&base_pass));
        }
        if pipeline_ok {
            pipe_s.push(one_pass(&pipe_pass));
        }
    }
    let mbps = |secs: f64| sample.bytes as f64 / secs / 1e6;
    let base_mbps = (!base_s.is_empty()).then(|| mbps(median_secs(base_s)));
    let pipe_mbps = (!pipe_s.is_empty()).then(|| mbps(median_secs(pipe_s)));

    eprintln!(
        "  {} decode: baseline {}, pipeline {}",
        f.name,
        base_mbps.map_or("—".into(), |v: f64| format!("{v:.1} MB/s")),
        pipe_mbps.map_or("pending".into(), |v: f64| format!("{v:.1} MB/s")),
    );

    json!({
        "decode_mbps": { "baseline": base_mbps, "pipeline": pipe_mbps },
        "text_match": text_match,
    })
}

/// The decode fields of a fixture row when there is no decode phase to run: the
/// release can't load this model, so there is no id stream and no oracle.
fn decode_null() -> Value {
    json!({
        "decode_mbps": { "baseline": Value::Null, "pipeline": Value::Null },
        "text_match": Value::Null,
    })
}

/// Multi-thread decode sweep over one fixture group's id stream — the decode twin
/// of [`bench_threads`], same `THREAD_COUNTS` and same output shape, so the report
/// renders both with one chart. `fresh` hands back the same instance every rep:
/// decode keeps no state, so there is nothing to reset between passes.
fn bench_decode_threads(
    baseline: &BaselineTokenizer,
    pipeline: &PipelineTokenizer,
    ids: &[&Vec<u32>],
    bytes: usize,
    pipeline_ok: bool,
    time_baseline: bool,
) -> Value {
    let counts = THREAD_COUNTS;
    let (mut pipe, mut base) = (Vec::new(), Vec::new());
    for &n in &counts {
        let b = time_baseline.then(|| {
            par_mbps(
                || |i: &&Vec<u32>| baseline.decode(i, false).unwrap().len(),
                ids,
                bytes,
                n,
            )
        });
        let p = pipeline_ok.then(|| {
            par_mbps(
                || |i: &&Vec<u32>| pipeline.decode(i, false).unwrap().len(),
                ids,
                bytes,
                n,
            )
        });
        eprintln!(
            "    decode {n} thread(s): pipeline {}{}",
            p.map_or("pending".into(), |v: f64| format!("{v:.1} MB/s")),
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

/// The `--memory` children's decode pass, the twin of [`encode_counting_allocs`]:
/// replay the capped decode sample with the counters snapshotted per fixture, and
/// return the pass's RSS delta alongside the rows.
///
/// The ids are minted *before* the bracketing RSS reading, so holding them lands
/// in neither `encode_bytes` (which must stay the tokenizer's own growth over the
/// corpus, not the size of the output) nor `decode_bytes`; they still show in
/// `peak_bytes`, which is honest — decoding a stream requires having one. Each
/// child mints with its own encoder instead of sharing one stream: `ids_match`
/// already requires the two to be identical, and CI fails when they are not.
fn decode_counting_allocs(
    encode_ids: &dyn Fn(&str) -> Vec<u32>,
    decode: &dyn Fn(&[u32]) -> usize,
    fixtures: &[Fixture],
) -> (i64, Vec<Value>) {
    let minted: Vec<(&str, usize, Vec<Vec<u32>>)> = fixtures
        .iter()
        .map(|f| {
            let chunks = decode_chunks(f);
            (
                f.name.as_str(),
                chunks.iter().map(String::len).sum(),
                chunks.iter().map(|c| encode_ids(c)).collect(),
            )
        })
        .collect();

    let before_pass = rss_now().unwrap_or(0);
    let mut rows = Vec::with_capacity(minted.len());
    let mut n = 0usize;
    for (name, bytes, ids) in &minted {
        let before = alloc_snap();
        for i in ids {
            n += decode(i);
        }
        let mut row = before.delta_json();
        row["fixture"] = json!(name);
        row["input_bytes"] = json!(bytes);
        rows.push(row);
    }
    black_box(n);
    (rss_now().unwrap_or(0) - before_pass, rows)
}

/// `--memory <impl> <model.json>` child entry: load one implementation and
/// encode the whole fixture corpus then decode the capped sample, printing
/// `{load_bytes, encode_bytes, decode_bytes, peak_bytes, allocs}`. One
/// implementation per process so the deltas attribute cleanly. `allocs` carries
/// the exact allocator traffic from [`CountingAlloc`]: the load phase,
/// per-fixture encode and decode rows ([`encode_counting_allocs`],
/// [`decode_counting_allocs`]), and the peak of live heap bytes. The decode
/// entries are `null` for a model this implementation can't decode.
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

    let (load_bytes, encode_bytes, load_allocs, encode_allocs, decode) = match which {
        "baseline" => {
            let before_load = alloc_snap();
            let mut tok = BaselineTokenizer::from_file(model).unwrap();
            inject_added_tokens_baseline(&mut tok);
            let load_allocs = before_load.delta_json();
            let after_load = rss_now().unwrap_or(0);
            let rows =
                encode_counting_allocs(&|s| tok.encode_fast(s, true).unwrap().len(), &fixtures);
            let after_encode = rss_now().unwrap_or(0);
            let decode = decode_counting_allocs(
                &|s| tok.encode_fast(s, true).unwrap().get_ids().to_vec(),
                &|i| tok.decode(i, false).unwrap().len(),
                &fixtures,
            );
            (
                after_load - rss0,
                after_encode - after_load,
                load_allocs,
                rows,
                Some(decode),
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
            // Same probe the timing phases use: a model the pipeline can't decode
            // reports `null` here rather than a bogus 0.
            let decode = pipeline.decode(&[0], false).is_ok().then(|| {
                decode_counting_allocs(
                    &|s| {
                        pipeline
                            .encode(s, true)
                            .wait()
                            .unwrap()
                            .first()
                            .unwrap()
                            .iter()
                            .map(|t| t.id)
                            .collect()
                    },
                    &|i| pipeline.decode(i, false).unwrap().len(),
                    &fixtures,
                )
            });
            (
                after_build - before_build,
                after_encode - after_build,
                load_allocs,
                rows,
                decode,
            )
        }
        other => panic!("unknown impl {:?}", other),
    };

    println!(
        "{}",
        json!({
            "load_bytes": load_bytes,
            "encode_bytes": encode_bytes,
            "decode_bytes": decode.as_ref().map(|(bytes, _)| *bytes),
            "peak_bytes": rss_peak().map(|p| p - rss0),
            "allocs": {
                "load": load_allocs,
                "encode": encode_allocs,
                "decode": decode.map(|(_, rows)| rows),
                "peak_live_bytes": PEAK_LIVE_BYTES.load(Relaxed) - live0,
            },
        })
    );
}

// ── environment stamp ───────────────────────────────────────────────────────

/// The measuring machine's identity, embedded in the output so the report can
/// tell an environment change from a code regression: a runner-image or
/// CPU-model change shifts wall time without any code changing.
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
    json!({ "cpu": cpu, "glibc": glibc })
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

// ── cached baseline numbers ─────────────────────────────────────────────────

/// A previous merged bench JSON, one that measured the release, indexed by model
/// name. `--baseline-from` mode copies the release's numbers out of it instead
/// of re-timing them; see the module docs.
struct BaselineCache {
    models: HashMap<String, Value>,
}

impl BaselineCache {
    fn load(path: &Path) -> Self {
        let data: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        // The CI cache key covers the pinned version, but numbers from another
        // release would silently corrupt every chart; check it outright.
        assert_eq!(
            data["baseline"]["version"].as_str(),
            Some(BASELINE_VERSION),
            "cached baseline numbers are for another release"
        );
        let models = data["models"]
            .as_array()
            .expect("no models in the cached bench JSON")
            .iter()
            .map(|m| (m["model"].as_str().unwrap().to_string(), m.clone()))
            .collect();
        Self { models }
    }

    /// The cached entry for one model. A missing model means the cache key
    /// failed to cover a manifest change; refuse rather than bench without a
    /// baseline.
    fn model(&self, name: &str) -> &Value {
        self.models
            .get(name)
            .unwrap_or_else(|| panic!("model {name:?} is not in the cached baseline"))
    }

    /// The cached baseline MB/s for one fixture, from the phase-1 (`mbps`) or the
    /// decode (`decode_mbps`) lane; `Null` when the release could not bench it.
    fn fixture_mbps(&self, model: &str, fixture: &str, lane: &str) -> Value {
        self.model(model)["results"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|r| r["fixture"] == fixture)
            .map_or(Value::Null, |r| r[lane]["baseline"].clone())
    }
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
        // Only the children count; see `COUNTING`.
        COUNTING.store(true, Relaxed);
        memory_child(&args[2], Path::new(&args[3]));
        return;
    }
    // `--model <name>` benches one manifest entry (CI runs one job per model);
    // `--baseline-from <json>` copies the release's numbers from a cached run
    // instead of re-timing them (see the module docs). No flags = the whole
    // manifest, everything measured.
    let mut model_filter: Option<String> = None;
    let mut baseline_from: Option<PathBuf> = None;
    let mut rest = args[1..].iter();
    while let Some(arg) = rest.next() {
        let mut value = || rest.next().unwrap_or_else(|| panic!("{arg} needs a value"));
        match arg.as_str() {
            "--model" => model_filter = Some(value().clone()),
            "--baseline-from" => baseline_from = Some(PathBuf::from(value())),
            other => panic!("unknown argument {other:?}"),
        }
    }
    let cache = baseline_from.as_deref().map(BaselineCache::load);

    let full: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(MANIFEST).unwrap()).unwrap();
    let manifest: Vec<&Value> = match &model_filter {
        Some(name) => {
            let picked: Vec<&Value> = full.iter().filter(|e| e["name"] == name.as_str()).collect();
            assert!(!picked.is_empty(), "model {name:?} is not in the manifest");
            picked
        }
        None => full.iter().collect(),
    };
    eprintln!(
        "benching {} of {} model(s){}",
        manifest.len(),
        full.len(),
        if cache.is_some() {
            ", baseline numbers from cache"
        } else {
            ""
        }
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

        // The canary, before any timed phase: re-measure the baseline on one
        // fixture in the phase-1 regime and pair it with the cached number, so
        // the report can tell whether the cache still describes this machine.
        let canary = cache.as_ref().zip(baseline.as_ref()).map(|(cache, b)| {
            let f = fixtures
                .iter()
                .find(|f| f.name == CANARY_FIXTURE)
                .expect("canary fixture missing from data/fixtures");
            let mut samples = Vec::with_capacity(REPS);
            for _ in 0..REPS {
                let cold = b.clone();
                samples.push(time_pass(
                    &|s| cold.encode_fast(s, true).unwrap().len(),
                    &f.chunks,
                ));
            }
            let measured = f.bytes as f64 / median_secs(samples) / 1e6;
            let cached = cache.fixture_mbps(&name, &f.name, "mbps");
            eprintln!(
                "  canary {}: baseline measured {measured:.1} MB/s, cached {cached} MB/s",
                f.name
            );
            json!({ "fixture": f.name, "measured_mbps": measured, "cached_mbps": cached })
        });

        let time_baseline = cache.is_none();
        let mut rows: Vec<Value> = fixtures
            .iter()
            .map(|f| bench_throughput(baseline.as_ref(), &tok, f, time_baseline))
            .collect();

        eprintln!("  input-size sweep:");
        let mut input_sizes = bench_sizes(baseline.as_ref(), &tok, &sample, time_baseline);

        let mut memory = measure_memory(&path, baseline.is_some() && time_baseline);
        let mut threads = Value::Object(
            group_chunks
                .iter()
                .map(|(g, chunks)| {
                    eprintln!("  encode thread sweep ({g}):");
                    (
                        g.to_string(),
                        bench_threads(baseline.as_ref(), &tok, chunks, time_baseline),
                    )
                })
                .collect(),
        );

        // Phase 4. The id stream is minted by the release, so a model it can't
        // load gets no decode phase either. One pipeline instance serves the whole
        // phase — decode keeps no state to reset between passes.
        let decode_pipeline = PipelineTokenizer::try_from(&tok).expect("probed at model load");
        let (decode_ok, decode_reason) = match decode_pipeline.decode(&[0], false) {
            Ok(_) => (true, None),
            Err(e) => {
                eprintln!("  pipeline can't decode this model ({shape}): {e}");
                (false, Some(format!("{e}")))
            }
        };
        let samples = baseline.as_ref().map(|b| decode_samples(b, &fixtures));
        let decode_input = baseline.as_ref().zip(samples.as_ref());
        for (i, (row, f)) in rows.iter_mut().zip(&fixtures).enumerate() {
            let dec = decode_input.map_or_else(decode_null, |(b, s)| {
                bench_decode(b, &decode_pipeline, f, &s[i], decode_ok, time_baseline)
            });
            let row = row.as_object_mut().unwrap();
            for (k, v) in dec.as_object().unwrap() {
                row.insert(k.clone(), v.clone());
            }
        }
        let mut decode_threads = decode_input.map_or(Value::Null, |(b, samples)| {
            Value::Object(
                GROUPS
                    .iter()
                    .map(|g| {
                        eprintln!("  decode thread sweep ({g}):");
                        let group: Vec<&DecodeSample> = fixtures
                            .iter()
                            .zip(samples)
                            .filter(|(f, _)| f.group == *g)
                            .map(|(_, s)| s)
                            .collect();
                        let ids: Vec<&Vec<u32>> = group.iter().flat_map(|s| s.ids.iter()).collect();
                        let bytes = group.iter().map(|s| s.bytes).sum();
                        let sweep = bench_decode_threads(
                            b,
                            &decode_pipeline,
                            &ids,
                            bytes,
                            decode_ok,
                            time_baseline,
                        );
                        (g.to_string(), sweep)
                    })
                    .collect(),
            )
        });

        // Stitch the cached baseline numbers into the shapes the measuring run
        // would have produced, so the merged JSON and the report never care
        // where they came from.
        if let Some(cache) = &cache {
            let cached = cache.model(&name);
            for (row, f) in rows.iter_mut().zip(&fixtures) {
                row["mbps"]["baseline"] = cache.fixture_mbps(&name, &f.name, "mbps");
                row["decode_mbps"]["baseline"] = cache.fixture_mbps(&name, &f.name, "decode_mbps");
            }
            input_sizes["baseline_mbps"] = cached["input_sizes"]["baseline_mbps"].clone();
            for (g, _) in &group_chunks {
                threads[*g]["baseline_mbps"] = cached["threads"][*g]["baseline_mbps"].clone();
                if decode_threads.is_object() {
                    decode_threads[*g]["baseline_mbps"] =
                        cached["decode_threads"][*g]["baseline_mbps"].clone();
                }
            }
            memory["baseline"] = cached["memory"]["baseline"].clone();
        }

        models.push(json!({
            "model": name, "desc": desc, "shape": shape,
            "results": rows, "memory": memory, "threads": threads,
            "input_sizes": input_sizes, "baseline_canary": canary,
            "decode_threads": decode_threads, "decode_reason": decode_reason,
        }));
    }

    let out = json!({
        "baseline": {
            "crate": "tokenizers",
            "version": BASELINE_VERSION,
            "cached": cache.is_some(),
        },
        "env": env_stamp(),
        "models": models,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
