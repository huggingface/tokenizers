//! Scaling bench for the `encode_batch` vs manual worker-pool comparison
//! described in https://github.com/huggingface/tokenizers/issues/1900.
//!
//! This is a deep-dive benchmark, not a CI watchdog. It is structured as a
//! standalone binary (with `harness = false`) rather than a criterion bench
//! because the regression it measures only manifests on machines with many
//! cores (≥16, ideally ≥64) under realistic batch shapes (large documents,
//! batch sizes ≥ 1024, hundreds of documents). The default GitHub Actions
//! `ubuntu-latest` runner has 4 vCPUs, where the asymmetry is small (≤1.2×)
//! and indistinguishable from noise; running the bench there would not
//! produce a useful regression signal. Maintainers run this manually on a
//! large machine; the existing `ci_benchmark.rs::concurrent-4t` group is
//! the in-CI watchdog.
//!
//! Two methods, both via the public Tokenizer API:
//!
//! * **`worker-pool`** — explicit `rayon::ThreadPoolBuilder` + `par_iter()`
//!   over `Tokenizer::encode`. Each `Encoding` is consumed (token count
//!   read, then dropped) inside the closure, so it is allocated and freed
//!   on the same worker thread.
//!
//! * **`encode-batch`** — stock `Tokenizer::encode_batch` per chunk of
//!   `--batch-size`. The returned `Vec<Encoding>` is iterated on the main
//!   thread and dropped there.
//!
//! Headline metric: `encode_batch_elapsed / worker_pool_elapsed`. On a
//! 128-core x86_64 box with glibc, against current `main` (which already
//! includes the per-thread BPE cache fix from #2028), this is ~2×. After
//! @sebpop's planned `Encoding`-recycling work lands, it should drop to
//! ~1×. See https://github.com/stargazerZJ/tokenizers-1900-repro for a
//! standalone reproducer with drop-site A/B, allocator-swap, and
//! `MALLOC_ARENA_MAX` experiments separating the contributing effects.
//!
//! Two workloads (run by default; pick one with `--workload`):
//!
//! * **`random-letters`** — synthesized random a-zA-Z with no whitespace,
//!   exercising the BPE merge path (low cache hit rate).
//! * **`repeated-words`** — synthesized short pseudo-words separated by
//!   spaces, exercising the BPE cache (high cache hit rate).
//!
//! The two regimes respond differently to fixes; running only one risks
//! over-fitting future PRs to one and silently regressing the other.
//!
//! Run it:
//!
//! ```text
//! cargo bench --bench scaling_bench --
//! cargo bench --bench scaling_bench -- --workers 32 --batch-size 1024 --count 1000 --length 8192
//! cargo bench --bench scaling_bench -- --quick     # small, finishes in < 30s
//! cargo bench --bench scaling_bench -- --workload repeated-words
//! cargo bench --bench scaling_bench -- --input data/big.txt    # real corpus
//! ```
//!
//! Allocator caveat: the magnitude of the ratio depends heavily on the
//! system allocator. On glibc the gap is widest; jemalloc roughly halves it;
//! mimalloc is workload-dependent. The *direction* of the asymmetry is
//! stable across allocators — that is the regression signal.

use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

/// Default tokenizer (already fetched by `make bench` into `data/`).
const DEFAULT_TOKENIZER: &str = "data/llama-3-tokenizer.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Method {
    WorkerPool,
    EncodeBatch,
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Workload {
    RandomLetters,
    RepeatedWords,
    Both,
}

#[derive(Debug)]
struct Args {
    tokenizer: PathBuf,
    workers: usize,
    batch_size: usize,
    count: usize,
    length: usize,
    word_vocab_size: usize,
    word_len: usize,
    seed: u64,
    method: Method,
    workload: Workload,
    input: Option<PathBuf>,
}

impl Args {
    fn defaults() -> Self {
        Self {
            tokenizer: PathBuf::from(DEFAULT_TOKENIZER),
            // 0 = num_cpus (resolved later)
            workers: 0,
            batch_size: 1024,
            count: 500,
            length: 51200,
            word_vocab_size: 2048,
            word_len: 8,
            seed: 42,
            method: Method::Both,
            workload: Workload::Both,
            input: None,
        }
    }

    fn quick() -> Self {
        Self {
            count: 100,
            length: 8192,
            batch_size: 256,
            ..Self::defaults()
        }
    }

    fn parse() -> Result<Self, String> {
        // First pass: was --quick passed? If so, start from quick defaults.
        let raw: Vec<String> = std::env::args().skip(1).collect();
        let mut args = if raw.iter().any(|a| a == "--quick") {
            Self::quick()
        } else {
            Self::defaults()
        };

        let mut i = 0;
        while i < raw.len() {
            let a = &raw[i];
            let take_value = |label: &str, i: &mut usize| -> Result<String, String> {
                *i += 1;
                raw.get(*i)
                    .cloned()
                    .ok_or_else(|| format!("{label} needs a value"))
            };
            match a.as_str() {
                "-h" | "--help" => {
                    print_help();
                    std::process::exit(0);
                }
                "--quick" => { /* handled above */ }
                "-t" | "--tokenizer" => {
                    args.tokenizer = PathBuf::from(take_value("--tokenizer", &mut i)?);
                }
                "-w" | "--workers" => {
                    args.workers = take_value("--workers", &mut i)?
                        .parse()
                        .map_err(|e| format!("--workers: {e}"))?;
                }
                "-b" | "--batch-size" => {
                    args.batch_size = take_value("--batch-size", &mut i)?
                        .parse()
                        .map_err(|e| format!("--batch-size: {e}"))?;
                }
                "--count" => {
                    args.count = take_value("--count", &mut i)?
                        .parse()
                        .map_err(|e| format!("--count: {e}"))?;
                }
                "--length" => {
                    args.length = take_value("--length", &mut i)?
                        .parse()
                        .map_err(|e| format!("--length: {e}"))?;
                }
                "--word-vocab-size" => {
                    args.word_vocab_size = take_value("--word-vocab-size", &mut i)?
                        .parse()
                        .map_err(|e| format!("--word-vocab-size: {e}"))?;
                }
                "--word-len" => {
                    args.word_len = take_value("--word-len", &mut i)?
                        .parse()
                        .map_err(|e| format!("--word-len: {e}"))?;
                }
                "--seed" => {
                    args.seed = take_value("--seed", &mut i)?
                        .parse()
                        .map_err(|e| format!("--seed: {e}"))?;
                }
                "--method" => {
                    args.method = match take_value("--method", &mut i)?.as_str() {
                        "worker-pool" | "worker_pool" => Method::WorkerPool,
                        "encode-batch" | "encode_batch" => Method::EncodeBatch,
                        "both" => Method::Both,
                        other => {
                            return Err(format!(
                                "--method: {other:?} (want worker-pool|encode-batch|both)"
                            ))
                        }
                    };
                }
                "--workload" => {
                    args.workload = match take_value("--workload", &mut i)?.as_str() {
                        "random-letters" | "random_letters" => Workload::RandomLetters,
                        "repeated-words" | "repeated_words" => Workload::RepeatedWords,
                        "both" => Workload::Both,
                        other => {
                            return Err(format!(
                                "--workload: {other:?} \
                                 (want random-letters|repeated-words|both)"
                            ))
                        }
                    };
                }
                "--input" => {
                    args.input = Some(PathBuf::from(take_value("--input", &mut i)?));
                }
                // Criterion-compatibility: ignore noise from `cargo bench`.
                "--bench" | "--test" | "--save-baseline" | "--baseline" => {
                    if a == "--save-baseline" || a == "--baseline" {
                        // these take values
                        let _ = take_value(a, &mut i);
                    }
                }
                other => return Err(format!("unknown argument: {other}")),
            }
            i += 1;
        }

        if args.workers == 0 {
            args.workers = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8);
        }
        Ok(args)
    }
}

fn print_help() {
    eprintln!(
        "scaling_bench — encode_batch vs worker-pool comparison for tokenizers#1900\n\
         \n\
         USAGE:\n  \
           cargo bench --bench scaling_bench -- [OPTIONS]\n\
         \n\
         OPTIONS:\n  \
           -t, --tokenizer <PATH>   Path to tokenizer.json [default: {DEFAULT_TOKENIZER}]\n  \
           -w, --workers <N>        Worker threads (0 = num_cpus) [default: num_cpus]\n  \
           -b, --batch-size <N>     Batch size for encode_batch [default: 1024]\n      \
               --count <N>          Number of synthetic documents [default: 500]\n      \
               --length <N>         Bytes per synthetic document [default: 51200]\n      \
               --word-vocab-size <N>  Vocab size (repeated-words workload) [default: 2048]\n      \
               --word-len <N>       Word length (repeated-words workload) [default: 8]\n      \
               --seed <N>           PRNG seed [default: 42]\n      \
               --method <M>         worker-pool|encode-batch|both [default: both]\n      \
               --workload <W>       random-letters|repeated-words|both [default: both]\n      \
               --input <PATH>       Use file lines instead of synthetic data\n      \
               --quick              Smaller defaults (~30s on a many-core box)\n      \
               --help               Show this help\n\
         \n\
         The headline metric is encode_batch_elapsed / worker_pool_elapsed.\n\
         See https://github.com/huggingface/tokenizers/issues/1900 for context\n\
         and https://github.com/stargazerZJ/tokenizers-1900-repro for the\n\
         standalone reproducer with allocator-swap and drop-site experiments."
    );
}

// ---------------------------------------------------------------------------
// Synthetic data generators
// ---------------------------------------------------------------------------

fn generate_random_letters(count: usize, length: usize, seed: u64) -> Vec<String> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    eprintln!("[scaling_bench] generating {count} random-letters strings × {length} bytes");
    (0..count)
        .map(|_| {
            (0..length)
                .map(|_| {
                    let idx: u8 = rng.random_range(0u8..52);
                    if idx < 26 {
                        (b'a' + idx) as char
                    } else {
                        (b'A' + idx - 26) as char
                    }
                })
                .collect()
        })
        .collect()
}

fn generate_repeated_words(
    count: usize,
    length: usize,
    seed: u64,
    vocab_size: usize,
    word_len: usize,
) -> Vec<String> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    eprintln!(
        "[scaling_bench] generating {count} repeated-words strings × {length} bytes \
         (vocab={vocab_size}, word_len={word_len})"
    );
    let vocab: Vec<String> = (0..vocab_size)
        .map(|_| {
            (0..word_len)
                .map(|_| {
                    let idx: u8 = rng.random_range(0u8..52);
                    if idx < 26 {
                        (b'a' + idx) as char
                    } else {
                        (b'A' + idx - 26) as char
                    }
                })
                .collect()
        })
        .collect();

    (0..count)
        .map(|_| {
            let mut s = String::with_capacity(length + word_len + 1);
            while s.len() < length {
                if !s.is_empty() {
                    s.push(' ');
                }
                s.push_str(&vocab[rng.random_range(0..vocab.len())]);
            }
            s.truncate(length);
            s
        })
        .collect()
}

fn load_input_file(path: &std::path::Path, target_count: usize) -> Result<Vec<String>, String> {
    let data = std::fs::read_to_string(path).map_err(|e| format!("read {path:?}: {e}"))?;
    let lines: Vec<String> = data.lines().map(|s| s.to_string()).collect();
    if lines.is_empty() {
        return Err(format!("{path:?}: no lines"));
    }
    eprintln!(
        "[scaling_bench] loaded {} lines from {path:?}; cycling to reach {target_count} docs",
        lines.len()
    );
    Ok((0..target_count)
        .map(|i| lines[i % lines.len()].clone())
        .collect())
}

// ---------------------------------------------------------------------------
// Stats accumulator
// ---------------------------------------------------------------------------

struct Stats {
    docs: AtomicUsize,
    tokens: AtomicU64,
    bytes: AtomicU64,
    start: Instant,
}

impl Stats {
    fn new() -> Self {
        Self {
            docs: AtomicUsize::new(0),
            tokens: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            start: Instant::now(),
        }
    }
    fn add(&self, t: usize, b: usize) {
        self.docs.fetch_add(1, Ordering::Relaxed);
        self.tokens.fetch_add(t as u64, Ordering::Relaxed);
        self.bytes.fetch_add(b as u64, Ordering::Relaxed);
    }
    fn report(&self, label: &str) -> f64 {
        let elapsed = self.start.elapsed().as_secs_f64();
        let d = self.docs.load(Ordering::Relaxed);
        let t = self.tokens.load(Ordering::Relaxed);
        let b = self.bytes.load(Ordering::Relaxed) as f64;
        println!(
            "[{label}] docs={d} ({:.1}/s) tokens={t} ({:.1}/s) \
             bytes={:.2}MiB ({:.2}MiB/s) elapsed={:.2}s",
            d as f64 / elapsed,
            t as f64 / elapsed,
            b / (1024.0 * 1024.0),
            b / (1024.0 * 1024.0) / elapsed,
            elapsed,
        );
        elapsed
    }
}

// ---------------------------------------------------------------------------
// Methods
// ---------------------------------------------------------------------------

/// Manual rayon worker-pool: encode in parallel, consume + drop each
/// `Encoding` inside the worker closure (alloc + free on the same thread).
fn run_worker_pool(
    tokenizer: &Tokenizer,
    texts: &[String],
    num_workers: usize,
) -> Result<f64, String> {
    let stats = Stats::new();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_workers)
        .build()
        .map_err(|e| format!("rayon pool: {e}"))?;

    pool.install(|| {
        texts.par_iter().for_each(|text| {
            if let Ok(enc) = tokenizer.encode(text.as_str(), false) {
                stats.add(enc.get_ids().len(), text.len());
                // enc dropped here, on the worker thread that allocated it.
            }
        });
    });

    Ok(stats.report("worker_pool"))
}

/// Stock `Tokenizer::encode_batch` per chunk; iterate the returned vec on
/// main and drop there.
fn run_encode_batch(
    tokenizer: &Tokenizer,
    texts: &[String],
    batch_size: usize,
) -> Result<f64, String> {
    let stats = Stats::new();
    for chunk in texts.chunks(batch_size) {
        let inputs: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
        let encs = tokenizer
            .encode_batch(inputs, false)
            .map_err(|e| format!("encode_batch: {e}"))?;
        for (i, e) in encs.iter().enumerate() {
            stats.add(e.get_ids().len(), chunk[i].len());
        }
        // encs drops here, on main; this is the cross-thread free site.
    }
    Ok(stats.report("encode_batch"))
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

fn run_workload(
    label: &str,
    args: &Args,
    tokenizer: &Tokenizer,
    texts: &[String],
) -> Result<(), String> {
    println!("\n=== workload: {label} ===");
    let mut wp: Option<f64> = None;
    let mut eb: Option<f64> = None;

    if matches!(args.method, Method::WorkerPool | Method::Both) {
        wp = Some(run_worker_pool(tokenizer, texts, args.workers)?);
    }
    if matches!(args.method, Method::EncodeBatch | Method::Both) {
        eb = Some(run_encode_batch(tokenizer, texts, args.batch_size)?);
    }

    if let (Some(wp), Some(eb)) = (wp, eb) {
        println!(
            "[ratio:{label}] encode_batch / worker_pool = {:.2}× \
             ({eb:.2}s / {wp:.2}s)",
            eb / wp,
        );
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let args = Args::parse().map_err(|e| {
        eprintln!("{e}\n");
        print_help();
        e
    })?;

    println!(
        "[scaling_bench] tokenizer={:?} workers={} batch_size={} count={} length={} method={:?} workload={:?}",
        args.tokenizer, args.workers, args.batch_size, args.count, args.length, args.method, args.workload,
    );
    if !args.tokenizer.exists() {
        return Err(format!(
            "tokenizer file not found: {:?}\n\
             Run `make bench` from the tokenizers/ directory to fetch it, \
             or pass --tokenizer <path>.",
            args.tokenizer
        ));
    }

    let tokenizer =
        Tokenizer::from_file(&args.tokenizer).map_err(|e| format!("load tokenizer: {e}"))?;
    let tokenizer = Arc::new(tokenizer);

    let workloads: &[Workload] = match args.workload {
        Workload::Both => &[Workload::RandomLetters, Workload::RepeatedWords],
        single => std::slice::from_ref(match &single {
            Workload::RandomLetters => &Workload::RandomLetters,
            Workload::RepeatedWords => &Workload::RepeatedWords,
            Workload::Both => unreachable!(),
        }),
    };

    for &w in workloads {
        let (label, texts) = match (w, &args.input) {
            (_, Some(path)) => ("input-file", load_input_file(path, args.count)?),
            (Workload::RandomLetters, _) => (
                "random-letters",
                generate_random_letters(args.count, args.length, args.seed),
            ),
            (Workload::RepeatedWords, _) => (
                "repeated-words",
                generate_repeated_words(
                    args.count,
                    args.length,
                    args.seed,
                    args.word_vocab_size,
                    args.word_len,
                ),
            ),
            (Workload::Both, _) => unreachable!(),
        };
        run_workload(label, &args, &tokenizer, &texts)?;
        // If --input is set, the workload distinction is moot — only run once.
        if args.input.is_some() {
            break;
        }
    }
    Ok(())
}
