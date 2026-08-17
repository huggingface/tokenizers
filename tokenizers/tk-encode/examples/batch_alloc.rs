//! Counts allocations and times `encode` for the three workload shapes, so the
//! per-document malloc traffic is a number rather than an argument.
//!
//! The glibc arena contention this is chasing is Linux-specific, but the
//! allocation *count* is not — that is the thing being reduced, and it is
//! measurable anywhere.
//!
//! usage: batch_alloc <tokenizer.json> <corpus.txt> [threads]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use tk_encode::Tokenizer;
use tk_encode::pipeline::PipelineTokenizer;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static FREES: AtomicUsize = AtomicUsize::new(0);
static COUNTING: AtomicUsize = AtomicUsize::new(0);

struct Counting;

// SAFETY: every method forwards to `System` with the same layout it was given;
// the counters are incidental and never affect the returned pointer.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) == 1 {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        if COUNTING.load(Ordering::Relaxed) == 1 {
            FREES.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) == 1 {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(p, l, new) }
    }
}

#[global_allocator]
static A: Counting = Counting;

fn main() {
    let mut args = std::env::args().skip(1);
    let tok_path = args
        .next()
        .expect("usage: batch_alloc <tokenizer.json> <corpus.txt> [threads]");
    let corpus_path = args
        .next()
        .expect("usage: batch_alloc <tokenizer.json> <corpus.txt> [threads]");
    if let Some(t) = args.next() {
        tk_encode::utils::parallelism::set_num_threads(
            t.parse().expect("threads must be a number"),
        );
    }

    let legacy = Tokenizer::from_file(&tok_path).expect("load tokenizer.json");
    let pipe = PipelineTokenizer::try_from(&legacy).expect("build pipeline");
    let text = std::fs::read_to_string(&corpus_path).expect("read corpus");

    // `batch`: one document per line — the shape that regresses with threads.
    let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
    // `mdoc`: lines grouped into ~8 KiB documents.
    let mut docs: Vec<String> = Vec::new();
    let mut cur = String::new();
    for l in &lines {
        cur.push_str(l);
        cur.push('\n');
        if cur.len() >= 8 * 1024 {
            docs.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        docs.push(cur);
    }

    let threads = tk_encode::utils::parallelism::num_threads();
    println!(
        "threads={threads}  batch={} tiny docs  mdoc={} docs  bytes={}",
        lines.len(),
        docs.len(),
        text.len()
    );

    for (name, inputs) in [
        (
            "batch",
            lines.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        ),
        ("mdoc", docs.clone()),
    ] {
        let bytes: usize = inputs.iter().map(|s| s.len()).sum();
        // Warm the pool and any lazily built state before counting.
        let _ = pipe.encode(inputs.clone(), false).wait().expect("warmup");

        // `encode` takes the batch by value, so the clone is unavoidable — but it
        // is the harness's cost, not the tokenizer's. Build it BEFORE the counter
        // starts, or every document is charged one extra String allocation and the
        // timing includes a full copy of the corpus.
        let owned = inputs.clone();

        ALLOCS.store(0, Ordering::Relaxed);
        FREES.store(0, Ordering::Relaxed);
        COUNTING.store(1, Ordering::Relaxed);
        let t0 = Instant::now();
        let out = pipe.encode(owned, false).wait().expect("encode");
        let dt = t0.elapsed();
        COUNTING.store(0, Ordering::Relaxed);

        let n = out.len();
        let toks: usize = out.iter().map(|e| e.len()).sum();
        let allocs = ALLOCS.load(Ordering::Relaxed);
        println!(
            "  {name:6} {n:7} docs  {toks:9} tok  {:8.2} ms  {:8.1} MiB/s  \
             allocs={allocs:9}  ({:.2} per doc)",
            dt.as_secs_f64() * 1e3,
            bytes as f64 / 1024.0 / 1024.0 / dt.as_secs_f64(),
            allocs as f64 / n as f64,
        );
    }
}
