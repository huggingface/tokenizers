//! End-to-end phase-split benchmark for the backtracking BPE encoder vs naive merges, on a
//! real tokenizer (normalizer + pre-tokenizer + BPE), across different kinds of text.
//!
//! Backtracking only changes the **BPE-merge** phase. This bench measures the three pipeline
//! phases separately — normalize / pre-tokenize / merge — so we can see how big the merge
//! phase actually is, and therefore how much a merge-only speedup moves end-to-end throughput.
//! The BPE cache is disabled (`resize_cache(0)`) so the merge phase reflects real compute, not
//! cache hits — the most generous case for backtracking (production caching shrinks it further).
//!
//!   cargo build --release -p tk-encode --example phase_bench
//!   cargo run --release -p tk-encode --example phase_bench -- \
//!       --tokenizer data/llama-3-tokenizer.json --passes 3 \
//!       data/big.txt data/unigram_wagahaiwa_nekodearu.txt data/agentic-traces.txt

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tk_encode::models::ModelWrapper;
use tk_encode::tokenizer::{OffsetReferential, OffsetType};
use tk_encode::{Model, Tokenizer};

struct CountingAlloc;
static LIVE: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            LIVE.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}
#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn live() -> usize {
    LIVE.load(Ordering::Relaxed)
}
fn mib(b: usize) -> f64 {
    b as f64 / (1024.0 * 1024.0)
}

#[derive(Default, Clone, Copy)]
struct Phases {
    normalize: Duration,
    pretokenize: Duration,
    merge: Duration,
    tokens: usize,
}
impl Phases {
    fn total(&self) -> Duration {
        self.normalize + self.pretokenize + self.merge
    }
}

/// One pass over the docs, timing each phase. Backtracking state is whatever is currently set
/// on the model.
fn one_pass(tok: &Tokenizer, docs: &[String]) -> Phases {
    let mut p = Phases::default();
    let model = tok.get_model();
    for doc in docs {
        let t0 = Instant::now();
        let normalized = tok.do_normalize(doc.as_str()).unwrap();
        p.normalize += t0.elapsed();

        let t1 = Instant::now();
        let pretok = tok.do_pre_tokenize(normalized).unwrap();
        // Collect the splits here so the merge phase times only `tokenize`, not get_splits.
        let pieces: Vec<&str> = pretok
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, _, _)| s)
            .collect();
        p.pretokenize += t1.elapsed();

        let t2 = Instant::now();
        for piece in &pieces {
            p.tokens += black_box(model.tokenize(piece).unwrap()).len();
        }
        p.merge += t2.elapsed();
    }
    p
}

/// Best-of-`passes` by total time, returning that pass's phase split.
fn measure(tok: &Tokenizer, docs: &[String], passes: usize) -> Phases {
    let mut best = one_pass(tok, docs);
    for _ in 1..passes {
        let p = one_pass(tok, docs);
        if p.total() < best.total() {
            best = p;
        }
    }
    best
}

fn set_backtracking(tok: &mut Tokenizer, on: bool) {
    if let ModelWrapper::BPE(bpe) = tok.get_model_mut() {
        bpe.set_backtracking(on);
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn report(name: &str, bytes: usize, naive: Phases, bt: Phases) {
    let pct = |d: Duration, p: Phases| 100.0 * d.as_secs_f64() / p.total().as_secs_f64();
    let mibps = |p: Phases| mib(bytes) / p.total().as_secs_f64();
    println!("\n===== {name}  ({:.1} MiB, {} tokens) =====", mib(bytes), naive.tokens);
    println!("  phase          naive          backtracking     (naive % of total)");
    println!(
        "  normalize    {:>8.1} ms     {:>8.1} ms      {:>5.1}%",
        ms(naive.normalize), ms(bt.normalize), pct(naive.normalize, naive)
    );
    println!(
        "  pretokenize  {:>8.1} ms     {:>8.1} ms      {:>5.1}%",
        ms(naive.pretokenize), ms(bt.pretokenize), pct(naive.pretokenize, naive)
    );
    println!(
        "  merge        {:>8.1} ms     {:>8.1} ms      {:>5.1}%   <- only phase backtracking changes",
        ms(naive.merge), ms(bt.merge), pct(naive.merge, naive)
    );
    println!(
        "  TOTAL        {:>8.1} ms     {:>8.1} ms",
        ms(naive.total()), ms(bt.total())
    );
    println!(
        "  throughput   {:>8.1} MiB/s   {:>8.1} MiB/s",
        mibps(naive), mibps(bt)
    );
    println!(
        "  merge speedup: {:.2}x   |   END-TO-END speedup: {:.2}x",
        naive.merge.as_secs_f64() / bt.merge.as_secs_f64(),
        naive.total().as_secs_f64() / bt.total().as_secs_f64(),
    );
}

fn main() {
    let mut tokenizer_path = "data/llama-3-tokenizer.json".to_string();
    let mut passes = 3usize;
    let mut max_bytes = 5 * 1024 * 1024usize;
    let mut corpora: Vec<String> = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--tokenizer" => tokenizer_path = it.next().unwrap(),
            "--passes" => passes = it.next().unwrap().parse().unwrap(),
            "--max-bytes" => max_bytes = it.next().unwrap().parse().unwrap(),
            other => corpora.push(other.to_string()),
        }
    }
    if corpora.is_empty() {
        corpora = vec![
            "data/big.txt".into(),
            "data/unigram_wagahaiwa_nekodearu.txt".into(),
        ];
    }

    let mut tok = Tokenizer::from_file(&tokenizer_path).expect("load tokenizer");
    assert!(
        matches!(tok.get_model(), ModelWrapper::BPE(_)),
        "phase_bench expects a BPE tokenizer"
    );
    // Drop the BPE cache entirely so merge timing is the raw algorithm (no cache machinery).
    if let ModelWrapper::BPE(bpe) = tok.get_model_mut() {
        bpe.disable_cache();
    }

    println!("tokenizer: {tokenizer_path}");

    // --- Init time + engine memory (build the backtracking engine once) ---
    set_backtracking(&mut tok, false);
    let before = live();
    let t0 = Instant::now();
    set_backtracking(&mut tok, true);
    let init = t0.elapsed();
    let engine_mem = live().saturating_sub(before);
    let breakdown = match tok.get_model() {
        ModelWrapper::BPE(bpe) => bpe.backtracking_memory_breakdown(),
        _ => None,
    };
    println!("\n== Backtracking engine cost (one-time) ==");
    println!("  init:   {init:.2?}");
    println!("  memory: +{:.2} MiB", mib(engine_mem));
    if let Some(b) = breakdown {
        for (n, bytes) in b {
            println!("      {n:<22} {:>7.2} MiB", mib(bytes));
        }
    }

    for path in &corpora {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("skip {path}: {e}");
                continue;
            }
        };
        let mut docs: Vec<String> = Vec::new();
        let mut total = 0usize;
        for line in text.lines() {
            if line.is_empty() {
                continue;
            }
            total += line.len();
            docs.push(line.to_string());
            if total >= max_bytes {
                break;
            }
        }
        let bytes: usize = docs.iter().map(|d| d.len()).sum();

        set_backtracking(&mut tok, false);
        let _ = one_pass(&tok, &docs); // warm
        let naive = measure(&tok, &docs, passes);

        set_backtracking(&mut tok, true);
        let _ = one_pass(&tok, &docs); // warm
        let bt = measure(&tok, &docs, passes);

        let name = std::path::Path::new(path)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.clone());
        report(&name, bytes, naive, bt);
    }
}
