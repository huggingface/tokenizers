//! Compares the classic (naive merge) BPE encoder against the backtracking encoder on the
//! same model and inputs, reporting:
//!   - initialization time (the extra cost to build the backtracking engine),
//!   - memory footprint (the engine's added data structures),
//!   - tokenization throughput.
//!
//! It benchmarks the *model* step in isolation (normalization / pre-tokenization are shared
//! and excluded): the corpus is pre-tokenized once into byte-level words, then encoded by
//! `BPE::tokenize` with backtracking off vs on. The cache is disabled so every call does real
//! work rather than serving repeats.
//!
//! Run from the workspace root (where `data/` lives):
//!   cargo run --release --example backtracking_bench
//!   cargo run --release --example backtracking_bench -- --corpus data/big.txt --max-words 200000 --passes 5

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tk_encode::models::bpe::BPE;
use tk_encode::utils::byte_level::BYTES_CHAR_LOOKUP;
use tk_encode::Model;

/// Global allocator that tracks live (allocated-minus-freed) bytes, so we can measure the net
/// heap retained by building the backtracking engine. Transient allocations (e.g. the scratch
/// vectors in `find_hash_factor`) are freed and therefore don't count toward the delta.
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

fn mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// Byte-level encodes a word the way GPT-2 does: prepend a space (`add_prefix_space`) then map
/// every byte through the byte-level alphabet (so a space becomes "Ġ", a 2-byte char). This is
/// faithful at the byte→char level; segmentation is approximated by whitespace (the GPT-2 regex
/// would also split on punctuation), which is fine here since both encoders see identical input.
fn byte_level_word(word: &str) -> String {
    format!(" {word}")
        .bytes()
        .map(|b| BYTES_CHAR_LOOKUP[b as usize])
        .collect()
}

struct Args {
    vocab: String,
    merges: String,
    corpus: String,
    max_words: usize,
    passes: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        vocab: "data/gpt2-vocab.json".to_string(),
        merges: "data/gpt2-merges.txt".to_string(),
        corpus: "data/big.txt".to_string(),
        max_words: 200_000,
        passes: 5,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().expect("missing value for flag");
        match flag.as_str() {
            "--vocab" => a.vocab = val(),
            "--merges" => a.merges = val(),
            "--corpus" => a.corpus = val(),
            "--max-words" => a.max_words = val().parse().expect("--max-words"),
            "--passes" => a.passes = val().parse().expect("--passes"),
            other => panic!("unknown flag: {}", other),
        }
    }
    a
}

/// Encodes every word `passes` times, returning the fastest pass and the total tokens emitted
/// (used to defeat dead-code elimination and to report tokens/s).
fn time_encode(bpe: &BPE, words: &[String], passes: usize) -> (Duration, usize) {
    let mut best = Duration::MAX;
    let mut tokens = 0;
    for _ in 0..passes {
        let mut count = 0usize;
        let start = Instant::now();
        for w in words {
            let encoded = bpe.tokenize(w).unwrap();
            count += encoded.len();
            black_box(&encoded);
        }
        best = best.min(start.elapsed());
        tokens = count;
    }
    (best, tokens)
}

fn throughput_row(label: &str, elapsed: Duration, bytes: usize, words: usize, tokens: usize) {
    let secs = elapsed.as_secs_f64();
    println!(
        "  {label:<13} {:>9.1} MiB/s  {:>11.0} tok/s  {:>8.1} ns/word",
        mib(bytes) / secs,
        tokens as f64 / secs,
        elapsed.as_nanos() as f64 / words as f64,
    );
}

fn main() {
    let args = parse_args();

    // --- Corpus: pre-tokenize once into byte-level words (shared, excluded from timing). ---
    let text = std::fs::read_to_string(&args.corpus)
        .unwrap_or_else(|e| panic!("cannot read corpus {}: {}", args.corpus, e));
    let words: Vec<String> = text
        .split_whitespace()
        .take(args.max_words)
        .map(byte_level_word)
        .collect();
    let total_bytes: usize = words.iter().map(|w| w.len()).sum();
    assert!(!words.is_empty(), "corpus produced no words");

    // --- Build the classic BPE model (cache disabled so every encode does real work). ---
    let pre_bpe = live();
    let build_start = Instant::now();
    let mut bpe = BPE::from_file(&args.vocab, &args.merges)
        .cache_capacity(0)
        .build()
        .unwrap_or_else(|e| panic!("cannot load BPE: {}", e));
    let classic_build = build_start.elapsed();
    let classic_mem = live().saturating_sub(pre_bpe);

    // --- Correctness gate: never benchmark a broken encoder. ---
    bpe.set_backtracking(false);
    let oracle: Vec<_> = words.iter().take(5000).map(|w| bpe.tokenize(w).unwrap()).collect();
    let before = live();
    let init_start = Instant::now();
    bpe.set_backtracking(true);
    let init_time = init_start.elapsed();
    let engine_mem = live().saturating_sub(before);
    assert!(bpe.backtracking(), "backtracking failed to enable for this model");
    for (w, exp) in words.iter().take(5000).zip(&oracle) {
        assert_eq!(&bpe.tokenize(w).unwrap(), exp, "backtracking disagrees with naive on {w:?}");
    }

    // --- Report: setup ---
    println!("== Setup ==");
    println!("  model:   {} + {}", args.vocab, args.merges);
    println!("  corpus:  {}", args.corpus);
    println!(
        "  input:   {} byte-level words, {:.1} MiB ({:.1} bytes/word avg)",
        words.len(),
        mib(total_bytes),
        total_bytes as f64 / words.len() as f64,
    );

    // --- Report: initialization ---
    println!("\n== Initialization ==");
    println!("  classic BPE build:        {classic_build:>8.2?}   (load + parse vocab/merges)");
    println!(
        "  backtracking engine init: {init_time:>8.2?}   (find_hash_factor + Aho-Corasick + tables)"
    );

    // --- Report: memory ---
    println!("\n== Memory (net live heap) ==");
    println!("  classic BPE model:    {:>8.2} MiB", mib(classic_mem));
    println!(
        "  backtracking engine: +{:>8.2} MiB   ({:.0}% on top of the model)",
        mib(engine_mem),
        100.0 * engine_mem as f64 / classic_mem as f64,
    );
    if let Some(breakdown) = bpe.backtracking_memory_breakdown() {
        for (name, bytes) in breakdown {
            println!("      {name:<24} {:>8.2} MiB", mib(bytes));
        }
    }

    // --- Report: throughput ---
    println!(
        "\n== Throughput (single-thread, cache disabled, best of {} passes) ==",
        args.passes
    );
    bpe.set_backtracking(false);
    let (naive_t, naive_tokens) = time_encode(&bpe, &words, args.passes);
    bpe.set_backtracking(true);
    let (bt_t, bt_tokens) = time_encode(&bpe, &words, args.passes);
    assert_eq!(naive_tokens, bt_tokens, "token counts differ between encoders");

    throughput_row("classic", naive_t, total_bytes, words.len(), naive_tokens);
    throughput_row("backtracking", bt_t, total_bytes, words.len(), bt_tokens);
    println!(
        "  speedup (backtracking / classic): {:.2}x",
        naive_t.as_secs_f64() / bt_t.as_secs_f64()
    );
}
