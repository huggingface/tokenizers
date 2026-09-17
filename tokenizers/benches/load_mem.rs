//! Heap-allocation benchmark: profiles the heap (peak live bytes and total
//! churn) via `dhat` while `Tokenizer::from_file` runs, to measure load memory.
//! Set `LOAD_MEM_TOKENIZER` / `LOAD_MEM_ITERS` to override the file and loop count.

use std::hint::black_box;

use tokenizers::Tokenizer;

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn main() {
    let tokenizer_path = std::env::var("LOAD_MEM_TOKENIZER")
        .unwrap_or_else(|_| "data/llama-3-tokenizer.json".to_string());
    let iters: usize = std::env::var("LOAD_MEM_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    if !std::path::Path::new(&tokenizer_path).exists() {
        eprintln!("load_mem: tokenizer file not found: {tokenizer_path}");
        eprintln!();
        eprintln!("Pass a path to any `tokenizer.json` via LOAD_MEM_TOKENIZER:");
        eprintln!("  LOAD_MEM_TOKENIZER=<tokenizer.json> cargo bench --bench load_mem");
        eprintln!();
        eprintln!("Or fetch the default llama-3 tokenizer (a large 128k-vocab BPE):");
        eprintln!("  make data       # downloads data/llama-3-tokenizer.json");
        std::process::exit(2);
    }

    let profiler = dhat::Profiler::new_heap();

    for _ in 0..iters {
        let tok = Tokenizer::from_file(&tokenizer_path)
            .unwrap_or_else(|e| panic!("failed to load {}: {}", tokenizer_path, e));
        black_box(&tok);
        drop(tok);
    }

    let stats = dhat::HeapStats::get();
    drop(profiler);

    println!();
    println!("load_mem: {tokenizer_path}  (iters={iters})");
    println!(
        "  peak  : {:>8.2} MiB  in {:>9} blocks   (max live, t-gmax)",
        mib(stats.max_bytes as u64),
        stats.max_blocks
    );
    println!(
        "  total : {:>8.2} MiB  in {:>9} blocks   (churn over region)",
        mib(stats.total_bytes),
        stats.total_blocks
    );
    println!("  wrote dhat-heap.json (open in https://nnethercote.github.io/dh_view/dh_view.html)");
}
