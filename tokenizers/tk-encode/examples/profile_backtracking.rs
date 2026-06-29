//! Sustained backtracking-only encode loop, for profiling with `samply` (or perf/dtrace).
//! Backtracking is the only thing running here, so the flamegraph is dominated by
//! `tokenize → tokenize_backtracking → encode_run → {find_overlapping_iter, is_valid_token_pair,
//! get_merged, peel}`.
//!
//!   cargo build --profile profiling -p tk-encode --example profile_backtracking
//!   samply record ./target/profiling/examples/profile_backtracking
//!
//! Env: PASSES (default 50) — number of times to encode the corpus. DATA (default "data").

use std::hint::black_box;

use tk_encode::models::bpe::BPE;
use tk_encode::utils::byte_level::BYTES_CHAR_LOOKUP;
use tk_encode::Model;

fn byte_level_word(word: &str) -> String {
    format!(" {word}")
        .bytes()
        .map(|b| BYTES_CHAR_LOOKUP[b as usize])
        .collect()
}

fn main() {
    let dir = std::env::var("DATA").unwrap_or_else(|_| "data".to_string());
    let passes: usize = std::env::var("PASSES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    let mut bpe = BPE::from_file(&format!("{dir}/gpt2-vocab.json"), &format!("{dir}/gpt2-merges.txt"))
        .cache_capacity(0)
        .build()
        .expect("load gpt2 BPE");
    bpe.set_backtracking(true);
    assert!(bpe.backtracking(), "backtracking failed to enable");

    let text = std::fs::read_to_string(format!("{dir}/big.txt")).expect("read big.txt");
    let words: Vec<String> = text
        .split_whitespace()
        .take(200_000)
        .map(byte_level_word)
        .collect();

    for w in &words {
        black_box(bpe.tokenize(w).unwrap());
    }

    let mut tokens = 0usize;
    for _ in 0..passes {
        for w in &words {
            tokens += black_box(bpe.tokenize(w).unwrap()).len();
        }
    }
    eprintln!(
        "encoded {tokens} tokens over {passes} passes of {} words",
        words.len()
    );
}
