//! Per-step profile of gigatoken's encode pipeline, decomposed via its public API.
//!
//! Steps (each isolated by which public entry we call):
//!   split      = drain `PretokenizerType::pretokenize` iterator (pure pretokenization)
//!   +hash+probe= `memoized_encode_flat(pretokenize(x))` warm  − split   (key hash + cache hit + emit)
//!   added-split= `encode_with_added_tokens_flat(x)` warm       − memoized-warm (added-token matcher pass)
//!   merge(miss)= `encode_with_added_tokens_flat(x)` COLD       − warm    (cache misses → BPE merge)
//!
//! run: cargo run --release --example profile_steps
use std::hint::black_box;
use std::time::Instant;
use gigatoken_rs::load_tokenizer::hf::load_hf_bpe;

const ROOT: &str = "/Users/arthurzucker/Work/tokenizers/tokenizers";

fn best(len: usize, iters: u32, mut f: impl FnMut()) -> f64 {
    for _ in 0..2 {
        f();
    }
    let mut b = f64::INFINITY;
    for _ in 0..7 {
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        b = b.min(t.elapsed().as_nanos() as f64 / (iters as usize * len) as f64);
    }
    b
}

fn corpus(rel: &str) -> Option<String> {
    let s = std::fs::read_to_string(format!("{ROOT}/{rel}")).ok()?;
    let mut c = s.len().min(1_000_000);
    while c > 0 && !s.is_char_boundary(c) {
        c -= 1;
    }
    Some(s[..c].to_string())
}

fn main() {
    let toks: &[(&str, &str)] = &[
        ("gpt2", "data/gpt2.json"),
        ("deepseek-v4", "data/deepseek-v4.json"),
        ("llama-3", "data/llama-3-tokenizer.json"),
    ];
    let corpora: &[(&str, &str)] = &[
        ("English", "data/big.txt"),
        ("French", "atomsplit/benches/data/fr.txt"),
        ("Russian", "atomsplit/benches/data/ru.txt"),
        ("Chinese", "atomsplit/benches/data/zh.txt"),
        ("Japanese", "data/unigram_wagahaiwa_nekodearu.txt"),
        ("Korean", "atomsplit/benches/data/ko.txt"),
    ];

    println!(
        "\nns/byte per step. split=pretokenize; probe=key-hash+cache-hit+emit; added=added-token\n\
         matcher pass; merge=cache-miss BPE (cold-warm). warm=full encode all-hits, cold=first touch.\n"
    );
    println!(
        "{:<12} {:<9} {:>7} {:>7} {:>7} {:>7} {:>7} | {:>7} {:>7}",
        "tokenizer", "lang", "bytes", "split", "probe", "added", "merge", "warm", "cold"
    );

    for (tname, tpath) in toks {
        let Ok(_) = load_hf_bpe(format!("{ROOT}/{tpath}")) else {
            eprintln!("skip {tname}: load failed");
            continue;
        };
        for (label, rel) in corpora {
            let Some(text) = corpus(rel) else { continue };
            let bytes = text.as_bytes();
            let n = bytes.len();
            let iters = (20_000_000 / n).clamp(2, 50) as u32;

            // fresh tokenizer, fill cache once (cold), then everything else is warm
            let mut tok = load_hf_bpe(format!("{ROOT}/{tpath}")).unwrap();
            let pt = tok.pretokenizer_type();

            // 1) split alone — drain the pretokenizer iterator
            let t_split = best(n, iters, || {
                black_box(pt.pretokenize(black_box(bytes)).count());
            });

            // 2) cold full encode (first touch: all cache misses → merges). Timed ONCE per fresh
            //    tok, with the tokenizer LOAD excluded (load parses/​seeds ~50k vocab — not encode).
            let mut out = Vec::with_capacity(n);
            let t_cold = {
                let mut fresh = load_hf_bpe(format!("{ROOT}/{tpath}")).unwrap();
                out.clear();
                let t0 = Instant::now();
                fresh.encode_with_added_tokens_flat(bytes, &mut out);
                t0.elapsed().as_nanos() as f64 / n as f64
            };

            // 3) warm full encode (cache now populated in `tok`)
            out.clear();
            tok.encode_with_added_tokens_flat(bytes, &mut out); // populate
            let t_warm = best(n, iters, || {
                out.clear();
                tok.encode_with_added_tokens_flat(black_box(bytes), &mut out);
            });

            // 4) warm memoized (no added-split): split + hash + probe + emit
            let t_memo = best(n, iters, || {
                out.clear();
                tok.memoized_encode_flat(pt.pretokenize(black_box(bytes)), &mut out);
            });

            let t_probe = (t_memo - t_split).max(0.0);
            let t_added = (t_warm - t_memo).max(0.0);
            let t_merge = (t_cold - t_warm).max(0.0);
            println!(
                "{tname:<12} {label:<9} {n:>7} {t_split:>7.3} {t_probe:>7.3} {t_added:>7.3} {t_merge:>7.3} | {t_warm:>7.3} {t_cold:>7.3}"
            );
        }
        println!();
    }
}
