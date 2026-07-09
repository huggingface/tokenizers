//! classify::<Atoms> throughput across three MB-fixup branch regimes, to check the per-chunk
//! `if any(out == MB)` fixup branch is predictable:
//!   none     — no astral (real text) → branch ALWAYS false → should be free.
//!   astral   — wall of emoji (every chunk has MB) → branch ALWAYS true → predictable, measures fixup cost.
//!   sprinkle — BMP text with an emoji every ~24 B → branch flips irregularly → misprediction stress.
//! Run before/after the fixup change and compare. Run: cargo bench --bench classify
use fast_split::classify::{Atoms, classify, classify_scalar};
use std::hint::black_box;
use std::time::Instant;

fn ns_per_byte(text: &[u8], tags: &mut [u8]) -> f64 {
    let iters = (8_000_000 / text.len().max(1)).clamp(20, 400) as u32;
    for _ in 0..3 {
        classify::<Atoms>(text, tags);
        black_box(tags[text.len() / 2]);
    }
    let mut best = f64::INFINITY;
    for _ in 0..9 {
        let t = Instant::now();
        for _ in 0..iters {
            classify::<Atoms>(text, tags);
            black_box(tags[text.len() / 2]);
        }
        best = best.min(t.elapsed().as_nanos() as f64 / (iters as usize * text.len()) as f64);
    }
    best
}

fn main() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let english = std::fs::read_to_string(format!("{manifest}/../data/big.txt")).unwrap_or_default();
    let english: String = english.chars().take(120_000).collect();

    let astral = "😀🎉🚀🔥🌍🐍".repeat(20_000); // pure 4-byte astral → MB in every chunk

    // BMP English with an emoji inserted every ~24 bytes → some chunks have MB, some don't, irregularly.
    let mut sprinkle = String::new();
    let mut acc = 0;
    for w in english.split_inclusive(' ') {
        sprinkle.push_str(w);
        acc += w.len();
        if acc >= 24 {
            sprinkle.push('🔥');
            acc = 0;
        }
    }

    println!("{:<10} {:>8} {:>10}  {:>10}", "input", "bytes", "clsSIMD", "clsScalar");
    for (label, s) in [("none", &english), ("astral", &astral), ("sprinkle", &sprinkle)] {
        if s.is_empty() {
            println!("{label:<10} (empty — big.txt missing?)");
            continue;
        }
        let text = s.as_bytes();
        let mut tags = vec![0u8; text.len()];
        // byte-exactness guard: SIMD == scalar (exercises the astral fixup path)
        let mut sc = vec![0u8; text.len()];
        classify::<Atoms>(text, &mut tags);
        classify_scalar::<Atoms>(text, &mut sc);
        let ok = if tags == sc { "✓" } else { "✗" };
        let simd = ns_per_byte(text, &mut tags);
        let scal = {
            let iters = (8_000_000 / text.len().max(1)).clamp(20, 400) as u32;
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let t = Instant::now();
                for _ in 0..iters {
                    classify_scalar::<Atoms>(text, &mut sc);
                    black_box(sc[text.len() / 2]);
                }
                best = best.min(t.elapsed().as_nanos() as f64 / (iters as usize * text.len()) as f64);
            }
            best
        };
        println!("{label:<10} {:>8} {simd:>10.3} {scal:>10.3}  {ok}", text.len());
    }
    println!("\n(ns/byte, lower better. clsSIMD is the path with the MB-fixup branch.)");
}
