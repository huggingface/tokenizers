//! WhitespaceSplit: SIMD boundary-detection (`whitespace_split_simd`) vs the scalar generic core
//! (`whitespace_split_scalar`), both checked byte-for-byte against a `char::is_whitespace` ground
//! truth. Whitespace-split is the simplest pre-tokenizer (one class, Removed) → the cleanest SIMD win.
//!
//! Run: cargo bench --bench wssplit
use fast_split::classify::{Atoms, classify};
use fast_split::fsm::{Span, whitespace_split_scalar, whitespace_split_simd};
use std::hint::black_box;
use std::time::Instant;

const CORPORA: &[(&str, &str)] = &[
    ("English", "../data/big.txt"),
    ("French", "benches/data/fr.txt"),
    ("Russian", "benches/data/ru.txt"),
    ("Greek", "benches/data/el.txt"),
    ("Arabic", "benches/data/ar.txt"),
    ("Hindi", "benches/data/hi.txt"),
    ("Thai", "benches/data/th.txt"),
    ("Chinese", "benches/data/zh.txt"),
    ("Japanese", "../data/unigram_wagahaiwa_nekodearu.txt"),
    ("Korean", "benches/data/ko.txt"),
];

// ground truth: WhitespaceSplit(Removed) = maximal runs of non-`char::is_whitespace`.
fn ws_ref(s: &str) -> Vec<Span> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    for (i, c) in s.char_indices() {
        if c.is_whitespace() {
            if let Some(st) = start.take() {
                out.push((st as u32, i as u32));
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(st) = start {
        out.push((st as u32, s.len() as u32));
    }
    out
}

fn ns_per_byte<F: FnMut() -> usize>(len: usize, iters: u32, mut f: F) -> f64 {
    for _ in 0..3 {
        black_box(f());
    }
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let t = Instant::now();
        let mut acc = 0usize;
        for _ in 0..iters {
            acc = acc.wrapping_add(f());
        }
        black_box(acc);
        best = best.min(t.elapsed().as_nanos() as f64 / (iters as usize * len) as f64);
    }
    best
}

fn main() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    println!(
        "{:<10} {:>7} {:>5}  {:>8} {:>8} | {:>7} {:>6}",
        "lang", "bytes", "b/tok", "simd", "scalar", "speedup", "parity"
    );

    for (label, rel) in CORPORA {
        let raw = match std::fs::read_to_string(format!("{manifest}/{rel}")) {
            Ok(s) if !s.trim().is_empty() => s,
            _ => {
                println!("{label:<10}  (skipped — {rel} missing)");
                continue;
            }
        };
        let mut c = raw.len().min(180_000);
        while c > 0 && !raw.is_char_boundary(c) {
            c -= 1;
        }
        let corpus = &raw[..c];
        let text = corpus.as_bytes();
        let n = text.len();
        let iters = (4_000_000 / n).clamp(3, 150) as u32;

        let mut tags = vec![0u8; n];
        classify::<Atoms>(text, &mut tags);

        // parity: simd == scalar == char::is_whitespace ground truth
        let reference = ws_ref(corpus);
        let (mut a, mut b) = (Vec::new(), Vec::new());
        whitespace_split_scalar(text, &tags, &mut a);
        whitespace_split_simd(text, &tags, &mut b);
        let parity = if a == reference && b == reference {
            "✓"
        } else {
            if a != reference {
                eprintln!("  {label}: scalar != ref ({} vs {} tokens)", a.len(), reference.len());
            }
            if b != reference {
                let k = a.iter().zip(&b).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
                eprintln!("  {label}: simd != ref @tok {k}: simd={:?} scal={:?}", b.get(k), a.get(k));
            }
            "✗"
        };
        let btok = n as f64 / a.len().max(1) as f64;

        let simd = ns_per_byte(n, iters, || {
            b.clear();
            whitespace_split_simd(text, &tags, &mut b);
            b.len()
        });
        let scal = ns_per_byte(n, iters, || {
            a.clear();
            whitespace_split_scalar(text, &tags, &mut a);
            a.len()
        });

        println!(
            "{label:<10} {n:>7} {btok:>5.1}  {simd:>8.3} {scal:>8.3} | {:>6.2}x {parity:>6}",
            scal / simd
        );
    }
    println!("\n(ns/byte, lower better. simd = whitespace_split_simd, scalar = generic fsm_split core.)");
}
