//! Class-family pre-tokenizers via the kept path: SIMD `classify::<Atoms>` + the no-push NEON
//! boundary extractor `class_runs_into` (movemask + homogeneous-chunk early-out, writing spans into a
//! preallocated slice). classify and each fsm are timed separately, per language.
//!
//! Run: cargo bench --bench class_runs
use fast_split::classify::{classify, mask, Atoms};
use fast_split::fsm::class_runs_into;
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
        "{:<10} {:>7}  {:>8}  {:>8} {:>8} {:>8} {:>8} {:>8}",
        "lang", "bytes", "classify", "WSsplit", "Punct", "Digits", "WS\\w", "Bert"
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
        let text = raw[..c].as_bytes();
        let n = text.len();
        let iters = (4_000_000 / n).clamp(3, 150) as u32;
        let mut tags = vec![0u8; n];
        let mut out = vec![(0u32, 0u32); n];
        classify::<Atoms>(text, &mut tags);

        let cls = ns_per_byte(n, iters, || {
            classify::<Atoms>(text, &mut tags);
            n
        });
        classify::<Atoms>(text, &mut tags);
        macro_rules! fsm {
            ($d:expr, $i:expr, $a:expr) => {
                ns_per_byte(n, iters, || class_runs_into::<$d, $i, $a>(text, &tags, &mut out))
            };
        }
        let wss = fsm!({ mask::WS }, 0, 0);
        let pun = fsm!(0, { mask::PUNCT }, 0);
        let dig = fsm!(0, 0, { mask::NUMERIC });
        let ws = fsm!({ mask::WS }, 0, { mask::WORD });
        let bert = fsm!({ mask::WS }, { mask::PUNCT }, 0);
        println!(
            "{label:<10} {n:>7}  {cls:>8.3}  {wss:>8.3} {pun:>8.3} {dig:>8.3} {ws:>8.3} {bert:>8.3}"
        );
    }
    println!(
        "\n(ns/byte, lower better. classify = SIMD classify::<Atoms>; the rest = class_runs_into fsm\n \
         (NEON movemask boundary-extract + early-out, no-push) for each class-family pre-tokenizer.)"
    );
}
