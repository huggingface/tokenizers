//! cl100k_base pretokenization on BIG real text (Wikipedia articles / big.txt), per language — no
//! tiny-repeated snippets, so L1 caching doesn't flatter the numbers. Per corpus: classify (SIMD vs
//! scalar) and fsm (scalar vs SIMD run-end) in ns/byte, plus the full pipeline vs onig AND vs
//! fancy-regex (the two regex engines). Byte-exactness of the pipeline vs onig is checked (✓/✗).
//!
//! Data: `../data/big.txt` (English) + `../data/unigram_wagahaiwa_nekodearu.txt` (Japanese) ship with
//! the repo; the rest are fetched by `benches/data/fetch.py` (gitignored). Missing files are skipped.
//!
//! Run: cargo bench --bench cl100k
use fast_split::classify::{Atoms, classify, classify_scalar};
use fast_split::fsm::{Span, fsm_cl100k, fsm_cl100k_simd};
use fancy_regex::Regex as Fancy;
use onig::Regex;
use std::hint::black_box;
use std::time::Instant;

const CL100K: &str = concat!(
    r"'(?i:[sdmt]|ll|ve|re)",
    r"|[^\r\n\p{L}\p{N}]?\p{L}+",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]|\s+(?!\S)|\s+",
);

// (label, path relative to CARGO_MANIFEST_DIR). Big real text, one file per language.
const CORPORA: &[(&str, &str)] = &[
    ("English", "../data/big.txt"),
    ("French", "benches/data/fr.txt"),
    ("Russian", "benches/data/ru.txt"),
    ("Greek", "benches/data/el.txt"),
    ("Hebrew", "benches/data/he.txt"),
    ("Arabic", "benches/data/ar.txt"),
    ("Hindi", "benches/data/hi.txt"),
    ("Thai", "benches/data/th.txt"),
    ("Chinese", "benches/data/zh.txt"),
    ("Japanese", "../data/unigram_wagahaiwa_nekodearu.txt"),
    ("Korean", "benches/data/ko.txt"),
];

// MIN over TRIALS timed loops — the fastest trial had the least CPU contention, so it's the truest
// estimate and is robust to thermal throttling / background load (which only ever make a trial slower).
fn ns_per_byte<F: FnMut() -> usize>(len: usize, iters: u32, mut f: F) -> f64 {
    const TRIALS: u32 = 7;
    for _ in 0..3 {
        black_box(f()); // warm
    }
    let mut best = f64::INFINITY;
    for _ in 0..TRIALS {
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
    let re = Regex::new(CL100K).expect("cl100k regex");
    let fancy = Fancy::new(CL100K).expect("fancy cl100k regex");
    let manifest = env!("CARGO_MANIFEST_DIR");

    println!(
        "{:<10} {:>7} {:>4} {:>5}  {:>8} {:>8} | {:>8} {:>8} | {:>8} {:>8} | {:>7} {:>7}",
        "lang", "bytes", "b/ch", "b/tok", "clsSIMD", "clsScal", "fsmScal", "fsmSIMD", "onig", "fancy", "vsOnig", "vsFncy"
    );

    for (label, rel) in CORPORA {
        let raw = match std::fs::read_to_string(format!("{manifest}/{rel}")) {
            Ok(s) if !s.trim().is_empty() => s,
            _ => {
                println!("{label:<10}  (skipped — {rel} missing; run benches/data/fetch.py)");
                continue;
            }
        };
        // UNIFORM cap (char boundary): every language same byte size → equal cache behaviour, so
        // per-byte compute is comparable across scripts (was unfair: English 1 MB vs Chinese 21 KB).
        // 180 KB > L1, and all corpora have ≥180 KB of real text.
        let mut c = raw.len().min(180_000);
        while c > 0 && !raw.is_char_boundary(c) {
            c -= 1;
        }
        let corpus = &raw[..c];
        let text = corpus.as_bytes();
        let n = text.len();
        let bpc = n as f64 / corpus.chars().count() as f64;
        let iters = (4_000_000 / n).clamp(3, 150) as u32; // × TRIALS inside ns_per_byte

        let onig_spans: Vec<Span> = re.find_iter(corpus).map(|(s, e)| (s as u32, e as u32)).collect();
        let mut tags = vec![0u8; n];
        let mut tsc = vec![0u8; n];
        classify::<Atoms>(text, &mut tags);
        let mut sc = Vec::new();
        fsm_cl100k(text, &tags, &mut sc);
        let ok = if sc == onig_spans { "✓" } else { "✗" };
        let btok = n as f64 / sc.len().max(1) as f64; // bytes per token = density (drives per-token cost)

        let mut buf = Vec::with_capacity(sc.len());
        let cls_simd = ns_per_byte(n, iters, || {
            classify::<Atoms>(text, &mut tags);
            tags[n / 2] as usize
        });
        let cls_scal = ns_per_byte(n, iters, || {
            classify_scalar::<Atoms>(text, &mut tsc);
            tsc[n / 2] as usize
        });
        classify::<Atoms>(text, &mut tags);
        let fsm_scal = ns_per_byte(n, iters, || {
            buf.clear();
            fsm_cl100k(text, &tags, &mut buf);
            buf.len()
        });
        let fsm_simd = ns_per_byte(n, iters, || {
            buf.clear();
            fsm_cl100k_simd(text, &tags, &mut buf);
            buf.len()
        });
        let onig_ns = ns_per_byte(n, iters, || re.find_iter(corpus).count());
        let fancy_ns = ns_per_byte(n, iters, || fancy.find_iter(corpus).count());

        let pipe = cls_simd + fsm_simd; // SIMD classify + SIMD fsm — the full pipeline
        println!(
            "{label:<10} {n:>7} {bpc:>4.1} {btok:>5.1}  {cls_simd:>8.3} {cls_scal:>8.3} | {fsm_scal:>8.3} {fsm_simd:>8.3} | {onig_ns:>8.2} {fancy_ns:>8.2} | {:>6.1}x {:>6.1}x {ok}",
            onig_ns / pipe,
            fancy_ns / pipe
        );
    }
    println!("\n(ns/byte, lower better. pipeline = SIMD classify + SIMD fsm; vs onig / vs fancy on that.)");
}
