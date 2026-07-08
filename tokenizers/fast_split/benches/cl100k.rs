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

fn ns_per_byte<F: FnMut() -> usize>(len: usize, iters: u32, mut f: F) -> f64 {
    for _ in 0..3 {
        black_box(f());
    }
    let t = Instant::now();
    let mut acc = 0usize;
    for _ in 0..iters {
        acc = acc.wrapping_add(f());
    }
    black_box(acc);
    t.elapsed().as_nanos() as f64 / (iters as usize * len) as f64
}

fn main() {
    let re = Regex::new(CL100K).expect("cl100k regex");
    let fancy = Fancy::new(CL100K).expect("fancy cl100k regex");
    let manifest = env!("CARGO_MANIFEST_DIR");

    println!(
        "{:<10} {:>8} {:>4}  {:>8} {:>8} | {:>8} {:>8} | {:>8} {:>8} | {:>7} {:>7}",
        "lang", "bytes", "b/ch", "clsSIMD", "clsScal", "fsmScal", "fsmSIMD", "onig", "fancy", "vsOnig", "vsFncy"
    );

    for (label, rel) in CORPORA {
        let raw = match std::fs::read_to_string(format!("{manifest}/{rel}")) {
            Ok(s) if !s.trim().is_empty() => s,
            _ => {
                println!("{label:<10}  (skipped — {rel} missing; run benches/data/fetch.py)");
                continue;
            }
        };
        // cap to ~1 MB on a char boundary: big enough to defeat L1, keeps the bench quick.
        let mut c = raw.len().min(1_000_000);
        while c > 0 && !raw.is_char_boundary(c) {
            c -= 1;
        }
        let corpus = &raw[..c];
        let text = corpus.as_bytes();
        let n = text.len();
        let bpc = n as f64 / corpus.chars().count() as f64;
        let iters = (15_000_000 / n).clamp(5, 400) as u32;

        let onig_spans: Vec<Span> = re.find_iter(corpus).map(|(s, e)| (s as u32, e as u32)).collect();
        let mut tags = vec![0u8; n];
        let mut tsc = vec![0u8; n];
        classify::<Atoms>(text, &mut tags);
        let mut sc = Vec::new();
        fsm_cl100k(text, &tags, &mut sc);
        let ok = if sc == onig_spans { "✓" } else { "✗" };

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
            "{label:<10} {n:>8} {bpc:>4.1}  {cls_simd:>8.3} {cls_scal:>8.3} | {fsm_scal:>8.3} {fsm_simd:>8.3} | {onig_ns:>8.2} {fancy_ns:>8.2} | {:>6.1}x {:>6.1}x {ok}",
            onig_ns / pipe,
            fancy_ns / pipe
        );
    }
    println!("\n(ns/byte, lower better. pipeline = SIMD classify + SIMD fsm; vs onig / vs fancy on that.)");
}
