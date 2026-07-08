//! cl100k_base pretokenization: **SIMD** (`classify::<Atoms>` + `fsm_cl100k`) vs **scalar**
//! (`classify_scalar::<Atoms>` + `fsm_cl100k`) vs the **onig** cl100k regex (the reference).
//! Whole-buffer tokenize; reports ns/byte, MB/s, and speed-up over onig. Also a byte-exactness gate:
//! all three must produce identical spans, else the bench panics.
//!
//! Run: cargo bench --bench cl100k
use fast_split::classify::{Atoms, classify, classify_scalar};
use fast_split::fsm::{Span, fsm_cl100k};
use onig::Regex;
use std::hint::black_box;
use std::time::Instant;

// The cl100k_base pattern (tiktoken). `?`/`+` are equivalent to tiktoken's possessive `?+`/`++` here
// because the prefix class excludes letters and the "other" run excludes newlines (no backtrack helps).
const CL100K: &str = concat!(
    r"'(?i:[sdmt]|ll|ve|re)",
    r"|[^\r\n\p{L}\p{N}]?\p{L}+",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]|\s+(?!\S)|\s+",
);

fn ns_per_byte<F: FnMut() -> usize>(len: usize, iters: u32, mut f: F) -> f64 {
    for _ in 0..3 {
        black_box(f()); // warm up
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
    // A realistic mix: prose, a contraction, code, punctuation/symbols, accented latin, a URL, digits.
    let unit = "The quick brown fox (x=42) jumps! Don't stop; e.g. 3.14 * 2 = 6.28...\n\tprintf(\"%d\\n\", i++); café naïve — 100% sure. https://example.com/path?q=1&r=2\n";
    let corpus = unit.repeat(500);
    let text = corpus.as_bytes();
    let n = text.len();

    // ── byte-exactness gate: SIMD == scalar == onig ──
    let re = Regex::new(CL100K).expect("cl100k regex");
    let onig: Vec<Span> = re.find_iter(&corpus).map(|(s, e)| (s as u32, e as u32)).collect();
    let mut tags = vec![0u8; n];
    let (mut simd, mut scalar) = (Vec::new(), Vec::new());
    classify::<Atoms>(text, &mut tags);
    fsm_cl100k(text, &tags, &mut simd);
    classify_scalar::<Atoms>(text, &mut tags);
    fsm_cl100k(text, &tags, &mut scalar);
    assert_eq!(simd, scalar, "SIMD classify != scalar classify");
    assert_eq!(simd, onig, "fast_split cl100k != onig cl100k (byte-exact gate)");
    println!("corpus {n} bytes, {} tokens — SIMD == scalar == onig ✓\n", simd.len());

    // ── timing (whole-buffer tokenize, reusing scratch buffers) ──
    let iters = 200;
    let mut buf = Vec::with_capacity(simd.len());
    let simd_ns = ns_per_byte(n, iters, || {
        classify::<Atoms>(text, &mut tags);
        buf.clear();
        fsm_cl100k(text, &tags, &mut buf);
        buf.len()
    });
    let scalar_ns = ns_per_byte(n, iters, || {
        classify_scalar::<Atoms>(text, &mut tags);
        buf.clear();
        fsm_cl100k(text, &tags, &mut buf);
        buf.len()
    });
    let onig_ns = ns_per_byte(n, iters, || re.find_iter(&corpus).count());

    let mbps = |ns: f64| 1000.0 / ns;
    println!("{:<26} {:>9} {:>10} {:>9}", "impl", "ns/byte", "MB/s", "vs onig");
    for (name, ns) in [
        ("SIMD (classify+fsm)", simd_ns),
        ("scalar (classify+fsm)", scalar_ns),
        ("onig regex", onig_ns),
    ] {
        println!("{name:<26} {ns:>9.3} {:>10.1} {:>8.1}x", mbps(ns), onig_ns / ns);
    }
}
