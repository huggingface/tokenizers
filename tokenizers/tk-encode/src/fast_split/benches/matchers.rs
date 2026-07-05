//! Custom-harness bench (no criterion): bitmap matchers vs the `unicode_categories` oracle.
//! Run with:  cargo bench --features unicode        (rtk strips stdout → use `rtk proxy` prefix)
use fast_split::matchers::{Digit, FastLetter, FastNumber, Letter, Matcher};
use std::hint::black_box;
use std::time::Instant;

/// ns/byte to scan one run from pos 0. black_box the input each call and accumulate the
/// result so LLVM can't hoist the (pure) scan out of the loop and report a fictitious 0.
fn time<F: Fn() -> usize>(bytes_len: usize, iters: u32, f: F) -> f64 {
    for _ in 0..3 {
        black_box(f());
    }
    let t = Instant::now();
    let mut acc = 0usize;
    for _ in 0..iters {
        acc = acc.wrapping_add(f());
    }
    black_box(acc);
    t.elapsed().as_nanos() as f64 / (iters as usize * bytes_len) as f64
}

/// Assert the two scanners agree (byte-exactness), then print both throughputs + speedup.
fn compare(
    name: &str,
    bytes: &[u8],
    iters: u32,
    fast: impl Fn() -> usize,
    oracle: impl Fn() -> usize,
) {
    assert_eq!(
        fast(),
        oracle(),
        "{name}: bitmap and unicode oracle disagree"
    );
    let f = time(bytes.len(), iters, &fast);
    let o = time(bytes.len(), iters, &oracle);
    println!("\n{name}  ({} bytes)", bytes.len());
    println!("  bitmap  : {f:.3} ns/byte ({:>6.0} MB/s)", 1000.0 / f);
    println!(
        "  unicode : {o:.3} ns/byte ({:>6.0} MB/s)  speedup {:.2}x",
        1000.0 / o,
        o / f
    );
}

fn main() {
    let iters = 2000u32;

    // Letters: all-letter, multi-script (1-byte ASCII, 2-byte Greek/Cyrillic/Arabic,
    // 3-byte CJK/Kana/Hangul), no separators → both scan the whole buffer from pos 0.
    let letters = "HelloΑλφαПриветمرحبا你好世界こんにちは한국어café".repeat(200);
    let lb = letters.as_bytes();
    compare(
        "FastLetter vs Letter",
        lb,
        iters,
        || FastLetter::run_end(black_box(lb), 0),
        || Letter::run_end(black_box(lb), 0),
    );

    // Digits: long ASCII-digit run. `Digit` oracle is ASCII-only, so keep the input ASCII
    // to stay apples-to-apples (a \p{N} comparison would need a \p{N} oracle).
    let digits = "1234567890 aor woipa jsajfsoiq 183012933 901200 , 09123".repeat(400);
    let db = digits.as_bytes();
    compare(
        "FastNumber vs Digit",
        db,
        iters,
        || FastNumber::run_end(black_box(db), 0),
        || Digit::run_end(black_box(db), 0),
    );
}
