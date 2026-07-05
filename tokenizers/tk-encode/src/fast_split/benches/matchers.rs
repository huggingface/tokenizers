//! Custom-harness bench (no criterion): bitmap matchers vs the `unicode_categories` oracle,
//! across several input shapes. Measures a WHOLE-BUFFER tokenize (loop `run_end` over every
//! position, like the split loop) so ns/byte reflects scanning the entire input — NOT a single
//! run from pos 0 (which, on mixed input, scans ~nothing and reports a fake ~0 ns/byte).
//! Run:  cargo bench --bench matchers --features unicode      (rtk proxy for stdout)
use fast_split::matchers::{Digit, FastLetter, FastNumber, Letter, Matcher};
use std::hint::black_box;
use std::time::Instant;

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

/// Split the whole buffer into runs of `M`, counting tokens. Every byte is visited once, so
/// ns/byte is the real cost of scanning the entire input (matching runs AND skipping non-matches).
fn tokenize<M: Matcher>(bytes: &[u8]) -> usize {
    let mut pos = 0;
    let mut n = 0;
    while pos < bytes.len() {
        let end = M::run_end(bytes, pos);
        if end > pos {
            n += 1;
            pos = end;
        } else {
            pos += 1;
        }
    }
    n
}

/// Assert bitmap and oracle produce the same tokenization (byte-exactness), then time both.
fn compare<Fast: Matcher, Oracle: Matcher>(name: &str, text: &str, iters: u32) {
    let b = text.as_bytes();
    let fast_n = tokenize::<Fast>(b);
    let orac_n = tokenize::<Oracle>(b);
    assert_eq!(
        fast_n, orac_n,
        "{name}: bitmap {fast_n} vs oracle {orac_n} tokens — they disagree"
    );
    let f = time(b.len(), iters, || tokenize::<Fast>(black_box(b)));
    let o = time(b.len(), iters, || tokenize::<Oracle>(black_box(b)));
    println!(
        "  {name:<21} {:>6} B {fast_n:>5} tok | bitmap {f:6.3} ns/B ({:>5.0} MB/s) | oracle {o:6.3} ns/B ({:>5.0} MB/s)  {:5.1}x",
        b.len(),
        1000.0 / f,
        1000.0 / o,
        o / f
    );
}

fn main() {
    let it = 400u32;

    println!("== FastLetter vs Letter (\\p{{L}}) ==");
    // no separators -> one long multiscript run (pure scan speed)
    compare::<FastLetter, Letter>(
        "dense multiscript",
        &"HelloΑλφαПривет你好Мир한국어café".repeat(300),
        it,
    );
    // words + spaces -> many tokens; the common shape
    compare::<FastLetter, Letter>(
        "prose (ascii+space)",
        &"the quick brown fox jumps over the lazy dog ".repeat(200),
        it,
    );
    // zero letters -> measures the reject/skip path over non-letters
    compare::<FastLetter, Letter>("no letters", &"12 34 567 89 !@# 0 , . 42 ".repeat(300), it);

    println!("\n== FastNumber vs Digit (\\p{{N}}) ==");
    // no separators -> one 20 KB digit run (this is the case that used to report 0 ns/byte)
    compare::<FastNumber, Digit>("dense ascii digits", &"1234567890".repeat(2000), it);
    // digits embedded in prose -> few digit runs, mostly skipping letters
    compare::<FastNumber, Digit>(
        "sparse digits",
        &"the fox ate 42 of 7 apples and 1337 pears ".repeat(200),
        it,
    );
    // zero digits -> reject/skip path over letters
    compare::<FastNumber, Digit>(
        "no digits",
        &"the quick brown fox jumps over the lazy dog ".repeat(200),
        it,
    );
    // Arabic-Indic (2-byte), Devanagari + fullwidth (3-byte) -> exercises the bitmap paths
    compare::<FastNumber, Digit>(
        "non-ascii digits",
        &"٠١٢٣٤٥ ०१२३४५ ０１２３４５ ".repeat(300),
        it,
    );
}
