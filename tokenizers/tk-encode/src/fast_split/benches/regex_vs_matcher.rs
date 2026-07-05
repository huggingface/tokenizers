//! Regex `\d{Q}` vs the monomorphized `run_matcher::<FastNumber, MIN, MAX>`, for the three
//! quantifier forms. Both *tokenize the whole buffer* (find every digit run), so ns/byte is
//! meaningful and the token counts show how the quantifier changes the split.
//!
//!   {1,3} -> run_matcher::<_, 1, 3>      (cap a run at 3)
//!   +     -> run_matcher::<_, 1, 255>    (>=1, unbounded; 255 = u8::MAX sentinel)
//!   *     -> run_matcher::<_, 0, 255>    (>=0, unbounded)
//!
//! Run: cargo bench --bench regex_vs_matcher --features unicode   (rtk proxy for stdout)
use fast_split::matchers::FastNumber;
use fast_split::rules::run_matcher;
use regex::Regex;
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

/// Tokenize the whole buffer with `run_matcher<MIN,MAX>`, mirroring the split loop; count tokens.
fn tok<const MIN: u8, const MAX: u8>(bytes: &[u8]) -> usize {
    let mut pos = 0;
    let mut n = 0;
    while pos < bytes.len() {
        match run_matcher::<FastNumber, MIN, MAX>(bytes, pos) {
            Some(end) if end > pos => {
                n += 1;
                pos = end;
            }
            _ => pos += 1,
        }
    }
    n
}

fn row(q: &str, text: &str, iters: u32, re: &Regex, matcher: impl Fn(&[u8]) -> usize, check: bool) {
    let bytes = text.as_bytes();
    let re_n = re.find_iter(text).count();
    let mt_n = matcher(bytes);
    // {1,3} and + must produce the same non-empty digit tokens; * legitimately differs
    // (regex \d* yields an empty match at every non-digit position; the split loop skips them).
    if check {
        assert_eq!(re_n, mt_n, "\\d{q}: regex and run_matcher disagree on token count");
    }
    let rf = time(bytes.len(), iters, || re.find_iter(black_box(text)).count());
    let mf = time(bytes.len(), iters, || matcher(black_box(bytes)));
    println!(
        "  \\d{q:<5} regex {rf:6.3} ns/B ({:>5.0} MB/s, {re_n:>5} tok) | run_matcher {mf:6.3} ns/B ({:>5.0} MB/s, {mt_n:>5} tok)  speedup {:5.1}x",
        1000.0 / rf,
        1000.0 / mf,
        rf / mf
    );
}

fn main() {
    let iters = 1000u32;
    // ASCII digit runs of length 1..6 separated by spaces, so {1,3} splits the long runs
    // (5-run -> "123","45"; 6-run -> "999","999") while + keeps each run whole.
    let text = "7 42 12345 999999 8 31 ".repeat(300);
    println!("digit split — {} bytes\n", text.len());
    row("{1,3}", &text, iters, &Regex::new(r"\d{1,3}").unwrap(), |x| tok::<1, 3>(x), true);
    row("+", &text, iters, &Regex::new(r"\d+").unwrap(), |x| tok::<1, 255>(x), true);
    row("*", &text, iters, &Regex::new(r"\d*").unwrap(), |x| tok::<0, 255>(x), false);
}
