//! Regex `CLASS{Q}` vs the monomorphized `run_matcher::<M, MIN, MAX>`, for the three quantifier
//! forms, across several input shapes (dense / sparse / none / non-ASCII), for BOTH digits
//! (`\d` vs `FastNumber`) and letters (`\p{L}` vs `FastLetter`). Both *tokenize the whole buffer*
//! (find every run), so ns/byte is real and the token counts show how the quantifier splits.
//!
//!   {1,3} -> run_matcher::<_, 1, 3>      (cap a run at 3)
//!   +     -> run_matcher::<_, 1, 255>    (>=1, unbounded; 255 = u8::MAX sentinel)
//!   *     -> run_matcher::<_, 0, 255>    (>=0, unbounded)
//!
//! Non-ASCII digit inputs are DECIMAL only (Arabic/Devanagari/fullwidth = \p{Nd}): regex `\d` ==
//! \p{Nd} but FastNumber == \p{N} (broader), so they agree only on decimals. Letter inputs use
//! long-assigned scripts so regex's `\p{L}` tables and `unicode_categories` (which baked the
//! bitmap) agree.
//!
//! Run: cargo bench --bench regex_vs_matcher --features unicode   (rtk proxy for stdout)
use fast_split::matchers::{FastLetter, FastNumber, Matcher};
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

/// Tokenize the whole buffer with `run_matcher::<M, MIN, MAX>`, mirroring the split loop.
fn tok<M: Matcher, const MIN: u8, const MAX: u8>(bytes: &[u8]) -> usize {
    let mut pos = 0;
    let mut n = 0;
    while pos < bytes.len() {
        match run_matcher::<M, MIN, MAX>(bytes, pos) {
            Some(end) if end > pos => {
                n += 1;
                pos = end;
            }
            _ => pos += 1,
        }
    }
    n
}

/// One quantifier row: build `CLASS{q}`, check both agree (when `check`), then time both.
fn row<M: Matcher, const MIN: u8, const MAX: u8>(class: &str, q: &str, text: &str, iters: u32, check: bool) {
    let re = Regex::new(&format!("{class}{q}")).unwrap();
    let b = text.as_bytes();
    let re_n = re.find_iter(text).count();
    let mt_n = tok::<M, MIN, MAX>(b);
    // {1,3} and + must produce the same non-empty runs; * legitimately differs (regex CLASS*
    // yields an empty match at every non-matching position; the split loop skips them).
    if check {
        assert_eq!(re_n, mt_n, "{class}{q}: regex {re_n} vs run_matcher {mt_n} tokens");
    }
    let rf = time(b.len(), iters, || re.find_iter(black_box(text)).count());
    let mf = time(b.len(), iters, || tok::<M, MIN, MAX>(black_box(b)));
    println!(
        "    {q:<5} regex {rf:8.3} ns/B ({:>5.0} MB/s, {re_n:>6} tok) | run_matcher {mf:8.3} ns/B ({:>5.0} MB/s, {mt_n:>6} tok)  {:6.1}x",
        1000.0 / rf,
        1000.0 / mf,
        rf / mf
    );
}

fn scenario<M: Matcher>(title: &str, class: &str, text: &str, iters: u32) {
    println!("\n  {title} — {} bytes  (regex {class})", text.len());
    row::<M, 1, 3>(class, "{1,3}", text, iters, true);
    row::<M, 1, 255>(class, "+", text, iters, true);
    row::<M, 0, 255>(class, "*", text, iters, false);
}

fn main() {
    let it = 300u32;

    println!("=== DIGITS: \\d vs FastNumber ===");
    scenario::<FastNumber>("dense ascii digits", r"\d", &"1234567890".repeat(800), it);
    scenario::<FastNumber>("sparse digits (prose)", r"\d", &"the fox ate 42 of 7 apples and 1337 pears ".repeat(150), it);
    scenario::<FastNumber>("no digits (prose)", r"\d", &"the quick brown fox jumps over the lazy dog ".repeat(150), it);
    scenario::<FastNumber>("dense non-ascii digits", r"\d", &"٠١٢٣٤٥٦٧٨٩ ०१२३४५६७८९ ０１２３４５６７８９ ".repeat(150), it);

    println!("\n=== LETTERS: \\p{{L}} vs FastLetter ===");
    scenario::<FastLetter>("dense ascii letters", r"\p{L}", &"abcdefghijklmnopqrstuvwxyz".repeat(300), it);
    scenario::<FastLetter>("prose letters (words)", r"\p{L}", &"the quick brown fox jumps over the lazy dog ".repeat(150), it);
    scenario::<FastLetter>("no letters (digits+punct)", r"\p{L}", &"12 34 567 !@# 0 , . 42 ".repeat(200), it);
    scenario::<FastLetter>("dense non-ascii letters", r"\p{L}", &"你好世界日本語한국어こんにちはΑλφαβήτα".repeat(150), it);
}
