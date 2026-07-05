use super::unicode::{letter2_hit, letter3_hit, number2_hit, number3_hit};
use std::str::from_utf8_unchecked;
use unicode_categories::UnicodeCategories;
pub trait Matcher {
    fn run_end(b: &[u8], from: usize) -> usize;
}
#[cfg(feature = "unicode")]
pub struct Letter;
impl Matcher for Letter {
    #[inline]
    #[cfg(feature = "unicode")]
    fn run_end(b: &[u8], from: usize) -> usize {
        // SAFETY: from has to always be a char boundary (loop advances by whole tokens)
        let s = unsafe { from_utf8_unchecked(&b[from..]) };
        for (off, c) in s.char_indices() {
            if !c.is_letter() {
                return from + off;
            }
        }
        b.len()
    }
}
#[cfg(feature = "unicode")]
pub struct Digit;
impl Matcher for Digit {
    #[inline]
    #[cfg(feature = "unicode")]
    fn run_end(b: &[u8], from: usize) -> usize {
        // SAFETY: from has to always be a char boundary (loop advances by whole tokens)
        let s = unsafe { from_utf8_unchecked(&b[from..]) };
        for (off, c) in s.char_indices() {
            if !c.is_ascii_digit() {
                return from + off;
            }
        }
        b.len()
    }
}
pub struct FastLetter;
impl Matcher for FastLetter {
    #[inline]
    fn run_end(b: &[u8], from: usize) -> usize {
        let mut off = from;
        while off < b.len() {
            let b0 = b[off];
            if b0 < 0x80 {
                if b0.is_ascii_alphabetic() {
                    off += 1
                } else {
                    return off;
                }
            } else if b0 < 0xE0 {
                if off + 1 < b.len() && letter2_hit(b[off], b[off + 1]) {
                    off += 2;
                } else {
                    return off;
                }
            } else if b0 < 0xF0 {
                if off + 2 < b.len() && letter3_hit(b[off], b[off + 1], b[off + 2]) {
                    off += 3;
                } else {
                    return off;
                }
            } else {
                return off;
            }
        }
        b.len()
    }
}
pub struct FastNumber;
impl Matcher for FastNumber {
    #[inline]
    fn run_end(b: &[u8], from: usize) -> usize {
        let mut off = from;
        while off < b.len() {
            let b0 = b[off];
            if b0 < 0x80 {
                if b0.is_ascii_digit() {
                    off += 1
                } else {
                    return off;
                }
            } else if b0 < 0xE0 {
                if off + 1 < b.len() && number2_hit(b[off], b[off + 1]) {
                    off += 2;
                } else {
                    return off;
                }
            } else if b0 < 0xF0 {
                if off + 2 < b.len() && number3_hit(b[off], b[off + 1], b[off + 2]) {
                    off += 3;
                } else {
                    return off;
                }
            } else {
                return off;
            }
        }
        b.len()
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use std::f32::DIGITS;
    use std::hint::black_box;
    use std::time::Instant;
    use unicode_categories::*;
    // Hoist-safe timer: black_box the input on every call, accumulate every result so LLVM
    // can't hoist the (pure) scan out of the loop and report a fictitious 0 ns/byte.
    fn time<F: Fn() -> usize>(bytes_len: usize, iters: u32, f: F) -> f64 {
        for _ in 0..2 {
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

    #[test]
    #[ignore]
    pub fn bench_fast_matcher() {
        let bytes = "Hey how ar eyou doing? 1298231".as_bytes();
        let iters = 20u32;
        let fast = time(bytes.len(), iters, || {
            FastNumber::run_end(black_box(bytes), 0)
        });
        let naive = time(bytes.len(), iters, || Digit::run_end(black_box(bytes), 0));
        println!(
            "  fast_run_end : {fast:.3} ns/byte ({:>5.0} MB/s)",
            1000.0 / fast
        );
        println!(
            "  unicode run_end  : {naive:.3} ns/byte ({:>5.0} MB/s)  match_bytes {:.2}x",
            1000.0 / naive,
            naive / fast
        );
    }
}
