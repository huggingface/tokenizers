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
            if !c.is_number() {
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
