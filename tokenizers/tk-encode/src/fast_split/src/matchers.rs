use std::{io::Read, str::from_utf8_unchecked};
use unicode_categories::UnicodeCategories;
pub trait Matcher {
    fn run_end(b: &[u8], from: usize) -> usize;
}
pub struct Letter;
impl Matcher for Letter {
    #[inline]
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
pub struct Digit;
impl Matcher for Digit {
    #[inline]
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
        // SAFETY: from has to always be a char boundary (loop advances by whole tokens)
        let off = from;
        while off < b.len() {
            if b[off] == 0 {
                off += 1;
            } else if off + 1 < b.len() && letter2_hit(b[off], b[off + 1]) {
                off += 1;
            } else if off + 2 < b.len() && letter3_hit(b[off], b[off + 1], b[off + 2]) {
                off += 2;
            } else {
                return from + off;
            }
        }
        b.len()
    }
}
