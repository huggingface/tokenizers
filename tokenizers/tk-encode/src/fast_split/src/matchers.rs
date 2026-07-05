use super::unicode::{letter2_hit, letter3_hit, number2_hit, number3_hit};
#[cfg(feature = "unicode")]
use std::str::from_utf8_unchecked;
#[cfg(feature = "unicode")]
use unicode_categories::UnicodeCategories;

pub trait Matcher {
    /// End (byte offset) of the ONE char at `from` if it matches the class, else `None`.
    /// `from` must be a char boundary; returns `None` when `from >= b.len()`.
    /// Callers advance by whole chars, so a run is just `step` looped — never a rescan.
    fn step(b: &[u8], from: usize) -> Option<usize>;
}

#[cfg(feature = "unicode")]
pub struct Letter;
#[cfg(feature = "unicode")]
impl Matcher for Letter {
    #[inline]
    fn step(b: &[u8], from: usize) -> Option<usize> {
        // SAFETY: from is always a char boundary (callers advance by whole chars).
        let c = unsafe { from_utf8_unchecked(&b[from..]) }.chars().next()?;
        c.is_letter().then_some(from + c.len_utf8())
    }
}

#[cfg(feature = "unicode")]
pub struct Digit;
#[cfg(feature = "unicode")]
impl Matcher for Digit {
    #[inline]
    fn step(b: &[u8], from: usize) -> Option<usize> {
        let c = unsafe { from_utf8_unchecked(&b[from..]) }.chars().next()?;
        c.is_number().then_some(from + c.len_utf8())
    }
}

pub struct FastLetter;
impl Matcher for FastLetter {
    #[inline]
    fn step(b: &[u8], from: usize) -> Option<usize> {
        let b0 = *b.get(from)?;
        if b0 < 0x80 {
            b0.is_ascii_alphabetic().then_some(from + 1)
        } else if b0 < 0xE0 {
            (from + 1 < b.len() && letter2_hit(b0, b[from + 1])).then_some(from + 2)
        } else if b0 < 0xF0 {
            (from + 2 < b.len() && letter3_hit(b0, b[from + 1], b[from + 2])).then_some(from + 3)
        } else {
            None // 4-byte / astral: not in the tables yet
        }
    }
}

pub struct FastNumber;
impl Matcher for FastNumber {
    #[inline]
    fn step(b: &[u8], from: usize) -> Option<usize> {
        let b0 = *b.get(from)?;
        if b0 < 0x80 {
            b0.is_ascii_digit().then_some(from + 1)
        } else if b0 < 0xE0 {
            (from + 1 < b.len() && number2_hit(b0, b[from + 1])).then_some(from + 2)
        } else if b0 < 0xF0 {
            (from + 2 < b.len() && number3_hit(b0, b[from + 1], b[from + 2])).then_some(from + 3)
        } else {
            None
        }
    }
}
