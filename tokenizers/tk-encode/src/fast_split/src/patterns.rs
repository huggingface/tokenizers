use crate::matchers::{Digit, Letter};
use crate::rules::{run_matcher, word_rule};
/// For each of these rules, we'll define fast SIMD optimized scanners
// Cl100k has 7 branches (separated by `|`)
// 1  (?i:'s|'t|'re|'ve|'m|'ll|'d)  -> match litterals, ?i case insensitive, `'` first byte to scan
// 2  [^\r\n\p{L}\p{N}]?\p{L}+      -> not match unicode general category: L: Letter N: number (both
//    ascii) from 0 to 1 then a letter from 1 to any
// 3  \p{N}{1,3}                    -> digits 1 min up to 3
// 4   ?[^\s\p{L}\p{N}]+[\r\n]*     -> maybe a space (' ?') followed by not a space, not a Letter
//     not a Number many time, and then \r or \n.
// 5  \s*[\r\n]+                    -> any number of space followed by at least \r or \n
// 6  \s+(?!\S)                     -> succession of \s chars up until a \S char
// 7  \s+                           -> any repetition of space char
// Each will be tried in order. But we should ont try them all at each pos. Dispatcher needs to
// In CL100K, we can see that the regex will in most cases be looking for whitespace boundaries,
// digits, letters or litterals.
// classify the char and say which rule to run. (first char + 1 peek can tell us everything )
pub trait PatternDef {
    fn next_token(bytes: &[u8], from: usize) -> Option<usize>;
}
pub enum SplitPatterns {
    Cl100k,
    Gpt2,
}
pub struct Cl100k;
impl PatternDef for Cl100k {
    fn next_token(bytes: &[u8], from: usize) -> Option<usize> {
        let b = bytes[from];
        if b >= 0x80 {
            todo!() // unicode
        } else if b.is_ascii_alphabetic() {
            word_rule::<Letter>(bytes, from)
        } else if b.is_ascii_digit() {
            run_matcher::<Digit, 1, 3>(bytes, from)
        } else {
            None
        }
    } // it will run the rule
}
pub fn split<P: PatternDef>(bytes: &[u8], out: &mut Vec<(usize, usize)>) {
    let mut pos = 0;
    while pos < bytes.len() {
        match P::next_token(bytes, pos) {
            Some(end) if end > pos => {
                out.push((pos, end));
                pos = end;
            }
            _ => pos += 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split() {
        let mut out = Vec::new();
        split::<Cl100k>("hey".as_bytes(), &mut out);
        assert_eq!(&out, &vec![(0, 3)]);
        out.clear();
        split::<Cl100k>("hey what".as_bytes(), &mut out);
        assert_eq!(&out, &vec![(0, 3), (4, 8)]);
    }
}
