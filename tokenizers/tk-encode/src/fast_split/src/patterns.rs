use crate::matchers::Letter;
use crate::rules::run_matcher;
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
    fn classify(bytes: &[u8], from: u8) -> u32;
    fn try_rule(rule_id: u8, bytes: &[u8], from: usize) -> Option<usize>;
}
pub enum SplitPatterns {
    Cl100k,
    Gpt2,
}
pub struct Cl100k {}
impl PatternDef for Cl100k {
    fn classify(bytes: &[u8], from: u8) -> u32 {
        return 0;
    } // it will return a RuleId to run next
    fn try_rule(rule_id: u8, bytes: &[u8], from: usize) -> Option<usize> {
        match rule_id {
            0 => run_matcher::<Letter, 0, 255>(bytes, from),
            _ => Some(from),
        }
    } // it will run the rule
}
pub fn split<P: PatternDef>(bytes: &[u8], out: &mut Vec<(usize, usize)>) {
    let mut pos = 0;
    while pos < bytes.len() {
        let rule_id = P::classify(bytes, pos as u8);
        match P::try_rule(rule_id as u8, bytes, pos) {
            Some(match_pos) => {
                if match_pos < bytes.len() {
                    out.push((pos, match_pos));
                    pos += match_pos;
                } else {
                    pos += 1;
                }
            }
            _ => pos += 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fancy_equivalence() {
        let mut out = Vec::new();
        split::<Cl100k>("hey".as_bytes(), &mut out);
        assert_eq!(&out, &vec![(0, 3)]);
    }
}
