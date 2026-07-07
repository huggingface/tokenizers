use std::error::Error;

use regex_automata::{meta::Regex, Anchored, Input};

/// A multi-pattern splitter that emulates single-character look-ahead by
/// dropping the last matched character.
///
/// GPT-style pre-tokenization regexes contain `\s+(?!\S)`, which has no single
/// lookaround-free equivalent (a DFA emits the whole match and can't drop the
/// trailing char). Instead we split such a pattern into several patterns: the
/// look-ahead alternative becomes `\s+\s` flagged "drop the last char" plus a
/// plain `\s+`, and run them together on a pure DFA (`regex-automata`
/// `new_many`) with no backtracking.
///
/// Matching is an anchored left-to-right walk: at each position the leftmost
/// pattern matches, and if it's a look-ahead pattern its final character is left
/// for the next piece. The patterns must cover the input contiguously (the GPT
/// patterns do), so the yielded ranges are the pre-token spans.
#[derive(Debug)]
pub struct MultiRegex {
    pat: Regex,
    /// Per-pattern flag: if set, drop the last char of a match by this pattern.
    lookahead: Vec<bool>,
}

impl MultiRegex {
    /// Builds from `(pattern, is_lookahead)` pairs. A look-ahead pattern must
    /// always match at least one character *beyond* its look-ahead char.
    pub fn new(patterns: &[(&str, bool)]) -> Result<Self, Box<dyn Error + Send + Sync + 'static>> {
        let (pats, lookahead): (Vec<_>, Vec<bool>) = patterns.iter().copied().unzip();
        let pat = Regex::new_many(&pats)?;
        Ok(Self { pat, lookahead })
    }

    /// If `pattern` is a recognized GPT pre-tokenization regex (gpt2 / cl100k /
    /// o200k), builds a `MultiRegex` from its vetted lookaround-free
    /// decomposition. Returns `None` for unknown patterns (caller keeps its
    /// backtracking engine). The decompositions are span-equivalent to the
    /// originals.
    pub fn for_gpt_pattern(
        pattern: &str,
    ) -> Option<Result<Self, Box<dyn Error + Send + Sync + 'static>>> {
        gpt_decomposition(pattern).map(Self::new)
    }

    /// Iterates the pre-token ranges `(start, end)` (byte offsets into `text`).
    pub fn split_ranges<'r, 't>(&'r self, text: &'t str) -> MultiSplits<'r, 't> {
        MultiSplits {
            pat: &self.pat,
            lookahead: &self.lookahead,
            text,
            last: 0,
        }
    }
}

pub struct MultiSplits<'r, 't> {
    pat: &'r Regex,
    lookahead: &'r [bool],
    text: &'t str,
    last: usize,
}

impl Iterator for MultiSplits<'_, '_> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        // Anchored: the next match must start exactly at `last`, so pieces are
        // contiguous (offsets are relative to the remaining slice).
        let input = Input::new(&self.text[self.last..]).anchored(Anchored::Yes);
        let m = self.pat.find(input)?;
        let start = self.last;
        let mut end = self.last + m.range().end;
        if self.lookahead[m.pattern().as_usize()] {
            // drop the look-ahead char (kept for the next piece)
            let last = self.text[start..end]
                .chars()
                .next_back()
                .expect("a look-ahead pattern matches at least one char");
            end -= last.len_utf8();
        }
        if end == start {
            // no progress: avoid an infinite loop (shouldn't happen for the GPT
            // patterns, where every alternative consumes ≥1 kept char)
            return None;
        }
        self.last = end;
        Some((start, end))
    }
}

// Canonical GPT pre-tokenization regexes (the look-ahead originals) used as
// recognition keys, each paired with a span-equivalent multi-pattern
// decomposition (from the `bpe` crate). In every decomposition the look-ahead
// `\s+(?!\S)|\s+` tail becomes `\s+$` in the first pattern, plus `\s+\s`
// (drop-last) and a plain `\s+`. Equivalence is checked in the tests.

const GPT2: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
const GPT2_PATS: &[(&str, bool)] = &[
    (
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+$",
        false,
    ),
    (r"\s+\s", true),
    (r"\s+", false),
];

const CL100K: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";
const CL100K_PATS: &[(&str, bool)] = &[
    (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+$",
        false,
    ),
    (r"\s+\s", true),
    (r"\s+", false),
];

const O200K: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const O200K_PATS: &[(&str, bool)] = &[
    (
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+$",
        false,
    ),
    (r"\s+\s", true),
    (r"\s+", false),
];

fn gpt_decomposition(pattern: &str) -> Option<&'static [(&'static str, bool)]> {
    if pattern == GPT2 {
        Some(GPT2_PATS)
    } else if pattern == CL100K {
        Some(CL100K_PATS)
    } else if pattern == O200K {
        Some(O200K_PATS)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::SysRegex;

    // corpus with the whitespace shapes the look-ahead cares about: single vs
    // multiple spaces, trailing spaces, tabs, blank lines, contractions, CJK.
    const CORPUS: &str = "The quick brown fox 123!!! café  résumé\n\n\
        double  spaces\tand\ttabs.  Trailing spaces   \n\
        MixedCASE words'll don't 42 3.14 @#$%^&*() 你好 世界 end. ";

    fn assert_equivalent(orig: &str, corpus: &str) {
        let pats = gpt_decomposition(orig).expect("known pattern");
        let multi = MultiRegex::new(pats).unwrap();
        let baseline = SysRegex::new(orig).unwrap();
        assert_eq!(
            multi.split_ranges(corpus).collect::<Vec<_>>(),
            baseline.find_iter(corpus).collect::<Vec<_>>(),
            "decomposition diverged from look-ahead regex",
        );
    }

    #[test]
    fn gpt2_decomposition_is_equivalent() {
        assert_equivalent(GPT2, CORPUS);
    }

    #[test]
    fn cl100k_decomposition_is_equivalent() {
        assert_equivalent(CL100K, CORPUS);
    }

    #[test]
    fn o200k_decomposition_is_equivalent() {
        assert_equivalent(O200K, CORPUS);
    }

    #[test]
    fn unknown_pattern_is_none() {
        assert!(gpt_decomposition(r"\s+").is_none());
        assert!(MultiRegex::for_gpt_pattern(r"\w+").is_none());
    }

    #[test]
    fn empty_and_edges() {
        let multi = MultiRegex::new(gpt_decomposition(CL100K).unwrap()).unwrap();
        assert_eq!(multi.split_ranges("").collect::<Vec<_>>(), Vec::new());
        assert_eq!(
            multi.split_ranges("hello").collect::<Vec<_>>(),
            vec![(0, 5)]
        );
        // trailing whitespace run stays whole (no following word to steal a space)
        assert_eq!(
            multi.split_ranges("hi   ").collect::<Vec<_>>(),
            SysRegex::new(CL100K)
                .unwrap()
                .find_iter("hi   ")
                .collect::<Vec<_>>(),
        );
    }
}
