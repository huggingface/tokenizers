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
            // drop the look-ahead char (kept for the next piece); skip if empty
            if let Some(last) = self.text[start..end].chars().next_back() {
                end -= last.len_utf8();
            }
        }
        if end == start {
            return None; // no progress (only reachable via a malformed pattern set)
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

const CL100K: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
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

/// A recognized GPT pre-tokenization regex that maps to a native `atomsplit` FSM (byte-exact).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GptFsm {
    /// GPT-2 / ByteLevel regex → `atomsplit::fsm::fsm_byte_level`.
    Gpt2,
    /// cl100k-family regex → `atomsplit::fsm::fsm_cl100k_cap`. `digit_cap` is rule 3's `\p{N}{1,cap}`
    /// bound: 3 = cl100k / Llama-3, 1 = Qwen2 (`\p{N}`), `usize::MAX` = an unbounded `\p{N}+`.
    Cl100k { digit_cap: usize },
    /// o200k / GPT-4o regex → `atomsplit::fsm::fsm_o200k`.
    O200k,
}

/// The cl100k-family template is fixed except rule 3's digit rule. If `pattern` is that template, return
/// the `\p{N}{1,cap}` bound (`\p{N}{1,3}`→3, `\p{N}{1,2}`→2, `\p{N}`→1, `\p{N}+`→`MAX`); else `None`.
/// This is what makes Qwen2 (cl100k with `\p{N}`) unroll without a per-tokenizer exact-string entry.
fn cl100k_digit_cap(pattern: &str) -> Option<usize> {
    // cl100k rules 1-2 (contraction + word) … <DIGIT RULE> … rules 4-7 (other + whitespace).
    const PRE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|";
    const SUF: &str = r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    match pattern.strip_prefix(PRE)?.strip_suffix(SUF)? {
        r"\p{N}{1,3}" => Some(3),
        r"\p{N}{1,2}" => Some(2),
        r"\p{N}" => Some(1),
        r"\p{N}+" => Some(usize::MAX),
        _ => None,
    }
}

/// If `pattern` is a recognized GPT pre-tokenization regex, name the native FSM that reproduces its
/// `Isolated` split byte-for-byte. GPT-2 and o200k are matched exactly; the cl100k family is matched
/// structurally ([`cl100k_digit_cap`]) so digit-cap variants (Qwen2 …) unroll too. An unrecognized
/// pattern → `None` (the SysRegex / fancy-regex path handles it).
pub fn gpt_fsm(pattern: &str) -> Option<GptFsm> {
    if pattern == GPT2 {
        Some(GptFsm::Gpt2)
    } else if pattern == O200K {
        Some(GptFsm::O200k)
    } else {
        cl100k_digit_cap(pattern).map(|digit_cap| GptFsm::Cl100k { digit_cap })
    }
}

/// `Pattern` that runs the native atomsplit FSM for a [`GptFsm`] — the legacy (`NormalizedString`)
/// split path's equivalent of the pipeline's native routing, so GPT pre-tokenizers need no
/// system-regex backend. `Isolated` behaviour keeps every span, and the FSM spans cover the input
/// contiguously, so this is byte-for-byte identical to the original GPT regex `Isolated` split.
pub struct GptFsmPattern(pub GptFsm);

impl crate::tokenizer::pattern::Pattern for GptFsmPattern {
    fn find_matches(
        &self,
        inside: &str,
    ) -> crate::tokenizer::Result<Vec<(crate::tokenizer::Offsets, bool)>> {
        if inside.is_empty() {
            return Ok(vec![((0, 0), false)]);
        }
        use atomsplit::classify::classify;
        let bytes = inside.as_bytes();
        let mut tags = vec![0u8; bytes.len()];
        classify(bytes, &mut tags);
        let mut spans = vec![(0u32, 0u32); bytes.len() + 1];
        let n = match self.0 {
            GptFsm::Gpt2 => atomsplit::fsm::fsm_byte_level(bytes, &tags, &mut spans),
            GptFsm::Cl100k { digit_cap } => {
                atomsplit::fsm::fsm_cl100k_cap(bytes, &tags, &mut spans, digit_cap)
            }
            GptFsm::O200k => atomsplit::fsm::fsm_o200k(bytes, &tags, &mut spans),
        };
        Ok(spans[..n]
            .iter()
            .map(|&(s, e)| ((s as usize, e as usize), true))
            .collect())
    }
}

// deepseek-v4's pre-tokenizer is a `Sequence` of these three Isolated `Split`s (+ a byte-map
// `ByteLevel`), which `atomsplit::fsm::fsm_deepseek` collapses into one pass. Byte-exact with the
// shipped tokenizer.json — the big pattern carries LITERAL CR/LF, spliced in via `concat!`.
const DS_NUM: &str = r"\p{N}{1,3}";
const DS_CJK: &str = "[\u{4E00}-\u{9FA5}\u{3040}-\u{309F}\u{30A0}-\u{30FF}]+";
const DS_BIG: &str = concat!(
    r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^"##,
    "\r\n",
    r##"\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+["##,
    "\r\n",
    r##"]*|\s*["##,
    "\r\n",
    r##"]+|\s+(?!\S)|\s+"##,
);

/// True iff three `Split` patterns are exactly deepseek's `[\p{N}{1,3}, CJK-range, big-regex]` prefix →
/// `atomsplit::fsm::fsm_deepseek` reproduces the whole composed Isolated split in one pass.
pub fn is_deepseek(p0: &str, p1: &str, p2: &str) -> bool {
    p0 == DS_NUM && p1 == DS_CJK && p2 == DS_BIG
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
    fn gpt_fsm_recognizes_family_and_extracts_digit_cap() {
        // Exact matches for gpt2 / o200k, structural (any digit cap) for the cl100k family.
        assert_eq!(gpt_fsm(GPT2), Some(GptFsm::Gpt2));
        assert_eq!(gpt_fsm(O200K), Some(GptFsm::O200k));
        assert_eq!(gpt_fsm(CL100K), Some(GptFsm::Cl100k { digit_cap: 3 }));
        // Qwen2: cl100k with rule 3 = `\p{N}` → cap 1.
        let qwen2 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        assert_eq!(gpt_fsm(qwen2), Some(GptFsm::Cl100k { digit_cap: 1 }));
        // Other in-family digit caps.
        let cap2 = CL100K.replace(r"\p{N}{1,3}", r"\p{N}{1,2}");
        assert_eq!(gpt_fsm(&cap2), Some(GptFsm::Cl100k { digit_cap: 2 }));
        let unbounded = CL100K.replace(r"\p{N}{1,3}", r"\p{N}+");
        assert_eq!(
            gpt_fsm(&unbounded),
            Some(GptFsm::Cl100k {
                digit_cap: usize::MAX
            })
        );
        // Out of family → None (fancy-regex fallback): a foreign digit rule, and a totally unrelated regex.
        assert_eq!(gpt_fsm(&CL100K.replace(r"\p{N}{1,3}", r"\p{N}{2,4}")), None);
        assert_eq!(gpt_fsm(r"\w+|\s+"), None);
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
