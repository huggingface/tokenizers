//! Splitting on an exact string — the Metaspace `▁` delimiter (llama2 / gemma), `CharDelimiterSplit`,
//! and any `Split` whose pattern is a plain string rather than a regex.
//!
//! - No atom classification here: a literal does not care what class a byte is, so the tag pass is
//!   skipped entirely and this works off the raw bytes.
//! - Same bitstream shape as the grammars: build a `u64` match bitmap per 64-byte block, then walk
//!   it with `trailing_zeros`. `shift-and` over the needle's bytes, which is the classic Baeza-Yates
//!   bitap specialised to an exact string — one AND per needle byte per 64 input bytes.
//! - Matches never overlap, so `"aa"` is found once in `"aaa"` — the same matches a regex reports.

use std::fmt;

/// The pattern handed to [`Literal::new`] was empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmptyPattern;

impl fmt::Display for EmptyPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("an empty pattern matches everywhere")
    }
}

impl std::error::Error for EmptyPattern {}

/// A literal string to split on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Literal {
    needle: Vec<u8>,
}

impl Literal {
    /// # Errors
    /// If `pattern` is empty, which would match everywhere.
    pub fn new(pattern: &[u8]) -> Result<Self, EmptyPattern> {
        if pattern.is_empty() {
            return Err(EmptyPattern);
        }
        Ok(Self {
            needle: pattern.to_vec(),
        })
    }

    /// The string being searched for.
    #[must_use]
    pub fn pattern(&self) -> &[u8] {
        &self.needle
    }

    /// Byte offset of every match, left to right, non-overlapping.
    pub fn matches<'t>(&'t self, text: &'t [u8]) -> Matches<'t> {
        Matches {
            needle: &self.needle,
            text,
            base: 0,
            word: 0,
            next_block: 0,
            min: 0,
        }
    }
}

/// Match positions of `needle` in `text[base..base + 64]`, as a bitmap. A match at bit `i` means
/// `text[base + i..][..needle.len()] == needle`, so the last `needle.len() - 1` bits of the block
/// depend on bytes past it — the shift-and simply reads them from `text`, which is why this takes
/// the whole slice rather than a block.
fn match_bits(text: &[u8], needle: &[u8], base: usize) -> u64 {
    let n = text.len();
    // positions in this block that could still fit the needle
    let span = n.saturating_sub(base).min(64);
    let last = n.saturating_sub(needle.len() - 1); // one past the last possible match start
    let room = last.saturating_sub(base).min(span);
    if room == 0 {
        return 0;
    }
    let valid = if room == 64 { !0u64 } else { (1u64 << room) - 1 };
    // First byte in SIMD -- this is the whole cost, everything after it runs on the survivors.
    let mut m = eq_bits(text, base, needle[0]) & valid;
    for (k, &b) in needle.iter().enumerate().skip(1) {
        let mut eq = 0u64;
        let mut bits = m;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            if text[base + i + k] == b {
                eq |= 1u64 << i;
            }
        }
        m &= eq;
        if m == 0 {
            return 0;
        }
    }
    m
}

/// `text[base..base + 64] == b`, as a bitmap. Falls back to a byte loop on a ragged tail (and on
/// targets with no kernel), which is correct everywhere and only pays on the last block.
#[inline]
fn eq_bits(text: &[u8], base: usize, b: u8) -> u64 {
    if base + 64 <= text.len() {
        #[cfg(target_arch = "aarch64")]
        // SAFETY: bounds checked directly above.
        return unsafe { crate::simd::eq64(text, base, b) };
        #[cfg(target_arch = "x86_64")]
        if crate::has_ssse3() {
            // SAFETY: bounds checked above, SSSE3 checked here.
            return unsafe { crate::simd_x86::eq64(text, base, b) };
        }
    }
    let mut m = 0u64;
    for i in 0..(text.len() - base).min(64) {
        if text[base + i] == b {
            m |= 1u64 << i;
        }
    }
    m
}

/// Iterator over non-overlapping match offsets.
pub struct Matches<'t> {
    needle: &'t [u8],
    text: &'t [u8],
    base: usize,
    word: u64,
    next_block: usize,
    min: usize, // matches before this would overlap the one just returned
}

impl Iterator for Matches<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        loop {
            while self.word != 0 {
                let i = self.word.trailing_zeros() as usize;
                self.word &= self.word - 1;
                let at = self.base + i;
                if at >= self.min {
                    self.min = at + self.needle.len();
                    return Some(at);
                }
            }
            if self.next_block >= self.text.len() {
                return None;
            }
            self.base = self.next_block;
            self.next_block += 64;
            self.word = match_bits(self.text, self.needle, self.base);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find(pat: &str, hay: &str) -> Vec<usize> {
        Literal::new(pat.as_bytes())
            .unwrap()
            .matches(hay.as_bytes())
            .collect()
    }

    /// The bitmap has to agree with a naive scan on overlap, block edges and multi-byte needles.
    #[test]
    fn matches_a_naive_scan() {
        let cases: &[(&str, String)] = &[
            ("a", "aaaa".into()),
            ("aa", "aaaaa".into()),
            ("ab", "abab".into()),
            ("▁", "▁a▁bb▁".into()),
            ("▁▁", "▁▁▁▁".into()),
            ("xyz", "xyxyz".into()),
            ("a", "b".repeat(200)),
            // needle straddling every 64-byte block edge
            ("▁w", format!("{}▁w{}", "q".repeat(63), "z".repeat(70))),
            ("abc", format!("{}abc", "a".repeat(62))),
        ];
        for (pat, hay) in cases {
            let (p, h) = (pat.as_bytes(), hay.as_bytes());
            let mut want = Vec::new();
            let mut i = 0;
            while i + p.len() <= h.len() {
                if &h[i..i + p.len()] == p {
                    want.push(i);
                    i += p.len(); // non-overlapping, like a regex
                } else {
                    i += 1;
                }
            }
            assert_eq!(find(pat, hay), want, "{pat:?} in {hay:?}");
        }
    }

    /// Every offset of the needle across a 3-block text, so it crosses each edge in every phase.
    #[test]
    fn finds_the_needle_at_every_offset() {
        let needle = "▁w";
        for off in 0..180usize {
            let hay = format!("{}{needle}{}", "q".repeat(off), "q".repeat(180 - off));
            assert_eq!(find(needle, &hay), vec![off], "offset {off}");
        }
    }

    #[test]
    fn an_empty_pattern_is_rejected() {
        assert_eq!(Literal::new(b""), Err(EmptyPattern));
    }
}
