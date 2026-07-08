//! FSM layer: turn the `Atom` tag stream into token spans. Every pre-tokenizer is one of these
//! shapes, parameterized by a class mask + behavior (all `const`-generic → fully monomorphized).
//! They all read the shared stream from `classify::classify::<Atoms>`; the delimiter *behavior* never
//! touches classification. See `TAG_CLASSIFY_SPEC.md` §4.
//!
//! Scalar cores are portable; the SIMD boundary-extract (`extract_boundaries`) is an aarch64
//! optimization for the RunSplit family (class-change → movemask → bit-iterate).
#![allow(dead_code)] // skeleton

use crate::classify::{Atoms, classify, in_mask, mask};

/// A token span: byte offsets `[start, end)` into the input.
/// TODO:i think we could use u16 for one of them if we stored start, offset5
pub type Span = (u32, u32);

/// Mirror of `tokenizer::SplitDelimiterBehavior`, kept here so it can be a `const`-generic argument.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Behavior {
    Removed = 0,
    Isolated = 1,
    Contiguous = 2,
    MergedWithPrevious = 3,
    MergedWithNext = 4,
}

// ── FSM shapes ───────────────────────────────────────────────────────────────────────────────

/// HF `split(delim, behavior)` over the tag stream. A char is a *match* iff `in_mask(tag, DELIM)`;
/// matches are per-char, non-matches are runs; `BEHAVIOR` places boundaries / drops matches exactly
/// as `SplitDelimiterBehavior`. Monomorphized on both params.
///
/// Covers: WhitespaceSplit, Punctuation, Digits, Metaspace (marker), CharDelimiterSplit, Split-literal.
///
/// ┌── OWNER: shared (scalar core); SIMD boundary-extract optional on aarch64 ──┐
pub fn fsm_split<const DELIM: u16, const BEHAVIOR: u8>(
    text: &[u8],
    tags: &[u8],
    out: &mut Vec<Span>,
) {
    // find_matches over the tag stream (HF Pattern for a char-fn): each matching char is its OWN
    // match; non-matching chars group into maximal runs. `segs` = (start, end, is_match), alternating.
    let n = text.len();
    let mut segs: Vec<(u32, u32, bool)> = Vec::new();
    let mut run_start = 0usize;
    let mut i = 0usize;
    while i < n {
        let s = i;
        i += 1;
        while i < n && text[i] & 0xC0 == 0x80 {
            i += 1; // consume continuation bytes → char is [s, i)
        }
        if in_mask(tags[s], DELIM) {
            if run_start != s {
                segs.push((run_start as u32, s as u32, false));
            }
            segs.push((s as u32, i as u32, true));
            run_start = i;
        }
    }
    if run_start != n {
        segs.push((run_start as u32, n as u32, false));
    }

    // Place / drop the match segments per SplitDelimiterBehavior (tokenizer::normalizer::split).
    match BEHAVIOR {
        b if b == Behavior::Isolated as u8 => {
            for &(s, e, _) in &segs {
                out.push((s, e));
            }
        }
        b if b == Behavior::Removed as u8 => {
            for &(s, e, m) in &segs {
                if !m {
                    out.push((s, e));
                }
            }
        }
        b if b == Behavior::Contiguous as u8 => {
            let mut prev_match = false;
            for &(s, e, m) in &segs {
                if m == prev_match && !out.is_empty() {
                    out.last_mut().unwrap().1 = e; // merge same-kind neighbours
                } else {
                    out.push((s, e));
                }
                prev_match = m;
            }
        }
        b if b == Behavior::MergedWithPrevious as u8 => {
            let mut prev_match = false;
            for &(s, e, m) in &segs {
                if m && !prev_match && !out.is_empty() {
                    out.last_mut().unwrap().1 = e; // delimiter joins the previous piece
                } else {
                    out.push((s, e));
                }
                prev_match = m;
            }
        }
        _ /* MergedWithNext */ => {
            let base = out.len();
            let mut prev_match = false;
            for &(s, e, m) in segs.iter().rev() {
                if m && !prev_match && out.len() > base {
                    out.last_mut().unwrap().0 = s; // delimiter joins the next piece
                } else {
                    out.push((s, e));
                }
                prev_match = m;
            }
            out[base..].reverse();
        }
    }
}

/// Cut at *every* class change; drop `DROP`-class runs, isolate `ISOLATE`-class per char. Keeps more
/// than one run type, so it isn't a single `fsm_split`. Covers: Whitespace (drop WS, keep Word+Symbol
/// runs), Bert (drop WS, isolate Punct).
///
/// Each char maps to one class: `DROP` → dropped run, `ISOLATE` → its own token, `KEEP_SPLIT` → a
/// "keep-A" run, everything else → a "keep-B" run. keep-A and keep-B are cut apart, so this expresses
/// Whitespace's Word(=keep-A)/Symbol(=keep-B) boundary while Bert (`KEEP_SPLIT=0`) keeps one run type.
///
/// ┌── OWNER: shared (scalar core); SIMD boundary-extract optional ──┐
pub fn fsm_class_runs<const DROP: u16, const ISOLATE: u16, const KEEP_SPLIT: u16>(
    text: &[u8],
    tags: &[u8],
    out: &mut Vec<Span>,
) {
    let n = text.len();
    let mut open: Option<(usize, u8)> = None; // (run_start, keep-class 2|3) — a currently-open keep run
    let mut i = 0usize;
    while i < n {
        let s = i;
        i += 1;
        while i < n && text[i] & 0xC0 == 0x80 {
            i += 1; // char is [s, i)
        }
        let tag = tags[s];
        let class = if in_mask(tag, DROP) {
            0
        } else if in_mask(tag, ISOLATE) {
            1
        } else if in_mask(tag, KEEP_SPLIT) {
            2 // keep-A
        } else {
            3 // keep-B
        };
        match class {
            0 => {
                if let Some((st, _)) = open.take() {
                    out.push((st as u32, s as u32)); // flush the open run; drop this char
                }
            }
            1 => {
                if let Some((st, _)) = open.take() {
                    out.push((st as u32, s as u32));
                }
                out.push((s as u32, i as u32)); // isolate this char
            }
            c => match open {
                Some((_, oc)) if oc == c => {} // same keep class → extend the run
                Some((st, _)) => {
                    out.push((st as u32, s as u32)); // class changed → flush, open new
                    open = Some((s, c));
                }
                None => open = Some((s, c)),
            },
        }
    }
    if let Some((st, _)) = open {
        out.push((st as u32, n as u32));
    }
}

/// cl100k pretokenization (7 rules, segmented `{1,3}` cap + whitespace-tail). Scalar FSM — the
/// measured winner vs the boundary bitmask. Peeks `text` for the ASCII contraction-suffix literals.
/// ┌── OWNER: shared (scalar) ──┐
pub fn fsm_cl100k(text: &[u8], tags: &[u8], out: &mut Vec<Span>) {
    let _ = (text, tags, out);
    todo!("7-rule scalar FSM over atoms; contraction suffix via ASCII byte compare; spec §4")
}

/// GPT-2 / ByteLevel pretokenization: like cl100k but `\p{N}+` unbounded and no `[\r\n]` split.
/// ┌── OWNER: shared (scalar) ──┐
pub fn fsm_byte_level(text: &[u8], tags: &[u8], out: &mut Vec<Span>) {
    let _ = (text, tags, out);
    todo!()
}

/// UnicodeScripts: split on script-id change, transparent set `{Common, Inherited, Any}` sticks to
/// the neighbouring run. Reads a `scripts` stream (the parallel classifier), not atoms.
/// ┌── OWNER: shared (scalar) ──┐
pub fn fsm_script_run(text: &[u8], scripts: &[u8], out: &mut Vec<Span>) {
    let _ = (text, scripts, out);
    todo!()
}

/// SIMD boundary-extract for the RunSplit family: `in_mask` class-change → `movemask` → bit-iterate
/// to spans. Optional aarch64 acceleration of `fsm_split`/`fsm_class_runs`' scalar core.
/// ┌── OWNER: SIMD path ──┐
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn extract_boundaries(tags: &[u8], delim: u16, out: &mut Vec<Span>) {
    let _ = (tags, delim, out);
    todo!()
}

// ── Composition recipes ────────────────────────────────────────────────────────────────────────
// Each pre-tokenizer = (classify::<Atoms> → fsm shape + params). `tags` is caller-owned scratch (reused
// across calls, no per-call alloc). In `tk-encode` these delegate from the `pipeline::PreTokenizer`
// impls (offset conversion Span → pipeline::Split happens there).

pub struct WhitespaceSplit;
impl WhitespaceSplit {
    #[inline]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut Vec<Span>) {
        classify::<Atoms>(text, tags);
        fsm_split::<{ mask::WS }, { Behavior::Removed as u8 }>(text, tags, out);
    }
}

pub struct Punctuation;
impl Punctuation {
    #[inline]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut Vec<Span>) {
        classify::<Atoms>(text, tags);
        fsm_split::<{ mask::PUNCT }, { Behavior::Isolated as u8 }>(text, tags, out);
    }
}

pub struct Digits;
impl Digits {
    #[inline]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut Vec<Span>) {
        classify::<Atoms>(text, tags);
        fsm_split::<{ mask::NUMERIC }, { Behavior::Contiguous as u8 }>(text, tags, out);
    }
}

pub struct Whitespace;
impl Whitespace {
    #[inline]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut Vec<Span>) {
        classify::<Atoms>(text, tags);
        // drop WS runs, keep Word and Symbol runs (isolate nothing)
        fsm_class_runs::<{ mask::WS }, 0, { mask::WORD }>(text, tags, out);
    }
}

pub struct Bert;
impl Bert {
    #[inline]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut Vec<Span>) {
        classify::<Atoms>(text, tags);
        // drop WS runs, isolate punctuation, keep everything else as single runs
        fsm_class_runs::<{ mask::WS }, { mask::PUNCT }, 0>(text, tags, out);
    }
}

pub struct Cl100k;
impl Cl100k {
    #[inline]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut Vec<Span>) {
        classify::<Atoms>(text, tags);
        fsm_cl100k(text, tags, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spans<const D: u16, const B: u8>(s: &str) -> Vec<Span> {
        let mut tags = vec![0u8; s.len()];
        classify::<Atoms>(s.as_bytes(), &mut tags);
        let mut out = Vec::new();
        fsm_split::<D, B>(s.as_bytes(), &tags, &mut out);
        out
    }

    fn class_spans<const DR: u16, const IS: u16, const KS: u16>(s: &str) -> Vec<Span> {
        let mut tags = vec![0u8; s.len()];
        classify::<Atoms>(s.as_bytes(), &mut tags);
        let mut out = Vec::new();
        fsm_class_runs::<DR, IS, KS>(s.as_bytes(), &tags, &mut out);
        out
    }

    #[test]
    fn class_runs_whitespace_and_bert() {
        // "×" is U+00D7 (2 bytes) — Symbol, not \w, not \s, not \p{P}. So Whitespace cuts Word/Symbol
        // but Bert keeps them together. Byte offsets: a=0, ×=[1,3), b=3, ' '=4, c=5, !=6, d=7.
        let s = "a×b c!d";
        // Whitespace `\w+|[^\w\s]+`: "a" | "×" | "b" | "c" | "!" | "d"  (ws dropped, Word/Symbol cut)
        assert_eq!(
            class_spans::<{ mask::WS }, 0, { mask::WORD }>(s),
            vec![(0, 1), (1, 3), (3, 4), (5, 6), (6, 7), (7, 8)]
        );
        // Bert: drop ws, isolate punct, keep the rest as runs: "a×b" | "c" | "!" | "d"
        assert_eq!(
            class_spans::<{ mask::WS }, { mask::PUNCT }, 0>(s),
            vec![(0, 4), (5, 6), (6, 7), (7, 8)]
        );
    }

    #[test]
    fn split_behaviors() {
        // WhitespaceSplit — drop whitespace, keep the runs around it: "ab," | "cd!"
        assert_eq!(spans::<{ mask::WS }, { Behavior::Removed as u8 }>("ab, cd!"), vec![(0, 3), (4, 7)]);
        // Punctuation — isolate each punct char: "ab" | "," | " cd" | "!"
        assert_eq!(
            spans::<{ mask::PUNCT }, { Behavior::Isolated as u8 }>("ab, cd!"),
            vec![(0, 2), (2, 3), (3, 6), (6, 7)]
        );
        // Digits (contiguous) — group the digit run: "a" | "12" | "b" | "3"
        assert_eq!(
            spans::<{ mask::NUMERIC }, { Behavior::Contiguous as u8 }>("a12b3"),
            vec![(0, 1), (1, 3), (3, 4), (4, 5)]
        );
        // MergedWithPrevious — the space joins the previous piece: "a " | "b"
        assert_eq!(spans::<{ mask::WS }, { Behavior::MergedWithPrevious as u8 }>("a b"), vec![(0, 2), (2, 3)]);
        // MergedWithNext — the space joins the next piece: "a" | " b"
        assert_eq!(spans::<{ mask::WS }, { Behavior::MergedWithNext as u8 }>("a b"), vec![(0, 1), (1, 3)]);
    }
}
