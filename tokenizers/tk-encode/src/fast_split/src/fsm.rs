//! FSM layer: turn the `Atom` tag stream into token spans. Every pre-tokenizer is one of these
//! shapes, parameterized by a class mask + behavior (all `const`-generic → fully monomorphized).
//! They all read the shared stream from `classify::classify_atoms`; the delimiter *behavior* never
//! touches classification. See `TAG_CLASSIFY_SPEC.md` §4.
//!
//! Scalar cores are portable; the SIMD boundary-extract (`extract_boundaries`) is an aarch64
//! optimization for the RunSplit family (class-change → movemask → bit-iterate).
#![allow(dead_code)] // skeleton

use crate::classify::{classify_atoms, mask};

/// A token span: byte offsets `[start, end)` into the input.
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
pub fn fsm_split<const DELIM: u16, const BEHAVIOR: u8>(text: &[u8], tags: &[u8], out: &mut Vec<Span>) {
    let _ = (text, tags, out);
    todo!("match-stream = in_mask(tag, DELIM); emit per Behavior::from(BEHAVIOR); spec §4")
}

/// Cut at *every* class change; drop `DROP`-class runs, isolate `ISOLATE`-class per char. Keeps more
/// than one run type, so it isn't a single `fsm_split`. Covers: Whitespace (drop WS, keep Word+Symbol
/// runs), Bert (drop WS, isolate Punct).
///
/// ┌── OWNER: shared (scalar core); SIMD boundary-extract optional ──┐
pub fn fsm_class_runs<const DROP: u16, const ISOLATE: u16>(
    text: &[u8],
    tags: &[u8],
    out: &mut Vec<Span>,
) {
    let _ = (text, tags, out);
    todo!("cut at class change; drop DROP runs, isolate ISOLATE per char; spec §4")
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
// Each pre-tokenizer = (classify_atoms → fsm shape + params). `tags` is caller-owned scratch (reused
// across calls, no per-call alloc). In `tk-encode` these delegate from the `pipeline::PreTokenizer`
// impls (offset conversion Span → pipeline::Split happens there).

pub struct WhitespaceSplit;
impl WhitespaceSplit {
    #[inline]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut Vec<Span>) {
        classify_atoms(text, tags);
        fsm_split::<{ mask::WS }, { Behavior::Removed as u8 }>(text, tags, out);
    }
}

pub struct Punctuation;
impl Punctuation {
    #[inline]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut Vec<Span>) {
        classify_atoms(text, tags);
        fsm_split::<{ mask::PUNCT }, { Behavior::Isolated as u8 }>(text, tags, out);
    }
}

pub struct Digits;
impl Digits {
    #[inline]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut Vec<Span>) {
        classify_atoms(text, tags);
        fsm_split::<{ mask::NUMERIC }, { Behavior::Contiguous as u8 }>(text, tags, out);
    }
}

pub struct Whitespace;
impl Whitespace {
    #[inline]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut Vec<Span>) {
        classify_atoms(text, tags);
        // drop WS runs, keep Word and Symbol runs (isolate nothing)
        fsm_class_runs::<{ mask::WS }, 0>(text, tags, out);
    }
}

pub struct Cl100k;
impl Cl100k {
    #[inline]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut Vec<Span>) {
        classify_atoms(text, tags);
        fsm_cl100k(text, tags, out);
    }
}
