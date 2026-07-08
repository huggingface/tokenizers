//! FSM layer: turn the `Atom` tag stream into token spans. Every pre-tokenizer is one of these
//! shapes, parameterized by a class mask + behavior (all `const`-generic → fully monomorphized).
//! They all read the shared stream from `classify::classify::<Atoms>`; the delimiter *behavior* never
//! touches classification. See `TAG_CLASSIFY_SPEC.md` §4.
//!
//! Scalar cores are portable; the SIMD boundary-extract (`extract_boundaries`) is an aarch64
//! optimization for the RunSplit family (class-change → movemask → bit-iterate).
#![allow(dead_code)] // skeleton

use crate::classify::{Atom, Atoms, char_len, classify, in_mask, mask};

/// Advance from `i` over maximal chars whose (lead) tag is in `m`; returns the byte index past the run.
#[inline]
fn run_end(text: &[u8], tags: &[u8], mut i: usize, end: usize, m: u16) -> usize {
    while i < end && in_mask(tags[i], m) {
        i += char_len(text[i]);
    }
    i
}

/// Membership LUT for the SIMD run-end: `lut[tag] = 0xFF` iff `tag` is in `m` OR is a continuation
/// byte (so a multibyte char's continuation bytes stay inside the run). Built ONCE per FSM call (not
/// per run) and reused — the per-call rebuild is what made an earlier version slow.
fn inmask_tbl(m: u16) -> [u8; 16] {
    let mut a = [0u8; 16];
    let mut t = 0u8;
    while t < 16 {
        if (m >> t) & 1 != 0 || t == Atom::Cont as u8 {
            a[t as usize] = 0xFF;
        }
        t += 1;
    }
    a
}

/// SIMD run-end (NEON): bulk-skip whole 16-tag chunks that stay in-run — `vqtbl1` membership then
/// `vminvq == 0xFF` (all lanes members) — and scalar-finish the partial tail. `tbl16` is the
/// precomputed `inmask_tbl(m)`. Same result as `run_end`; wins on run-heavy text.
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn run_end_simd(tags: &[u8], mut i: usize, end: usize, m: u16, tbl16: &[u8; 16]) -> usize {
    use core::arch::aarch64::*;
    let t = vld1q_u8(tbl16.as_ptr());
    while i + 16 <= end {
        let chunk = vld1q_u8(tags.as_ptr().add(i));
        if vminvq_u8(vqtbl1q_u8(t, chunk)) == 0xFF {
            i += 16;
        } else {
            break;
        }
    }
    while i < end && (in_mask(tags[i], m) || tags[i] == Atom::Cont as u8) {
        i += 1;
    }
    i
}

/// Pick SIMD or scalar run-end at monomorphization time. `lut` is the precomputed membership LUT for
/// `m`; only the SIMD path reads it (the scalar path ignores it).
#[inline]
fn run_end_sel<const SIMD: bool>(text: &[u8], tags: &[u8], i: usize, end: usize, m: u16, lut: &[u8; 16]) -> usize {
    #[cfg(target_arch = "aarch64")]
    if SIMD {
        return unsafe { run_end_simd(tags, i, end, m, lut) };
    }
    let _ = (SIMD, lut);
    run_end(text, tags, i, end, m)
}

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

/// cl100k pretokenization (7 rules, segmented `{1,3}` cap + whitespace-tail). Peeks `text` for the
/// ASCII contraction-suffix literals. Scalar run-ends.
/// ┌── OWNER: shared (scalar) ──┐
pub fn fsm_cl100k(text: &[u8], tags: &[u8], out: &mut Vec<Span>) {
    cl100k::<false>(text, tags, out)
}

/// Same, with SIMD (NEON) run-ends (`run_end_simd`) for the letter / symbol / whitespace runs — wins
/// on run-heavy text. Byte-identical output to `fsm_cl100k`.
#[cfg(target_arch = "aarch64")]
pub fn fsm_cl100k_simd(text: &[u8], tags: &[u8], out: &mut Vec<Span>) {
    cl100k::<true>(text, tags, out)
}

fn cl100k<const SIMD: bool>(text: &[u8], tags: &[u8], out: &mut Vec<Span>) {
    const AP: u8 = Atom::Apostrophe as u8;
    const L: u8 = Atom::Letter as u8;
    const SP: u8 = Atom::Space as u8;
    const NL: u8 = Atom::Newline as u8;
    // rule-2 optional prefix `[^\r\n\p{L}\p{N}]` = every atom except Newline / Letter / Number.
    const PREFIX2: u16 = Atom::Space.bit()
        | Atom::WsOther.bit()
        | Atom::Mark.bit()
        | Atom::Connector.bit()
        | Atom::Punct.bit()
        | Atom::Apostrophe.bit()
        | Atom::SymOther.bit()
        | Atom::Control.bit()
        | Atom::NumericOther.bit();
    // membership LUTs for the SIMD run-ends, built once (scalar path ignores them).
    let (lut_l, lut_o, lut_w) = (
        inmask_tbl(mask::LETTER),
        inmask_tbl(mask::NOT_WS_L_N),
        inmask_tbl(mask::WS),
    );

    let end = text.len();
    let mut i = 0;
    while i < end {
        let start = i;
        let b = text[i];
        // rule 1: `'(?i:[sdmt]|ll|ve|re)` — apostrophe + ASCII contraction suffix (peek bytes)
        if tags[i] == AP && i + 1 < end && text[i + 1] < 0x80 {
            let lc = text[i + 1] | 0x20;
            let adv = match lc {
                b's' | b't' | b'm' | b'd' => 2,
                b'r' | b'v' | b'l' if i + 2 < end && text[i + 2] < 0x80 => {
                    let l2 = text[i + 2] | 0x20;
                    if (lc == b'r' && l2 == b'e') || (lc == b'v' && l2 == b'e') || (lc == b'l' && l2 == b'l') {
                        3
                    } else {
                        0
                    }
                }
                _ => 0,
            };
            if adv > 0 {
                out.push((start as u32, (start + adv) as u32));
                i += adv;
                continue;
            }
        }
        let c = tags[i];
        // rule 2: `[^\r\n\p{L}\p{N}]?\p{L}+` — letters, with at most one non-(nl/l/n) prefix char
        if c == L {
            i = run_end_sel::<SIMD>(text, tags, i, end, mask::LETTER, &lut_l);
            out.push((start as u32, i as u32));
            continue;
        }
        let l0 = char_len(b);
        if in_mask(c, PREFIX2) {
            let a = i + l0;
            if a < end && tags[a] == L {
                i = run_end_sel::<SIMD>(text, tags, a, end, mask::LETTER, &lut_l);
                out.push((start as u32, i as u32));
                continue;
            }
        }
        // rule 3: `\p{N}{1,3}` — 1..3 number chars
        if in_mask(c, mask::NUMBER) {
            let (mut p, mut cnt) = (i, 0);
            while p < end && cnt < 3 && in_mask(tags[p], mask::NUMBER) {
                p += char_len(text[p]);
                cnt += 1;
            }
            out.push((start as u32, p as u32));
            i = p;
            continue;
        }
        // rule 4: ` ?[^\s\p{L}\p{N}]+[\r\n]*` — optional space, run of "other", trailing newlines
        let sp0 = if c == SP { i + l0 } else { i };
        let mut p = run_end_sel::<SIMD>(text, tags, sp0, end, mask::NOT_WS_L_N, &lut_o);
        if p > sp0 {
            while p < end && tags[p] == NL {
                p += char_len(text[p]);
            }
            out.push((start as u32, p as u32));
            i = p;
            continue;
        }
        // rules 5-7: whitespace — `\s*[\r\n]` | `\s+(?!\S)` | `\s+`
        if in_mask(c, mask::WS) {
            let re = run_end_sel::<SIMD>(text, tags, i, end, mask::WS, &lut_w);
            if let Some(r) = text[i..re].iter().rposition(|&x| x == 0x0A || x == 0x0D) {
                i = i + r + 1; // rule 5: up to & including the last newline in the run
            } else if re == end {
                i = re; // rule 7: trailing whitespace at EOF → take it all
            } else {
                // rule 6: whitespace before a non-space → leave the last ws char for the next token
                let mut last = re - 1;
                while last > i && text[last] & 0xC0 == 0x80 {
                    last -= 1;
                }
                i = if last > i { last } else { re };
            }
            out.push((start as u32, i as u32));
            continue;
        }
        i += l0;
        out.push((start as u32, i as u32));
    }
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

    fn cl100k(s: &str) -> Vec<Span> {
        let mut tags = vec![0u8; s.len()];
        classify::<Atoms>(s.as_bytes(), &mut tags);
        let mut out = Vec::new();
        fsm_cl100k(s.as_bytes(), &tags, &mut out);
        out
    }

    #[test]
    fn cl100k_rules() {
        // hand-verified against the tiktoken cl100k_base regex
        assert_eq!(cl100k("Hello world"), vec![(0, 5), (5, 11)]); // "Hello" | " world" (space attaches)
        assert_eq!(cl100k("don't"), vec![(0, 3), (3, 5)]); // "don" | "'t" (contraction, rule 1)
        assert_eq!(cl100k("a1234"), vec![(0, 1), (1, 4), (4, 5)]); // "a" | "123" | "4" ({1,3} digit cap)
        assert_eq!(cl100k("  hi"), vec![(0, 1), (1, 4)]); // " " | " hi" (leave 1 ws for next word)
        assert_eq!(cl100k("a, b"), vec![(0, 1), (1, 2), (2, 4)]); // "a" | "," | " b"
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
