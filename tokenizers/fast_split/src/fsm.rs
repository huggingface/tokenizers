//! FSM layer: turn the `Atom` tag stream into token spans. Every pre-tokenizer is one of these
//! shapes, parameterized by a class mask + behavior (all `const`-generic → fully monomorphized).
//! They all read the shared stream from `classify::classify::<Atoms>`; the delimiter *behavior* never
//! touches classification. See `TAG_CLASSIFY_SPEC.md` §4.
//!
//! Scalar cores are portable; the SIMD boundary-extract (`extract_boundaries`) is an aarch64
//! optimization for the RunSplit family (class-change → movemask → bit-iterate).
#![allow(dead_code)] // skeleton

use crate::classify::{char_len, classify, in_mask, mask, Atom, Atoms};

/// Advance over a maximal `m`-membership run; returns the byte index past it. Byte-wise (`i += 1`),
/// treating continuation bytes as in-run — so NO `char_len` branch per char and no `text` access. This
/// is THE hot inner loop of the dense (English/code) FSM; the earlier `char_len`-per-char version was
/// ~2× slower there (English fsm 2.3 → ~1.1 ns/byte). Keep this a tight, inlinable byte scan.
/// `inline(always)`: it's called once per token (~200K/MB on English) — a real call here doubles fsm cost.
#[inline(always)]
fn run_end(tags: &[u8], mut i: usize, end: usize, m: u16) -> usize {
    while i < end && (in_mask(tags[i], m) || tags[i] == Atom::Cont as u8) {
        i += 1;
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
    // Phase 1 — scalar for the first ≤16 bytes. Short runs (the dense English/code case) end here with
    // ZERO vector ops, so no SIMD tax; only a run that survives all 16 is "long" enough to vectorize.
    let lim = (i + 16).min(end);
    while i < lim && (in_mask(tags[i], m) || tags[i] == Atom::Cont as u8) {
        i += 1;
    }
    if i < lim {
        return i; // run ended within 16 → done
    }
    // Phase 2 — long run: skip 16 in-run tags at a time (vqtbl1 membership, vminvq == 0xFF = all member).
    let t = vld1q_u8(tbl16.as_ptr());
    while i + 16 <= end {
        if vminvq_u8(vqtbl1q_u8(t, vld1q_u8(tags.as_ptr().add(i)))) == 0xFF {
            i += 16;
        } else {
            break;
        }
    }
    // Phase 3 — scalar tail.
    while i < end && (in_mask(tags[i], m) || tags[i] == Atom::Cont as u8) {
        i += 1;
    }
    i
}

/// Pick SIMD or scalar run-end at monomorphization time. `lut` is the precomputed membership LUT for
/// `m`; only the SIMD path reads it (the scalar path ignores it).
#[inline(always)]
fn run_end_sel<const SIMD: bool>(
    tags: &[u8],
    i: usize,
    end: usize,
    m: u16,
    lut: &[u8; 16],
) -> usize {
    #[cfg(target_arch = "aarch64")]
    if SIMD {
        return unsafe { run_end_simd(tags, i, end, m, lut) };
    }
    let _ = (SIMD, lut);
    run_end(tags, i, end, m)
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
/// on run-heavy text. Byte-identical output to `fsm_cl100k`. On non-aarch64 the run-end falls back to
/// scalar (so this always exists; no `cfg` at call sites).
pub fn fsm_cl100k_simd(text: &[u8], tags: &[u8], out: &mut Vec<Span>) {
    cl100k::<true>(text, tags, out)
}

fn cl100k<const SIMD: bool>(text: &[u8], tags: &[u8], out: &mut Vec<Span>) {
    // Leading-atom values, as `const` so the `match` below is a dense jump table (not an if-cascade):
    // the dispatch is O(1) and a token never pays for a rule it can't start (e.g. non-number tokens
    // never test the number rule — which is what the POC's const-gating removed by hand; here it's free).
    const LET: u8 = Atom::Letter as u8;
    const NW: u8 = Atom::NumWord as u8;
    const NO: u8 = Atom::NumOther as u8;
    const NLN: u8 = Atom::Newline as u8;
    const SPC: u8 = Atom::Space as u8;
    const WSO: u8 = Atom::WsOther as u8;
    const MRK: u8 = Atom::Mark as u8;
    const CON: u8 = Atom::Connector as u8;
    const PUN: u8 = Atom::Punct as u8;
    const APO: u8 = Atom::Apostrophe as u8;
    const SYM: u8 = Atom::SymOther as u8;
    const NMO: u8 = Atom::NumericOther as u8;
    const CTL: u8 = Atom::Control as u8;
    // membership LUTs for the SIMD run-ends, built once (scalar path ignores them).
    let (lut_l, lut_o, lut_w) = (
        inmask_tbl(mask::LETTER),
        inmask_tbl(mask::NOT_WS_L_N),
        inmask_tbl(mask::WS),
    );
    let end = text.len();
    let letters = |a: usize| run_end_sel::<SIMD>(tags, a, end, mask::LETTER, &lut_l);
    // rule 4 body: `[^\s\p{L}\p{N}]+[\r\n]*` from `sp0` (any leading space already consumed). Returns
    // the run end, or `sp0` if there is no "other" run there (caller then treats it as whitespace).
    let other = |sp0: usize| -> usize {
        let mut p = run_end_sel::<SIMD>(tags, sp0, end, mask::NOT_WS_L_N, &lut_o);
        if p > sp0 {
            while p < end && tags[p] == NLN {
                p += char_len(text[p]);
            }
        }
        p
    };
    // rules 5-7: `\s*[\r\n]` | `\s+(?!\S)` | `\s+` — end of the whitespace token starting at `i`.
    let ws = |i: usize| -> usize {
        let re = run_end_sel::<SIMD>(tags, i, end, mask::WS, &lut_w);
        if let Some(r) = text[i..re].iter().rposition(|&x| x == 0x0A || x == 0x0D) {
            i + r + 1 // rule 5: up to & including the last newline
        } else if re == end {
            re // rule 7: trailing whitespace at EOF
        } else {
            // rule 6: before a non-space, leave the last ws char for the next token
            let mut last = re - 1;
            while last > i && text[last] & 0xC0 == 0x80 {
                last -= 1;
            }
            if last > i { last } else { re }
        }
    };

    let mut i = 0;
    while i < end {
        let start = i;
        let b = text[i];
        match tags[i] {
            // rule 2: `\p{L}+`
            LET => i = letters(i),
            // rule 3: `\p{N}{1,3}`
            NW | NO => {
                let (mut p, mut cnt) = (i, 0);
                while p < end && cnt < 3 && in_mask(tags[p], mask::NUMBER) {
                    p += char_len(text[p]);
                    cnt += 1;
                }
                i = p;
            }
            // Space: rule 2 (space prefix + `\p{L}+`) | rule 4 (` ` + "other") | rules 5-7
            SPC => {
                let a = i + 1; // Space is ASCII (0x20)
                i = if a < end && tags[a] == LET {
                    letters(a)
                } else {
                    let p = other(a);
                    if p > a { p } else { ws(i) }
                };
            }
            // WsOther: rule 2 (prefix + `\p{L}+`) | whitespace (never rule 4 — not in NOT_WS_L_N)
            WSO => {
                let a = i + char_len(b);
                i = if a < end && tags[a] == LET { letters(a) } else { ws(i) };
            }
            // Newline: whitespace (rule 5 ends at the last newline)
            NLN => i = ws(i),
            // Apostrophe: rule 1 (contraction) | rule 2 (prefix + `\p{L}+`) | rule 4
            APO => {
                let mut adv = 0;
                if i + 1 < end && text[i + 1] < 0x80 {
                    let lc = text[i + 1] | 0x20;
                    adv = match lc {
                        b's' | b't' | b'm' | b'd' => 2,
                        b'r' | b'v' | b'l' if i + 2 < end && text[i + 2] < 0x80 => {
                            let l2 = text[i + 2] | 0x20;
                            usize::from(
                                (lc == b'r' && l2 == b'e')
                                    || (lc == b'v' && l2 == b'e')
                                    || (lc == b'l' && l2 == b'l'),
                            ) * 3
                        }
                        _ => 0,
                    };
                }
                i = if adv > 0 {
                    i + adv
                } else {
                    let a = i + 1; // Apostrophe is ASCII (0x27)
                    if a < end && tags[a] == LET { letters(a) } else { other(i) } // c ∈ NOT_WS_L_N ⇒ > i
                };
            }
            // Mark | Connector | Punct | SymOther | NumericOther | Control (∈ PREFIX2 ∩ NOT_WS_L_N):
            // rule 2 (prefix + `\p{L}+`) | rule 4
            MRK | CON | PUN | SYM | NMO | CTL => {
                let a = i + char_len(b);
                i = if a < end && tags[a] == LET { letters(a) } else { other(i) }; // c ∈ NOT_WS_L_N ⇒ > i
            }
            // Sentinel / MultiByte / Cont — never a char-start atom; emit one char defensively.
            _ => i += char_len(b),
        }
        out.push((start as u32, i as u32));
    }
}

/// The specific CJK ranges deepseek's Split-2 isolates: Han U+4E00..9FA5 ∪ Hiragana U+3040..309F ∪
/// Katakana U+30A0..30FF. (All 3-byte, leads E3..E9.) Not "all letters" — only these.
#[inline]
fn ds_is_cjk(cp: u32) -> bool {
    (0x4E00..=0x9FA5).contains(&cp) || (0x3040..=0x30FF).contains(&cp)
}
/// Codepoint of a 3-byte UTF-8 char at `text[i]` (only called for leads E2..E9 → always 3-byte).
#[inline]
fn cp3(text: &[u8], i: usize) -> u32 {
    ((text[i] as u32 & 0x0F) << 12)
        | ((text[i + 1] as u32 & 0x3F) << 6)
        | (text[i + 2] as u32 & 0x3F)
}

/// A char that ENDS a deepseek alt-2 `[\p{L}\p{M}]+` run despite its atom being Letter/Mark: the
/// CJK-range chars (Split-2 isolated them first) and ZWJ/ZWNJ (U+200C/200D — `\p{Cf}`, so not
/// `\p{L}∪\p{M}`, but atom-folded into `Mark`). Both are 3-byte (leads E2..E9); cheap cp check.
/// `inline(always)`: called per lead byte in `letter_run`'s hot loop — a real call there is costly.
#[inline(always)]
fn ds_breaks(text: &[u8], p: usize) -> bool {
    let b = text[p];
    if (0xE3..=0xE9).contains(&b) {
        return ds_is_cjk(cp3(text, p));
    }
    if b == 0xE2 {
        let cp = cp3(text, p);
        return cp == 0x200C || cp == 0x200D;
    }
    false
}

/// A CJK-range char that is also a LETTER/MARK (`\p{L}∪\p{M}`). Split-2 `[…]+` isolates the whole CJK
/// range — including CJK-range *punctuation/symbols* (・ U+30FB, ゠ U+30A0, ゛゜ U+309B/C) — but Split-3
/// then RE-splits that piece, peeling those non-letters out via its punct rule. So the net Split-2⇒3
/// unit is a maximal CJK *letter* run; the non-letters must not extend it (they fall to the punct rule).
#[inline(always)]
fn ds_is_cjk_letter(text: &[u8], tags: &[u8], p: usize) -> bool {
    let b = text[p];
    (0xE3..=0xE9).contains(&b) && ds_is_cjk(cp3(text, p)) && in_mask(tags[p], mask::LETTER_MARK)
}

/// deepseek-v3 pretokenization: the `Sequence` of `[N{1,3}]`, `[CJK]+`, `<big regex>` (all Isolated)
/// collapsed into ONE scalar FSM over the atom stream. Precedence (= the Sequence order): digits →
/// CJK-range runs → the big-regex alts. Because Split-2 isolates CJK *before* the letter rule, the
/// letter run stops at CJK-range codepoints. Peeks bytes for the ASCII `[punct][A-Za-z]+` alt.
///
/// Byte-exact vs the real composed Sequence (onig ×3, each Isolated) on 10 languages — see
/// `benches/deepseek.rs`. The subtleties the single pass replicates: (1) ws *followed by* a digit/CJK
/// is its own Sequence piece → the whole run is one token (`\s+(?!\S)`); (2) ZWJ/ZWNJ are `\p{Cf}`, not
/// `\p{L}∪\p{M}`, so they end a letter run (`ds_breaks`); (3) Split-2 isolates the whole CJK range but
/// Split-3 re-splits it, so CJK-range punct (・) breaks the CJK *letter* run (`ds_is_cjk_letter`).
/// ┌── OWNER: shared (scalar) ──┐
pub fn fsm_deepseek(text: &[u8], tags: &[u8], out: &mut Vec<Span>) {
    // Leading-atom values as `const` → the `match` is a dense jump table (see `cl100k`). The Split
    // precedence (digits → CJK → big-regex alts) is preserved because the atom partition is disjoint.
    const LET: u8 = Atom::Letter as u8;
    const MRK: u8 = Atom::Mark as u8;
    const NW: u8 = Atom::NumWord as u8;
    const NO: u8 = Atom::NumOther as u8;
    const NLN: u8 = Atom::Newline as u8;
    const SPC: u8 = Atom::Space as u8;
    const WSO: u8 = Atom::WsOther as u8;
    const CON: u8 = Atom::Connector as u8;
    const PUN: u8 = Atom::Punct as u8;
    const APO: u8 = Atom::Apostrophe as u8;
    const SYM: u8 = Atom::SymOther as u8;
    const NMO: u8 = Atom::NumericOther as u8;
    const CTL: u8 = Atom::Control as u8;

    const CONT: u8 = Atom::Cont as u8;
    let end = text.len();
    // maximal `[\p{L}\p{M}]+` run from `a`, stopping at CJK-range chars (Split-2 took those) and
    // ZWJ/ZWNJ (not `\p{L}∪\p{M}` — see `ds_breaks`). BYTE-wise (`p += 1`, continuation bytes stay
    // in-run, `ds_breaks` only fires at a lead) — the `char_len`-per-char form was ~2× slower (see
    // `run_end`'s note). This is the hot inner loop of the dense (latin) deepseek path.
    let letter_run = |a: usize| -> usize {
        let mut p = a;
        while p < end {
            let t = tags[p];
            if t == CONT {
                p += 1;
            } else if in_mask(t, mask::LETTER_MARK) && !ds_breaks(text, p) {
                p += 1;
            } else {
                break;
            }
        }
        p
    };
    // is `text[a]` the start of a deepseek letter/mark char (alt-2 run body / space-prefix target)?
    let is_lm = |a: usize| a < end && in_mask(tags[a], mask::LETTER_MARK) && !ds_breaks(text, a);
    // Split-3 alt-3 tail `[\p{P}\p{S}]+[\r\n]*` from `sp0` (a leading space is already consumed); `sp0`
    // if there is no punct/sym run there.
    let punct = |sp0: usize| -> usize {
        let mut p = run_end(tags, sp0, end, mask::PUNCT_SYM);
        if p > sp0 {
            while p < end && tags[p] == NLN {
                p += char_len(text[p]);
            }
        }
        p
    };
    // Split-3 alts d/e/f (whitespace). Unlike cl100k: a ws run FOLLOWED BY a digit/CJK is its own
    // Sequence piece (Split-1/2 isolated the next match) → `\s+(?!\S)` takes the WHOLE run; only a
    // following letter/punct (same Split-3 piece) leaves the last ws char for its ` ?`/`[^…]?` prefix.
    let ws = |i: usize| -> usize {
        let re = run_end(tags, i, end, mask::WS);
        let next_isolated =
            re < end && (in_mask(tags[re], mask::NUMBER) || ds_is_cjk_letter(text, tags, re));
        if let Some(r) = text[i..re].iter().rposition(|&x| x == 0x0A || x == 0x0D) {
            i + r + 1
        } else if re == end || next_isolated {
            re // whole ws run is one token
        } else {
            let mut last = re - 1;
            while last > i && text[last] & 0xC0 == 0x80 {
                last -= 1;
            }
            if last > i { last } else { re }
        }
    };

    let mut i = 0;
    while i < end {
        let start = i;
        let b = text[i];
        match tags[i] {
            // Split-1: `\p{N}{1,3}`
            NW | NO => {
                let (mut p, mut cnt) = (i, 0);
                while p < end && cnt < 3 && in_mask(tags[p], mask::NUMBER) {
                    p += char_len(text[p]);
                    cnt += 1;
                }
                i = p;
            }
            // Split-2 CJK *letter* run (⇒ Split-3 re-split), else Split-3 alt-2 `[\p{L}\p{M}]+`. ZWJ/ZWNJ
            // aren't `\p{L}∪\p{M}`, but they ARE a valid alt-2 prefix `[^\r\n\p{L}\p{P}\p{S}]?`, so a ZWJ
            // FOLLOWED by a letter starts a "ZWJ + letters" token; otherwise it's its own gap token.
            LET | MRK => {
                i = if ds_is_cjk_letter(text, tags, i) {
                    let mut p = i + 3;
                    while p < end && ds_is_cjk_letter(text, tags, p) {
                        p += 3;
                    }
                    p
                } else if !ds_breaks(text, i) {
                    letter_run(i)
                } else {
                    let a = i + char_len(b); // ZWJ/ZWNJ as alt-2 prefix, else own token
                    if is_lm(a) { letter_run(a) } else { a }
                };
            }
            // Space: alt-2 (space prefix + `[\p{L}\p{M}]+`) | alt-3 (` ` + `[\p{P}\p{S}]+`) | whitespace
            SPC => {
                let a = i + 1; // Space is ASCII (0x20)
                i = if is_lm(a) {
                    letter_run(a)
                } else {
                    let p = punct(a);
                    if p > a { p } else { ws(i) }
                };
            }
            // WsOther: alt-2 (prefix + `[\p{L}\p{M}]+`) | whitespace (not `\p{P}∪\p{S}` → no alt-3)
            WSO => {
                let a = i + char_len(b);
                i = if is_lm(a) { letter_run(a) } else { ws(i) };
            }
            // Newline: whitespace
            NLN => i = ws(i),
            // Connector | Punct | Apostrophe | SymOther (∈ `\p{P}∪\p{S}`): alt-1 `[ascii_punct][A-Za-z]+`
            // | alt-3 `[\p{P}\p{S}]+[\r\n]*`
            CON | PUN | APO | SYM => {
                i = if b.is_ascii_punctuation() && i + 1 < end && text[i + 1].is_ascii_alphabetic() {
                    let mut p = i + 1;
                    while p < end && text[p].is_ascii_alphabetic() {
                        p += 1;
                    }
                    p
                } else {
                    punct(i) // c ∈ PUNCT_SYM ⇒ > i
                };
            }
            // NumericOther | Control (prefix candidates, not `\p{P}∪\p{S}`/ws): alt-2 prefix | own token
            NMO | CTL => {
                let a = i + char_len(b);
                i = if is_lm(a) { letter_run(a) } else { a };
            }
            // Sentinel / MultiByte / Cont — never a char-start atom; emit one char defensively.
            _ => i += char_len(b),
        }
        out.push((start as u32, i as u32));
    }
}

/// GPT-2 / ByteLevel pretokenization. Regex (same jump-table shape as cl100k, with 3 differences):
///   `'s|'t|'re|'ve|'m|'ll|'d | ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+ | \s+(?!\S) | \s+`
/// vs cl100k: (1) contractions are case-SENSITIVE (lowercase only, no `(?i:)`); (2) the ` ?` prefix is
/// a literal SPACE only (not any non-l/n char) and it applies to letters, numbers AND "other"; (3) no
/// `\p{N}{1,3}` cap (numbers are unbounded) and no `\s*[\r\n]`/trailing-`[\r\n]*` rules.
/// ┌── OWNER: shared (scalar) ──┐
pub fn fsm_byte_level(text: &[u8], tags: &[u8], out: &mut Vec<Span>) {
    const LET: u8 = Atom::Letter as u8;
    const NW: u8 = Atom::NumWord as u8;
    const NO: u8 = Atom::NumOther as u8;
    const NLN: u8 = Atom::Newline as u8;
    const SPC: u8 = Atom::Space as u8;
    const WSO: u8 = Atom::WsOther as u8;
    const MRK: u8 = Atom::Mark as u8;
    const CON: u8 = Atom::Connector as u8;
    const PUN: u8 = Atom::Punct as u8;
    const APO: u8 = Atom::Apostrophe as u8;
    const SYM: u8 = Atom::SymOther as u8;
    const NMO: u8 = Atom::NumericOther as u8;
    const CTL: u8 = Atom::Control as u8;

    let end = text.len();
    // `\s+(?!\S)|\s+`: the whole run at EOF, else leave the last ws char for the next ` ?`-prefixed run.
    let ws = |i: usize| -> usize {
        let re = run_end(tags, i, end, mask::WS);
        if re == end {
            re
        } else {
            let mut last = re - 1;
            while last > i && text[last] & 0xC0 == 0x80 {
                last -= 1;
            }
            if last > i { last } else { re }
        }
    };

    let mut i = 0;
    while i < end {
        let start = i;
        match tags[i] {
            LET => i = run_end(tags, i, end, mask::LETTER), // ` ?\p{L}+` (space taken by the Space arm)
            NW | NO => i = run_end(tags, i, end, mask::NUMBER), // ` ?\p{N}+` — UNBOUNDED
            MRK | CON | PUN | SYM | NMO | CTL => i = run_end(tags, i, end, mask::NOT_WS_L_N), // ` ?[^…]+`
            // `'s|'t|'re|'ve|'m|'ll|'d` (case-sensitive), else `[^\s\p{L}\p{N}]+` (apostrophe ∈ that set)
            APO => {
                let adv = match (text.get(i + 1), text.get(i + 2)) {
                    (Some(b's' | b't' | b'm' | b'd'), _) => 2,
                    (Some(b'r'), Some(b'e')) | (Some(b'v'), Some(b'e')) | (Some(b'l'), Some(b'l')) => 3,
                    _ => 0,
                };
                i = if adv > 0 { i + adv } else { run_end(tags, i, end, mask::NOT_WS_L_N) };
            }
            // Space: the ` ?` prefix — attach one space to a following letter / number / "other" run,
            // else it's whitespace (rules `\s+(?!\S)|\s+`, which leave one space for the next run).
            SPC => {
                let a = i + 1; // Space is ASCII (0x20)
                i = match tags.get(a) {
                    Some(&LET) => run_end(tags, a, end, mask::LETTER),
                    Some(&NW) | Some(&NO) => run_end(tags, a, end, mask::NUMBER),
                    Some(&t) if in_mask(t, mask::NOT_WS_L_N) => run_end(tags, a, end, mask::NOT_WS_L_N),
                    _ => ws(i),
                };
            }
            // WsOther / Newline: whitespace only — the ` ?` prefix is a literal 0x20, so tabs/newlines
            // never prefix a run.
            WSO | NLN => i = ws(i),
            // Sentinel / MultiByte / Cont — never a char-start atom; emit one char defensively.
            _ => i += char_len(text[i]),
        }
        out.push((start as u32, i as u32));
    }
}

/// UnicodeScripts: split on script-id change; the transparent set `{Common, Inherited, Any}` (id
/// [`SCRIPT_ANY`]) sticks to the current run without changing or splitting it. Reads a `scripts` stream
/// (from `classify::<Scripts>`), NOT atoms — `Scripts::CONT` (0xFF) marks continuation bytes, which
/// stay in-run byte-wise. A run's script is fixed by its first real (non-Any) char; adjacent real
/// scripts that differ are a boundary. `text` is unused (offsets only) but kept for a uniform signature.
/// ┌── OWNER: shared (scalar) ──┐
pub fn fsm_script_run(text: &[u8], scripts: &[u8], out: &mut Vec<Span>) {
    let _ = text;
    const CONT: u8 = 0xFF; // Scripts::CONT
    let n = scripts.len();
    let mut run_start = 0usize;
    let mut cur = SCRIPT_ANY; // the current run's fixed script (Any until a real one is seen)
    let mut i = 0;
    while i < n {
        let s = scripts[i];
        if s != CONT && s != SCRIPT_ANY {
            if cur != SCRIPT_ANY && s != cur {
                out.push((run_start as u32, i as u32)); // real script changed → boundary
                run_start = i;
            }
            cur = s;
        }
        i += 1; // byte-wise: CONT / Any chars stay in the current run
    }
    if run_start < n {
        out.push((run_start as u32, n as u32));
    }
}
/// Transparent script id for `{Common, Inherited, Any}` (the Scripts scheme must map those to 0).
pub const SCRIPT_ANY: u8 = 0;

/// SIMD boundary-extract for the RunSplit family: `in_mask` class-change → `movemask` → bit-iterate
/// to spans. Optional aarch64 acceleration of `fsm_split`/`fsm_class_runs`' scalar core.
/// ┌── OWNER: SIMD path ──┐
#[cfg(target_arch = "aarch64")]
pub(crate) unsafe fn extract_boundaries(tags: &[u8], delim: u16, out: &mut Vec<Span>) {
    let _ = (tags, delim, out);
    todo!()
}

/// Scalar `WhitespaceSplit` (drop WS runs, keep non-WS runs) — thin alias over the generic core, so the
/// SIMD version has a same-signature scalar twin to bench against.
pub fn whitespace_split_scalar(text: &[u8], tags: &[u8], out: &mut Vec<Span>) {
    fsm_split::<{ mask::WS }, { Behavior::Removed as u8 }>(text, tags, out)
}

/// SIMD (NEON) `WhitespaceSplit`: emit maximal non-whitespace byte runs (WS dropped), byte-identical to
/// [`whitespace_split_scalar`]. This is the simplest RunSplit — one class, `Removed` — so it's pure
/// boundary detection: a per-lane keep/drop `vqtbl1` + NEON movemask, then bit-iterate runs. No per-char
/// rule dispatch, no `segs` scratch, no double pass; whole non-WS chunks (words/CJK) cost one movemask.
///
/// `keep[tag] = 0xFF` for a kept byte, `0x00` for a whitespace *lead*. Continuation bytes (`Cont`) are
/// kept (→ multibyte non-WS chars stay whole); the sole cost is that the continuation bytes of a
/// non-ASCII whitespace char (NBSP, U+3000, …) also read as "kept" — `push` strips those leading
/// continuation bytes so the emitted token still starts on a char boundary, keeping it byte-exact.
#[cfg(target_arch = "aarch64")]
pub fn whitespace_split_simd(text: &[u8], tags: &[u8], out: &mut Vec<Span>) {
    use core::arch::aarch64::*;
    let n = text.len();

    // emit [s,e) after skipping leading continuation bytes (orphaned conts of a dropped multibyte WS
    // char). The `while` ~never fires (only at non-ASCII whitespace) — no cost on ASCII-spaced text.
    let push = |mut s: usize, e: usize, out: &mut Vec<Span>| {
        while s < e && text[s] & 0xC0 == 0x80 {
            s += 1;
        }
        if s < e {
            out.push((s as u32, e as u32));
        }
    };

    let mut run_start = usize::MAX; // usize::MAX = not currently in a run
    let mut i = 0;
    if n >= 16 {
        // keep LUT + NEON movemask weights (lane j → bit j; low 8 → low byte, high 8 → high byte).
        let mut keep = [0xFFu8; 16];
        let mut t = 0u8;
        while t < 16 {
            if (mask::WS >> t) & 1 != 0 {
                keep[t as usize] = 0x00; // Newline | Space | WsOther leads
            }
            t += 1;
        }
        const POW: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        unsafe {
            let keepv = vld1q_u8(keep.as_ptr());
            let powv = vld1q_u8(POW.as_ptr());
            while i + 16 <= n {
                let bits = vandq_u8(vqtbl1q_u8(keepv, vld1q_u8(tags.as_ptr().add(i))), powv);
                let km = (vaddv_u8(vget_low_u8(bits)) as u16)
                    | ((vaddv_u8(vget_high_u8(bits)) as u16) << 8);
                if km == 0xFFFF {
                    if run_start == usize::MAX {
                        run_start = i; // whole chunk kept → (re)open a run
                    }
                } else if km == 0 {
                    if run_start != usize::MAX {
                        push(run_start, i, out); // whole chunk dropped → close the run
                        run_start = usize::MAX;
                    }
                } else {
                    let mut j = 0;
                    while j < 16 {
                        if (km >> j) & 1 != 0 {
                            if run_start == usize::MAX {
                                run_start = i + j;
                            }
                        } else if run_start != usize::MAX {
                            push(run_start, i + j, out);
                            run_start = usize::MAX;
                        }
                        j += 1;
                    }
                }
                i += 16;
            }
        }
    }
    // scalar tail (Cont ∉ WS → kept, matching the LUT)
    while i < n {
        if in_mask(tags[i], mask::WS) {
            if run_start != usize::MAX {
                push(run_start, i, out);
                run_start = usize::MAX;
            }
        } else if run_start == usize::MAX {
            run_start = i;
        }
        i += 1;
    }
    if run_start != usize::MAX {
        push(run_start, n, out);
    }
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

pub struct DeepSeek;
impl DeepSeek {
    #[inline]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut Vec<Span>) {
        classify::<Atoms>(text, tags);
        fsm_deepseek(text, tags, out);
    }
}

pub struct ByteLevel;
impl ByteLevel {
    #[inline]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut Vec<Span>) {
        classify::<Atoms>(text, tags);
        fsm_byte_level(text, tags, out);
    }
}

/// `Split(char, Removed)` — the only pre-tokenizer that keys on a *literal char* rather than an atom
/// class, so it scans bytes directly (no classify pass). UTF-8 is self-synchronizing, so the
/// delimiter's byte pattern only matches on char boundaries.
pub struct CharDelimiterSplit(pub char);
impl CharDelimiterSplit {
    pub fn pre_tokenize(&self, text: &[u8], _tags: &mut [u8], out: &mut Vec<Span>) {
        let mut buf = [0u8; 4];
        let delim = self.0.encode_utf8(&mut buf).as_bytes();
        let (n, dl) = (text.len(), delim.len());
        let (mut start, mut i) = (0usize, 0usize);
        while i + dl <= n {
            // memchr the first delimiter byte, then confirm the full pattern.
            match memchr::memchr(delim[0], &text[i..n - dl + 1]) {
                Some(off) if text[i + off..i + off + dl] == *delim => {
                    let m = i + off;
                    if m > start {
                        out.push((start as u32, m as u32)); // gap before the delimiter (Removed)
                    }
                    i = m + dl;
                    start = i;
                }
                Some(off) => i += off + 1, // first byte matched mid-pattern; keep scanning
                None => break,
            }
        }
        if start < n {
            out.push((start as u32, n as u32));
        }
    }
}

// UnicodeScripts (`classify::<Scripts>` → `fsm_script_run`) is intentionally NOT wired up as a struct
// yet: the `Scripts` scheme's SCRIPT_TABLES are Phase 2 (a new unicode-script data source in
// `bitmap_gen`), so `classify::<Scripts>` would panic. `fsm_script_run` above is done and tested.

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

    fn deepseek(s: &str) -> Vec<Span> {
        let mut tags = vec![0u8; s.len()];
        classify::<Atoms>(s.as_bytes(), &mut tags);
        let mut out = Vec::new();
        fsm_deepseek(s.as_bytes(), &tags, &mut out);
        out
    }

    #[test]
    fn deepseek_rules() {
        // 中 = U+4E2D (3 bytes) is CJK-range → isolated out of the letter runs (Split-2 before Split-3)
        assert_eq!(deepseek("abc中def"), vec![(0, 3), (3, 6), (6, 9)]); // letters | CJK | letters
        assert_eq!(deepseek("abc123"), vec![(0, 3), (3, 6)]); // letters | digits {1,3}
        assert_eq!(deepseek("_abc"), vec![(0, 4)]); // alt-1: ASCII punct + ASCII letters
        assert_eq!(deepseek("hello world"), vec![(0, 5), (5, 11)]); // word | space+word (alt-2 prefix)
        assert_eq!(deepseek("!!!"), vec![(0, 3)]); // \p{P}∪\p{S} run (alt-3)
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
        assert_eq!(
            spans::<{ mask::WS }, { Behavior::Removed as u8 }>("ab, cd!"),
            vec![(0, 3), (4, 7)]
        );
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
        assert_eq!(
            spans::<{ mask::WS }, { Behavior::MergedWithPrevious as u8 }>("a b"),
            vec![(0, 2), (2, 3)]
        );
        // MergedWithNext — the space joins the next piece: "a" | " b"
        assert_eq!(
            spans::<{ mask::WS }, { Behavior::MergedWithNext as u8 }>("a b"),
            vec![(0, 1), (1, 3)]
        );
    }

    fn byte_level(s: &str) -> Vec<Span> {
        let mut tags = vec![0u8; s.len()];
        classify::<Atoms>(s.as_bytes(), &mut tags);
        let mut out = Vec::new();
        fsm_byte_level(s.as_bytes(), &tags, &mut out);
        out
    }

    #[test]
    fn byte_level_rules() {
        // ` ?\p{L}+` (space attaches), lowercase contraction, ` ?\p{N}+` UNBOUNDED (no {1,3} cap).
        // "I" | "'m" | " 12345" | " ok"
        assert_eq!(byte_level("I'm 12345 ok"), vec![(0, 1), (1, 3), (3, 9), (9, 12)]);
        // contractions are CASE-SENSITIVE: uppercase 'S is not one → "IT" | "'" | "S"
        assert_eq!(byte_level("IT'S"), vec![(0, 2), (2, 3), (3, 4)]);
        // multi-space: `\s+(?!\S)` leaves one space for the next word → "hi" | "  " ... wait: "hi   ok"
        // → "hi" | "  " (2 of 3, rule \s+(?!\S)) | " ok" (rule ` ?\p{L}+`)
        assert_eq!(byte_level("hi   ok"), vec![(0, 2), (2, 4), (4, 7)]);
    }

    #[test]
    fn script_run_basic() {
        // ids per byte; ANY(0) sticks, CONT(0xFF) stays in-run. "aa" Latin | "." Any | "bb" Cyr.
        assert_eq!(
            {
                let mut o = Vec::new();
                fsm_script_run(b"aa.bb", &[1, 1, 0, 2, 2], &mut o);
                o
            },
            vec![(0, 3), (3, 5)] // the "." (Any) sticks to the preceding Latin run
        );
        // multibyte: "é"(Latin, lead+CONT) | "Б"(Cyr, lead+CONT)
        assert_eq!(
            {
                let mut o = Vec::new();
                fsm_script_run("éБ".as_bytes(), &[1, 0xFF, 2, 0xFF], &mut o);
                o
            },
            vec![(0, 2), (2, 4)]
        );
    }

    #[test]
    fn char_delimiter_split() {
        let mut out = Vec::new();
        // split on '/', Removed → drop delimiters, drop the empty gap between "//"
        CharDelimiterSplit('/').pre_tokenize(b"a/bc//d", &mut [], &mut out);
        assert_eq!(out, vec![(0, 1), (2, 4), (6, 7)]);
    }
}
