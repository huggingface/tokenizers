//! TODO: this is the only file left to review and push to the bring of performances.
//!
//! FSM layer: turn the `Atom` tag stream into token spans. Every pre-tokenizer is one of these
//! shapes, parameterized by a class mask + behavior (all `const`-generic → fully monomorphized).
//! They all read the shared stream from `classify::classify::<Atoms>`; the delimiter *behavior* never
//! touches classification. See `TAG_CLASSIFY_SPEC.md` §4.
//!
//! Scalar cores are portable; the SIMD boundary-extract (`extract_boundaries`) is an aarch64
//! optimization for the RunSplit family (class-change → movemask → bit-iterate).
#![allow(dead_code)] // skeleton

use crate::classify::{Atom, Atoms, char_len, classify, in_mask, mask};

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

/// No-`push` class-family pre-tokenizer core: writes spans into the preallocated `out` slice and returns
/// the count — no `Vec`, no realloc. ONE shape covers the whole class family via `<DROP, ISOLATE, KEEP_A>`:
///   WhitespaceSplit `<{WS},0,0>` · Punctuation `<0,{PUNCT},0>` · Digits `<0,0,{NUMERIC}>` ·
///   Whitespace `<{WS},0,{WORD}>` · Bert `<{WS},{PUNCT},0>`.
/// Class of a char: `DROP`→dropped, `ISOLATE`→own token, `KEEP_A`→run "A", else→run "B" (A/B cut apart).
/// aarch64 uses the NEON boundary extractor ([`class_runs_neon`]); elsewhere the run-end core. Byte-exact
/// with the `Vec` fsms and across both paths (see `class_runs_into_matches`). `out.len()` ≥ `text.len()`.
#[inline]
pub fn class_runs_into<const DROP: u16, const ISOLATE: u16, const KEEP_A: u16>(
    text: &[u8],
    tags: &[u8],
    out: &mut [Span],
) -> usize {
    #[cfg(target_arch = "aarch64")]
    {
        class_runs_neon::<DROP, ISOLATE, KEEP_A>(text, tags, out)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        class_runs_runend::<DROP, ISOLATE, KEEP_A>(text, tags, out)
    }
}

/// Run-end core: skip each homogeneous run with `run_end` (bulk skip; NEON `vqtbl` on aarch64). Wins on
/// long runs (Digits/Punct); the portable path and a test oracle. No `Vec` — writes into `out`.
pub fn class_runs_runend<const DROP: u16, const ISOLATE: u16, const KEEP_A: u16>(
    text: &[u8],
    tags: &[u8],
    out: &mut [Span],
) -> usize {
    let mb: u16 = !(DROP | ISOLATE | KEEP_A); // keep-B = everything else (Cont rides along in run_end)
    let (lut_d, lut_a, lut_b) = (inmask_tbl(DROP), inmask_tbl(KEEP_A), inmask_tbl(mb));
    let n = text.len();
    let (mut w, mut i) = (0usize, 0usize);
    while i < n {
        let t = tags[i];
        if in_mask(t, DROP) {
            i = run_end_sel::<true>(tags, i, n, DROP, &lut_d); // skip the whole drop run at once
        } else if in_mask(t, ISOLATE) {
            let s = i;
            i += char_len(text[i]);
            out[w] = (s as u32, i as u32); // isolate: one char = one token
            w += 1;
        } else {
            let s = i;
            i = if in_mask(t, KEEP_A) {
                run_end_sel::<true>(tags, i, n, KEEP_A, &lut_a)
            } else {
                run_end_sel::<true>(tags, i, n, mb, &lut_b)
            };
            out[w] = (s as u32, i as u32);
            w += 1;
        }
    }
    w
}

/// NEON movemask boundary-extract: per 16 tags → class via `vqtbl1` (Cont→`0xFF`), fill Cont lanes with
/// the left neighbour's class (≤3 iters = max continuation bytes), then boundary = class-change | isolate
/// lead, restricted to leads → `movemask` → iterate set bits, emitting one span per non-`DROP` segment
/// into `out`. Finds every boundary in a 16-chunk in parallel, so short-run text (English words) isn't
/// per-char. Open segment + fill/prev carries cross chunks; a scalar tail finishes the < 16-byte remainder.
#[cfg(target_arch = "aarch64")]
fn class_runs_neon<const DROP: u16, const ISOLATE: u16, const KEEP_A: u16>(
    text: &[u8],
    tags: &[u8],
    out: &mut [Span],
) -> usize {
    use core::arch::aarch64::*;
    const CONT: u8 = Atom::Cont as u8;
    // class LUT: tag → 0 drop / 1 isolate / 2 keep-A / 3 keep-B; Cont → 0xFF (fill sentinel).
    let mut lut = [3u8; 16];
    let mut t = 0u8;
    while t < 16 {
        lut[t as usize] = if t == CONT {
            0xFF
        } else if (DROP >> t) & 1 != 0 {
            0
        } else if (ISOLATE >> t) & 1 != 0 {
            1
        } else if (KEEP_A >> t) & 1 != 0 {
            2
        } else {
            3
        };
        t += 1;
    }
    const POW: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let n = text.len();
    let (mut w, mut i) = (0usize, 0usize);
    let (mut seg_start, mut seg_class) = (0usize, 0u8); // seg_class 0 (drop) → first close emits nothing
    let mut carry: u8 = 0xFE; // class "before" pos 0 — impossible → forces a boundary at 0
    let mut cls_arr = [0u8; 16];
    unsafe {
        let lutv = vld1q_u8(lut.as_ptr());
        let powv = vld1q_u8(POW.as_ptr());
        let contv = vdupq_n_u8(0xFF);
        let onev = vdupq_n_u8(1);
        while i + 16 <= n {
            let v = vld1q_u8(tags.as_ptr().add(i));
            let raw = vqtbl1q_u8(lutv, v); // class per lane, Cont → 0xFF
            // fast path (long runs): whole chunk is the current segment's class or continuation → no
            // boundary; extend and skip all the fill/boundary work. Recovers the run-end bulk-skip for
            // Digits/Punct/CJK. Not for isolate segments (those need a boundary per char).
            if seg_class != 1 {
                let ok = vorrq_u8(vceqq_u8(raw, vdupq_n_u8(seg_class)), vceqq_u8(raw, contv));
                if vminvq_u8(ok) == 0xFF {
                    carry = seg_class;
                    i += 16;
                    continue;
                }
            }
            // fill Cont lanes with the left neighbour's class (≤3 iters covers a 4-byte char's 3 conts)
            let mut cls = raw;
            let mut k = 0;
            while k < 3 {
                let shifted = vextq_u8::<15>(vdupq_n_u8(carry), cls); // [carry, cls0..cls14]
                cls = vbslq_u8(vceqq_u8(cls, contv), shifted, cls);
                k += 1;
            }
            let prev = vextq_u8::<15>(vdupq_n_u8(carry), cls);
            let changed = vmvnq_u8(vceqq_u8(cls, prev)); // class changed vs left neighbour
            let is_iso = vceqq_u8(cls, onev);
            let is_lead = vmvnq_u8(vceqq_u8(raw, contv)); // raw != 0xFF (a char lead)
            let bnd = vandq_u8(vorrq_u8(changed, vandq_u8(is_iso, is_lead)), is_lead);
            let bits = vandq_u8(bnd, powv);
            let mm = (vaddv_u8(vget_low_u8(bits)) as u16)
                | ((vaddv_u8(vget_high_u8(bits)) as u16) << 8);
            vst1q_u8(cls_arr.as_mut_ptr(), cls);
            let mut m = mm;
            while m != 0 {
                let j = m.trailing_zeros() as usize;
                let pos = i + j;
                if seg_class != 0 {
                    out[w] = (seg_start as u32, pos as u32);
                    w += 1;
                }
                seg_start = pos;
                seg_class = cls_arr[j];
                m &= m - 1;
            }
            carry = cls_arr[15];
            i += 16;
        }
    }
    // scalar tail (< 16 bytes; MAY start mid-char — the chunk loop steps by 16, not by char): per-char
    // boundary, continuing the open segment. A continuation byte stays in the current segment; advance
    // ONE byte (`char_len` is only valid on a lead, so never call it on a cont).
    while i < n {
        let s = i;
        let r = tags[s];
        if r == CONT {
            i += 1;
            continue;
        }
        let c = if in_mask(r, DROP) {
            0
        } else if in_mask(r, ISOLATE) {
            1
        } else if in_mask(r, KEEP_A) {
            2
        } else {
            3
        };
        i += char_len(text[s]);
        if c != seg_class || c == 1 || seg_class == 1 {
            if seg_class != 0 {
                out[w] = (seg_start as u32, s as u32);
                w += 1;
            }
            seg_start = s;
            seg_class = c;
        }
    }
    if seg_class != 0 {
        out[w] = (seg_start as u32, n as u32);
        w += 1;
    }
    w
}

/// cl100k pretokenization (7 rules, segmented `{1,3}` cap + whitespace-tail). Peeks `text` for the
/// ASCII contraction-suffix literals. Scalar run-ends.
/// ┌── OWNER: shared (scalar) ──┐
pub fn fsm_cl100k(text: &[u8], tags: &[u8], out: &mut [Span]) -> usize {
    cl100k::<false>(text, tags, out)
}

/// Same, with SIMD (NEON) run-ends (`run_end_simd`) for the letter / symbol / whitespace runs — wins
/// on run-heavy text. Byte-identical output to `fsm_cl100k`. On non-aarch64 the run-end falls back to
/// scalar (so this always exists; no `cfg` at call sites).
pub fn fsm_cl100k_simd(text: &[u8], tags: &[u8], out: &mut [Span]) -> usize {
    cl100k::<true>(text, tags, out)
}

fn cl100k<const SIMD: bool>(text: &[u8], tags: &[u8], out: &mut [Span]) -> usize {
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
    let mut w = 0usize;
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
                i = if a < end && tags[a] == LET {
                    letters(a)
                } else {
                    ws(i)
                };
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
                    if a < end && tags[a] == LET {
                        letters(a)
                    } else {
                        other(i)
                    } // c ∈ NOT_WS_L_N ⇒ > i
                };
            }
            // Mark | Connector | Punct | SymOther | NumericOther | Control (∈ PREFIX2 ∩ NOT_WS_L_N):
            // rule 2 (prefix + `\p{L}+`) | rule 4
            MRK | CON | PUN | SYM | NMO | CTL => {
                let a = i + char_len(b);
                i = if a < end && tags[a] == LET {
                    letters(a)
                } else {
                    other(i)
                }; // c ∈ NOT_WS_L_N ⇒ > i
            }
            // Sentinel / MultiByte / Cont — never a char-start atom; emit one char defensively.
            _ => i += char_len(b),
        }
        out[w] = (start as u32, i as u32);
        w += 1;
    }
    w
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
pub fn fsm_deepseek(text: &[u8], tags: &[u8], out: &mut [Span]) -> usize {
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
    let mut w = 0usize;
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
                i = if b.is_ascii_punctuation() && i + 1 < end && text[i + 1].is_ascii_alphabetic()
                {
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
        out[w] = (start as u32, i as u32);
        w += 1;
    }
    w
}

/// GPT-2 / ByteLevel pretokenization. Regex (same jump-table shape as cl100k, with 3 differences):
///   `'s|'t|'re|'ve|'m|'ll|'d | ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+ | \s+(?!\S) | \s+`
/// vs cl100k: (1) contractions are case-SENSITIVE (lowercase only, no `(?i:)`); (2) the ` ?` prefix is
/// a literal SPACE only (not any non-l/n char) and it applies to letters, numbers AND "other"; (3) no
/// `\p{N}{1,3}` cap (numbers are unbounded) and no `\s*[\r\n]`/trailing-`[\r\n]*` rules.
/// ┌── OWNER: shared (scalar) ──┐
pub fn fsm_byte_level(text: &[u8], tags: &[u8], out: &mut [Span]) -> usize {
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
    let mut w = 0usize;
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
                    (Some(b'r'), Some(b'e'))
                    | (Some(b'v'), Some(b'e'))
                    | (Some(b'l'), Some(b'l')) => 3,
                    _ => 0,
                };
                i = if adv > 0 {
                    i + adv
                } else {
                    run_end(tags, i, end, mask::NOT_WS_L_N)
                };
            }
            // Space: the ` ?` prefix — attach one space to a following letter / number / "other" run,
            // else it's whitespace (rules `\s+(?!\S)|\s+`, which leave one space for the next run).
            SPC => {
                let a = i + 1; // Space is ASCII (0x20)
                i = match tags.get(a) {
                    Some(&LET) => run_end(tags, a, end, mask::LETTER),
                    Some(&NW) | Some(&NO) => run_end(tags, a, end, mask::NUMBER),
                    Some(&t) if in_mask(t, mask::NOT_WS_L_N) => {
                        run_end(tags, a, end, mask::NOT_WS_L_N)
                    }
                    _ => ws(i),
                };
            }
            // WsOther / Newline: whitespace only — the ` ?` prefix is a literal 0x20, so tabs/newlines
            // never prefix a run.
            WSO | NLN => i = ws(i),
            // Sentinel / MultiByte / Cont — never a char-start atom; emit one char defensively.
            _ => i += char_len(text[i]),
        }
        out[w] = (start as u32, i as u32);
        w += 1;
    }
    w
}

// ── Composition recipes ────────────────────────────────────────────────────────────────────────
// Each pre-tokenizer = (classify::<Atoms> → fsm shape + params). `tags` and `out` are caller-owned
// scratch, reused across calls — NO per-call alloc, NO push. The class family writes spans into the
// preallocated `out: &mut [Span]` (len ≥ text.len()) via `class_runs_into` and returns the token count.
// In `tk-encode` these delegate from the `pipeline::PreTokenizer` impls (offset conversion happens there).

pub struct WhitespaceSplit;
impl WhitespaceSplit {
    #[inline]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut [Span]) -> usize {
        classify::<Atoms>(text, tags);
        class_runs_into::<{ mask::WS }, 0, 0>(text, tags, out)
    }
}

pub struct Punctuation;
impl Punctuation {
    #[inline]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut [Span]) -> usize {
        classify::<Atoms>(text, tags);
        class_runs_into::<0, { mask::PUNCT }, 0>(text, tags, out)
    }
}

pub struct Digits;
impl Digits {
    #[inline]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut [Span]) -> usize {
        classify::<Atoms>(text, tags);
        class_runs_into::<0, 0, { mask::NUMERIC }>(text, tags, out)
    }
}

pub struct Whitespace;
impl Whitespace {
    #[inline]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut [Span]) -> usize {
        classify::<Atoms>(text, tags);
        // drop WS runs, keep Word and Symbol runs (isolate nothing)
        class_runs_into::<{ mask::WS }, 0, { mask::WORD }>(text, tags, out)
    }
}

pub struct Bert;
impl Bert {
    #[inline]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut [Span]) -> usize {
        classify::<Atoms>(text, tags);
        // drop WS runs, isolate punctuation, keep everything else as single runs
        class_runs_into::<{ mask::WS }, { mask::PUNCT }, 0>(text, tags, out)
    }
}

pub struct Cl100k;
impl Cl100k {
    #[inline]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut [Span]) -> usize {
        classify::<Atoms>(text, tags);
        fsm_cl100k_simd(text, tags, out)
    }
}

pub struct DeepSeek;
impl DeepSeek {
    #[inline]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut [Span]) -> usize {
        classify::<Atoms>(text, tags);
        fsm_deepseek(text, tags, out)
    }
}

pub struct ByteLevel;
impl ByteLevel {
    #[inline]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut [Span]) -> usize {
        classify::<Atoms>(text, tags);
        fsm_byte_level(text, tags, out)
    }
}

/// `Split(char, Removed)` — the only pre-tokenizer that keys on a *literal char* rather than an atom
/// class, so it scans bytes directly (no classify pass). UTF-8 is self-synchronizing, so the
/// delimiter's byte pattern only matches on char boundaries.
pub struct CharDelimiterSplit(pub char);
impl CharDelimiterSplit {
    pub fn pre_tokenize(&self, text: &[u8], _tags: &mut [u8], out: &mut [Span]) -> usize {
        let mut buf = [0u8; 4];
        let delim = self.0.encode_utf8(&mut buf).as_bytes();
        let (n, dl) = (text.len(), delim.len());
        let (mut start, mut i, mut w) = (0usize, 0usize, 0usize);
        while i + dl <= n {
            // memchr the first delimiter byte, then confirm the full pattern.
            match memchr::memchr(delim[0], &text[i..n - dl + 1]) {
                Some(off) if text[i + off..i + off + dl] == *delim => {
                    let m = i + off;
                    if m > start {
                        out[w] = (start as u32, m as u32); // gap before the delimiter (Removed)
                        w += 1;
                    }
                    i = m + dl;
                    start = i;
                }
                Some(off) => i += off + 1, // first byte matched mid-pattern; keep scanning
                None => break,
            }
        }
        if start < n {
            out[w] = (start as u32, n as u32);
            w += 1;
        }
        w
    }
}
