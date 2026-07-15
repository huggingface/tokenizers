//! FSM layer: turn the `Atom` tag stream (from [`crate::classify`]) into token spans. Every fsm is
//! NO-PUSH — it writes spans into a caller-preallocated `&mut [Span]` (len ≥ `text.len()`) and returns
//! the token count; no `Vec`, no realloc. Inputs must be well-formed UTF-8 (see the crate-level docs).
//!
//! The class family (WhitespaceSplit / Punctuation / Digits / Whitespace / Bert) goes through
//! [`class_runs_into`]: on aarch64/wasm the SIMD movemask boundary-extractor + homogeneous-chunk
//! early-out (in `simd_fsm`), elsewhere the scalar run-end core ([`emit_class_spans`]). The
//! regex-shaped ones ([`fsm_cl100k`] / [`fsm_o200k`] / [`fsm_deepseek`] / [`fsm_byte_level`]) are scalar
//! jump-tables, with some SIMD for the occasional * or + patterns

use crate::classify::{Atom, char_len, classify, in_mask, mask};
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
const CONT: u8 = Atom::Cont as u8;
const ASM: u8 = Atom::AlphaSymMark as u8;

/// Advance over a maximal `m`-membership run (m is a mask); returns the byte index past it.
/// `inline(always)`: it's called once per token (~200K/MB on English) — a real call here doubles fsm cost.
#[inline(always)]
fn run_end(tags: &[u8], mut i: usize, end: usize, mut m: u16) -> usize {
    m |= Atom::Cont.bit();
    while i < end && in_mask(tags[i], m) {
        i += 1;
    }
    i
}

/// A token span: byte offsets `[start, end)` into the input.
pub type Span = (u32, u32);

/// No-`push` class-family pre-tokenizer core: writes spans into the preallocated `out` slice and returns
/// the count. ONE shape covers the whole class family via `<DROP, ISOLATE, KEEP_A>`:
///   WhitespaceSplit `<{WS},0,0>` · Punctuation `<0,{PUNCT},0>` · Digits `<0,0,{NUMERIC}>` ·
///   Whitespace `<{WS},0,{WORD}>` · Bert `<{WS},{PUNCT},0>`.
/// Class of a char: `DROP`→dropped, `ISOLATE`→own token, `KEEP_A`→run "A", else→run "B" (A/B cut apart).
/// TODO: find a better explanation
#[inline]
#[must_use]
pub fn class_runs_into<const DROP: u16, const ISOLATE: u16, const KEEP_A: u16>(
    text: &[u8],
    tags: &[u8],
    out: &mut [Span],
) -> usize {
    debug_assert!(out.len() >= text.len() && tags.len() >= text.len());
    #[cfg(target_arch = "aarch64")]
    {
        crate::simd_fsm::class_runs_neon::<DROP, ISOLATE, KEEP_A>(text, tags, out)
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        crate::simd_fsm::class_runs_wasm::<DROP, ISOLATE, KEEP_A>(text, tags, out)
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        emit_class_spans::<DROP, ISOLATE, KEEP_A>(text, tags, out, 0, 0, 0, None)
    }
}

/// This is the most important function as it's the core of the scalar finite state machine.
/// It allows to emit class spans with different behaviours for tags we want to drop, tags we want
/// to isolate and tags we want to keep. Any other tags are assumed to be keept.
///
/// This function is used as a fallback to the SIMD fast fsm. It is used for most pre tokenizers
/// but the unrolled regex, which have more complex variations that cannot be expressed with drop,
/// isolate, keep. These 3 generic parameters are u16 bitmap masks over the 16 classes we have and
/// define the behaviour. They are usally one of the [`crate::classify::mask`]. They allow dropping
/// words, isolating whitespace and keeping new line for example.
#[must_use]
#[inline]
pub fn emit_class_spans<const DROP: u16, const ISOLATE: u16, const KEEP_A: u16>(
    text: &[u8],
    tags: &[u8],
    out: &mut [Span],
    mut write_index: usize,         // in the out slice
    mut text_pointer: usize,        // in the text slice
    segment_start: usize,       // previous segment_start
    segment_class: Option<u16>, // previous segment's class
) -> usize {
    debug_assert!(out.len() >= text.len() && tags.len() >= text.len());
    let n = text.len();
    let other = !(DROP | ISOLATE | KEEP_A); // None of the above correspond to a continuation
    if let Some(segment_class) = segment_class {
        // this will usually be at the tail of a SIMD call.
        text_pointer = run_end(tags, text_pointer, n, segment_class); // skip the whole drop run at once
        if segment_class != DROP {
            out[write_index] = (segment_start as u32, text_pointer as u32);
            if text_pointer == n {
                return write_index + 1;
            }
            write_index += 1;
        }
    }
    while text_pointer < n {
        let t = tags[text_pointer];
        if t == Atom::Cont as u8 {
            text_pointer += 1;
            continue;
        }
        // classify the first char.
        if in_mask(t, DROP) {
            text_pointer = run_end(tags, text_pointer, n, DROP); // skip the whole drop run at once
        } else if in_mask(t, ISOLATE) {
            let s = text_pointer;
            text_pointer += char_len(text[text_pointer]);
            out[write_index] = (s as u32, text_pointer as u32); // isolate: one char = one token
            write_index += 1;
        } else {
            let s = text_pointer;
            text_pointer = if in_mask(t, KEEP_A) {
                run_end(tags, text_pointer, n, KEEP_A)
            } else {
                run_end(tags, text_pointer, n, other)
            };
            out[write_index] = (s as u32, text_pointer as u32);
            write_index += 1;
        }
    }
    write_index
}

/// TODO: NONE OF THE FOLLOWING HAS BEEN REVIEWED
/// cl100k / Llama-3 pretokenization (7 rules + whitespace-tail, rule-3 cap `\p{N}{1,3}`). Peeks `text`
/// for the ASCII contraction literals. Scalar run-ends. See [`fsm_cl100k_cap`] for the variable-cap
/// family (Qwen2 etc.). ┌── OWNER: shared (scalar) ──┐
#[must_use]
pub fn fsm_cl100k(text: &[u8], tags: &[u8], out: &mut [Span]) -> usize {
    fsm_cl100k_cap(text, tags, out, 3)
}

/// The cl100k family with an explicit rule-3 digit cap: 3 = cl100k / Llama-3, 1 = Qwen2's `\p{N}` (each
/// digit its own token), `usize::MAX` = an unbounded `\p{N}+`. Only the digit rule differs across these.
#[must_use]
pub fn fsm_cl100k_cap(text: &[u8], tags: &[u8], out: &mut [Span], digit_cap: usize) -> usize {
    debug_assert!(out.len() >= text.len() && tags.len() >= text.len());
    cl100k(text, tags, out, digit_cap)
}

fn cl100k(text: &[u8], tags: &[u8], out: &mut [Span], digit_cap: usize) -> usize {
    // Leading-atom values, as `const` so the `match` below is a dense jump table (not an if-cascade):
    // the dispatch is O(1) and a token never pays for a rule it can't start (e.g. non-number tokens
    // never test the number rule — which is what the POC's const-gating removed by hand; here it's free).
    let end = text.len();
    let letters = |a: usize| run_end(tags, a, end, mask::LETTER);
    // rule 4 body: `[^\s\p{L}\p{N}]+[\r\n]*` from `sp0` (any leading space already consumed). Returns
    // the run end, or `sp0` if there is no "other" run there (caller then treats it as whitespace).
    let other = |sp0: usize| -> usize {
        let mut p = run_end(tags, sp0, end, mask::NOT_WS_L_N);
        if p > sp0 {
            while p < end && tags[p] == NLN {
                p += char_len(text[p]);
            }
        }
        p
    };
    // rules 5-7: `\s*[\r\n]` | `\s+(?!\S)` | `\s+` — end of the whitespace token starting at `i`.
    let ws = |i: usize| -> usize {
        let re = run_end(tags, i, end, mask::WS);
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
        match tags[i] & 0x0F {
            // rule 2: `\p{L}+`
            LET => i = letters(i),
            // rule 3: `\p{N}{1,cap}` (cap = 3 cl100k, 1 Qwen2, MAX for `\p{N}+`)
            NW | NO => {
                let (mut p, mut cnt) = (i, 0);
                while p < end && cnt < digit_cap && in_mask(tags[p], mask::NUMBER) {
                    p += char_len(text[p]);
                    cnt += 1;
                }
                i = p;
            }
            // Space: rule 2 (space prefix + `\p{L}+`) | rule 4 (` ` + "other") | rules 5-7
            SPC => {
                let a = i + 1; // Space is ASCII (0x20)
                i = if a < end && (tags[a] & 0x0F) == LET {
                    letters(a)
                } else {
                    let p = other(a);
                    if p > a { p } else { ws(i) }
                };
            }
            // WsOther: rule 2 (prefix + `\p{L}+`) | whitespace (never rule 4 — not in NOT_WS_L_N)
            WSO => {
                let a = i + char_len(b);
                i = if a < end && (tags[a] & 0x0F) == LET {
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
                                (l2 == b'e' && matches!(lc, b'r' | b'v'))
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
                    if a < end && (tags[a] & 0x0F) == LET {
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
                i = if a < end && (tags[a] & 0x0F) == LET {
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

/// Any CJK-range char (letter OR punct/sym) at `p` — the full `[一-龥぀-ゟ゠-ヿ]` set Split-2 `[…]+`
/// isolates, INCLUDING CJK punctuation (・ U+30FB, ゠ U+30A0, ゛゜ U+309B/C). Split-3 then re-splits that
/// isolated run into same-kind sub-runs (letters `[\p{L}\p{M}]+` vs punct/sym `[\p{P}\p{S}]+`); because
/// the run is a CLOSED unit, none of it steals a surrounding space or merges with non-CJK punct.
#[inline(always)]
fn ds_is_cjk_at(text: &[u8], p: usize) -> bool {
    (0xE3..=0xE9).contains(&text[p]) && ds_is_cjk(cp3(text, p))
}

/// ZWJ/ZWNJ (U+200C/200D, `\p{Cf}`) at `p` — atom-folded into `Mark` but NOT `\p{L}∪\p{M}`, so they end a
/// deepseek letter run and (not prefixing a letter) form part of an unmatched gap. 3-byte, lead E2.
#[inline(always)]
fn ds_is_zwj(text: &[u8], p: usize) -> bool {
    text[p] == 0xE2 && matches!(cp3(text, p), 0x200C | 0x200D)
}

/// deepseek-v3 pretokenization: the `Sequence` of `[N{1,3}]`, `[CJK]+`, `<big regex>` (all Isolated)
/// collapsed into ONE scalar FSM over the atom stream. Precedence (= the Sequence order): digits →
/// CJK-range runs → the big-regex alts. Because Split-2 isolates CJK *before* the letter rule, the
/// letter run stops at CJK-range codepoints. Peeks bytes for the ASCII `[punct][A-Za-z]+` alt.
///
/// Byte-exact vs the real composed Sequence (onig ×3, each Isolated) on 10 languages — see
/// `benches/deepseek.rs` (plus Hebrew/Arabic via `tk-encode`'s corpus test). The subtleties the single
/// pass replicates: (1) ws *followed by* a digit/CJK is its own Sequence piece → the whole run is one
/// token (`\s+(?!\S)`); (2) ZWJ/ZWNJ are `\p{Cf}`, not `\p{L}∪\p{M}`, so they end a letter run
/// (`ds_breaks`); (3) Split-2 isolates a maximal CJK-range run and Split-3 re-splits it into same-kind
/// sub-runs — the top-of-loop handler consumes that run as a CLOSED unit (`ds_is_cjk_at`), so CJK punct
/// (・) never steals a surrounding space nor merges with non-CJK punct; (4) chars matching no alt
/// (Control / NumericOther / ZWJ) group into ONE gap piece, and Other_Alphabetic symbols (`ALPHA_SYM`:
/// `\w` but categorically `\p{S}`) take the `[\p{P}\p{S}]` path, not the letter run.
#[must_use]
pub fn fsm_deepseek(text: &[u8], tags: &[u8], out: &mut [Span]) -> usize {
    debug_assert!(out.len() >= text.len() && tags.len() >= text.len());
    // Leading-atom values as `const` → the `match` is a dense jump table (see `cl100k`). The Split
    // precedence (digits → CJK → big-regex alts) is preserved because the atom partition is disjoint.
    // `Mark` refined as an Other_Alphabetic symbol (Ⓘ …): coarse `LETTER_MARK`, but categorically `\p{S}`
    // — excluded from `[\p{L}\p{M}]`, routed to the `[\p{P}\p{S}]+` run instead (see `punct`).
    let end = text.len();
    // maximal `[\p{L}\p{M}]+` run from `a`, stopping at CJK-range chars (Split-2 took those), ZWJ/ZWNJ
    // (not `\p{L}∪\p{M}` — see `ds_breaks`), and Other_Alphabetic symbols (`ASM`, categorically `\p{S}`).
    // BYTE-wise (`p += 1`, continuation bytes stay in-run, `ds_breaks` only fires at a lead) — the
    // `char_len`-per-char form was ~2× slower (see `run_end`'s note). Hot inner loop of the latin path.
    let letter_run = |a: usize| -> usize {
        let mut p = a;
        while p < end {
            let t = tags[p];
            if t == CONT || (in_mask(t, mask::LETTER_MARK) && t != ASM && !ds_breaks(text, p)) {
                p += 1;
            } else {
                break;
            }
        }
        p
    };
    // is `text[a]` the start of a deepseek letter/mark char (alt-2 run body / space-prefix target)?
    let is_lm = |a: usize| {
        a < end && in_mask(tags[a], mask::LETTER_MARK) && tags[a] != ASM && !ds_breaks(text, a)
    };
    // Split-3 alt-3 tail `[\p{P}\p{S}]+[\r\n]*` from `sp0` (a leading space is already consumed); `sp0`
    // if there is no punct/sym run there. STOPS at CJK-range chars — Split-1 isolated those, so a CJK
    // punct (・) is never merged into a non-CJK punct run (`!・` → `!`, `・`, not `!・`).
    let punct = |sp0: usize| -> usize {
        let mut p = sp0;
        while p < end
            && (in_mask(tags[p], mask::PUNCT_SYM) || tags[p] == ASM)
            && !ds_is_cjk_at(text, p)
        {
            p += char_len(text[p]);
        }
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
        let next_isolated = re < end && (in_mask(tags[re], mask::NUMBER) || ds_is_cjk_at(text, re));
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
        // Split-2 isolated a maximal CJK-range run; Split-3 re-splits it into same-kind sub-runs
        // (letters `[\p{L}\p{M}]+` vs punct/sym `[\p{P}\p{S}]+`) — a CLOSED unit, handled before the atom
        // arms so CJK punct (・) never leaks into alt-3 (stealing a space / merging with non-CJK punct).
        if ds_is_cjk_at(text, i) {
            let is_letter = in_mask(tags[i], mask::LETTER_MARK);
            let mut p = i + 3; // CJK-range chars are all 3-byte (leads E3..E9)
            while p < end
                && ds_is_cjk_at(text, p)
                && in_mask(tags[p], mask::LETTER_MARK) == is_letter
            {
                p += 3;
            }
            out[w] = (start as u32, p as u32);
            w += 1;
            i = p;
            continue;
        }
        // Gap run: maximal Control / NumericOther / ZWJ — none matches a Split-3 alt, so the composed
        // Split emits the whole run as ONE unmatched piece. Exception: if it's immediately followed by a
        // letter run, the LAST gap char is that run's alt-2 `[^\r\n\p{L}\p{P}\p{S}]?` prefix (splits off).
        if matches!(tags[i] & 0x0F, NMO | CTL) || ds_is_zwj(text, i) {
            let (mut p, mut last) = (i, i);
            while p < end && (matches!(tags[p] & 0x0F, NMO | CTL) || ds_is_zwj(text, p)) {
                last = p;
                p += char_len(text[p]);
            }
            if is_lm(p) {
                if last > i {
                    out[w] = (start as u32, last as u32); // gap = all but the prefix char
                    w += 1;
                }
                let e = letter_run(p);
                out[w] = (last as u32, e as u32); // prefix char + `[\p{L}\p{M}]+`
                w += 1;
                i = e;
            } else {
                out[w] = (start as u32, p as u32); // whole gap run is one piece
                w += 1;
                i = p;
            }
            continue;
        }
        match tags[i] & 0x0F {
            // Split-1: `\p{N}{1,3}`
            NW | NO => {
                let (mut p, mut cnt) = (i, 0);
                while p < end && cnt < 3 && in_mask(tags[p], mask::NUMBER) {
                    p += char_len(text[p]);
                    cnt += 1;
                }
                i = p;
            }
            // Split-3 alt-2 `[\p{L}\p{M}]+` (CJK letters + ZWJ/gap chars were consumed above the match).
            // An Other_Alphabetic symbol (`ASM`, coarse `Mark`) is categorically `\p{S}` → the alt-3 run.
            LET | MRK => {
                i = if tags[i] == ASM {
                    punct(i)
                } else {
                    letter_run(i)
                }
            }
            // Space: alt-2 (space prefix + `[\p{L}\p{M}]+`) | alt-3 (` ` + `[\p{P}\p{S}]+`) | whitespace
            SPC => {
                let a = i + 1; // Space is ASCII (0x20)
                i = if is_lm(a) {
                    letter_run(a)
                } else if a < end && ds_is_cjk_at(text, a) {
                    ws(i) // next is a Split-1-isolated CJK char → the space is standalone whitespace
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
#[must_use]
pub fn fsm_byte_level(text: &[u8], tags: &[u8], out: &mut [Span]) -> usize {
    debug_assert!(out.len() >= text.len() && tags.len() >= text.len());
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
        match tags[i] & 0x0F {
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
                i = match tags.get(a).map(|&t| t & 0x0F) {
                    Some(LET) => run_end(tags, a, end, mask::LETTER),
                    Some(NW) | Some(NO) => run_end(tags, a, end, mask::NUMBER),
                    Some(t) if in_mask(t, mask::NOT_WS_L_N) => {
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

// ── o200k (GPT-4o): case-aware letter split ──────────────────────────────────────────────────────

/// o200k class of a letter-run char (all chars in a run are real `[\p{L}\p{M}]`, never ALPHA_SYM/ZWJ):
/// `refine::UPPER` → U (`\p{Lu}\p{Lt}`), `refine::LOWER` → L (`\p{Ll}`), else C (caseless `\p{Lm}\p{Lo}
/// \p{M}`). The two alt char-classes are `[UC]` = "not L" (`!o_is_lower`) and `[LC]` = "not U".
#[inline]
fn o_is_upper(t: u8) -> bool {
    t == Atom::UpperLetter as u8 // 0x10: coarse Letter (low nibble 0) + UPPER refine (\p{Lu}∪\p{Lt})
}
#[inline]
fn o_is_lower(t: u8) -> bool {
    t == Atom::LowerLetter as u8 // 0x20: coarse Letter + LOWER refine (\p{Ll})
}

/// One o200k letter sub-token from `p` within the run `[.., re)`: alt-1 `[UC]*[LC]+` (tried first) else
/// alt-2 `[UC]+[LC]*` (reached only for an all-U run). Greedy with Perl backtracking — `[UC]*` gives back
/// to the last C so `[LC]+` can take ≥1. Returns the sub-token end, always in `(p, re]`. BYTE-wise (`+=1`,
/// like `run_end`): continuation bytes are tag `Cont`(15) → neither U nor L → transparent to `[UC]`/`[LC]`.
#[inline(always)]
fn o200k_letter_match(tags: &[u8], p: usize, re: usize) -> usize {
    // alt-1 `[UC]*`: greedy over "not L" (stops at the next lowercase *lead*), tracking the last C
    // char-start so a no-L run needs no separate backtrack pass (`Cont`=15 is neither U nor L, so it's
    // "C-like" — the `!= CONT` guard keeps `last_c` on a real char start).
    let mut q = p;
    let mut last_c = usize::MAX;
    while q < re && !o_is_lower(tags[q]) {
        if tags[q] != CONT && !o_is_upper(tags[q]) {
            last_c = q;
        }
        q += 1;
    }
    if q < re {
        // tags[q] is L → `[LC]+` from q: greedy over "not U"
        let mut e = q;
        while e < re && !o_is_upper(tags[e]) {
            e += 1;
        }
        return e;
    }
    if last_c == usize::MAX {
        return re; // no L and no C → all U → alt-2 `[UC]+[LC]*` (empty `[LC]*`) = the whole run
    }
    // no L: `[UC]*` gives back to the last C, which begins the `[LC]+`
    let mut e = last_c;
    while e < re && !o_is_upper(tags[e]) {
        e += 1;
    }
    e
}

/// Length (incl the `'`) of an o200k contraction suffix `(?i:'s|'t|'re|'ve|'m|'ll|'d)` at `i`, else 0.
#[inline]
fn o200k_contraction(text: &[u8], i: usize) -> usize {
    let end = text.len();
    if i >= end || text[i] != 0x27 || i + 1 >= end || text[i + 1] >= 0x80 {
        return 0;
    }
    let lc = text[i + 1] | 0x20;
    match lc {
        b's' | b't' | b'm' | b'd' => 2,
        b'r' | b'v' | b'l' if i + 2 < end && text[i + 2] < 0x80 => {
            let l2 = text[i + 2] | 0x20;
            usize::from((matches!(lc, b'r' | b'v') && l2 == b'e') || (lc == b'l' && l2 == b'l')) * 3
        }
        _ => 0,
    }
}

/// Emit the o200k case-split of the letter run `[ls, re)` into `out[*w..]`: the first sub-token starts at
/// `pfx` (the optional `[^\r\n\p{L}\p{N}]?` prefix; `pfx == ls` when none), the last absorbs a trailing
/// contraction. Returns the new cursor (past the contraction). `ls < re` (caller-guaranteed).
#[inline(always)]
fn emit_o200k_letters(
    text: &[u8],
    tags: &[u8],
    pfx: usize,
    ls: usize,
    re: usize,
    out: &mut [Span],
    w: &mut usize,
) -> usize {
    let (mut p, mut first, mut cursor) = (ls, true, re);
    while p < re {
        let e = o200k_letter_match(tags, p, re);
        let start = if first { pfx } else { p };
        let tok_end = if e == re {
            e + o200k_contraction(text, e)
        } else {
            e
        };
        out[*w] = (start as u32, tok_end as u32);
        *w += 1;
        first = false;
        cursor = tok_end;
        p = e;
    }
    cursor
}

/// o200k (GPT-4o) pretokenization. Same skeleton as cl100k, but the letter body is `[\p{L}\p{M}]+`
/// split into case sub-runs (`emit_o200k_letters`), the contraction is a *suffix* on letter tokens (not
/// a leading rule), and rule 4 is `[^\s\p{L}\p{N}]+[\r\n/]*`. The letter body excludes ALPHA_SYM symbols
/// and ZWJ/ZWNJ (coarse `Mark` but categorically `\p{S}`/`\p{Cf}`) — they take the prefix / rule-4 path.
/// Unlike deepseek there are no gaps: rule 4's `[^\s\p{L}\p{N}]+` is a catch-all. Scalar; ┌ OWNER: shared ┐
#[must_use]
pub fn fsm_o200k(text: &[u8], tags: &[u8], out: &mut [Span]) -> usize {
    debug_assert!(out.len() >= text.len() && tags.len() >= text.len());
    let end = text.len();

    // Is tag `t` (at byte `p`) a real `[\p{L}\p{M}]` member? Coarse `Letter` (any case) is always in;
    // coarse `Mark` is in only as a true `\p{M}` — ALPHA_SYM (`\p{S}`) and ZWJ/ZWNJ (`\p{Cf}`) are `\w`
    // but not `[\p{L}\p{M}]`. The `ds_is_zwj` byte-peek is thus paid ONLY for Marks, never for letters.
    let member = |t: u8, p: usize| -> bool {
        let c = t & 0x0F;
        c == LET || (c == MRK && t != ASM && !ds_is_zwj(text, p))
    };
    let is_lm = |a: usize| a < end && member(tags[a], a);
    // maximal `[\p{L}\p{M}]+` run from `a` (byte-wise; continuation bytes ride along — see `run_end`).
    let letter_end = |a: usize| -> usize {
        let mut p = a;
        while p < end && (tags[p] == CONT || member(tags[p], p)) {
            p += 1;
        }
        p
    };
    // rule 4 `[^\s\p{L}\p{N}]+[\r\n/]*` from `sp0` (any leading space already consumed); `sp0` if none.
    // `/` is in the `+` body too — the trailing class only matters after the `+` stops at a `\r\n`.
    let other = |sp0: usize| -> usize {
        let mut p = run_end(tags, sp0, end, mask::NOT_WS_L_N);
        if p > sp0 {
            while p < end && (tags[p] == NLN || text[p] == b'/') {
                p += char_len(text[p]);
            }
        }
        p
    };
    // rules 5-7 `\s*[\r\n]+ | \s+(?!\S) | \s+` — identical to cl100k.
    let ws = |i: usize| -> usize {
        let re = run_end(tags, i, end, mask::WS);
        if let Some(r) = text[i..re].iter().rposition(|&x| x == 0x0A || x == 0x0D) {
            i + r + 1
        } else if re == end {
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
        let b = text[i];
        match tags[i] & 0x0F {
            // rule 3: `\p{N}{1,3}` (no prefix)
            NW | NO => {
                let (mut p, mut cnt) = (i, 0);
                while p < end && cnt < 3 && in_mask(tags[p], mask::NUMBER) {
                    p += char_len(text[p]);
                    cnt += 1;
                }
                i = p;
            }
            // letters `[\p{L}\p{M}]+` (case-split) — but ALPHA_SYM/ZWJ (coarse Mark, not `[\p{L}\p{M}]`)
            // take the `[^\r\n\p{L}\p{N}]?` prefix / rule-4 path instead.
            LET | MRK => {
                if tags[i] != ASM && !ds_is_zwj(text, i) {
                    i = emit_o200k_letters(text, tags, i, i, letter_end(i), out, &mut w);
                    continue;
                }
                let a = i + char_len(b);
                if is_lm(a) {
                    i = emit_o200k_letters(text, tags, i, a, letter_end(a), out, &mut w);
                    continue;
                }
                i = other(i); // ∈ NOT_WS_L_N ⇒ > i
            }
            // Space: ` ?` prefix + letters | ` ?` + rule-4 other | whitespace
            SPC => {
                let a = i + 1; // Space is ASCII (0x20)
                if is_lm(a) {
                    i = emit_o200k_letters(text, tags, i, a, letter_end(a), out, &mut w);
                    continue;
                }
                let p = other(a);
                i = if p > a { p } else { ws(i) };
            }
            // WsOther: prefix + letters | whitespace (∈ `\s` ⇒ never starts rule 4)
            WSO => {
                let a = i + char_len(b);
                if is_lm(a) {
                    i = emit_o200k_letters(text, tags, i, a, letter_end(a), out, &mut w);
                    continue;
                }
                i = ws(i);
            }
            NLN => i = ws(i),
            // punct / sym / … (∈ `[^\r\n\p{L}\p{N}]` and `[^\s\p{L}\p{N}]`): prefix + letters | rule-4 other
            CON | PUN | APO | SYM | NMO | CTL => {
                let a = i + char_len(b);
                if is_lm(a) {
                    i = emit_o200k_letters(text, tags, i, a, letter_end(a), out, &mut w);
                    continue;
                }
                i = other(i); // ∈ NOT_WS_L_N ⇒ > i
            }
            // Sentinel / MultiByte / Cont — never a char-start atom; emit one char defensively.
            _ => i += char_len(b),
        }
        out[w] = (start as u32, i as u32);
        w += 1;
    }
    w
}

// ── Composition recipes ────────────────────────────────────────────────────────────────────────
// Each pre-tokenizer = (classify → fsm shape + params). `tags` and `out` are caller-owned
// scratch, reused across calls — NO per-call alloc, NO push. The class family writes spans into the
// preallocated `out: &mut [Span]` (len ≥ text.len()) via `class_runs_into` and returns the token count.
// In `tk-encode` these delegate from the `pipeline::PreTokenizer` impls (offset conversion happens there).

/// `WhitespaceSplit` — split on Unicode whitespace and drop it; keeps maximal non-whitespace runs.
pub struct WhitespaceSplit;
impl WhitespaceSplit {
    /// Classify then split; writes spans into `out` (len ≥ `text.len()`) and returns the token count.
    #[inline]
    #[must_use]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut [Span]) -> usize {
        classify(text, tags);
        class_runs_into::<{ mask::WS }, 0, 0>(text, tags, out)
    }
}

/// `Punctuation` — isolate each punctuation char as its own token; non-punct grouped into runs.
pub struct Punctuation;
impl Punctuation {
    /// Classify then split; writes spans into `out` (len ≥ `text.len()`) and returns the token count.
    #[inline]
    #[must_use]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut [Span]) -> usize {
        classify(text, tags);
        class_runs_into::<0, { mask::PUNCT }, 0>(text, tags, out)
    }
}

/// `Digits` — cut numeric runs apart from non-numeric runs (contiguous), keeping both.
pub struct Digits;
impl Digits {
    /// Classify then split; writes spans into `out` (len ≥ `text.len()`) and returns the token count.
    #[inline]
    #[must_use]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut [Span]) -> usize {
        classify(text, tags);
        class_runs_into::<0, 0, { mask::NUMERIC }>(text, tags, out)
    }
}

/// `Whitespace` — the `\w+|[^\w\s]+` pre-tokenizer: drop whitespace, cut word runs from symbol runs.
pub struct Whitespace;
impl Whitespace {
    /// Classify then split; writes spans into `out` (len ≥ `text.len()`) and returns the token count.
    #[inline]
    #[must_use]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut [Span]) -> usize {
        classify(text, tags);
        class_runs_into::<{ mask::WS }, 0, { mask::WORD }>(text, tags, out)
    }
}

/// `Bert` — the BERT basic pre-tokenizer: drop whitespace, isolate punctuation, keep the rest as runs.
pub struct Bert;
impl Bert {
    /// Classify then split; writes spans into `out` (len ≥ `text.len()`) and returns the token count.
    #[inline]
    #[must_use]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut [Span]) -> usize {
        classify(text, tags);
        class_runs_into::<{ mask::WS }, { mask::PUNCT }, 0>(text, tags, out)
    }
}

/// `Cl100k` — the tiktoken cl100k_base / o200k GPT-4 pre-tokenizer (7-rule regex).
pub struct Cl100k;
impl Cl100k {
    /// Classify then split; writes spans into `out` (len ≥ `text.len()`) and returns the token count.
    /// Uses the scalar run-end core: cl100k's letter/ws runs are short on Latin/code (the common case),
    /// where a SIMD run-end's setup would lose; only long CJK runs would benefit.
    #[inline]
    #[must_use]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut [Span]) -> usize {
        classify(text, tags);
        fsm_cl100k(text, tags, out)
    }
}

/// `DeepSeek` — the DeepSeek-V3/R1 pre-tokenizer (digits{1,3} → CJK-range → big regex, composed).
pub struct DeepSeek;
impl DeepSeek {
    /// Classify then split; writes spans into `out` (len ≥ `text.len()`) and returns the token count.
    #[inline]
    #[must_use]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut [Span]) -> usize {
        classify(text, tags);
        fsm_deepseek(text, tags, out)
    }
}

/// `ByteLevel` — the GPT-2 / Llama / Qwen byte-level pre-tokenizer regex (before byte-mapping).
pub struct ByteLevel;
impl ByteLevel {
    /// Classify then split; writes spans into `out` (len ≥ `text.len()`) and returns the token count.
    #[inline]
    #[must_use]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut [Span]) -> usize {
        classify(text, tags);
        fsm_byte_level(text, tags, out)
    }
}

/// `Split(char, Removed)` — the only pre-tokenizer that keys on a *literal char* rather than an atom
/// class, so it scans bytes directly (no classify pass). UTF-8 is self-synchronizing, so the
/// delimiter's byte pattern only matches on char boundaries.
pub struct CharDelimiterSplit(pub char);
impl CharDelimiterSplit {
    /// Split on the literal char (Removed); writes spans into `out` (len ≥ `text.len()`), returns count.
    #[inline]
    #[must_use]
    pub fn pre_tokenize(&self, text: &[u8], _tags: &mut [u8], out: &mut [Span]) -> usize {
        debug_assert!(out.len() >= text.len());
        let mut buf = [0u8; 4];
        let delim = self.0.encode_utf8(&mut buf).as_bytes();
        let (n, dl) = (text.len(), delim.len());
        let (mut start, mut i, mut w) = (0usize, 0usize, 0usize);
        while i + dl <= n {
            // memchr the first delimiter byte, then confirm the full pattern. memchr (already a
            // workspace dep) beats a scalar scan 1.4–23× here — the gap widening as the delimiter
            // gets rarer over large inputs, since its SIMD skips whole 16/32/64-byte strides.
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
