use super::*;

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

/// Any CJK-range char (letter OR punct/sym) at `p` — the full `[一-龥぀-ゟ゠-ヿ]` set Split-2 `[…]+`
/// isolates, INCLUDING CJK punctuation (・ U+30FB, ゠ U+30A0, ゛゜ U+309B/C). Split-3 then re-splits that
/// isolated run into same-kind sub-runs (letters `[\p{L}\p{M}]+` vs punct/sym `[\p{P}\p{S}]+`); because
/// the run is a CLOSED unit, none of it steals a surrounding space or merges with non-CJK punct.
#[inline(always)]
fn ds_is_cjk_at(text: &[u8], p: usize) -> bool {
    (0xE3..=0xE9).contains(&text[p]) && ds_is_cjk(cp3(text, p))
}

/// Does the char tagged `t` match NO alternative of deepseek's big regex? Control / NumericOther / ZWJ
/// never do. `\p{N}` also doesn't — but only when Split-1 is absent (`NUM == false`), since the big
/// regex has no digit rule of its own; there a digit is either a gap char or a letter run's optional
/// `[^\r\n\p{L}\p{P}\p{S}]?` prefix. Consecutive such chars become ONE unmatched piece.
#[inline(always)]
fn ds_is_gap<const NUM: bool>(t: u8) -> bool {
    matches!(t & 0x0F, NMO | CTL) || t == ZWJ || (!NUM && in_mask(t, mask::NUMBER))
}

/// deepseek Split-1 on its own: `\p{N}{1,3}` under an `Isolated` split — numeric runs cut into
/// ≤3-*char* tokens, every maximal non-numeric run emitted as ONE gap piece. `fsm_deepseek` fuses this
/// in; the standalone form is what a lone `Split` with that pattern needs.
#[must_use]
pub fn fsm_deepseek_num(text: &[u8], tags: &[u8], out: &mut [Span]) -> usize {
    debug_assert!(out.len() >= text.len() && tags.len() >= text.len());
    let end = text.len();
    let tags = &tags[..end];
    let (mut i, mut w) = (0usize, 0usize);
    while i < end {
        let start = i;
        if in_mask(tags[i], mask::NUMBER) {
            let mut cnt = 0;
            while i < end && cnt < 3 && in_mask(tags[i], mask::NUMBER) {
                i += char_len(text[i]);
                cnt += 1;
            }
        } else {
            i = run_end(tags, i, end, !mask::NUMBER);
        }
        out[w] = Span::new(start as u32, i as u32);
        w += 1;
    }
    w
}

/// deepseek Split-2 on its own: `[一-龥぀-ゟ゠-ヿ]+` under an `Isolated` split — maximal CJK-range runs,
/// every maximal non-CJK run as ONE gap piece. Unlike the fused [`fsm_deepseek`], the run is NOT cut
/// into letter / punct sub-runs: that cut comes from Split-3 re-splitting the isolated piece.
///
/// The one fsm that reads no tags: a codepoint *range* is not an atom class, so it works off the text.
/// It keeps the shared signature anyway, so callers can dispatch over any fsm without a special case.
#[must_use]
pub fn fsm_deepseek_cjk(text: &[u8], _tags: &[u8], out: &mut [Span]) -> usize {
    debug_assert!(out.len() >= text.len());
    let end = text.len();
    let (mut i, mut w) = (0usize, 0usize);
    while i < end {
        let start = i;
        if ds_is_cjk_at(text, i) {
            while i < end && ds_is_cjk_at(text, i) {
                i += 3; // CJK-range chars are all 3-byte
            }
        } else {
            while i < end && !ds_is_cjk_at(text, i) {
                i += char_len(text[i]);
            }
        }
        out[w] = Span::new(start as u32, i as u32);
        w += 1;
    }
    w
}

/// deepseek pretokenization over the atom stream, as one scalar pass. The two flags say which of the
/// `Sequence`'s earlier Splits are fused in — they are pure compile-time switches, so each instantiation
/// is the straight-line FSM for its grammar:
/// * `<true, true>` = [`fsm_deepseek`], the whole `Sequence`. Precedence (= Sequence order): digits →
///   CJK-range runs → the big-regex alts, so the letter run stops at CJK codepoints and a `\p{N}{1,3}`
///   token wins over them all.
/// * `<false, false>` = [`fsm_deepseek_big`], Split-3's regex alone: digits match no alt (they become
///   gap chars, or the leading `[^\r\n\p{L}\p{P}\p{S}]?` of a following letter run) and CJK is just
///   `\p{L}`, free to join an adjacent letter run.
///
/// Peeks bytes for the ASCII `[punct][A-Za-z]+` alt. The subtleties either instantiation replicates:
/// (1) a ws run *followed by* a fused-Split match is its own Sequence piece → `\s+(?!\S)` takes the
/// WHOLE run; (2) ZWJ/ZWNJ are `\p{Cf}`, not `\p{L}∪\p{M}`, so they end a letter run; (3) a CJK run is
/// consumed as a CLOSED unit, so CJK punct (・) never steals a surrounding space nor merges with
/// non-CJK punct; (4) chars matching no alt group into ONE gap piece, and Other_Alphabetic symbols
/// (`ASM`: `\w` but categorically `\p{S}`) take the `[\p{P}\p{S}]` path, not the letter run.
#[must_use]
#[inline] // so each wrapper below gets its own flat copy, as if hand-written, rather than a shared call
fn fsm_ds<const NUM: bool, const CJK: bool>(text: &[u8], tags: &[u8], out: &mut [Span]) -> usize {
    debug_assert!(out.len() >= text.len() && tags.len() >= text.len());
    // Leading-atom values as `const` → the `match` is a dense jump table (see `cl100k`). The Split
    // precedence (digits → CJK → big-regex alts) is preserved because the atom partition is disjoint.
    // `Mark` refined as an Other_Alphabetic symbol (Ⓘ …): coarse `LETTER_MARK`, but categorically `\p{S}`
    // — excluded from `[\p{L}\p{M}]`, routed to the `[\p{P}\p{S}]+` run instead (see `punct`).
    let end = text.len();
    // Tie `tags.len() == end` so the optimizer drops the per-byte bounds check on every interior
    // `tags[i]` in this fsm + its `run_end`/`letter_*` scans. (Callers guarantee `tags.len() >= end`.)
    let tags = &tags[..end];
    // The CJK test below is spelled out at each use site as `CJK && ds_is_cjk_at(…)` rather than hoisted
    // into one local closure: it sits in `letter_run`'s per-byte loop, and a closure captured by another
    // closure stopped inlining there — worth ~7% on the multilingual bench.
    //
    // maximal `[\p{L}\p{M}]+` run from `a`, stopping at CJK-range chars (Split-2 took those), ZWJ/ZWNJ
    // (not `\p{L}∪\p{M}` — see `ds_breaks`), and Other_Alphabetic symbols (`ASM`, categorically `\p{S}`).
    // BYTE-wise (`p += 1`, continuation bytes stay in-run, `ds_breaks` only fires at a lead) — the
    // `char_len`-per-char form was ~2× slower (see `run_end`'s note). Hot inner loop of the latin path.
    let letter_run = |a: usize| -> usize {
        let mut p = a;
        // ZWJ/ASM are now tags (no text peek); only the CJK-range exclusion still peeks text.
        while p < end {
            let t = tags[p];
            if t == CONT
                || (in_mask(t, mask::LETTER_MARK)
                    && t != ASM
                    && t != ZWJ
                    && !(CJK && ds_is_cjk_at(text, p)))
            {
                p += 1;
            } else {
                break;
            }
        }
        p
    };
    // is `text[a]` the start of a deepseek letter/mark char (alt-2 run body / space-prefix target)?
    let is_lm = |a: usize| {
        a < end
            && in_mask(tags[a], mask::LETTER_MARK)
            && tags[a] != ASM
            && tags[a] != ZWJ
            && !(CJK && ds_is_cjk_at(text, a))
    };
    // Split-3 alt-3 tail `[\p{P}\p{S}]+[\r\n]*` from `sp0` (a leading space is already consumed); `sp0`
    // if there is no punct/sym run there. STOPS at CJK-range chars — Split-2 isolated those, so a CJK
    // punct (・) is never merged into a non-CJK punct run (`!・` → `!`, `・`, not `!・`).
    let punct = |sp0: usize| -> usize {
        let mut p = sp0;
        while p < end
            && (in_mask(tags[p], mask::PUNCT_SYM) || tags[p] == ASM)
            && !(CJK && ds_is_cjk_at(text, p))
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
        let next_isolated = re < end
            && ((NUM && in_mask(tags[re], mask::NUMBER)) || (CJK && ds_is_cjk_at(text, re)));
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
        if CJK && ds_is_cjk_at(text, i) {
            let is_letter = in_mask(tags[i], mask::LETTER_MARK);
            let mut p = i + 3; // CJK-range chars are all 3-byte (leads E3..E9)
            while p < end
                && ds_is_cjk_at(text, p)
                && in_mask(tags[p], mask::LETTER_MARK) == is_letter
            {
                p += 3;
            }
            unsafe {
                *out.get_unchecked_mut(w) = Span {
                    start: start as u32,
                    end: p as u32,
                }
            };
            w += 1;
            i = p;
            continue;
        }
        // A maximal run of chars that match no alt (see `ds_is_gap`) is ONE unmatched piece. Exception:
        // if a letter run follows immediately, the LAST gap char is that run's alt-2
        // `[^\r\n\p{L}\p{P}\p{S}]?` prefix, so it splits off the gap and joins the letters.
        if ds_is_gap::<NUM>(tags[i]) {
            let (mut p, mut last) = (i, i);
            while p < end && ds_is_gap::<NUM>(tags[p]) {
                last = p;
                p += char_len(text[p]);
            }
            if is_lm(p) {
                if last > i {
                    unsafe {
                        *out.get_unchecked_mut(w) = Span {
                            start: start as u32,
                            end: last as u32,
                        }
                    }; // gap sans prefix char
                    w += 1;
                }
                let e = letter_run(p);
                unsafe {
                    *out.get_unchecked_mut(w) = Span {
                        start: last as u32,
                        end: e as u32,
                    }
                }; // prefix char + `[\p{L}\p{M}]+`
                w += 1;
                i = e;
            } else {
                unsafe {
                    *out.get_unchecked_mut(w) = Span {
                        start: start as u32,
                        end: p as u32,
                    }
                }; // whole gap run is one piece
                w += 1;
                i = p;
            }
            continue;
        }
        match tags[i] & 0x0F {
            // Split-1: `\p{N}{1,3}` — reachable only with `NUM`; otherwise `ds_is_gap` took these above.
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
                } else if CJK && a < end && ds_is_cjk_at(text, a) {
                    ws(i) // next is a Split-2-isolated CJK char → the space is standalone whitespace
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
        // SAFETY: tokens partition the input, so `w < #tokens <= end < out.len()` (out ≥ text.len()+? ; callers size n+1).
        unsafe {
            *out.get_unchecked_mut(w) = Span {
                start: start as u32,
                end: i as u32,
            }
        };
        w += 1;
    }
    w
}

/// deepseek pretokenization: the `Sequence` of `[\p{N}{1,3}]`, `[CJK]+`, `<big regex>` (all `Isolated`)
/// as ONE pass — see [`fsm_ds`]. Byte-exact vs the real composed Sequence (onig ×3) on both parity
/// corpora and 9 Wikipedia languages (`tests/parity.rs`, `benches/regex.rs`).
#[must_use]
pub fn fsm_deepseek(text: &[u8], tags: &[u8], out: &mut [Span]) -> usize {
    fsm_ds::<true, true>(text, tags, out)
}

/// deepseek Split-3 on its own: the big regex under an `Isolated` split — see [`fsm_ds`]. Without the
/// earlier Splits, digits are unmatched gap chars and CJK is plain `\p{L}`.
#[must_use]
pub fn fsm_deepseek_big(text: &[u8], tags: &[u8], out: &mut [Span]) -> usize {
    fsm_ds::<false, false>(text, tags, out)
}
