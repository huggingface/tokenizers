use super::*;

// ── o200k (GPT-4o): case-aware letter split ──────────────────────────────────────────────────────

/// o200k class of a letter-run char (all chars in a run are real `[\p{L}\p{M}]`, never ALPHA_SYM/ZWJ):
/// `Atom::UpperLetter` → U (`\p{Lu}\p{Lt}`), `Atom::LowerLetter` → L (`\p{Ll}`), else C (caseless `\p{Lm}\p{Lo}
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

/// Emit the o200k case-split of the letter run `[ls, re)`: the first sub-token starts at
/// `pfx` (the optional `[^\r\n\p{L}\p{N}]?` prefix; `pfx == ls` when none), the last absorbs a trailing
/// contraction (`CONTRACTION` — off for tekken). Returns the new cursor (past the contraction).
/// `ls < re` (caller-guaranteed).
#[inline(always)]
fn emit_o200k_letters<const CONTRACTION: bool, E: FnMut(Span)>(
    text: &[u8],
    tags: &[u8],
    pfx: usize,
    ls: usize,
    re: usize,
    emit: &mut E,
) -> usize {
    let (mut p, mut first, mut cursor) = (ls, true, re);
    while p < re {
        let e = o200k_letter_match(tags, p, re);
        let start = if first { pfx } else { p };
        let tok_end = if CONTRACTION && e == re {
            e + contraction(text, e)
        } else {
            e
        };
        emit(Span {
            start: start as u32,
            end: tok_end as u32,
        });
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
    debug_assert!(out.len() >= text.len());
    let mut w = 0usize;
    scan_o200k(text, tags, |span| {
        // SAFETY: tokens partition the input, so `w < #tokens <= text.len() <= out.len()`.
        unsafe { *out.get_unchecked_mut(w) = span };
        w += 1;
    });
    w
}

/// Mistral tekken ([`crate::regexes::TEKKEN`]) — the o200k FSM with the contraction suffix off and one
/// token per digit. Every other rule is shared, so both are the same code monomorphized twice.
#[must_use]
pub fn fsm_tekken(text: &[u8], tags: &[u8], out: &mut [Span]) -> usize {
    debug_assert!(out.len() >= text.len());
    let mut w = 0usize;
    scan_tekken(text, tags, |span| {
        // SAFETY: tokens partition the input, so `w < #tokens <= text.len() <= out.len()`.
        unsafe { *out.get_unchecked_mut(w) = span };
        w += 1;
    });
    w
}

/// The scan under [`fsm_o200k`]: hands each token to `emit` the moment it is cut,
/// so a caller can consume tokens in place instead of collecting a span buffer first.
pub fn scan_o200k(text: &[u8], tags: &[u8], emit: impl FnMut(Span)) {
    o200k::<true, 3, _>(text, tags, emit);
}

/// The scan under [`fsm_tekken`]; see [`scan_o200k`].
pub fn scan_tekken(text: &[u8], tags: &[u8], emit: impl FnMut(Span)) {
    o200k::<false, 1, _>(text, tags, emit);
}

/// Is tag `t` a real `[\p{L}\p{M}]` member? Coarse `Letter` (any case) is always in; coarse
/// `Mark` is in only as a true `\p{M}` — ALPHA_SYM (`\p{S}`) and ZWJ/ZWNJ (`\p{Cf}`) are `\w`
/// but not `[\p{L}\p{M}]`.
#[inline(always)]
fn member(t: u8) -> bool {
    let c = t & 0x0F;
    c == LET || (c == MRK && t != ASM && t != ZWJ)
}

#[inline(always)]
fn is_lm(tags: &[u8], a: usize, end: usize) -> bool {
    a < end && member(tags[a])
}

/// Maximal `[\p{L}\p{M}]+` run from `a` (byte-wise; continuation bytes ride along — see `run_end`).
#[inline(always)]
fn letter_end(tags: &[u8], a: usize, end: usize) -> usize {
    let mut p = a;
    // logos-style fast loop: 16 tags/chunk, one bounds check, unchecked reads. A plain `Letter`
    // (low nibble 0, incl Han — o200k keeps all letters) or a `Cont` byte stays in-run; only a
    // coarse `Mark` lane pays the refinement test. Byte-exact with the plain scan below.
    // SAFETY: `p + 16 <= end <= tags.len()` in the body.
    while p + 16 <= end {
        let mut brk = 16;
        for k in 0..16 {
            let t = unsafe { *tags.get_unchecked(p + k) };
            if t == CONT || t & 0x0F == LET {
                continue;
            }
            if t & 0x0F == MRK && t != ASM && t != ZWJ {
                continue;
            }
            brk = k;
            break;
        }
        if brk < 16 {
            return p + brk;
        }
        p += 16;
    }
    while p < end && (tags[p] == CONT || member(tags[p])) {
        p += 1;
    }
    p
}

/// rule 4 `[^\s\p{L}\p{N}]+[\r\n/]*` from `sp0` (any leading space already consumed); `sp0` if
/// none. `/` is in the `+` body too — the trailing class only matters after the `+` stops at a
/// `\r\n`.
#[inline(always)]
fn other(text: &[u8], tags: &[u8], sp0: usize, end: usize) -> usize {
    let mut p = run_end(tags, sp0, end, mask::NOT_WS_L_N);
    if p > sp0 {
        while p < end && (tags[p] == NLN || text[p] == b'/') {
            p += char_len(text[p]);
        }
    }
    p
}

/// End of the token starting at `i` (`i < end`, `i` on a token boundary): one rule dispatch of
/// the o200k-family regex. A letter-path token is the FIRST case sub-run from its start (the
/// case split restarts at every sub-token, so one dispatch per sub-token equals the run loop in
/// [`emit_o200k_letters`]); the masked scanner re-derives tokens with this where its batch
/// masks are not trustworthy.
#[inline(always)]
pub(super) fn advance_o200k<const CONTRACTION: bool, const DIGIT_CAP: usize>(
    text: &[u8],
    tags: &[u8],
    i: usize,
    end: usize,
) -> usize {
    let one_letter = |ls: usize| -> usize {
        let re = letter_end(tags, ls, end);
        let e = o200k_letter_match(tags, ls, re);
        if CONTRACTION && e == re {
            e + contraction(text, e)
        } else {
            e
        }
    };
    let b = text[i];
    match tags[i] & 0x0F {
        NW | NO => {
            let (mut p, mut cnt) = (i, 0);
            while p < end && cnt < DIGIT_CAP && in_mask(tags[p], mask::NUMBER) {
                p += char_len(text[p]);
                cnt += 1;
            }
            p
        }
        LET | MRK => {
            if tags[i] != ASM && tags[i] != ZWJ {
                one_letter(i)
            } else {
                let a = i + char_len(b);
                if is_lm(tags, a, end) {
                    one_letter(a)
                } else {
                    other(text, tags, i, end)
                }
            }
        }
        SPC => {
            let a = i + 1; // Space is ASCII (0x20)
            if is_lm(tags, a, end) {
                one_letter(a)
            } else {
                let p = other(text, tags, a, end);
                if p > a {
                    p
                } else {
                    ws_tail(text, tags, i, end)
                }
            }
        }
        WSO => {
            let a = i + char_len(b);
            if is_lm(tags, a, end) {
                one_letter(a)
            } else {
                ws_tail(text, tags, i, end)
            }
        }
        NLN => ws_tail(text, tags, i, end),
        CON | PUN | APO | SYM | NMO | CTL => {
            let a = i + char_len(b);
            if is_lm(tags, a, end) {
                one_letter(a)
            } else {
                other(text, tags, i, end)
            }
        }
        _ => i + char_len(b),
    }
}

fn o200k<const CONTRACTION: bool, const DIGIT_CAP: usize, E: FnMut(Span)>(
    text: &[u8],
    tags: &[u8],
    mut emit: E,
) {
    debug_assert!(tags.len() >= text.len());
    let end = text.len();
    // Tie `tags.len() == end` so the optimizer drops the per-byte bounds check on every interior
    // `tags[i]` in this fsm + its `run_end`/`letter_*` scans. (Callers guarantee `tags.len() >= end`.)
    let tags = &tags[..end];

    let is_lm = |a: usize| is_lm(tags, a, end);
    // rules 5-7 (`\s*[\r\n]+ | \s+(?!\S) | \s+`) → the shared `ws_tail` (identical to cl100k).
    let ws = |i: usize| -> usize { ws_tail(text, tags, i, end) };
    let other = |sp0: usize| -> usize { other(text, tags, sp0, end) };
    // The letter rules: case-split the run starting at `ls`, first sub-token starting at the prefix `pfx`.
    let letters = |pfx: usize, ls: usize, emit: &mut E| -> usize {
        emit_o200k_letters::<CONTRACTION, E>(text, tags, pfx, ls, letter_end(tags, ls, end), emit)
    };

    let mut i = 0;
    while i < end {
        let start = i;
        let b = text[i];
        match tags[i] & 0x0F {
            // rule 3: `\p{N}{1,DIGIT_CAP}`, no prefix (3 = o200k, 1 = tekken)
            NW | NO => {
                let (mut p, mut cnt) = (i, 0);
                while p < end && cnt < DIGIT_CAP && in_mask(tags[p], mask::NUMBER) {
                    p += char_len(text[p]);
                    cnt += 1;
                }
                i = p;
            }
            // letters `[\p{L}\p{M}]+` (case-split) — but ALPHA_SYM/ZWJ (coarse Mark, not `[\p{L}\p{M}]`)
            // take the `[^\r\n\p{L}\p{N}]?` prefix / rule-4 path instead.
            LET | MRK => {
                if tags[i] != ASM && tags[i] != ZWJ {
                    i = letters(i, i, &mut emit);
                    continue;
                }
                let a = i + char_len(b);
                if is_lm(a) {
                    i = letters(i, a, &mut emit);
                    continue;
                }
                i = other(i); // ∈ NOT_WS_L_N ⇒ > i
            }
            // Space: ` ?` prefix + letters | ` ?` + rule-4 other | whitespace
            SPC => {
                let a = i + 1; // Space is ASCII (0x20)
                if is_lm(a) {
                    i = letters(i, a, &mut emit);
                    continue;
                }
                let p = other(a);
                i = if p > a { p } else { ws(i) };
            }
            // WsOther: prefix + letters | whitespace (∈ `\s` ⇒ never starts rule 4)
            WSO => {
                let a = i + char_len(b);
                if is_lm(a) {
                    i = letters(i, a, &mut emit);
                    continue;
                }
                i = ws(i);
            }
            NLN => i = ws(i),
            // punct / sym / … (∈ `[^\r\n\p{L}\p{N}]` and `[^\s\p{L}\p{N}]`): prefix + letters | rule-4 other
            CON | PUN | APO | SYM | NMO | CTL => {
                let a = i + char_len(b);
                if is_lm(a) {
                    i = letters(i, a, &mut emit);
                    continue;
                }
                i = other(i); // ∈ NOT_WS_L_N ⇒ > i
            }
            // Sentinel / MultiByte / Cont — never a char-start atom; emit one char defensively.
            _ => i += char_len(b),
        }
        emit(Span {
            start: start as u32,
            end: i as u32,
        });
    }
}
