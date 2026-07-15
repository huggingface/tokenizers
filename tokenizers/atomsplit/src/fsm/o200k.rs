use super::*;

// ── o200k (GPT-4o): case-aware letter split ──────────────────────────────────────────────────────

/// o200k class of a letter-run char (all chars in a run are real `[\p{L}\p{M}]`, never ALPHA_SYM/ZWJ):
/// `Atom::UpperLetter` → U (`\p{Lu}\p{Lt}`), `Atom::LowerLetter` → L (`\p{Ll}`), else C (caseless `\p{Lm}\p{Lo}
/// \p{M}`). The two alt char-classes are `[UC]` = "not L" (`!o_is_lower`) and `[LC]` = "not U".
#[inline(always)]
fn o_is_upper(t: u8) -> bool {
    t == Atom::UpperLetter as u8 // 0x10: coarse Letter (low nibble 0) + UPPER refine (\p{Lu}∪\p{Lt})
}
#[inline(always)]
fn o_is_lower(t: u8) -> bool {
    t == Atom::LowerLetter as u8 // 0x20: coarse Letter + LOWER refine (\p{Ll})
}

/// Is `t` a real `[\p{L}\p{M}]` member? Coarse `Letter` (any case) is always in; coarse `Mark` is in only
/// as a true `\p{M}` — ALPHA_SYM (`\p{S}`) and ZWJ/ZWNJ (`\p{Cf}`) are `\w` but not `[\p{L}\p{M}]`. `Cont`
/// (low nibble 15) is not a member — the scan loops handle it separately as a transparent ride-along byte.
#[inline(always)]
fn o_member(t: u8) -> bool {
    let c = t & 0x0F;
    c == LET || (c == MRK && t != ASM && t != ZWJ)
}

/// Phase A `[UC]*`: first index ≥ `p` that STOPS the run — a lowercase lead (→ phase B, same sub-token)
/// or a non-member (the `[\p{L}\p{M}]+` run end). `Cont` rides along (transparent).
/// Same unrolled "fast loop" as [`run_end`] (16 tags/chunk, one bounds check, unchecked reads) — a stride
/// of 16 already skips whole cont spans wholesale, so cont bytes are not a per-char cost; a wider stride
/// (measured at 48 = 16×3) only bloats the code and loses on short/2-byte runs.
#[inline(always)]
fn o200k_uc_run(tags: &[u8], mut p: usize, end: usize) -> usize {
    while p + 16 <= end {
        for k in 0..16 {
            let t = unsafe { *tags.get_unchecked(p + k) };
            if t != CONT && (o_is_lower(t) || !o_member(t)) {
                return p + k;
            }
        }
        p += 16;
    }
    while p < end {
        let t = tags[p];
        if t != CONT && (o_is_lower(t) || !o_member(t)) {
            return p;
        }
        p += 1;
    }
    p
}

/// Phase B `[LC]+`: first index ≥ `e` that STOPS — an uppercase lead (sub-token boundary) or a non-member
/// (run end). `Cont` rides along. Same 16-wide unrolled scan as [`o200k_uc_run`].
#[inline(always)]
fn o200k_lc_run(tags: &[u8], mut e: usize, end: usize) -> usize {
    while e + 16 <= end {
        for k in 0..16 {
            let t = unsafe { *tags.get_unchecked(e + k) };
            if t != CONT && (o_is_upper(t) || !o_member(t)) {
                return e + k;
            }
        }
        e += 16;
    }
    while e < end {
        let t = tags[e];
        if t != CONT && (o_is_upper(t) || !o_member(t)) {
            return e;
        }
        e += 1;
    }
    e
}

/// The give-back target: last non-uppercase member char-start in `[p, q)` (all members, no lowercase), or
/// `usize::MAX` if the whole span is uppercase. A backward hop — O(1) for the common all-caseless run
/// (CJK: the last char is the target) — which lets phase A stay a pure skippable scan (no per-char track).
#[inline(always)]
fn o200k_last_c(tags: &[u8], p: usize, mut c: usize) -> usize {
    while c > p {
        c -= 1;
        while c > p && tags[c] == CONT {
            c -= 1; // step back to the char start
        }
        if tags[c] != Atom::UpperLetter as u8 {
            return c; // caseless letter / mark = C
        }
    }
    usize::MAX
}

/// Emit the o200k case-split of the letter run starting at `ls` into `out[*w..]` — ONE forward pass that
/// discovers the `[\p{L}\p{M}]+` run end inline (no separate `letter_end` scan; the case tags are already
/// exact — `Atom::UpperLetter`/`LowerLetter`/caseless `Letter`). The first sub-token starts at `pfx` (the
/// `[^\r\n\p{L}\p{N}]?` prefix; `pfx == ls` when none); the last absorbs a trailing contraction. Returns
/// the cursor past it. `ls < end` (caller-guaranteed via `is_lm`). Each sub-token is `[UC]*[LC]+` (alt-1,
/// greedy with the give-back that lets `[LC]+` take ≥1), an all-U tail being alt-2 `[UC]+[LC]*`.
#[inline(always)]
fn emit_o200k_letters(
    text: &[u8],
    tags: &[u8],
    pfx: usize,
    ls: usize,
    end: usize,
    out: &mut [Span],
    w: &mut usize,
) -> usize {
    let (mut p, mut first) = (ls, true);
    loop {
        let q = o200k_uc_run(tags, p, end); // end of `[UC]*` — a lowercase lead or the run end

        // Resolve the sub-token end and whether another sub-token follows (a U at the boundary vs run end).
        let (tok_end, next_p, more) = if q < end && o_is_lower(tags[q]) {
            let e = o200k_lc_run(tags, q, end); // `[LC]+` from the first L
            (e, e, e < end && o_is_upper(tags[e]))
        } else {
            // no L in [p, q) → q is the run end. Give back to the last caseless/mark char (if any) so
            // `[LC]+` can take ≥1; trailing U's after it re-enter the loop as an alt-2 `[UC]+` token.
            let last_c = o200k_last_c(tags, p, q);
            if last_c == usize::MAX {
                (q, q, false) // all-U alt-2 `[UC]+` = the whole [p, q)
            } else {
                let e = o200k_lc_run(tags, last_c, end);
                (e, e, e < end && o_is_upper(tags[e]))
            }
        };

        let start = if first { pfx } else { p };
        let tok_end = if more {
            tok_end
        } else {
            tok_end + contraction(text, tok_end)
        };
        // SAFETY: tokens partition the input, so `*w < #tokens <= end <= out.len()` (callers size n+1).
        unsafe {
            *out.get_unchecked_mut(*w) = Span {
                start: start as u32,
                end: tok_end as u32,
            }
        };
        *w += 1;
        first = false;
        p = next_p;
        if !more {
            return tok_end;
        }
    }
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
    // Tie `tags.len() == end` so the optimizer drops the per-byte bounds check on every interior
    // `tags[i]` in this fsm + its `run_end`/letter scans. (Callers guarantee `tags.len() >= end`.)
    let tags = &tags[..end];

    // Does a char-start at `a` begin a `[\p{L}\p{M}]` run? (`a` is always a char-start here, never `Cont`.)
    let is_lm = |a: usize| a < end && o_member(tags[a]);
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
    // rules 5-7 (`\s*[\r\n]+ | \s+(?!\S) | \s+`) → the shared `ws_tail` (identical to cl100k).
    let ws = |i: usize| -> usize { ws_tail(text, tags, i, end) };

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
                if tags[i] != ASM && tags[i] != ZWJ {
                    i = emit_o200k_letters(text, tags, i, i, end, out, &mut w);
                    continue;
                }
                let a = i + char_len(b);
                if is_lm(a) {
                    i = emit_o200k_letters(text, tags, i, a, end, out, &mut w);
                    continue;
                }
                i = other(i); // ∈ NOT_WS_L_N ⇒ > i
            }
            // Space: ` ?` prefix + letters | ` ?` + rule-4 other | whitespace
            SPC => {
                let a = i + 1; // Space is ASCII (0x20)
                if is_lm(a) {
                    i = emit_o200k_letters(text, tags, i, a, end, out, &mut w);
                    continue;
                }
                let p = other(a);
                i = if p > a { p } else { ws(i) };
            }
            // WsOther: prefix + letters | whitespace (∈ `\s` ⇒ never starts rule 4)
            WSO => {
                let a = i + char_len(b);
                if is_lm(a) {
                    i = emit_o200k_letters(text, tags, i, a, end, out, &mut w);
                    continue;
                }
                i = ws(i);
            }
            NLN => i = ws(i),
            // punct / sym / … (∈ `[^\r\n\p{L}\p{N}]` and `[^\s\p{L}\p{N}]`): prefix + letters | rule-4 other
            CON | PUN | APO | SYM | NMO | CTL => {
                let a = i + char_len(b);
                if is_lm(a) {
                    i = emit_o200k_letters(text, tags, i, a, end, out, &mut w);
                    continue;
                }
                i = other(i); // ∈ NOT_WS_L_N ⇒ > i
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
