use super::*;

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
    // Tie `tags.len() == end` so the optimizer drops the per-byte bounds check on every interior
    // `tags[i]` in this fsm + its `run_end`/`letter_*` scans. (Callers guarantee `tags.len() >= end`.)
    let tags = &tags[..end];
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
    // rules 5-7 (`\s*[\r\n] | \s+(?!\S) | \s+`) → the shared `ws_tail`.
    let ws = |i: usize| -> usize { ws_tail(text, tags, i, end) };

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
                let adv = contraction(text, i); // rule 1: `'s 't 're 've 'm 'll 'd` (case-insensitive)
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
            // Mark | Connector | Punct | SymOther | NumericOther | Control (all in NOT_WS_L_N):
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
