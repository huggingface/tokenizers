use super::*;

/// GPT-2 / ByteLevel pretokenization. Regex (same jump-table shape as cl100k, with 3 differences):
///   `'s|'t|'re|'ve|'m|'ll|'d | ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+ | \s+(?!\S) | \s+`
/// vs cl100k: (1) contractions are case-SENSITIVE (lowercase only, no `(?i:)`); (2) the ` ?` prefix is
/// a literal SPACE only (not any non-l/n char) and it applies to letters, numbers AND "other"; (3) no
/// `\p{N}{1,3}` cap (numbers are unbounded) and no `\s*[\r\n]`/trailing-`[\r\n]*` rules.
#[must_use]
pub fn fsm_byte_level(text: &[u8], tags: &[u8], out: &mut [Span]) -> usize {
    debug_assert!(out.len() >= text.len() && tags.len() >= text.len());
    let mut w = 0usize;
    scan_byte_level(text, tags, |span| {
        // SAFETY: tokens partition the input, so `w < #tokens <= text.len() <= out.len()`.
        unsafe { *out.get_unchecked_mut(w) = span };
        w += 1;
    });
    w
}

/// The scan under [`fsm_byte_level`]: hands each token to `emit` the moment it is cut,
/// so a caller can consume tokens in place instead of collecting a span buffer first.
pub fn scan_byte_level(text: &[u8], tags: &[u8], mut emit: impl FnMut(Span)) {
    debug_assert!(tags.len() >= text.len());
    let end = text.len();
    // Tie `tags.len() == end` so the optimizer drops the per-byte bounds check on every interior
    // `tags[i]` in this fsm + its `run_end`/`letter_*` scans. (Callers guarantee `tags.len() >= end`.)
    let tags = &tags[..end];
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
        emit(Span {
            start: start as u32,
            end: i as u32,
        });
    }
}
