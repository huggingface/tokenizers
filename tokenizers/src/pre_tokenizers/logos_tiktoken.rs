//! Compile-time DFA for the tiktoken `cl100k_base` split pattern used by
//! Llama-3, GPT-3.5/4-class tokenizers, and other modern LLMs that use
//! `Split` + a (basically) tiktoken-style regex in their serialized config.
//!
//! The target pattern (verbatim from `data/llama-3-tokenizer.json`):
//!
//! ```text
//! (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//! ```
//!
//! Implementation notes (same structural trick as `byte_level::BlTok`):
//!   - Lookahead `\s+(?!\S)` can't be expressed in logos. We emit plain
//!     `\s+` and replay the lookahead's backtrack semantics in a post-
//!     processing pass on the token stream.
//!   - Longest-match (logos) vs leftmost-first (onig/fancy-regex) differ
//!     when a `Letters` match with a non-whitespace prefix (e.g. `(hello`)
//!     or a `Contraction` match (`'t`) overlaps with a shorter `Other`
//!     match the legacy engine would have picked earlier in its
//!     alternation. Those get fixed up in the same post-processing pass.

#![cfg(feature = "logos-pretok")]

use logos::Logos;

use crate::tokenizer::pattern::Pattern;
use crate::tokenizer::{Offsets, Result};

/// The exact regex string this logos enum is equivalent to. `Split::new`
/// compares the user's pattern string against this to decide whether to
/// dispatch through the logos DFA instead of `SysRegex`.
#[doc(hidden)]
pub const CL100K_PATTERN: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Logos-derived token categories mirroring the cl100k alternation in
/// declaration order. Source order is the priority tiebreaker when two
/// patterns match the same span.
#[derive(Logos, Debug, Clone, Copy, PartialEq, Eq)]
enum Cl100kTok {
    #[regex(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)")]
    Contraction,

    #[regex(r"[^\r\n\p{L}\p{N}]?\p{L}+")]
    Letters,

    #[regex(r"\p{N}{1,3}")]
    Numbers,

    #[regex(r" ?[^\s\p{L}\p{N}]+[\r\n]*")]
    Other,

    #[regex(r"\s*[\r\n]+", priority = 3)]
    NewlineRun,

    #[regex(r"\s+")]
    Whitespace,
}

/// Zero-sized `Pattern` impl. Hidden re-export: used by the equivalence
/// integration test to drive the logos path directly side-by-side with
/// the legacy `SysRegex`.
#[doc(hidden)]
pub struct LogosCl100k;

impl Pattern for &LogosCl100k {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        if inside.is_empty() {
            return Ok(vec![((0, 0), false)]);
        }

        // First pass: consume the input with logos, collecting (variant, start, end).
        let mut tokens: Vec<(Option<Cl100kTok>, usize, usize)> = Vec::with_capacity(inside.len());
        let mut lex = Cl100kTok::lexer(inside);
        while let Some(result) = lex.next() {
            let span = lex.span();
            tokens.push((result.ok(), span.start, span.end));
        }

        split_letters_contractions(inside, &mut tokens);
        replay_lookahead(inside, &mut tokens);

        // Flatten to (Offsets, true) sequence; fill any gaps with `false`.
        let mut prev = 0;
        let mut splits = Vec::with_capacity(tokens.len());
        for (_variant, start, end) in tokens {
            if start == end {
                continue;
            }
            if prev != start {
                splits.push(((prev, start), false));
            }
            splits.push(((start, end), true));
            prev = end;
        }
        if prev != inside.len() {
            splits.push(((prev, inside.len()), false));
        }
        Ok(splits)
    }
}

/// Legacy's alternation puts `(?i:'s|'t|'re|'ve|'m|'ll|'d)` *before*
/// `[^\r\n\p{L}\p{N}]?\p{L}+`, so at a position like `'store` it matches
/// `'s` (2 chars) even though `'store` (6 chars) would match the later
/// `Letters` alternative. Logos's longest-match picks `'store`. Detect
/// `Letters` spans starting with `'<contraction suffix>` and split them
/// into `Contraction('X)` + `Letters(rest)`.
fn split_letters_contractions(
    inside: &str,
    tokens: &mut Vec<(Option<Cl100kTok>, usize, usize)>,
) {
    let mut i = 0;
    while i < tokens.len() {
        let (variant, start, end) = tokens[i];
        if !matches!(variant, Some(Cl100kTok::Letters)) {
            i += 1;
            continue;
        }
        if !inside[start..end].starts_with('\'') {
            i += 1;
            continue;
        }
        // At least one char after the quote is required (we're inside a
        // Letters match which means `\p{L}+` matched ≥ 1 letter after
        // the quote prefix).
        let after = &inside.as_bytes()[start + 1..end];
        // Contraction suffixes by byte pattern, longest first so `re`
        // wins over `r`-only (which isn't a contraction anyway).
        let suffix_len = match after {
            [b'r' | b'R', b'e' | b'E', ..] => 2,
            [b'v' | b'V', b'e' | b'E', ..] => 2,
            [b'l' | b'L', b'l' | b'L', ..] => 2,
            [b's' | b'S', ..] => 1,
            [b't' | b'T', ..] => 1,
            [b'm' | b'M', ..] => 1,
            [b'd' | b'D', ..] => 1,
            _ => 0,
        };
        if suffix_len == 0 {
            i += 1;
            continue;
        }
        let contraction_end = start + 1 + suffix_len; // 1 byte for `'`
        if contraction_end == end {
            // Whole Letters span is exactly the contraction.
            tokens[i] = (Some(Cl100kTok::Contraction), start, end);
        } else {
            tokens[i] = (Some(Cl100kTok::Contraction), start, contraction_end);
            tokens.insert(
                i + 1,
                (Some(Cl100kTok::Letters), contraction_end, end),
            );
            i += 1;
        }
        i += 1;
    }
}

/// Replays the `\s+(?!\S)` lookahead + leftmost-first-priority side effects
/// by rewriting the token stream in place.
fn replay_lookahead(inside: &str, tokens: &mut Vec<(Option<Cl100kTok>, usize, usize)>) {
    let mut i = 0;
    while i < tokens.len().saturating_sub(1) {
        if !matches!(tokens[i].0, Some(Cl100kTok::Whitespace)) {
            i += 1;
            continue;
        }
        let (ws_start, ws_end) = (tokens[i].1, tokens[i].2);
        let ws_slice = &inside[ws_start..ws_end];
        let ws_chars = ws_slice.chars().count();

        // Byte offset of the last char inside the ws span.
        let last_char_off = ws_slice
            .char_indices()
            .last()
            .map(|(b, _)| b)
            .unwrap_or(0);
        let shrunk_end = ws_start + last_char_off;

        match tokens[i + 1].0 {
            Some(Cl100kTok::Letters) => {
                if ws_chars < 2 {
                    i += 1;
                    continue;
                }
                // Check the first char of the Letters span. If it's a letter,
                // the logos match had no prefix — just shrink ws, extend
                // Letters. If it's non-letter, the prefix is a char that
                // legacy would have routed through `Other` (leftmost-first:
                // `Other` is later in the alternation but `Letters` in
                // tiktoken DOESN'T require a space prefix, so this branch
                // is tricky). Split Letters into Other(prefix) +
                // Letters(rest), extend Other to claim the freed ws char,
                // merge Letters(rest) with any following contiguous Letters.
                let (_, lstart, lend) = tokens[i + 1];
                let first_char = inside[lstart..lend].chars().next().unwrap();
                if first_char.is_alphabetic() {
                    tokens[i].2 = shrunk_end;
                    tokens[i + 1].1 = shrunk_end;
                    i += 1;
                } else {
                    let prefix_end = lstart + first_char.len_utf8();
                    tokens[i].2 = shrunk_end;
                    tokens[i + 1] = (Some(Cl100kTok::Other), shrunk_end, prefix_end);

                    let letters_seg = (Some(Cl100kTok::Letters), prefix_end, lend);
                    merge_or_insert_letters(inside, tokens, i + 2, letters_seg);
                    i += 2;
                }
            }
            Some(Cl100kTok::Other) => {
                if ws_chars < 2 {
                    i += 1;
                    continue;
                }
                tokens[i].2 = shrunk_end;
                tokens[i + 1].1 = shrunk_end;
                i += 1;
            }
            Some(Cl100kTok::Contraction) => {
                // Legacy's leftmost-first `Other` alternative picks up ` '`
                // before the contraction literal gets tried when there's a
                // preceding whitespace. Split the contraction into
                // `Other(')` (extended leftward to claim the freed ws char)
                // + `Letters(rest)`, merging `Letters(rest)` with any
                // contiguous alphabetic `Letters` span that follows.
                let (_, cstart, cend) = tokens[i + 1];
                let quote_end = cstart + 1;
                let letters_span = (quote_end, cend);
                // Can the following Letters span absorb our synthetic
                // letters segment? (See `merge_or_insert_letters` for the
                // rules — must be alphabetic-first, contiguous Letters.)
                let can_merge_next = tokens
                    .get(i + 2)
                    .map(|next| {
                        matches!(next.0, Some(Cl100kTok::Letters))
                            && next.1 == cend
                            && inside[next.1..next.2]
                                .chars()
                                .next()
                                .map(|c| c.is_alphabetic())
                                .unwrap_or(false)
                    })
                    .unwrap_or(false);

                if ws_chars == 1 {
                    // ws(1) fully consumed: Other covers (ws_start, quote_end).
                    tokens[i] = (Some(Cl100kTok::Other), ws_start, quote_end);
                    // tokens[i+1] was the Contraction. Replace with Letters
                    // or merge with neighbor.
                    if can_merge_next {
                        tokens[i + 2].1 = letters_span.0;
                        tokens.remove(i + 1);
                    } else if letters_span.0 < letters_span.1 {
                        tokens[i + 1] =
                            (Some(Cl100kTok::Letters), letters_span.0, letters_span.1);
                    } else {
                        tokens.remove(i + 1);
                    }
                } else {
                    // ws_chars >= 2: shrink ws, insert Other between ws and
                    // Contraction, replace Contraction with Letters(rest).
                    tokens[i].2 = shrunk_end;
                    tokens.insert(
                        i + 1,
                        (Some(Cl100kTok::Other), shrunk_end, quote_end),
                    );
                    // Original Contraction is now at i+2; tokens[i+3] was
                    // the old tokens[i+2] (the merge candidate).
                    if can_merge_next {
                        tokens[i + 3].1 = letters_span.0;
                        tokens.remove(i + 2);
                    } else if letters_span.0 < letters_span.1 {
                        tokens[i + 2] =
                            (Some(Cl100kTok::Letters), letters_span.0, letters_span.1);
                    } else {
                        tokens.remove(i + 2);
                    }
                }
                i += 2;
            }
            Some(Cl100kTok::Numbers) => {
                // Numbers has no ` ?` prefix — can't absorb a leading space.
                // Legacy produces ws(N-1) + ws(1) + Numbers; mirror that by
                // splitting the ws run.
                if ws_chars < 2 {
                    i += 1;
                    continue;
                }
                tokens[i].2 = shrunk_end;
                tokens.insert(
                    i + 1,
                    (Some(Cl100kTok::Whitespace), shrunk_end, ws_end),
                );
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }
}

/// If `tokens[at]` is a `Letters` span starting at `seg.2` whose content
/// begins with a letter (i.e. no prefix char), extend it leftward to
/// `seg.1` so the split `seg` merges into it. Otherwise insert `seg` at
/// position `at`.
fn merge_or_insert_letters(
    inside: &str,
    tokens: &mut Vec<(Option<Cl100kTok>, usize, usize)>,
    at: usize,
    seg: (Option<Cl100kTok>, usize, usize),
) {
    if seg.1 >= seg.2 {
        return;
    }
    if let Some(next) = tokens.get(at).copied() {
        if matches!(next.0, Some(Cl100kTok::Letters)) && next.1 == seg.2 {
            let first = inside[next.1..next.2].chars().next().unwrap();
            if first.is_alphabetic() {
                tokens[at].1 = seg.1;
                return;
            }
        }
    }
    tokens.insert(at, seg);
}
