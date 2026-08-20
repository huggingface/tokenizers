//! o200k_base / GPT-4o — byte-for-byte the regex **Llama-4, gpt-oss and MiniMax-M2** also ship:
//! `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|…)?`
//! `|` same with `+`/`*` swapped `| \p{N}{1,3} | ?[^\s\p{L}\p{N}]+[\r\n/]* | \s*[\r\n]+ | \s+(?!\S) | \s+`
//!
//! Same skeleton as cl100k; only the letter half differs. The grammar itself lives in
//! [`super::family_o200k`] — tekken and kimi are the same one at different knob settings.

use crate::{AUX_SLASH, Span};

/// o200k_base / GPT-4o — and byte-for-byte the same regex Llama-4, gpt-oss and MiniMax-M2 ship.
#[must_use]
pub fn bitsplit_o200k(
    text: &[u8],
    tags: &[u8],
    starts: &mut [u64],
    flag: &mut [u64],
    later: &mut [u64],
    out: &mut [Span],
) -> usize {
    super::family_o200k::run::<{ AUX_SLASH }, true, 3, false>(text, tags, starts, flag, later, out)
}
