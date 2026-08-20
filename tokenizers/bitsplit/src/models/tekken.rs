//! Mistral `tekken` (mistral-small-4 / mistral-4). o200k's grammar with two changes: letter tokens
//! take no contraction suffix, and the digit rule is a bare `\p{N}` — one token per digit.
//!
//! The grammar lives in [`super::family_o200k`]; this is o200k's regex at `CONTR = false, DIGITS = 1`.

use crate::{AUX_SLASH, Span};

/// Mistral tekken pre-tokenization.
#[must_use]
pub fn bitsplit_tekken(
    text: &[u8],
    tags: &[u8],
    starts: &mut [u64],
    flag: &mut [u64],
    out: &mut [Span],
) -> usize {
    super::family_o200k::run::<{ AUX_SLASH }, false, 1>(text, tags, starts, flag, out)
}
