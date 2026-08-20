//! kimi-k2 / k3 — `moonshotai/Kimi-K2-Instruct`'s `tokenization_kimi.py` `pat_str`. o200k plus a
//! leading `[\p{Han}]+` arm, Han subtracted from both letter classes, and a `[\r\n]*` rule-4 tail
//! (o200k has `[\r\n/]*`). Kimi ships `tiktoken.model` rather than a `tokenizer.json`, so this is
//! the pattern as a converted tokenizer would spell it.
//!
//! The grammar lives in [`super::family_o200k`]; this is o200k's regex at `AUX = AUX_HAN`, which both
//! turns on the Han arm and drops `/` from the rule-4 tail.

use crate::{AUX_HAN, Span};

/// kimi-k2 pre-tokenization.
#[must_use]
pub fn bitsplit_kimi(
    text: &[u8],
    tags: &[u8],
    starts: &mut [u64],
    flag: &mut [u64],
    out: &mut [Span],
) -> usize {
    super::family_o200k::run::<{ AUX_HAN }, true, 3>(text, tags, starts, flag, out)
}
