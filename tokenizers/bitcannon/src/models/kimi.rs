//! kimi-k2 / k3 — `moonshotai/Kimi-K2-Instruct`'s `tokenization_kimi.py` `pat_str`. o200k plus a
//! leading `[\p{Han}]+` arm, Han subtracted from both letter classes, and a `[\r\n]*` rule-4 tail
//! (o200k has `[\r\n/]*`). Kimi ships `tiktoken.model` rather than a `tokenizer.json`, so this is
//! the pattern as a converted tokenizer would spell it.
//!
//! The bitstream grammar lives in [`super::family_o200k`]. For Han characters whose general
//! categories overlap other alternatives, the scalar path preserves ordered matching at each
//! token start, including greedy number and punctuation runs.

use crate::classify::char_len;
use crate::{AUX_NONE, Span, contr_len};

/// kimi-k2 pre-tokenization.
#[must_use]
pub fn bitcannon_kimi(
    text: &[u8],
    tags: &[u8],
    starts: &mut [u64],
    flag: &mut [u64],
    later: &mut [u64],
    out: &mut [Span],
) -> usize {
    let nblk = text.len().div_ceil(64);
    assert!(
        tags.len() >= text.len()
            && starts.len() >= nblk
            && later.len() >= 2 * nblk
            && out.len() >= text.len()
    );
    if tags[..text.len()]
        .iter()
        .any(|tag| matches!(tag, 0x31 | 0x36 | 0x38 | 0x3A))
    {
        // Script=Han overlaps the number and punctuation arms. Resolve their ordered,
        // greedy matches at token starts rather than removing Han from those classes.
        return overlapping_han(text, tags, out);
    }
    super::family_o200k::run::<{ AUX_NONE }, true, 3, true>(text, tags, starts, flag, later, out)
}

fn letter_end(text: &[u8], tags: &[u8], start: usize, first: bool) -> usize {
    let mut end = start;
    let mut last_common = None;
    while end < text.len() && matches!(tags[end], 0x00 | 0x06 | 0x10) {
        let tag = tags[end];
        end += char_len(text[end]);
        if tag != 0x10 {
            last_common = Some(end);
        }
    }
    if !first && end == start {
        return start;
    }
    if end < text.len() && matches!(tags[end], 0x00 | 0x06 | 0x20) {
        while end < text.len() && matches!(tags[end], 0x00 | 0x06 | 0x20) {
            end += char_len(text[end]);
        }
        end
    } else if first {
        last_common.unwrap_or(start)
    } else {
        end
    }
}

fn overlapping_han(text: &[u8], tags: &[u8], out: &mut [Span]) -> usize {
    let (mut pos, mut count) = (0, 0);
    while pos < text.len() {
        let start = pos;
        if tags[pos] & 0xF0 == 0x30 {
            while pos < text.len() && tags[pos] & 0xF0 == 0x30 {
                pos += char_len(text[pos]);
            }
        } else {
            let prefix = !matches!(tags[pos] & 0x0F, 0..=3);
            let mut end = pos;
            for first in [true, false] {
                if prefix {
                    let after = pos + char_len(text[pos]);
                    let matched = letter_end(text, tags, after, first);
                    if matched > after {
                        end = matched;
                    }
                }
                if end == pos {
                    end = letter_end(text, tags, pos, first);
                }
                if end > pos {
                    break;
                }
            }
            if end > pos {
                pos = end + contr_len(text, end, true);
            } else if matches!(tags[pos] & 0x0F, 1 | 2) {
                for _ in 0..3 {
                    if pos == text.len() || !matches!(tags[pos] & 0x0F, 1 | 2) {
                        break;
                    }
                    pos += char_len(text[pos]);
                }
            } else {
                let mut end = pos + usize::from(text[pos] == b' ');
                while end < text.len() && !matches!(tags[end] & 0x0F, 0..=5) {
                    end += char_len(text[end]);
                }
                if end > pos + usize::from(text[pos] == b' ') {
                    while end < text.len() && tags[end] == 3 {
                        end += char_len(text[end]);
                    }
                    pos = end;
                } else {
                    let mut last = pos;
                    let mut newline_end = pos;
                    while end < text.len() && matches!(tags[end], 3..=5) {
                        last = end;
                        end += char_len(text[end]);
                        if tags[last] == 3 {
                            newline_end = end;
                        }
                    }
                    pos = if newline_end > pos {
                        newline_end
                    } else if end < text.len() && last > pos {
                        last
                    } else {
                        end
                    };
                }
            }
        }
        out[count] = Span::new(start as u32, pos as u32);
        count += 1;
    }
    count
}
