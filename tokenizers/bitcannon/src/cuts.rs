//! **Safe cuts**: positions `p` where `split(text) == split(text[..p]) ++ split(text[p..]) + p`, so
//! a large input can be pre-tokenized (and encoded) in independent chunks.
//!
//! Decided from the two chars around `p` only, so a planner can probe near any offset without
//! scanning from the start. A strict subset of each grammar's token starts: a missed cut costs
//! parallelism, a wrong one changes ids.

use crate::classify::{CONT, classify, in_mask, mask};

/// Bytes classified per window (stack buffer).
const WINDOW: usize = 256;

/// The first position `p` in `from..to` where the char before `p` is a letter (`\p{L}`) and the
/// char at `p` is not.
///
/// Sound for **cl100k** (Llama-3, GLM, Qwen) and **GPT-2**: there every token containing a letter
/// ends with it (`\p{L}+` runs, contractions), so a token ends at `p`; the right half restarts the
/// match at `p` (no lookbehind) and nothing before `p` looked past it (the only lookahead,
/// `\s+(?!\S)`, peeks at the letter). Checked in `tests/cuts.rs`.
///
/// Unsound for o200k (contractions are word suffixes) and deepseek (`[\p{L}\p{M}]+`).
///
/// `text` must be valid UTF-8, and `from` a char boundary. Returns `None` if there is no such cut
/// before `to` (clamped to `text.len()`).
#[must_use]
pub fn after_letter(text: &[u8], from: usize, to: usize) -> Option<usize> {
    let to = to.min(text.len());
    if from >= to {
        return None;
    }
    // start at the lead byte of the char before `from`
    let mut pos = from.saturating_sub(1);
    while pos > 0 && text[pos] & 0xC0 == 0x80 {
        pos -= 1;
    }
    let mut tags = [0u8; WINDOW + 3];
    let mut prev_letter = false;
    while pos < to {
        // whole chars only
        let mut end = (pos + WINDOW).min(text.len());
        while end < text.len() && text[end] & 0xC0 == 0x80 {
            end += 1;
        }
        let tags = &mut tags[..end - pos];
        classify(&text[pos..end], tags);
        for (i, &tag) in tags.iter().enumerate() {
            if tag == CONT {
                continue;
            }
            let letter = in_mask(tag, mask::LETTER);
            let p = pos + i;
            if prev_letter && !letter && p >= from {
                return (p < to).then_some(p);
            }
            prev_letter = letter;
        }
        pos = end;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::after_letter;

    fn all(text: &str) -> Vec<usize> {
        let mut out = Vec::new();
        let mut from = 1;
        while let Some(p) = after_letter(text.as_bytes(), from, text.len()) {
            out.push(p);
            from = p + 1;
            while !text.is_char_boundary(from) {
                from += 1;
            }
        }
        out
    }

    #[test]
    fn cuts_where_a_letter_run_ends() {
        assert_eq!(all("ab cd"), vec![2]);
        assert_eq!(all("don't stop"), vec![3, 5]);
        assert_eq!(all("abc123def!"), vec![3, 9]);
        // no letter → nothing to end
        assert_eq!(all("123 456 !!"), Vec::<usize>::new());
        // a run ending at the very end of the text is not a cut: `p` must be < `to`
        assert_eq!(all("abc"), Vec::<usize>::new());
    }

    #[test]
    fn multibyte_letters_cut_after_their_last_byte() {
        // `é` is 2 bytes, `中文` 3 each: the cut lands after the whole char
        assert_eq!(all("café."), vec![5]);
        assert_eq!(all("中文。"), vec![6]);
        // a combining mark is not `\p{L}`: the base letter's run ends before it
        assert_eq!(all("a\u{0301}b"), vec![1]);
    }

    #[test]
    fn window_bounds() {
        // `from` past the run's end still sees the letter before it
        assert_eq!(after_letter(b"ab cd", 2, 5), Some(2));
        assert_eq!(after_letter(b"ab cd", 3, 5), None);
        // `to` is exclusive
        assert_eq!(after_letter(b"ab cd", 1, 2), None);
        // a cut far past the first window
        let text = format!("{}x.", "1".repeat(1000));
        assert_eq!(after_letter(text.as_bytes(), 1, text.len()), Some(1001));
    }
}
