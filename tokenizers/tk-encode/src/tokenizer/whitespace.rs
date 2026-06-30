use unicode_categories::UnicodeCategories;

/// Somewhat ai generated, but its not super complicated
/// We don't do a full check of the bytes, we preemptibely compute the length of the whitespace
/// based on a "table". TODO: we can probably push the perfs even more by precomputing instead of
/// branching?
/// If a Unicode `White_Space` char starts at `bytes[i]`, return its byte length, else `None`.
/// `bytes` must be valid UTF-8 and `i` a char boundary (true for any `&str`-derived slice).
/// Matches `char::is_whitespace` — incl. VT (0x0B) and the multi-byte Unicode spaces.
#[inline]
fn whitespace_len_at(bytes: &[u8], i: usize) -> Option<usize> {
    let b = *bytes.get(i)?;
    if b < 0x80 {
        // ASCII White_Space = 0x09..=0x0D, 0x20
        return (b == 0x20 || b.wrapping_sub(0x09) < 5).then_some(1);
    }
    let len = if b < 0xE0 {
        2
    } else if b < 0xF0 {
        3
    } else {
        4
    };
    let ws = match b {
        0xC2 => matches!(bytes[i + 1], 0x85 | 0xA0), // NEL, NBSP
        0xE1 => bytes[i + 1] == 0x9A && bytes[i + 2] == 0x80, // U+1680
        0xE2 => {
            let (c1, c2) = (bytes[i + 1], bytes[i + 2]);
            (c1 == 0x80 && matches!(c2, 0x80..=0x8A | 0xA8 | 0xA9 | 0xAF))    // U+2000..200A,2028,2029,202F
                || (c1 == 0x81 && c2 == 0x9F) // U+205F
        }
        0xE3 => bytes[i + 1] == 0x80 && bytes[i + 2] == 0x80, // U+3000
        _ => false,
    };
    ws.then_some(len)
}

/// rstrip: index one past the trailing whitespace run starting at `from`.
#[inline]
pub fn skip_whitespace_forward(bytes: &[u8], from: usize) -> usize {
    let mut i = from;
    while let Some(len) = whitespace_len_at(bytes, i) {
        i += len;
    }
    i
}

/// lstrip: start index of the whitespace run ending at `from` (exclusive). Walks backward over
/// char boundaries. Clamp the result at the previous emitted span so you don't steal its bytes.
#[inline]
pub fn skip_whitespace_backward(bytes: &[u8], from: usize) -> usize {
    let mut i = from;
    while i > 0 {
        let mut j = i - 1; // step back over continuation bytes (0x80..=0xBF) to the lead byte
        while j > 0 && bytes[j] & 0xC0 == 0x80 {
            j -= 1;
        }
        if whitespace_len_at(bytes, j).is_some() {
            i = j;
        } else {
            break;
        }
    }
    i
}

/// `\w`-style word char (Unicode), matching HF's regex `\w`: alphanumeric, the connector `_`, or a
/// combining/other mark. Marks matter — e.g. U+0330 is a word char, which `is_alphanumeric` misses.
#[inline]
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c.is_mark()
}

/// Is the char starting at `bytes[i]` a word char? Decodes straight from the bytes (no `&str`
/// slicing). `i` must be a char boundary in valid UTF-8.
#[inline]
fn word_char_at(bytes: &[u8], i: usize) -> bool {
    let b = bytes[i];
    let cp = if b < 0x80 {
        b as u32
    } else if b < 0xE0 {
        (((b & 0x1F) as u32) << 6) | (bytes[i + 1] & 0x3F) as u32
    } else if b < 0xF0 {
        (((b & 0x0F) as u32) << 12)
            | (((bytes[i + 1] & 0x3F) as u32) << 6)
            | (bytes[i + 2] & 0x3F) as u32
    } else {
        (((b & 0x07) as u32) << 18)
            | (((bytes[i + 1] & 0x3F) as u32) << 12)
            | (((bytes[i + 2] & 0x3F) as u32) << 6)
            | (bytes[i + 3] & 0x3F) as u32
    };
    // SAFETY: `i` is a char boundary in valid UTF-8, so `cp` is a valid Unicode scalar value.
    is_word_char(unsafe { char::from_u32_unchecked(cp) })
}

/// `single_word` boundary test, byte-native (~1.3x faster than the `chars()` version): the match
/// `[start, end)` is a standalone word iff neither neighbour char is a word char. Only the region
/// since the last match (`lower`, e.g. the previous match end) is searched on the left — at
/// `start <= lower` the previous boundary already bounds it. `lower`/`start`/`end` are char boundaries.
#[inline]
pub fn is_single_word(bytes: &[u8], lower: usize, start: usize, end: usize) -> bool {
    let left_ok = start <= lower || {
        let mut j = start - 1; // step back over continuation bytes to the previous lead byte
        while j > lower && bytes[j] & 0xC0 == 0x80 {
            j -= 1;
        }
        !word_char_at(bytes, j)
    };
    let right_ok = end == bytes.len() || !word_char_at(bytes, end);
    left_ok && right_ok
}

#[cfg(test)]
mod ws_tests {
    use super::*;
    #[test]
    fn strip() {
        let b = "x \u{00A0}\u{3000}\ty".as_bytes(); // x, space, NBSP, ideographic, tab, y
        assert_eq!(skip_whitespace_forward(b, 1), b.len() - 1); // consumes through '\t', stops at 'y'
        assert_eq!(skip_whitespace_backward(b, b.len() - 1), 1); // walks back to just after 'x'
        assert_eq!(skip_whitespace_forward(b"\x0b\x0bz", 0), 2); // VT counts (unlike is_ascii_whitespace)
        assert_eq!(skip_whitespace_forward(b"abc", 0), 0); // no leading ws
    }

    #[test]
    fn single_word() {
        // "XmaskY": token "mask" at [1,5). left 'X' is a word char -> reject.
        assert!(!is_single_word(b"XmaskY", 0, 1, 5));
        // " mask ": neighbours are spaces (non-word) -> standalone.
        assert!(is_single_word(b" mask ", 0, 1, 5));
        // combining mark U+0330 (CC B0) before the token is a word char -> reject.
        let s = "\u{0330}mask"; // 2 bytes mark + "mask" -> token at [2,6)
        assert!(!is_single_word(s.as_bytes(), 0, 2, 6));
        // match sitting right at `lower` -> previous boundary bounds the left -> ok (right is edge).
        assert!(is_single_word(b" mask", 1, 1, 5));
    }
}
