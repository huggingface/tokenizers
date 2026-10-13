use std::ops::Range;

use wide::u8x16;

use super::CharacterCache;

/// Given a slice bytes, return the length of the largest ASCII prefix.
/// Leverages SIMD to check the input 16 bytes at a time.
pub(crate) fn ascii_prefix_len(input: &[u8]) -> usize {
    let mut chunks = input.chunks_exact(16);
    let mut offset = 0;
    for chunk in &mut chunks {
        // In unicode, only ASCII bytes have their high bit set to 0 (meaning they are a single-byte codepoint)
        // and .to_bitmask packs the top-bits into a u32, the lowest set bit is then the first non-ascii byte
        let mask = u8x16::from(<[u8; 16]>::try_from(chunk).unwrap()).to_bitmask();
        if mask != 0 {
            return offset + mask.trailing_zeros() as usize;
        }
        offset += 16;
    }
    let remainder = chunks.remainder();
    let mut window = [0u8; 16];
    window[..remainder.len()].copy_from_slice(remainder);
    let mask = u8x16::from(window).to_bitmask();
    offset + (mask.trailing_zeros() as usize).min(remainder.len())
}

/// Finds the offset of the next chunk of 16 bytes that is only ascii characters, or input.len()
/// Strides input 16 bytes per 16 bytes
pub(crate) fn next_ascii_chunk(input: &[u8]) -> usize {
    let mut offset: usize = 0;
    while let Some(chunk) = input.get(offset..offset + 16) {
        if u8x16::from(<[u8; 16]>::try_from(chunk).unwrap()).to_bitmask() == 0 {
            return offset;
        }
        offset += 16
    }
    input.len()
}

pub(crate) fn non_ascii_segments_iter(bytes: &[u8]) -> impl Iterator<Item = Range<usize>> {
    let mut cursor = ascii_prefix_len(bytes);
    std::iter::from_fn(move || {
        if cursor >= bytes.len() {
            return None;
        }
        let start = cursor;
        let end = cursor + next_ascii_chunk(&bytes[cursor..]);
        cursor = end + ascii_prefix_len(&bytes[end..]);
        Some(start..end)
    })
}

pub(crate) fn unstable_spans_iter(
    input: &str,
    unstable: &CharacterCache,
    keep_previous_char: bool,
) -> impl Iterator<Item = Range<usize>> {
    non_ascii_segments_iter(input.as_bytes()).flat_map(move |segment| {
        unstable_spans_in_segment(
            &input[segment.clone()],
            segment.start,
            unstable,
            keep_previous_char,
        )
    })
}

fn unstable_spans_in_segment(
    segment: &str,
    offset: usize,
    unstable: &CharacterCache,
    keep_previous_char: bool,
) -> impl Iterator<Item = Range<usize>> {
    let mut chars = segment.char_indices();
    // The byte before an unstable span is always ascii, hence always a stable starter
    let mut previous_stable = offset.checked_sub(1);
    std::iter::from_fn(move || {
        let (run_start, _) = chars.find(|&(i, c)| {
            let is_unstable = unstable.get(c);
            if !is_unstable {
                previous_stable = Some(offset + i);
            }
            is_unstable
        })?;
        let start = match previous_stable {
            Some(previous) if keep_previous_char => previous,
            _ => offset + run_start,
        };
        let end = match chars.find(|&(_, c)| !unstable.get(c)) {
            Some((i, _)) => {
                previous_stable = Some(offset + i);
                i
            }
            None => segment.len(),
        };
        Some(start..offset + end)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalizers::utils::random_texts;

    #[test]
    fn ascii_prefix_len_stops_at_the_first_high_byte() {
        for len in 0..=70 {
            let ascii = vec![0x7F; len];
            assert_eq!(ascii_prefix_len(&ascii), len, "all ASCII, len {len}");
            for at in 0..len {
                for high in [0x80, 0xFF] {
                    let mut bytes = ascii.clone();
                    bytes[at..].fill(high);
                    assert_eq!(
                        ascii_prefix_len(&bytes),
                        at,
                        "len {len}, {high:#x} from {at}"
                    );
                }
            }
        }
    }

    #[test]
    fn non_ascii_segments_skip_long_ascii_runs_only() {
        let a = |n| "a".repeat(n);
        let cases = [
            (String::new(), vec![]),
            (a(40), vec![]),
            ("é".to_string(), vec![(0, 2)]),
            ("éa".to_string(), vec![(0, 3)]),
            ("aé".to_string(), vec![(1, 3)]),
            ("Привет мир".to_string(), vec![(0, 19)]),
            (format!("é{}é", a(16)), vec![(0, 20)]),
            (format!("é{}é", a(31)), vec![(0, 16), (33, 35)]),
        ];
        for (text, expected) in cases {
            let got: Vec<_> = non_ascii_segments_iter(text.as_bytes())
                .map(|range| (range.start, range.end))
                .collect();
            assert_eq!(got, expected, "{text:?}");
        }
    }

    #[test]
    fn non_ascii_segments_cover_every_non_ascii_byte() {
        let pieces = ["a", " ", &"x".repeat(40), "é", "Ж", "中", "😀"];
        for text in random_texts(4, &pieces, 2000, 200) {
            let bytes = text.as_bytes();
            let mut previous_end = 0;
            for segment in non_ascii_segments_iter(bytes) {
                assert!(previous_end <= segment.start, "{text:?}");
                assert!(bytes[previous_end..segment.start].is_ascii(), "{text:?}");
                assert!(bytes[segment.start] >= 0x80, "{text:?}");
                assert!(text.is_char_boundary(segment.end), "{text:?}");
                previous_end = segment.end;
            }
            assert!(bytes[previous_end..].is_ascii(), "{text:?}");
        }
    }

    #[test]
    fn unstable_spans_are_the_runs_of_flagged_chars_in_non_ascii_segments() {
        let uppercase = CharacterCache::new(char::is_uppercase);
        let a = |n| "a".repeat(n);
        let cases = [
            (String::new(), vec![]),
            ("aÉÉb Жж".to_string(), vec![(1, 5), (7, 9)]),
            ("ÉЖ".to_string(), vec![(0, 4)]),
            ("éÉ".to_string(), vec![(2, 4)]),
            ("éA".to_string(), vec![(2, 3)]),
            (format!("{}é", "A".repeat(40)), vec![]),
            (format!("É{}Ж", a(40)), vec![(0, 2), (42, 44)]),
        ];
        for (text, expected) in cases {
            let got: Vec<_> = unstable_spans_iter(&text, &uppercase, false)
                .map(|span| (span.start, span.end))
                .collect();
            assert_eq!(got, expected, "{text:?}");
        }
    }

    #[test]
    fn unstable_spans_cover_every_flagged_non_ascii_char() {
        let uppercase = CharacterCache::new(char::is_uppercase);
        let pieces = [
            "a",
            "A",
            " ",
            &"x".repeat(40),
            "é",
            "É",
            "ж",
            "Ж",
            "中",
            "Σ",
        ];
        for text in random_texts(5, &pieces, 2000, 200) {
            let spans: Vec<_> = unstable_spans_iter(&text, &uppercase, false).collect();
            let mut previous_end = 0;
            for span in &spans {
                assert!(
                    previous_end <= span.start && span.start < span.end,
                    "{text:?}"
                );
                assert!(
                    text[span.clone()].chars().all(char::is_uppercase),
                    "{text:?}"
                );
                previous_end = span.end;
            }
            for (i, c) in text.char_indices() {
                if !c.is_ascii() && c.is_uppercase() {
                    assert!(
                        spans.iter().any(|span| span.contains(&i)),
                        "{c:?} at {i} in {text:?}"
                    );
                }
            }
        }
    }
}
