use std::borrow::Cow;
use std::ops::Range;
use std::sync::{LazyLock, OnceLock};

use wide::u8x16;

use crate::pipeline;
use crate::tokenizer::Result;

/// A cache / memoization structure for unicode normalizers.
/// Organized in pages of 256 unicode characters, lazily initiated on first lookup with [`CharacterCache::classifier`].
///
/// For unicode normalizers, the `classifier` is an expensive binary search in unicode class tables.
/// This structure allows to amortize this cost by computing this once lazily and replacing the expensive binary search with a simple lookup.
///
/// Unicode locality properties help here:
/// - Characters of a given script (eg latin, cyrillic, hangul, etc) are often contiguous in codepoint space
/// - Scripts are also often local in a given text, meaning there's a good chance most characters of a given text chunk land in the same cache page
pub(crate) struct CharacterCache {
    pages: Box<[OnceLock<Box<[bool; 256]>>]>,
    classifier: Box<dyn Fn(char) -> bool + Send + Sync>,
}

impl CharacterCache {
    /// Number of codepoints to store per cache page.
    const CHARS_PER_PAGE: usize = 256;

    /// Number of cache pages required to cache all of unicode codepoints (0x1100)
    const NUM_PAGES: usize = (char::MAX as usize + 1) / Self::CHARS_PER_PAGE;

    pub(crate) fn new(classifier: impl Fn(char) -> bool + Send + Sync + 'static) -> Self {
        Self {
            pages: (0..Self::NUM_PAGES).map(|_| OnceLock::new()).collect(),
            classifier: Box::new(classifier),
        }
    }

    fn init_page(&self, page_index: u32) -> [bool; Self::CHARS_PER_PAGE] {
        std::array::from_fn(|idx| {
            char::from_u32(page_index * Self::CHARS_PER_PAGE as u32 + idx as u32)
                .is_some_and(self.classifier.as_ref())
        })
    }

    #[inline]
    pub(crate) fn get(&self, c: char) -> bool {
        let page_index = c as u32 / Self::CHARS_PER_PAGE as u32;
        let index = c as usize % Self::CHARS_PER_PAGE;

        let page =
            self.pages[page_index as usize].get_or_init(|| Box::new(self.init_page(page_index)));
        page[index]
    }
}

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

/// Builds a normalizer's output when only parts of the input must change.
/// Inputs that don't require changes are returned as Cow::Borrowed
pub(crate) struct Rewriter<'a> {
    input: &'a str,
    cursor: usize,
    output: Option<String>,
}

impl<'a> Rewriter<'a> {
    pub(crate) fn new(input: &'a str) -> Self {
        Self {
            input,
            cursor: 0,
            output: None,
        }
    }

    /// Copies the input bytes up to range, then replaces the range by calling `with`.
    /// Ranges must come in ascending order and must not overlap.
    pub(crate) fn replace_range_with(
        &mut self,
        range: Range<usize>,
        with: impl FnOnce(&mut String),
    ) {
        debug_assert!(self.cursor <= range.start && range.start <= range.end);
        let out = self
            .output
            .get_or_insert_with(|| String::with_capacity(self.input.len()));
        out.push_str(&self.input[self.cursor..range.start]);
        with(out);
        self.cursor = range.end;
    }

    pub(crate) fn finish(self) -> Cow<'a, str> {
        if let Some(mut owned) = self.output {
            owned.push_str(&self.input[self.cursor..]);
            Cow::Owned(owned)
        } else {
            Cow::Borrowed(self.input)
        }
    }
}

static LOWERCASE_CACHE: LazyLock<CharacterCache> =
    LazyLock::new(|| CharacterCache::new(|c| !lowercases_to_self(c)));

/// Lowercases the input
#[derive(Copy, Clone, Debug)]
pub struct Lowercase;

/// Whether lowercasing `c` leaves it unchanged (a single, identical char)
pub(crate) fn lowercases_to_self(c: char) -> bool {
    let mut it = c.to_lowercase();
    matches!((it.next(), it.next()), (Some(first), None) if first == c)
}

impl pipeline::Normalizer for Lowercase {
    fn normalize<'a>(&self, input: &'a str, _offset: usize) -> Result<Cow<'a, str>> {
        let mut rewriter = Rewriter::new(input);
        let bytes = input.as_bytes();

        // First pass: normalize text chunks that contains non-ascii characters
        let mut cursor = ascii_prefix_len(bytes);
        while cursor < bytes.len() {
            let end = cursor + next_ascii_chunk(&bytes[cursor..]);
            let non_ascii_segment = &input[cursor..end];

            let mut run_start = None;
            for (i, c) in non_ascii_segment.char_indices() {
                if LOWERCASE_CACHE.get(c) {
                    run_start.get_or_insert(i);
                } else if let Some(start) = run_start.take() {
                    rewriter.replace_range_with(cursor + start..cursor + i, |out| {
                        out.extend(
                            non_ascii_segment[start..i]
                                .chars()
                                .flat_map(char::to_lowercase),
                        )
                    });
                }
            }
            if let Some(start) = run_start {
                rewriter.replace_range_with(cursor + start..end, |out| {
                    out.extend(
                        non_ascii_segment[start..]
                            .chars()
                            .flat_map(char::to_lowercase),
                    )
                });
            }
            cursor = end + ascii_prefix_len(&bytes[end..]);
        }
        let mut result = rewriter.finish();

        // Second pass: use make_ascii_lowercase to lowercase ascii characters
        if result.bytes().any(|b| b.is_ascii_uppercase()) {
            result.to_mut().make_ascii_lowercase();
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn character_cache_matches_its_classifier() {
        let cache = CharacterCache::new(char::is_alphabetic);
        for c in (0..=0x10FFFFu32).filter_map(char::from_u32) {
            assert_eq!(cache.get(c), c.is_alphabetic(), "{c:?}");
        }
    }

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
    fn rewriter_borrows_without_replace() {
        for input in ["The lazy dog jumps over", ""] {
            assert!(
                matches!(Rewriter::new(input).finish(), Cow::Borrowed(_)),
                "{input:?}"
            );
        }
    }

    #[test]
    fn rewriter_owns_after_a_replace() {
        let mut rewriter = Rewriter::new("The lazy dog jumps over");
        rewriter.replace_range_with(0.."The lazy".len(), |out| out.push_str("catzz"));
        assert!(matches!(rewriter.finish(), Cow::Owned(s) if s == "catzz dog jumps over"));
    }

    #[test]
    fn rewriter_splices_replacements_into_the_input() {
        let cases: [(&str, &[(Range<usize>, &str)], &str); 6] = [
            ("hello world", &[(0..5, "HI")], "HI world"),
            ("hello world", &[(6..11, "")], "hello "),
            ("hello world", &[(0..11, "x")], "x"),
            ("hello world", &[(0..1, "HH"), (6..11, "w")], "HHello w"),
            (
                "hello world",
                &[(0..5, "a"), (5..6, "b"), (6..6, "c")],
                "abcworld",
            ),
            ("café 中文", &[(3..5, "e"), (6..9, "")], "cafe 文"),
        ];
        for (input, replacements, expected) in cases {
            let mut rewriter = Rewriter::new(input);
            for (range, text) in replacements {
                rewriter.replace_range_with(range.clone(), |out| out.push_str(text));
            }
            assert_eq!(rewriter.finish(), expected, "{input:?} {replacements:?}");
        }
    }

    /// Expected values were captured from the legacy `NormalizedString` normalizer this test used
    /// to compare against, on the commit that removed it -- so they still pin exactly what the two
    /// implementations agreed on, without keeping the legacy code alive to ask.
    #[test]
    fn pipeline_lowercase() {
        let n = Lowercase;
        for (input, expected) in [
            ("HELLO", "hello"),
            ("Hello World", "hello world"),
            ("abc", "abc"),
            ("", ""),
            ("ÀÉ", "àé"),
            ("ΟΔΟΣ", "οδοσ"),
        ] {
            assert_eq!(
                &*pipeline::Normalizer::normalize(&n, input, 0).unwrap(),
                expected,
                "input={input:?}"
            );
        }
    }

    /// `count` texts of up to `max_len` bytes, each a random concatenation of `pieces`.
    fn random_texts(seed: u64, pieces: &[&str], count: usize, max_len: usize) -> Vec<String> {
        let mut state = seed;
        let mut next = |bound: usize| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as usize % bound
        };
        (0..count)
            .map(|_| {
                let len = next(max_len);
                let mut text = String::new();
                while text.len() < len {
                    text.push_str(pieces[next(pieces.len())]);
                }
                text
            })
            .collect()
    }

    #[test]
    fn lowercase_matches_the_char_chain() {
        let pieces = [
            "A",
            "z",
            " ",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "abcdefghijklmnopqrstuvwxyz",
            "\u{c9}",
            "\u{e9}",
            "\u{130}",
            "\u{1c5}",
            "\u{212a}",
            "\u{3a3}",
            "\u{416}",
            "\u{436}",
            "\u{4e2d}",
        ];
        for text in random_texts(1, &pieces, 2000, 90) {
            let expected: String = text.chars().flat_map(char::to_lowercase).collect();
            let got = pipeline::Normalizer::normalize(&Lowercase, &text, 0).unwrap();
            assert_eq!(got, expected, "{text:?}");
        }
    }

    #[test]
    fn lowercase_borrows_text_that_is_already_lowercase() {
        for text in [
            "",
            "the lazy dog jumps over the lazy dog",
            "\u{43f}\u{440}\u{438}\u{432}\u{435}\u{442} \u{43c}\u{438}\u{440}",
            "caf\u{e9} \u{4e2d}\u{6587} ok",
        ] {
            let got = pipeline::Normalizer::normalize(&Lowercase, text, 0).unwrap();
            assert!(matches!(got, Cow::Borrowed(_)), "{text:?}");
        }
    }
}
