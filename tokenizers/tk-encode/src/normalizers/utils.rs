use std::borrow::Cow;
use std::sync::OnceLock;

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
        if input.chars().all(lowercases_to_self) {
            Ok(input.into())
        } else {
            Ok(Cow::Owned(
                input.chars().flat_map(|c| c.to_lowercase()).collect(),
            ))
        }
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
}
