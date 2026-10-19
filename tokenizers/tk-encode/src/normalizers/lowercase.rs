use std::{borrow::Cow, sync::LazyLock};

use crate::{
    Result,
    normalizers::utils::{CharacterCache, Rewriter, unstable_spans_iter},
    pipeline,
};

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

        for unstable_span in unstable_spans_iter(input, &LOWERCASE_CACHE, false) {
            let unstable_text = &input[unstable_span.clone()];
            rewriter
                .rewrite(unstable_span)
                .extend(unstable_text.chars().flat_map(char::to_lowercase));
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
    use crate::normalizers::utils::random_texts;

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
