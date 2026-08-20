use crate::pipeline::{self, PreTokenizerScratch};
use crate::tokenizer::pattern::{Invert, Pattern};
use crate::tokenizer::{Offsets, PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior};
#[cfg(feature = "serde")]
use crate::utils::macro_rules_attribute;

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", macro_rules_attribute(impl_serde_type!))]
pub struct Whitespace;

use atomsplit::classify::mask;

impl Default for Whitespace {
    fn default() -> Self {
        Self
    }
}

/// The three character classes `\w+|[^\w\s]+` is built out of.
///
/// Naming them is the whole trick to spelling that regex out by hand: the alternation is not a
/// search, it is "one maximal run of one class at a time", and the chars the regex leaves uncovered
/// are exactly the runs of the third class.
#[derive(Copy, Clone, PartialEq, Eq)]
enum WordClass {
    /// `\w` — see [`is_word_char`], which is this crate's definition of it.
    Word,
    /// `\s`. `char::is_whitespace` is Unicode `White_Space`, which is what `regex`'s `\s` is too.
    Space,
    /// `[^\w\s]`: everything else. Punctuation, symbols, emoji.
    Symbol,
}

fn word_class(c: char) -> WordClass {
    if is_word_char(c) {
        WordClass::Word
    } else if c.is_whitespace() {
        WordClass::Space
    } else {
        WordClass::Symbol
    }
}

/// `\w+|[^\w\s]+` as a [`Pattern`], with no regex engine behind it.
///
/// This exists because `Whitespace`'s legacy `NormalizedString` path used to hold a
/// `LazyLock<regex::Regex>` and split on `Invert` of it, which made it the last non-test caller of
/// `impl Pattern for &regex::Regex` — and therefore the last reason for `regex` to be reachable from
/// this crate at all.
///
/// Nothing in that regex needs an engine. It has no captures, no backtracking, and no alternation
/// the engine has to *search*: at every position exactly one of its two branches can start, decided
/// by the class of the character sitting there. So a match is a maximal run of [`WordClass::Word`]
/// or a maximal run of [`WordClass::Symbol`], and the gaps the regex leaves uncovered are the
/// maximal runs of [`WordClass::Space`]. One pass over `char_indices`, emitting a run whenever the
/// class changes, reproduces `find_matches` for it exactly — including the two things that are easy
/// to get wrong:
///
///   * **adjacent word and symbol runs stay separate entries.** `"a!b"` is three matches, not one.
///     Merging them would lose the word↔symbol cut, which is the entire difference between this
///     pre-tokenizer and [`WhitespaceSplit`].
///   * **the empty string is one non-match, `[((0, 0), false)]`**, not an empty list. That is what
///     the `&Regex` impl returned, and [`Pattern`] requires the output to cover the whole input.
///
/// `legacy_matches_the_regex_it_replaced` in the tests below pins the equivalence against a real
/// `regex::Regex` (a dev-dependency, so the check runs in every feature rung).
struct WordAndSymbolRuns;

impl Pattern for WordAndSymbolRuns {
    fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        if inside.is_empty() {
            return Ok(vec![((0, 0), false)]);
        }

        let mut splits = Vec::with_capacity(inside.len());
        let mut run_start = 0;
        let mut run_class: Option<WordClass> = None;

        for (offset, c) in inside.char_indices() {
            let class = word_class(c);
            match run_class {
                // Still inside the same run: nothing to emit yet.
                Some(open) if open == class => continue,
                // The class changed, so the run that was open ends here. A `Space` run is what the
                // regex did *not* match; the other two are what it did.
                Some(open) => splits.push(((run_start, offset), open != WordClass::Space)),
                None => {}
            }
            run_start = offset;
            run_class = Some(class);
        }

        // `inside` is non-empty, so the loop opened a run and it reaches the end of the string.
        if let Some(open) = run_class {
            splits.push(((run_start, inside.len()), open != WordClass::Space));
        }

        Ok(splits)
    }
}

/// The legacy `NormalizedString` path. Nothing on the encode path reaches it: the pipeline
/// dispatches to the `pipeline::PreTokenizer` impl below, which does the same split on `atomsplit`'s
/// classification masks. It is kept because the umbrella crate and both bindings still expose this
/// trait.
///
/// It used to be gated on `config`, purely because the regex it split on was. [`WordAndSymbolRuns`]
/// needs no engine, so the gate is gone and the impl is available in every build — which is also how
/// `Whitespace` stops being the reason `regex` is reachable from this crate.
impl PreTokenizer for Whitespace {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        // `Invert` is kept rather than folded into `WordAndSymbolRuns`: the pattern is the *words*,
        // and what gets removed is everything between them.
        pretokenized.split(|_, normalized| {
            normalized.split(Invert(WordAndSymbolRuns), SplitDelimiterBehavior::Removed)
        })
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", macro_rules_attribute(impl_serde_type!))]
pub struct WhitespaceSplit;

impl PreTokenizer for WhitespaceSplit {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, normalized| {
            normalized.split(char::is_whitespace, SplitDelimiterBehavior::Removed)
        })
    }
}

// SAFETY: the spans come from an `atomsplit` fsm, which cuts only at character boundaries of `text`.
// See "What the spans guarantee" in the `atomsplit::fsm` docs.
unsafe impl pipeline::PreTokenizer for WhitespaceSplit {
    fn pre_tokenize(
        &self,
        text: &str,
        scratch: &mut PreTokenizerScratch,
        out: &mut Vec<pipeline::Span>,
    ) -> Result<()> {
        // drop whitespace runs, keep everything else as runs — atomsplit SIMD classify + class-runs FSM.
        // atom `WS` == `char::is_whitespace`, so byte-exact with the scalar path.
        scratch.split_on_tags(
            text.as_bytes(),
            atomsplit::fsm::class_runs_into::<{ mask::WS }, 0, 0>,
            out,
        );
        Ok(())
    }
}

/// Matches the same characters as the `\w` regex class (Unicode-aware).
/// This is: Alphabetic + Nd (decimal digit) + Pc (connector punctuation) +
/// M (marks) + Join_Control — NOT Nl/No (which Rust's is_alphanumeric includes).
pub fn is_word_char(ch: char) -> bool {
    use unicode_categories::UnicodeCategories;

    ch.is_alphabetic()
        || ch.is_number_decimal_digit()
        || ch.is_punctuation_connector()
        || ch.is_mark()
        || ch == '\u{200c}' // Zero-Width Non-Joiner
        || ch == '\u{200d}' // Zero-Width Joiner
}

// SAFETY: the spans come from an `atomsplit` fsm, which splits only at character boundaries of `text`.
// See `atomsplit::fsm` docs.
unsafe impl pipeline::PreTokenizer for Whitespace {
    #[inline(never)]
    fn pre_tokenize(
        &self,
        text: &str,
        scratch: &mut PreTokenizerScratch,
        out: &mut Vec<pipeline::Span>,
    ) -> Result<()> {
        // `\w+|[^\w\s]+`: drop whitespace, cut at the word↔symbol boundary, each run one token —
        // atomsplit classify + class-runs FSM (`WORD` = `\w`; keep-A = word, keep-B = symbol).
        scratch.split_on_tags(
            text.as_bytes(),
            atomsplit::fsm::class_runs_into::<{ mask::WS }, 0, { mask::WORD }>,
            out,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer};

    fn pretokenize(text: &str) -> Vec<(&str, (u32, u32))> {
        let pretok = Whitespace;
        let mut scratch = PreTokenizerScratch::default();
        let mut splits = Vec::new();
        crate::pipeline::PreTokenizer::pre_tokenize(&pretok, text, &mut scratch, &mut splits)
            .unwrap();
        splits
            .iter()
            .map(|s| (&text[s.range()], (s.start, s.end)))
            .collect()
    }

    #[test]
    fn basic() {
        let tests = vec![
            (
                "Hey man!",
                vec![("Hey", (0, 3)), ("man", (4, 7)), ("!", (7, 8))],
            ),
            (
                "How are you doing?",
                vec![
                    ("How", (0, 3)),
                    ("are", (4, 7)),
                    ("you", (8, 11)),
                    ("doing", (12, 17)),
                    ("?", (17, 18)),
                ],
            ),
            ("\n", vec![]),
        ];
        for (s, res) in tests {
            assert_eq!(pretokenize(s), res, "input: {s:?}");
        }
    }

    #[test]
    fn whitespace_split() {
        let tests = vec![
            ("Hey man!", vec![("Hey", (0, 3)), ("man!", (4, 8))]),
            (
                "Hey, man, Good?",
                vec![("Hey,", (0, 4)), ("man,", (5, 9)), ("Good?", (10, 15))],
            ),
        ];
        let pretok = WhitespaceSplit;
        for (s, res) in tests {
            let mut pretokenized = PreTokenizedString::from(s);
            pretok.pre_tokenize(&mut pretokenized).unwrap();
            assert_eq!(
                pretokenized
                    .get_splits(OffsetReferential::Original, OffsetType::Byte)
                    .into_iter()
                    .map(|(s, o, _)| (s, o))
                    .collect::<Vec<_>>(),
                res
            );
        }
    }

    fn pretokenize_split(text: &str) -> Vec<(&str, (u32, u32))> {
        let pretok = WhitespaceSplit;
        let mut scratch = PreTokenizerScratch::default();
        let mut splits = Vec::new();
        crate::pipeline::PreTokenizer::pre_tokenize(&pretok, text, &mut scratch, &mut splits)
            .unwrap();
        splits
            .iter()
            .map(|s| (&text[s.range()], (s.start, s.end)))
            .collect()
    }

    #[test]
    fn whitespace_split_pipeline() {
        // same cases as `whitespace_split`, via the pipeline path: only whitespace
        // splits, punctuation stays attached to its run
        assert_eq!(
            pretokenize_split("Hey man!"),
            vec![("Hey", (0, 3)), ("man!", (4, 8))],
        );
        assert_eq!(
            pretokenize_split("Hey, man, Good?"),
            vec![("Hey,", (0, 4)), ("man,", (5, 9)), ("Good?", (10, 15))],
        );
    }

    #[test]
    fn whitespace_split_pipeline_edge_cases() {
        let empty = Vec::<(&str, (u32, u32))>::new();
        assert_eq!(pretokenize_split(""), empty);
        assert_eq!(pretokenize_split("   "), empty);
        assert_eq!(pretokenize_split("a"), vec![("a", (0, 1))]);
        // leading/trailing/multiple whitespace dropped, runs kept whole
        assert_eq!(
            pretokenize_split(" a  b "),
            vec![("a", (1, 2)), ("b", (4, 5))]
        );
        assert_eq!(
            pretokenize_split("a\tb\nc"),
            vec![("a", (0, 1)), ("b", (2, 3)), ("c", (4, 5))],
        );
        // multibyte: é is 2 bytes -> byte offsets
        assert_eq!(
            pretokenize_split("café au lait"),
            vec![("café", (0, 5)), ("au", (6, 8)), ("lait", (9, 13))],
        );
    }

    #[test]
    fn edge_cases() {
        // word / symbol / whitespace transitions; whitespace dropped, splits kept whole
        let edge_cases = vec![
            ("", vec![]),
            (" ", vec![]),
            ("  ", vec![]),
            ("a", vec![("a", (0, 1))]),
            ("!", vec![("!", (0, 1))]),
            ("a!", vec![("a", (0, 1)), ("!", (1, 2))]),
            ("!a", vec![("!", (0, 1)), ("a", (1, 2))]),
            ("a b", vec![("a", (0, 1)), ("b", (2, 3))]),
            ("a  b", vec![("a", (0, 1)), ("b", (3, 4))]),
            ("a\tb", vec![("a", (0, 1)), ("b", (2, 3))]),
            ("a\nb", vec![("a", (0, 1)), ("b", (2, 3))]),
            ("a\r\nb", vec![("a", (0, 1)), ("b", (3, 4))]),
        ];

        for (input, expected) in edge_cases {
            assert_eq!(pretokenize(input), expected, "input: {input:?}");
        }
    }

    #[test]
    fn multibyte_offsets() {
        // offsets are byte offsets into the input; classification is Unicode-aware.
        // é is 2 bytes, so "café" = 0..5 and "résumé" = 6..14.
        assert_eq!(
            pretokenize("café résumé"),
            vec![("café", (0, 5)), ("résumé", (6, 14))],
        );
        // CJK ideographs are alphabetic (word chars): one split, no inner boundary.
        assert_eq!(
            pretokenize("中文 text"),
            vec![("中文", (0, 6)), ("text", (7, 11))],
        );
        // '_' is connector punctuation (a word char) -> a single word token.
        assert_eq!(pretokenize("hello_world"), vec![("hello_world", (0, 11))]);
        // word and symbol groups are each one split; only the boundary splits.
        assert_eq!(
            pretokenize("ab!!cd"),
            vec![("ab", (0, 2)), ("!!", (2, 4)), ("cd", (4, 6))],
        );
    }

    /// The corpus the two equivalence tests below run over. Deliberately heavy on the places where
    /// "is this a word character" is not obvious: the connector punctuation and the join controls
    /// that `\w` includes, and the `Nl`/`No` numerals it does *not*.
    const CORPUS: &[&str] = &[
        "",
        " ",
        "  ",
        "\t",
        "\r\n",
        "\u{a0}", // NO-BREAK SPACE: White_Space, so `\s`
        "a",
        "!",
        "a!",
        "!a",
        "a b",
        "a  b",
        "a\tb",
        "ab!!cd",
        "Hey man!",
        "How are you doing?",
        "Hey, man, Good?",
        "hello_world", // '_' is Pc (connector punctuation), so a word char
        "café résumé",
        "中文 text",
        "中 文 text",
        "野口里佳 Noguchi Rika",
        "\u{200c}\u{200d}", // ZWNJ / ZWJ: Join_Control, so word chars
        "a\u{200d}b",
        "e\u{301}", // combining acute: a Mark, so a word char
        "Ⅷ",        // Nl but Alphabetic -> a word char
        "½",        // No and not Alphabetic -> a symbol
        "3.14",
        "a1_b2",
        "🙂",
        "a🙂b",
        "  leading and trailing  ",
        "emoji🙂 and 中文, mixed_with_1 punctuation!!",
    ];

    /// `WordAndSymbolRuns` replaced a `regex::Regex` for `\w+|[^\w\s]+`, and the only thing that
    /// makes that replacement safe is that it produces the identical `find_matches` output. Assert
    /// it directly, against a real `regex` (a dev-dependency, so this runs in every feature rung).
    #[test]
    fn legacy_matches_the_regex_it_replaced() {
        use crate::tokenizer::pattern::Pattern;
        use regex::Regex;

        let re = Regex::new(r"\w+|[^\w\s]+").unwrap();
        let re_ref: &Regex = &re;

        for input in CORPUS {
            assert_eq!(
                WordAndSymbolRuns.find_matches(input).unwrap(),
                re_ref.find_matches(input).unwrap(),
                "input: {input:?}",
            );
        }
    }

    /// And the split the legacy path actually performs has to agree with the `atomsplit` FSM the
    /// pipeline runs, which is what the rest of this module's tests exercise. This is the same claim
    /// the `// SAFETY` comment on the `pipeline::PreTokenizer` impl makes, checked over the corpus.
    #[test]
    fn legacy_matches_the_pipeline() {
        for input in CORPUS {
            let mut pretokenized = PreTokenizedString::from(*input);
            Whitespace.pre_tokenize(&mut pretokenized).unwrap();
            // `get_splits` reports `usize` offsets; the pipeline's `Span` carries `u32`. Same
            // numbers, so narrow rather than widen -- a mismatch would show up as a failed
            // comparison, not as a silent truncation, at these lengths.
            let legacy: Vec<(&str, (u32, u32))> = pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, (o.0 as u32, o.1 as u32)))
                .collect();

            assert_eq!(legacy, pretokenize(input), "input: {input:?}");
        }
    }

    // TODO: add xnli test:
    // - either as an integration test
    // - either as a unit test that triggers only if the xnli file is present in the data/ dir
    // #[test]
    // fn xnli() {}
}
