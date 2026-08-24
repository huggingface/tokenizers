//! The pre-tokenizer half of the pipeline: the [`PreTokenizer`] trait, its scratch
//! buffers, the [`PipelinePreTokenizer`] enum, and the shared `split*` primitives
//! the concrete pre-tokenizers are built from.

use crate::pre_tokenizers::{
    bert::BertPreTokenizer,
    delimiter::CharDelimiterSplit,
    digits::Digits,
    fixed_length::FixedLength,
    punctuation::Punctuation,
    sequence::PipelineSequence,
    split::Split as SplitPretok,
    whitespace::{Whitespace, WhitespaceSplit},
};
use crate::tokenizer::{Result, SplitDelimiterBehavior};
use atomsplit::classify::classify;
use atomsplit::fsm::Span;

/// Range-based pre-tokenization: yields spans into the input rather than owned
/// substrings, so the pipeline can pre-tokenize without allocating.
///
/// # Safety
///
/// [`PreTokenizer::pre_tokenize`] produces [`Span`] objects that are later consumed by [`PipelineTokenizer::encode_sequence`].
/// For performance reasons, [`PipelineTokenizer::encode_sequence`] turns every span into a `&str` with [`str::get_unchecked`].
/// As a consequence, every implementation of this trait *MUST* ensure the following invariants for each [`Span`]:
///
/// * `span.end <= text.len()`: the span is inside the text
/// * `span.start <= span.end`: the span is not reversed
/// * `text.is_char_boundary(start)` and `text.is_char_boundary(end)`: the start and end offset must be UTF-8 char boundaries
/// * The offsets are relative to the `text`.
pub unsafe trait PreTokenizer {
    /// Split `text` into pre-tokens, appending to `out`. Ranges are into `text`.
    /// `scratch` holds the working buffers, see [`PreTokenizerScratch`].
    fn pre_tokenize(
        &self,
        text: &str,
        scratch: &mut PreTokenizerScratch,
        out: &mut Vec<Span>,
    ) -> Result<()>;
}

/// The working buffers a [`PreTokenizer`] needs to split a text into pre-tokens.
#[derive(Default)]
pub struct PreTokenizerScratch {
    /// One [`atomsplit`] atom tag per input byte, what [`atomsplit::classify::classify`] writes
    /// and the FSMs read.
    tags: Vec<u8>,
    /// An intermediate buffer in which the FSM writes the Span before they get appended to the output
    spans: Vec<Span>,
    /// The two buffers a [`PipelineSequence`] reads and writes as each of its children subdivides
    /// the spans the child before it produced. Handed out by [`Self::take_pair`] rather than
    /// borrowed, so the sequence can pass the rest of the scratch to its children while it holds
    /// them.
    pub(super) pair: [Vec<Span>; 2],
}

impl PreTokenizerScratch {
    /// Tag every byte of `bytes` with its [`atomsplit`] atom class, run `fsm` over the tags, and
    /// append the spans to `out`.
    pub fn split_on_tags(
        &mut self,
        bytes: &[u8],
        fsm: impl FnOnce(&[u8], &[u8], &mut [Span]) -> usize,
        out: &mut Vec<Span>,
    ) {
        let n = bytes.len();
        let Self { tags, spans, .. } = self;
        if tags.len() < n {
            tags.resize(n, 0);
        }
        if spans.len() < n + 1 {
            spans.resize(n + 1, Span::default());
        }
        // Assign a tag to each byte
        classify(bytes, &mut tags[..n]);
        // Run the fsm on the tags to determine where to cut
        let k = fsm(bytes, &tags[..n], &mut spans[..n + 1]);
        // Copy spans to output
        out.extend_from_slice(&spans[..k]);
    }

    /// Run `fsm` over `bytes` and append the spans it cut to `out`, for an FSM that scans the bytes
    /// itself and so has no use for the atom tags. Skips the [`classify`]
    pub fn split_on_bytes(
        &mut self,
        bytes: &[u8],
        fsm: impl FnOnce(&[u8], &mut [Span]) -> usize,
        out: &mut Vec<Span>,
    ) {
        let n = bytes.len();
        let spans = &mut self.spans;
        if spans.len() < n + 1 {
            spans.resize(n + 1, Span::default());
        }
        let k = fsm(bytes, &mut spans[..n + 1]);
        out.extend_from_slice(&spans[..k]);
    }

    /// Take the sequence's two buffers, leaving empty ones behind.
    pub fn take_pair(&mut self) -> [Vec<Span>; 2] {
        std::mem::take(&mut self.pair)
    }

    /// Give back what [`Self::take_pair`] handed out, so the next sequence reuses the allocations.
    pub fn put_pair(&mut self, pair: [Vec<Span>; 2]) {
        self.pair = pair;
    }
}

/// The pre-tokenizers a [`PipelineTokenizer`] can run.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum PipelinePreTokenizer {
    Bert(BertPreTokenizer),
    Delimiter(CharDelimiterSplit),
    Digits(Digits),
    FixedLength(FixedLength),
    Punctuation(Punctuation),
    Sequence(PipelineSequence),
    Split(SplitPretok),
    #[cfg(feature = "unicode-scripts")]
    UnicodeScripts(crate::pre_tokenizers::unicode_scripts::UnicodeScripts),
    Whitespace(Whitespace),
    WhitespaceSplit(WhitespaceSplit),
    None,
}

// SAFETY: every arm but `None` forwards to another `PreTokenizer`, which upholds the contract itself.
// `None` emits the single span covering all of `text`, whose ends are `0` and `text.len()`.
unsafe impl PreTokenizer for PipelinePreTokenizer {
    fn pre_tokenize(
        &self,
        text: &str,
        scratch: &mut PreTokenizerScratch,
        out: &mut Vec<Span>,
    ) -> Result<()> {
        match self {
            Self::None => {
                out.push(Span {
                    start: 0,
                    end: text.len() as u32,
                });
                Ok(())
            }
            Self::Bert(pretok) => pretok.pre_tokenize(text, scratch, out),
            Self::Delimiter(pretok) => pretok.pre_tokenize(text, scratch, out),
            Self::Digits(pretok) => pretok.pre_tokenize(text, scratch, out),
            Self::FixedLength(pretok) => pretok.pre_tokenize(text, scratch, out),
            Self::Punctuation(pretok) => pretok.pre_tokenize(text, scratch, out),
            Self::Sequence(pretok) => pretok.pre_tokenize(text, scratch, out),
            Self::Split(pretok) => pretok.pre_tokenize(text, scratch, out),
            #[cfg(feature = "unicode-scripts")]
            Self::UnicodeScripts(pretok) => pretok.pre_tokenize(text, scratch, out),
            Self::Whitespace(pretok) => pretok.pre_tokenize(text, scratch, out),
            Self::WhitespaceSplit(pretok) => pretok.pre_tokenize(text, scratch, out),
        }
    }
}

/// What [`split`] does with each split it forms
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SplitPolicy {
    /// Drop it, emit no split
    Remove,
    /// Emit it whole as one split
    Keep,
    /// Emit each character as its own split
    Isolate,
}

/// Splits `text` into same-class groups, emitting each as a [`Span`]
/// according to its [`SplitPolicy`].
///
/// `classify` maps each char to a small `Copy + Eq` class, the current
/// split ends whenever the class changes (or on every char of an `Isolate`
/// class), and `policy` decides what becomes of it. Ranges are byte offsets
/// into `text`.
#[inline(always)]
pub fn split<C: Copy + PartialEq>(
    text: &str,
    out: &mut Vec<Span>,
    classify: impl Fn(char) -> C,
    policy: impl Fn(C) -> SplitPolicy,
) {
    let mut start: u32 = 0;
    let mut prev: Option<C> = None;

    for (i, ch) in text.char_indices() {
        let c = classify(ch);
        if let Some(p) = prev
            && (p != c || policy(c) == SplitPolicy::Isolate)
        {
            if policy(p) != SplitPolicy::Remove {
                out.push(Span {
                    start,
                    end: i as u32,
                });
            }
            start = i as u32;
        }
        prev = Some(c);
    }

    if let Some(p) = prev
        && policy(p) != SplitPolicy::Remove
    {
        out.push(Span {
            start,
            end: text.len() as u32,
        });
    }
}

/// Splits `text` around a single delimiter predicate, honoring the full
/// [`SplitDelimiterBehavior`] contract. The pipeline-side equivalent of
/// `NormalizedString::split(pattern, behavior)` for a char predicate.
///
/// The three non-merging behaviors reduce to a [`SplitPolicy`] on the delimiter
/// class and reuse [`split`]. The two merge variants are their own single pass:
/// - `MergedWithPrevious` cuts the split *after* each delimiter, so a delimiter
///   joins the run before it (`"the-final"` -> `["the-", "final"]`).
/// - `MergedWithNext` cuts *before* each delimiter, so it joins the run after it
///   (`"the-final"` -> `["the", "-final"]`).
pub fn split_delimiter(
    text: &str,
    out: &mut Vec<Span>,
    is_delim: impl Fn(char) -> bool,
    behavior: SplitDelimiterBehavior,
) {
    use SplitDelimiterBehavior::*;

    let delim_policy = match behavior {
        Removed => SplitPolicy::Remove,
        Isolated => SplitPolicy::Isolate,
        Contiguous => SplitPolicy::Keep,
        MergedWithPrevious => {
            let mut start: u32 = 0;
            for (i, ch) in text.char_indices() {
                if is_delim(ch) {
                    let end = (i + ch.len_utf8()) as u32;
                    out.push(Span { start, end });
                    start = end;
                }
            }
            if (start as usize) < text.len() {
                out.push(Span {
                    start,
                    end: text.len() as u32,
                });
            }
            return;
        }
        MergedWithNext => {
            let mut start: u32 = 0;
            for (i, ch) in text.char_indices() {
                if is_delim(ch) {
                    let i = i as u32;
                    // skip the empty span before a leading run of delimiters
                    if i > start {
                        out.push(Span { start, end: i });
                    }
                    start = i;
                }
            }
            if (start as usize) < text.len() {
                out.push(Span {
                    start,
                    end: text.len() as u32,
                });
            }
            return;
        }
    };

    split(text, out, is_delim, |d| {
        if d { delim_policy } else { SplitPolicy::Keep }
    });
}

/// Applies a [`SplitDelimiterBehavior`] to a match segmentation and appends the
/// resulting pieces to `out`.
///
/// `matches` is the `(offsets, is_match)` sequence covering the whole input,
/// so regex matches interleaved with the gaps between them (exactly what
/// `Pattern::find_matches` produces). This is the pipeline-side equivalent of
/// the fold in `NormalizedString::split`; the arms mirror it exactly. Empty and
/// removed pieces are dropped.
pub fn split_matches(
    out: &mut Vec<Span>,
    matches: Vec<((usize, usize), bool)>,
    behavior: SplitDelimiterBehavior,
) {
    use SplitDelimiterBehavior::*;

    // (offsets, should_remove), mirroring `NormalizedString::split`.
    let splits: Vec<((usize, usize), bool)> = match behavior {
        Isolated => matches.into_iter().map(|(o, _)| (o, false)).collect(),
        Removed => matches, // should_remove == is_match
        Contiguous => {
            let mut previous_match = false;
            matches
                .into_iter()
                .fold(vec![], |mut acc, (offsets, is_match)| {
                    if is_match == previous_match {
                        if let Some(((_, end), _)) = acc.last_mut() {
                            *end = offsets.1;
                        } else {
                            acc.push((offsets, false));
                        }
                    } else {
                        acc.push((offsets, false));
                    }
                    previous_match = is_match;
                    acc
                })
        }
        MergedWithPrevious => {
            let mut previous_match = false;
            matches
                .into_iter()
                .fold(vec![], |mut acc, (offsets, is_match)| {
                    if is_match && !previous_match {
                        if let Some(((_, end), _)) = acc.last_mut() {
                            *end = offsets.1;
                        } else {
                            acc.push((offsets, false));
                        }
                    } else {
                        acc.push((offsets, false));
                    }
                    previous_match = is_match;
                    acc
                })
        }
        MergedWithNext => {
            let mut previous_match = false;
            let mut splits =
                matches
                    .into_iter()
                    .rev()
                    .fold(vec![], |mut acc, (offsets, is_match)| {
                        if is_match && !previous_match {
                            if let Some(((start, _), _)) = acc.last_mut() {
                                *start = offsets.0;
                            } else {
                                acc.push((offsets, false));
                            }
                        } else {
                            acc.push((offsets, false));
                        }
                        previous_match = is_match;
                        acc
                    });
            splits.reverse();
            splits
        }
    };

    for ((start, end), should_remove) in splits {
        if !should_remove && start != end {
            out.push(Span {
                start: start as u32,
                end: end as u32,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pre_tokenizers::split::SplitPattern;
    use crate::utils::byte_level::GPT2_REGEX_STR;

    fn variant_name(pre_tokenizer: &PipelinePreTokenizer) -> &'static str {
        match pre_tokenizer {
            PipelinePreTokenizer::Bert(_) => "Bert",
            PipelinePreTokenizer::Delimiter(_) => "Delimiter",
            PipelinePreTokenizer::Digits(_) => "Digits",
            PipelinePreTokenizer::FixedLength(_) => "FixedLength",
            PipelinePreTokenizer::Punctuation(_) => "Punctuation",
            PipelinePreTokenizer::Sequence(_) => "Sequence",
            PipelinePreTokenizer::Split(_) => "Split",
            #[cfg(feature = "unicode-scripts")]
            PipelinePreTokenizer::UnicodeScripts(_) => "UnicodeScripts",
            PipelinePreTokenizer::Whitespace(_) => "Whitespace",
            PipelinePreTokenizer::WhitespaceSplit(_) => "WhitespaceSplit",
            PipelinePreTokenizer::None => "None",
        }
    }

    const HOSTILE: &[&str] = &[
        "",
        " ",
        "\r",
        "\r\n\n",
        "hello world",
        "café naïve",
        "a\u{0301}b\u{0301}\u{0301}c",
        "中文分词。ひらがな 한글",
        "😀👍🏽 👨‍👩‍👧‍👦 x",
        "\u{FEFF}leading bom",
        "مرحبا\u{200F} العربية",
        "नरेंद्र मोदी",
        "hello▁world▁▁",
        "a1b22c333d4444 42 ½²¼ Ⅷ",
        "!!!...?! a, b; c",
        "  \t  trailing   ",
        "Ⓘ\u{200D}x_y a-b'c",
    ];

    /// Every variant must produce spans that [`PipelineTokenizer::encode_sequence`] can hand to `str::get_unchecked`
    #[test]
    fn every_pre_tokenizer_emits_sliceable_spans() {
        use crate::pre_tokenizers::digits::Digits;
        use crate::pre_tokenizers::fixed_length::FixedLength;
        use crate::pre_tokenizers::punctuation::Punctuation;
        use crate::tokenizer::SplitDelimiterBehavior::*;

        let literal_split = |pattern: &str, behavior| {
            SplitPretok::new(SplitPattern::String(pattern.to_owned()), behavior, false).unwrap()
        };
        // The gpt2 regex is recognized, so this routes to `fsm_byte_level` with no regex backend.
        let gpt2_split = SplitPretok::new(
            SplitPattern::Regex(GPT2_REGEX_STR.to_owned()),
            Isolated,
            false,
        )
        .unwrap();

        // `mut` + a cfg'd push rather than a `#[cfg]` on a `vec!` element: attributes on
        // expression position are not stable.
        #[allow(unused_mut)]
        let mut cases = vec![
            PipelinePreTokenizer::Bert(BertPreTokenizer),
            PipelinePreTokenizer::Delimiter(CharDelimiterSplit::new(' ')),
            PipelinePreTokenizer::Delimiter(CharDelimiterSplit::new('▁')),
            PipelinePreTokenizer::Digits(Digits::new(true)),
            PipelinePreTokenizer::Digits(Digits::new(false)),
            PipelinePreTokenizer::FixedLength(FixedLength::new(3)),
            PipelinePreTokenizer::FixedLength(FixedLength::new(0)),
            PipelinePreTokenizer::Punctuation(Punctuation::new(Removed)),
            PipelinePreTokenizer::Punctuation(Punctuation::new(Isolated)),
            PipelinePreTokenizer::Punctuation(Punctuation::new(Contiguous)),
            PipelinePreTokenizer::Punctuation(Punctuation::new(MergedWithPrevious)),
            PipelinePreTokenizer::Punctuation(Punctuation::new(MergedWithNext)),
            PipelinePreTokenizer::Sequence(PipelineSequence::new(vec![
                PipelinePreTokenizer::WhitespaceSplit(WhitespaceSplit),
                PipelinePreTokenizer::Punctuation(Punctuation::new(Isolated)),
            ])),
            PipelinePreTokenizer::Split(gpt2_split),
            PipelinePreTokenizer::Split(literal_split("▁", MergedWithPrevious)),
            PipelinePreTokenizer::Split(literal_split("▁", MergedWithNext)),
            PipelinePreTokenizer::Split(literal_split(" ", Removed)),
            PipelinePreTokenizer::Whitespace(Whitespace),
            PipelinePreTokenizer::WhitespaceSplit(WhitespaceSplit),
            PipelinePreTokenizer::None,
        ];
        #[cfg(feature = "unicode-scripts")]
        cases.push(PipelinePreTokenizer::UnicodeScripts(
            crate::pre_tokenizers::unicode_scripts::UnicodeScripts::new(),
        ));

        let mut covered: Vec<&str> = cases.iter().map(variant_name).collect();
        covered.sort_unstable();
        covered.dedup();
        #[allow(unused_mut)]
        let mut expected: Vec<&str> = vec![
            "Bert",
            "Delimiter",
            "Digits",
            "FixedLength",
            "None",
            "Punctuation",
            "Sequence",
            "Split",
            "Whitespace",
            "WhitespaceSplit",
        ];
        #[cfg(feature = "unicode-scripts")]
        expected.push("UnicodeScripts");
        expected.sort_unstable();
        assert_eq!(
            covered, expected,
            "every PipelinePreTokenizer variant needs a case above",
        );

        let mut scratch = PreTokenizerScratch::default();
        let mut spans = Vec::new();
        for pre_tokenizer in &cases {
            for text in HOSTILE {
                spans.clear();
                pre_tokenizer
                    .pre_tokenize(text, &mut scratch, &mut spans)
                    .unwrap();
                for span in &spans {
                    let range = span.range();
                    assert!(
                        range.start <= range.end,
                        "{} reversed {span:?} on {text:?}",
                        variant_name(pre_tokenizer),
                    );
                    assert!(
                        text.is_char_boundary(range.start) && text.is_char_boundary(range.end),
                        "{} cut {span:?} off a character boundary of {text:?}",
                        variant_name(pre_tokenizer),
                    );
                    // What the pipeline does with the span. Slicing checked here is the point: it
                    // panics on exactly the ranges `get_unchecked` would turn into undefined behavior.
                    assert!(text.get(range).is_some());
                }
            }
        }
    }
}
