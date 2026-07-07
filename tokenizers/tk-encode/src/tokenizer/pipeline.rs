use std::convert::TryInto;
use std::ops::Range;
use std::{borrow::Cow, convert::TryFrom};

use crate::vocab::bucket_added_vocabulary::{
    AddedToken as BucketAddedToken, AddedVocabulary as BucketAddedVocabulary,
};
use crate::models::bpe::BPE;
use crate::models::unigram::Unigram;
use crate::models::wordlevel::WordLevel;
use crate::models::wordpiece::WordPiece;
use crate::{
    normalizers::NormalizerWrapper,
    pre_tokenizers::{
        bert::BertPreTokenizer,
        delimiter::CharDelimiterSplit,
        digits::Digits,
        fixed_length::FixedLength,
        punctuation::Punctuation,
        sequence::Sequence,
        split::Split as SplitPretok,
        unicode_scripts::UnicodeScripts,
        whitespace::{Whitespace, WhitespaceSplit},
    },
    Model as LegacyModelTrait, ModelWrapper, PostProcessorWrapper, PreTokenizerWrapper, Token,
    Tokenizer,
};

use super::{Result, SplitDelimiterBehavior};

/// A pre-token split, a range into the input text.
#[derive(Copy, Clone)]
pub struct Split {
    pub start: u32,
    pub end: u32,
}

impl Split {
    #[inline]
    pub fn range(self) -> Range<usize> {
        self.start as usize..self.end as usize
    }
}

pub trait Normalizer {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>>;
}

/// Range-based pre-tokenization: yields spans into the input rather than owned
/// substrings, so the pipeline can pre-tokenize without allocating.
pub trait PreTokenizer {
    /// Split `text` into pre-tokens, appending to `out`. Ranges are into `text`.
    fn pre_tokenize(&self, text: &str, out: &mut Vec<Split>) -> Result<()>;
}

/// The pre-tokenizers a [`PipelineTokenizer`] can run.
#[allow(clippy::large_enum_variant)]
pub enum PipelinePreTokenizer {
    Bert(BertPreTokenizer),
    Delimiter(CharDelimiterSplit),
    Digits(Digits),
    FixedLength(FixedLength),
    Punctuation(Punctuation),
    Sequence(Sequence),
    Split(SplitPretok),
    UnicodeScripts(UnicodeScripts),
    Whitespace(Whitespace),
    WhitespaceSplit(WhitespaceSplit),
    None,
}

impl PreTokenizer for PipelinePreTokenizer {
    fn pre_tokenize(&self, text: &str, out: &mut Vec<Split>) -> Result<()> {
        match self {
            Self::None => Ok(()),
            Self::Bert(pretok) => pretok.pre_tokenize(text, out),
            Self::Delimiter(pretok) => pretok.pre_tokenize(text, out),
            Self::Digits(pretok) => pretok.pre_tokenize(text, out),
            Self::FixedLength(pretok) => pretok.pre_tokenize(text, out),
            Self::Punctuation(pretok) => pretok.pre_tokenize(text, out),
            Self::Sequence(pretok) => pretok.pre_tokenize(text, out),
            Self::Split(pretok) => pretok.pre_tokenize(text, out),
            Self::UnicodeScripts(pretok) => pretok.pre_tokenize(text, out),
            Self::Whitespace(pretok) => pretok.pre_tokenize(text, out),
            Self::WhitespaceSplit(pretok) => pretok.pre_tokenize(text, out),
        }
    }
}

/// An output token. Carries only the vocabulary `id` — offsets and the token
/// string are dropped, which is all an encode-only caller needs.
#[derive(Debug, Clone, Copy)]
pub struct PipelineToken {
    pub id: u32,
}

impl From<Token> for PipelineToken {
    fn from(value: Token) -> Self {
        Self { id: value.id }
    }
}

/// Finds special/added tokens in a text segment so the pipeline can carve them
/// out before running the model.
pub trait PipelinePatternMatcher {
    /// Return the first special token in `input` as `Some(((start, end), id))`, where
    /// `start..end` is its byte range. `normalized` selects whether to match the
    /// tokens declared on normalized or on raw text.
    /// Returns `None` if there is no special tokens in input.
    fn extract_next(
        &self,
        full_input: &[u8],
        search_offset: usize,
        normalized: bool,
    ) -> Option<((usize, usize), u32)>;
}

/// A piece of the input produced by [`SpecialSegmentIterator`].
pub enum Segment<'a> {
    /// Ordinary text still to be (optonally normalized), pre-tokenized and run through the model.
    Text(&'a str),
    /// A matched special token, identified by its vocabulary id.
    SpecialToken(u32),
}

/// Splits `input` into [`Segment`]s, in order: runs of ordinary text
/// ([`Segment::Text`]) interleaved with the special tokens
/// ([`Segment::SpecialToken`]) matched by the [`PipelinePatternMatcher`].
///
/// ```ignore
/// for segment in SpecialSegmentIterator::new(input, pattern_matcher, false) {
///     match segment {
///         Segment::SpecialToken(id) => { /* emit the special token */ }
///         Segment::Text(chunk) => { /* tokenize this chunk */ }
///     }
/// }
/// ```
pub struct SpecialSegmentIterator<'a, 'b, PatternMatcher: PipelinePatternMatcher> {
    /// The chunk of text from which we want to extract special tokens
    input: &'a str,
    /// Implementor of [`PipelinePatternMatcher`] - the engine to match special tokens
    pattern_matcher: &'b PatternMatcher,
    /// Whether the input is normalized
    normalized: bool,
    offset: usize,
    pending: Option<u32>,
}

impl<'a, 'b, PatternMatcher: PipelinePatternMatcher>
    SpecialSegmentIterator<'a, 'b, PatternMatcher>
{
    /// Create a new iterator over [`Segment`] of the [`input`].
    /// This iterator will yield [`Segment`] in order.
    pub(crate) fn new(
        input: &'a str,
        pattern_matcher: &'b PatternMatcher,
        normalized: bool,
    ) -> Self {
        Self {
            input,
            pattern_matcher,
            normalized,
            pending: None,
            offset: 0,
        }
    }
}

impl<'a, 'b, PatternMatcher: PipelinePatternMatcher> Iterator
    for SpecialSegmentIterator<'a, 'b, PatternMatcher>
{
    type Item = Segment<'a>;

    /// Get the next segment of the input.
    fn next(&mut self) -> Option<Self::Item> {
        // take resets the pending option to None
        if let Some(special_token) = self.pending.take() {
            return Some(Segment::SpecialToken(special_token));
        }

        let remaining_input = &self.input[self.offset..];
        if remaining_input.is_empty() {
            // We've processed all the input string, return
            return None;
        }
        if let Some(((start, end), token)) =
            self.pattern_matcher
                .extract_next(self.input.as_bytes(), self.offset, self.normalized)
        {
            // `extract_next` positions are absolute in `input`, not relative to `offset`.
            let before_token = &self.input[self.offset..start];
            self.offset = end;
            if !before_token.is_empty() {
                // The iterator returns segments in order: we need to return the chunk of text and then the special token.
                // Store the special token to return in the next call and return a [`Segment::Text`]
                self.pending = Some(token);
                return Some(Segment::Text(before_token));
            } else {
                return Some(Segment::SpecialToken(token));
            }
        }
        self.offset = self.input.len();
        Some(Segment::Text(remaining_input))
    }
}

/// Experimental encode-only pipeline built from a [`Tokenizer`]. Runs the same
/// stages (special-token split → normalize → pre-tokenize → model) over borrowed
/// ranges to avoid the reference path's allocations.
pub struct PipelineTokenizer {
    added_vocabulary: BucketAddedVocabulary,
    normalizer: Option<NormalizerWrapper>,
    pre_tokenizer: PipelinePreTokenizer,
    model: PipelineModel,
    _post_processor: Option<PostProcessorWrapper>,
}

impl TryFrom<&Tokenizer> for PipelineTokenizer {
    type Error = super::Error;

    /// Build a pipeline from an existing [`Tokenizer`], cloning its components.
    ///
    /// The base [`Tokenizer`] carries the legacy [`crate::AddedVocabulary`]; the pipeline uses the
    /// fast bucket [`BucketAddedVocabulary`], so we rebuild it from the tokenizer's added tokens.
    /// Adding them in id order preserves ids (tokens present in the model reuse their model id, the
    /// rest keep their dense order), so the pipeline emits the same ids as the reference tokenizer.
    fn try_from(tok: &Tokenizer) -> Result<Self> {
        let pre_tokenizer = match tok.get_pre_tokenizer() {
            None => PipelinePreTokenizer::None,
            Some(PreTokenizerWrapper::BertPreTokenizer(p)) => PipelinePreTokenizer::Bert(*p),
            Some(PreTokenizerWrapper::Delimiter(p)) => PipelinePreTokenizer::Delimiter(*p),
            Some(PreTokenizerWrapper::Digits(p)) => PipelinePreTokenizer::Digits(p.clone()),
            Some(PreTokenizerWrapper::FixedLength(p)) => PipelinePreTokenizer::FixedLength(*p),
            Some(PreTokenizerWrapper::Punctuation(p)) => PipelinePreTokenizer::Punctuation(*p),
            Some(PreTokenizerWrapper::Sequence(p)) => PipelinePreTokenizer::Sequence(p.clone()),
            Some(PreTokenizerWrapper::Split(p)) => PipelinePreTokenizer::Split(p.clone()),
            Some(PreTokenizerWrapper::UnicodeScripts(p)) => PipelinePreTokenizer::UnicodeScripts(*p),
            Some(PreTokenizerWrapper::Whitespace(p)) => PipelinePreTokenizer::Whitespace(p.clone()),
            Some(PreTokenizerWrapper::WhitespaceSplit(p)) => PipelinePreTokenizer::WhitespaceSplit(*p),
            Some(other) => {
                return Err(format!(
                    "PipelineTokenizer only supports Bert/Whitespace/None pre-tokenizers, got: {other:?}"
                )
                .into())
            }
        };

        let legacy_av = tok.get_added_vocabulary();
        let mut added_tokens: Vec<_> = legacy_av.get_added_tokens_decoder().iter().collect();
        added_tokens.sort_by_key(|(id, _)| **id);
        let mut added_vocabulary = BucketAddedVocabulary::new();
        added_vocabulary.add_tokens(
            added_tokens.into_iter().map(|(_, t)| BucketAddedToken {
                content: t.content.clone(),
                single_word: t.single_word,
                lstrip: t.lstrip,
                rstrip: t.rstrip,
                normalized: t.normalized,
                special: t.special,
            }),
            tok.get_model(),
            tok.get_normalizer(),
        )?;
        added_vocabulary.set_encode_special_tokens(legacy_av.get_encode_special_tokens());

        let model = tok.get_model().clone().try_into()?;

        Ok(Self {
            added_vocabulary,
            normalizer: tok.get_normalizer().cloned(),
            pre_tokenizer,
            model,
            _post_processor: tok.get_post_processor().cloned(),
        })
    }
}

impl PipelineTokenizer {
    /// Encode `input` into token ids.
    ///
    /// Special tokens are matched in two passes:
    ///  1. on the raw input,
    ///  2. then on each segment after normalization
    ///
    /// This way, special / added tokens declared on raw or normalized text are both caught.
    /// The remaining text is pre-tokenized and run through the model span by span.
    ///
    /// todo: wire the post-processing
    pub fn encode(&self, input: &str, _add_special_tokens: bool) -> Result<Vec<PipelineToken>> {
        let mut output: Vec<PipelineToken> = vec![];
        let mut pre_tokens: Vec<Split> = vec![];

        // First, we extract all special tokens from the non-normalized input
        for segment in SpecialSegmentIterator::new(input, &self.added_vocabulary, false) {
            match segment {
                Segment::SpecialToken(token) => {
                    output.push(PipelineToken { id: token });
                }
                Segment::Text(chunk) => {
                    let normalized: &str = if let Some(normalizer) = &self.normalizer {
                        &normalizer.normalize(chunk)?
                    } else {
                        chunk
                    };

                    // Extract special tokens from the normalized input
                    for segment in
                        SpecialSegmentIterator::new(normalized, &self.added_vocabulary, true)
                    {
                        match segment {
                            Segment::SpecialToken(token) => {
                                output.push(PipelineToken { id: token });
                            }
                            Segment::Text(normalized_chunk) => {
                                // Pre-tokenize the chunk of normalized text
                                pre_tokens.clear();
                                self.pre_tokenizer
                                    .pre_tokenize(normalized_chunk, &mut pre_tokens)?;

                                // Tokenize each chunk
                                for pre_token in pre_tokens.iter() {
                                    self.model.tokenize_bytes(
                                        &normalized_chunk[pre_token.range()].as_bytes(),
                                        &mut output,
                                    )?;
                                }
                            }
                        }
                    }
                }
            };
        }
        Ok(output)
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

/// Splits `text` into same-class groups, emitting each as a [`Split`]
/// according to its [`SplitPolicy`].
///
/// `classify` maps each char to a small `Copy + Eq` class, the current
/// split ends whenever the class changes (or on every char of an `Isolate`
/// class), and `policy` decides what becomes of it. Ranges are byte offsets
/// into `text`.
#[inline(always)]
pub fn split<C: Copy + PartialEq>(
    text: &str,
    out: &mut Vec<Split>,
    classify: impl Fn(char) -> C,
    policy: impl Fn(C) -> SplitPolicy,
) {
    let mut start: u32 = 0;
    let mut prev: Option<C> = None;

    for (i, ch) in text.char_indices() {
        let c = classify(ch);
        if let Some(p) = prev {
            if p != c || policy(c) == SplitPolicy::Isolate {
                if policy(p) != SplitPolicy::Remove {
                    out.push(Split {
                        start,
                        end: i as u32,
                    });
                }
                start = i as u32;
            }
        }
        prev = Some(c);
    }

    if let Some(p) = prev {
        if policy(p) != SplitPolicy::Remove {
            out.push(Split {
                start,
                end: text.len() as u32,
            });
        }
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
    out: &mut Vec<Split>,
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
                    out.push(Split { start, end });
                    start = end;
                }
            }
            if (start as usize) < text.len() {
                out.push(Split {
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
                        out.push(Split { start, end: i });
                    }
                    start = i;
                }
            }
            if (start as usize) < text.len() {
                out.push(Split {
                    start,
                    end: text.len() as u32,
                });
            }
            return;
        }
    };

    split(text, out, is_delim, |d| {
        if d {
            delim_policy
        } else {
            SplitPolicy::Keep
        }
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
    out: &mut Vec<Split>,
    matches: Vec<((usize, usize), bool)>,
    behavior: SplitDelimiterBehavior,
) {
    use SplitDelimiterBehavior::*;

    // (offsets, should_remove) — mirrors `NormalizedString::split`.
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
            out.push(Split {
                start: start as u32,
                end: end as u32,
            });
        }
    }
}

pub trait Model {
    fn tokenize_bytes(&self, bytes: &[u8], output: &mut Vec<PipelineToken>) -> Result<()>;
}

pub enum PipelineModel {
    BPE(BPE),
    Unigram(Unigram),
    WordLevel(WordLevel),
    WordPiece(WordPiece),
}

impl Model for PipelineModel {
    fn tokenize_bytes(&self, bytes: &[u8], output: &mut Vec<PipelineToken>) -> Result<()> {
        let sequence = str::from_utf8(bytes)?;
        let tokens = match self {
            Self::BPE(model) => model.tokenize(sequence),
            Self::Unigram(model) => model.tokenize(sequence),
            Self::WordLevel(model) => model.tokenize(sequence),
            Self::WordPiece(model) => model.tokenize(sequence),
        }?;
        output.extend(tokens.iter().map(|&Token { id, .. }| PipelineToken { id }));
        Ok(())
    }
}

impl TryFrom<ModelWrapper> for PipelineModel {
    type Error = crate::Error;

    fn try_from(value: ModelWrapper) -> std::prelude::v1::Result<Self, Self::Error> {
        Ok(match value {
            ModelWrapper::BPE(model) => Self::BPE(model),
            ModelWrapper::Unigram(model) => Self::Unigram(model),
            ModelWrapper::WordLevel(model) => Self::WordLevel(model),
            ModelWrapper::WordPiece(model) => Self::WordPiece(model),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedMatcher(Vec<((usize, usize), u32)>);
    impl PipelinePatternMatcher for FixedMatcher {
        fn extract_next(
            &self,
            _bytes: &[u8],
            search_offset: usize,
            _normalized: bool,
        ) -> Option<((usize, usize), u32)> {
            self.0
                .iter()
                .find(|((start, _), _)| *start >= search_offset)
                .copied()
        }
    }

    #[test]
    fn segment_iterator_yields_text_and_specials_in_order() {
        let input = "aa<s>bb<s>cc";
        let matcher = FixedMatcher(vec![((2, 5), 0), ((7, 10), 1)]);

        let segments: Vec<_> = SpecialSegmentIterator::new(input, &matcher, false)
            .map(|segment| match segment {
                Segment::Text(text) => (Some(text), None),
                Segment::SpecialToken(id) => (None, Some(id)),
            })
            .collect();

        assert_eq!(
            segments,
            vec![
                (Some("aa"), None),
                (None, Some(0)),
                (Some("bb"), None),
                (None, Some(1)),
                (Some("cc"), None),
            ]
        );
    }
}
