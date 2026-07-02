use std::convert::TryFrom;
use std::ops::Range;

use crate::{
    pre_tokenizers::{bert::BertPreTokenizer, whitespace::Whitespace},
    AddedVocabulary, Model, ModelWrapper, NormalizedString, Normalizer, NormalizerWrapper,
    PostProcessorWrapper, PreTokenizerWrapper, Token, Tokenizer,
};

use super::Result;

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

/// Range-based pre-tokenization: yields spans into the input rather than owned
/// substrings, so the pipeline can pre-tokenize without allocating.
pub trait PreTokenizer {
    /// Split `text` into pre-tokens, appending to `out`. Ranges are into `text`.
    fn pre_tokenize(&self, text: &str, out: &mut Vec<Split>) -> Result<()>;
}

/// The pre-tokenizers a [`PipelineTokenizer`] can run.
pub enum PipelinePreTokenizer {
    Bert(BertPreTokenizer),
    Whitespace(Whitespace),
    None,
}

impl PreTokenizer for PipelinePreTokenizer {
    fn pre_tokenize(&self, text: &str, out: &mut Vec<Split>) -> Result<()> {
        match self {
            Self::None => Ok(()),
            Self::Bert(pretok) => pretok.pre_tokenize(text, out),
            Self::Whitespace(pretok) => pretok.pre_tokenize(text, out),
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
    fn new(input: &'a str, pattern_matcher: &'b PatternMatcher, normalized: bool) -> Self {
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
            let before_token = &self.input[self.offset..self.offset + start];
            if !before_token.is_empty() {
                // The iterator returns segments in order: we need to return the chunk of text and then the special token.
                // Store the special token to return in the next call and return a [`Segment::Text`]
                self.pending = Some(token);
                self.offset += end;
                return Some(Segment::Text(before_token));
            } else {
                self.offset += end;
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
    added_vocabulary: AddedVocabulary,
    normalizer: Option<NormalizerWrapper>,
    pre_tokenizer: PipelinePreTokenizer,
    model: ModelWrapper,
    _post_processor: Option<PostProcessorWrapper>,
}

impl TryFrom<&Tokenizer> for PipelineTokenizer {
    type Error = super::Error;

    /// Build a pipeline from an existing [`Tokenizer`], cloning its components.
    fn try_from(tok: &Tokenizer) -> Result<Self> {
        let pre_tokenizer = match tok.get_pre_tokenizer() {
            None => PipelinePreTokenizer::None,
            Some(PreTokenizerWrapper::BertPreTokenizer(p)) => PipelinePreTokenizer::Bert(*p),
            Some(PreTokenizerWrapper::Whitespace(p)) => PipelinePreTokenizer::Whitespace(p.clone()),
            Some(other) => {
                return Err(format!(
                    "PipelineTokenizer only supports Bert/Whitespace/None pre-tokenizers, got: {other:?}"
                )
                .into())
            }
        };

        Ok(Self {
            added_vocabulary: tok.get_added_vocabulary().clone(),
            normalizer: tok.get_normalizer().cloned(),
            pre_tokenizer,
            model: tok.get_model().clone(),
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
                    // Normalize the text segment
                    let mut normalized: NormalizedString = chunk.into();
                    if let Some(normalizer) = &self.normalizer {
                        normalizer.normalize(&mut normalized)?;
                    }

                    // Extract special tokens from the normalized input
                    for segment in
                        SpecialSegmentIterator::new(normalized.get(), &self.added_vocabulary, true)
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
                                    output.extend(
                                        self.model
                                            .tokenize(&normalized_chunk[pre_token.range()])?
                                            .into_iter()
                                            .map(|Token { id, .. }| PipelineToken { id }),
                                    );
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
