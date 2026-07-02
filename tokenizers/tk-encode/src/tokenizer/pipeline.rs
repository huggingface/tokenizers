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

pub trait PreTokenizer {
    /// Split `text` into pre-tokens, appending to `out`. Ranges are into `text`.
    fn pre_tokenize(&self, text: &str, out: &mut Vec<Split>) -> Result<()>;
}

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

#[derive(Debug, Clone, Copy)]
pub struct PipelineToken {
    pub id: u32,
}

impl From<Token> for PipelineToken {
    fn from(value: Token) -> Self {
        Self { id: value.id }
    }
}

pub trait PipelinePatternMatcher {
    fn get_next_special_token(
        &self,
        input: &str,
        normalized: bool,
    ) -> Option<((usize, usize), u32)>;
}

pub enum Segment<'a> {
    Text(&'a str),
    SpecialToken(u32),
}

/// This iterator wraps the logic to extract special tokens from the input text
/// It yields [`Segment`] of the input text, in order
///
/// [`Segment`] can either be a chunk of text ([`Segment::Text`]) that needs to be tokenized by the
/// model, or a [`Segment::SpecialToken`] holding the id (u32) of the special token
/// 
/// Usage:
/// 
/// ```rust
/// let mut segment_iterator = SpecialSegmentIterator::new(input, pattern_matcher, false);
/// 
/// for segment in segment_iterator {
///     match segment {
///         Segment::SpecialToken(token_id) => println!("Segment is a special token!"),
///         Segment::Text(chunk) => println!("Segment is a chunk of text!"),
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
        if let Some(((start, end), token)) = self
            .pattern_matcher
            .get_next_special_token(remaining_input, self.normalized)
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

pub struct PipelineTokenizer {
    added_vocabulary: AddedVocabulary,
    normalizer: Option<NormalizerWrapper>,
    pre_tokenizer: PipelinePreTokenizer,
    model: ModelWrapper,
    _post_processor: Option<PostProcessorWrapper>,
}

impl TryFrom<&Tokenizer> for PipelineTokenizer {
    type Error = super::Error;

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
