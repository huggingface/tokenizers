use std::ops::Range;
use std::{borrow::Cow, convert::TryFrom};

use crate::{
    normalizers::NormalizerWrapper,
    pre_tokenizers::{bert::BertPreTokenizer, whitespace::Whitespace},
    AddedVocabulary, Model, ModelWrapper, PostProcessorWrapper, PreTokenizerWrapper, Token,
    Tokenizer,
};

use super::Result;

/// A pre-token split, a range into the input text.
#[derive(Clone, Debug)]
pub struct Split {
    pub start: u32,
    pub end: u32,
}

impl Split {
    #[inline]
    pub fn range(&self) -> Range<usize> {
        self.start as usize..self.end as usize
    }
}

pub trait Normalizer {
    fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str>;
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

pub struct PipelineTokenizer {
    added_vocabulary: AddedVocabulary,
    normalizer: Option<NormalizerWrapper>,
    pre_tokenizer: PipelinePreTokenizer,
    model: ModelWrapper,
    _post_processor: Option<PostProcessorWrapper>,
}

/// Build a `PipelineTokenizer` from a fully-configured `Tokenizer` (the oracle),
/// cloning its components. Fails if the oracle uses a pre-tokenizer the pipeline
/// does not yet support (only `Bert`, `Whitespace`, or none).
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

        let normalizer = tok.get_normalizer().cloned();

        Ok(Self {
            added_vocabulary: tok.get_added_vocabulary().clone(),
            normalizer,
            pre_tokenizer,
            model: tok.get_model().clone(),
            _post_processor: tok.get_post_processor().cloned(),
        })
    }
}

impl PipelineTokenizer {
    fn tokenize_chunk(
        &self,
        chunk: &str,
        pre_tokens: &mut Vec<Split>,
        output: &mut Vec<PipelineToken>,
    ) -> Result<()> {
        // todo: no more NormalizedString
        let normalized: &str = if let Some(normalizer) = &self.normalizer {
            &normalizer.normalize(chunk)
        } else {
            chunk
        };
        for segment in SpecialSegmentIterator::new(normalized, &self.added_vocabulary, true) {
            match segment {
                Segment::SpecialToken(token) => {
                    output.push(PipelineToken { id: token });
                }
                Segment::Text(normalized_chunk) => {
                    pre_tokens.clear();
                    self.pre_tokenizer
                        .pre_tokenize(normalized_chunk, pre_tokens)?;
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
        Ok(())
    }

    pub fn encode(&self, input: &str, _add_special_tokens: bool) -> Result<Vec<PipelineToken>> {
        let mut output: Vec<PipelineToken> = vec![];
        let mut pre_tokens: Vec<Split> = vec![];

        for segment in SpecialSegmentIterator::new(input, &self.added_vocabulary, false) {
            match segment {
                Segment::SpecialToken(token) => {
                    output.push(PipelineToken { id: token });
                }
                Segment::Text(chunk) => {
                    self.tokenize_chunk(chunk, &mut pre_tokens, &mut output)?;
                }
            };
        }
        Ok(output)
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

pub struct SpecialSegmentIterator<'a, 'b, PatternMatcher: PipelinePatternMatcher> {
    input: &'a str,
    offset: usize,
    pending: Option<u32>,
    pattern_matcher: &'b PatternMatcher,
    normalized: bool,
}

impl<'a, 'b, PatternMatcher: PipelinePatternMatcher>
    SpecialSegmentIterator<'a, 'b, PatternMatcher>
{
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
                // Store the special token to return in the next call
                self.pending = Some(token);
                self.offset = self.offset + end;
                return Some(Segment::Text(before_token));
            } else {
                self.offset = self.offset + end;
                return Some(Segment::SpecialToken(token));
            }
        }
        self.offset = self.input.len();
        return Some(Segment::Text(remaining_input));
    }
}
