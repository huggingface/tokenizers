use std::convert::TryFrom;
use std::ops::Range;

use crate::{
    pre_tokenizers::{bert::BertPreTokenizer, whitespace::Whitespace},
    AddedVocabulary, Model, ModelWrapper, NormalizedString, Normalizer, NormalizerWrapper,
    PostProcessorWrapper, PreTokenizerWrapper, Token, Tokenizer,
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
    fn tokenize_chunk(
        &self,
        chunk: &str,
        pre_tokens: &mut Vec<Split>,
        output: &mut Vec<PipelineToken>,
    ) -> Result<()> {
        // todo: no more NormalizedString
        let mut normalized: NormalizedString = chunk.into();
        if let Some(normalizer) = &self.normalizer {
            normalizer.normalize(&mut normalized)?;
        }
        let normalized_chunk = normalized.get();

        // todo: extract special tokens from normalized input

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
        Ok(())
    }

    pub fn encode(&self, input: &str, _add_special_tokens: bool) -> Result<Vec<PipelineToken>> {
        let mut output: Vec<PipelineToken> = vec![];
        let mut pre_tokens: Vec<Split> = vec![];

        let mut offset: usize = 0;

        loop {
            if let Some(((start, end), token)) = self
                .added_vocabulary
                .get_next_special_token(&input[offset..], false)
            {
                let chunk_to_tokenize = &input[offset..offset + start];
                if !chunk_to_tokenize.is_empty() {
                    self.tokenize_chunk(chunk_to_tokenize, &mut pre_tokens, &mut output)?;
                }
                output.push(PipelineToken { id: token });
                offset += end;
            } else {
                let chunk_to_tokenize = &input[offset..];
                if !chunk_to_tokenize.is_empty() {
                    self.tokenize_chunk(chunk_to_tokenize, &mut pre_tokens, &mut output)?;
                }
                break;
            }
        }

        Ok(output)
    }
}
