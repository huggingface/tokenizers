use std::ops::Range;

use itertools::Itertools;

use crate::{
    pre_tokenizers::{bert::BertPreTokenizer, whitespace::Whitespace},
    AddedVocabulary, Model, ModelWrapper, Normalizer, NormalizerWrapper, PostProcessorWrapper,
    Token,
};

use super::Result;

/// A pre-token split, a range into the input text.
#[derive(Clone)]
pub struct Split {
    pub start: u32,
    pub end: u32,
    pub tokens: Option<Vec<PipelineToken>>,
}

impl Split {
    #[inline]
    pub fn range(&self) -> Range<usize> {
        self.start as usize..self.end as usize
    }
}

pub trait PreTokenizer {
    /// Split `text` into pre-tokens, appending to `out`. Ranges are into `text`.
    fn pre_tokenize(
        &self,
        text: &str,
        out: &mut Vec<Split>,
        start_offset: Option<u32>,
    ) -> Result<()>;
}

pub enum PipelinePreTokenizer {
    Bert(BertPreTokenizer),
    Whitespace(Whitespace),
    None,
}

impl PreTokenizer for PipelinePreTokenizer {
    fn pre_tokenize(
        &self,
        text: &str,
        out: &mut Vec<Split>,
        start_offset: Option<u32>,
    ) -> Result<()> {
        match self {
            Self::None => Ok(()),
            Self::Bert(pretok) => pretok.pre_tokenize(text, out, start_offset),
            Self::Whitespace(pretok) => pretok.pre_tokenize(text, out, start_offset),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PipelineToken {
    id: u32,
    start: u32,
    end: u32,
}

impl From<Token> for PipelineToken {
    fn from(value: Token) -> Self {
        Self {
            id: value.id,
            start: value.offsets.0 as u32,
            end: value.offsets.1 as u32,
        }
    }
}

pub struct PipelineTokenizer {
    added_vocabulary: AddedVocabulary,
    normalizer: NormalizerWrapper,
    pre_tokenizer: PipelinePreTokenizer,
    model: ModelWrapper,
    _post_processor: PostProcessorWrapper,
}

impl PipelineTokenizer {
    pub fn encode(&self, input: &str, _add_special_tokens: bool) -> Result<Vec<PipelineToken>> {
        let mut normalized = input.into();
        let mut output: Vec<PipelineToken> = vec![];
        let mut pre_tokens: Vec<Split> = vec![];

        // todo: maybe get rid of NormalizedString
        self.normalizer.normalize(&mut normalized)?;
        // todo: plug the VocabStore optimized special token extract
        let pre_tokenized = self
            .added_vocabulary
            .extract_and_normalize::<NormalizerWrapper>(None, normalized.get());

        pre_tokenized.into_splits().into_iter().for_each(
            |((start_offset, end_offset), maybe_tokens)| {
                if let Some(tokens) = maybe_tokens {
                    // Already tokenized (special tokens etc)
                    output.extend(tokens.into_iter().map_into::<PipelineToken>());
                } else {
                    let slice = &normalized.get()[start_offset..end_offset];
                    self.pre_tokenizer
                        .pre_tokenize(slice, &mut pre_tokens, Some(start_offset as u32))
                        .expect("Failed to pre-tokenize slice");

                    pre_tokens.iter().for_each(|pre_token| {
                        if pre_token.tokens.is_none() {
                            // Pre-token needs to be tokenized
                            let slice = &normalized.get()[pre_token.range()];
                            let tokens = self
                                .model
                                .tokenize(slice)
                                .expect("Failed to tokenize pretoken");
                            output.extend(tokens.into_iter().map_into::<PipelineToken>());
                        }
                    });
                    pre_tokens.clear();
                }
            },
        );

        // todo: plug post-processor
        // self.post_processor.process(encoding, pair_encoding, add_special_tokens);

        Ok(output)
    }
}
