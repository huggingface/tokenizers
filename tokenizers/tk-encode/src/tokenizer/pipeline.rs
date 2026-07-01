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
    pub start: u32,
    pub end: u32,
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
    pub fn encode(&self, input: &str, _add_special_tokens: bool) -> Result<Vec<PipelineToken>> {
        let mut output: Vec<PipelineToken> = vec![];
        let mut pre_tokens: Vec<Split> = vec![];

        for (split, maybe_token) in self.added_vocabulary.extract_special_tokens(input) {
            // todo: compute offsets properly: split here is in the original string space
            if let Some(id) = maybe_token {
                output.push(PipelineToken {
                    id,
                    start: split.start,
                    end: split.end,
                });
                continue;
            }

            let base = split.start;

            // todo: NormalizedString allocates 3 times: original String, normalized String + offsets Vec
            // original String alloc is not necessary
            let mut normalized: NormalizedString = input[split.range()].into();
            if let Some(normalizer) = &self.normalizer {
                normalizer.normalize(&mut normalized)?;
            }
            let normalized_chunk = normalized.get();

            for (split, maybe_token) in self
                .added_vocabulary
                .extract_normalized_tokens(normalized_chunk)
            {
                // todo: compute offsets properly: split here is in the _normalized_ string space
                if let Some(id) = maybe_token {
                    output.push(PipelineToken {
                        id,
                        start: base + split.start,
                        end: base + split.end,
                    });
                    continue;
                }
                self.tokenize_chunk(
                    &normalized_chunk[split.range()],
                    base + split.start,
                    &mut pre_tokens,
                    &mut output,
                )?;
            }
        }
        // todo: post-processing
        // if let Some(post_processor) = self._post_processor {
        //     post_processor.process_encodings(encodings, add_special_tokens);
        // }
        Ok(output)
    }

    /// Pre-tokenize `chunk` and run each pre-token through the model, appending the
    /// resulting tokens to `output`. `base_offset` is the offset of `chunk` within the input.
    fn tokenize_chunk(
        &self,
        chunk: &str,
        base_offset: u32,
        pre_tokens: &mut Vec<Split>,
        output: &mut Vec<PipelineToken>,
    ) -> Result<()> {
        pre_tokens.clear();
        self.pre_tokenizer.pre_tokenize(chunk, pre_tokens)?;

        for pre_token in pre_tokens.iter() {
            let pt_base = base_offset + pre_token.start;
            output.extend(
                self.model
                    .tokenize(&chunk[pre_token.range()])?
                    .into_iter()
                    .map(|Token { id, offsets, .. }| PipelineToken {
                        id,
                        start: pt_base + offsets.0 as u32,
                        end: pt_base + offsets.1 as u32,
                    }),
            );
        }
        Ok(())
    }
}
