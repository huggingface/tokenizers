mod normalize;
mod pretokenize;

use normalize::NormalizerError;
use pretokenize::PreTokenizerError;
use thiserror::Error;
use std::{borrow::Cow, result};

use crate::{normalize::{NormalizePlan, Normalizer}, pretokenize::{PreTokenizePlan, PreTokenizer}};

#[derive(Debug, Error)]
enum TokenizerError {
    #[error("Normalizer error")]
    NormalizerError(#[from] NormalizerError),
    #[error("PreTokenizer error")]
    PreTokenizerError(#[from] PreTokenizerError),
}

type Result<T> = result::Result<T, TokenizerError>;

pub struct Tokenizer {
    normalizer: NormalizePlan,
    pre_tokenizer: PreTokenizePlan,
    model: TokenizerModel,
    // post_process: PostProcessPlan
}

#[derive(Debug)]
pub struct TokenizerModel {

}

impl TokenizerModel {
    pub fn tokenize(&self, _pre_token: &[u8]) -> Vec<Token> {
        unimplemented!("not implemented")
    }
}

pub struct Token {
    id: u32
}

impl Tokenizer {
    pub fn encode(&mut self, input: &str) -> Result<Vec<Token>> {
        let normalized = self.normalizer.normalize(Cow::from(input))?;
        let pre_tokenized = self.pre_tokenizer.pre_tokenize(&normalized)?;
        let mut tokens = Vec::with_capacity(pre_tokenized.len());
        for pre_token in pre_tokenized.iter() {
            tokens.extend(self.model.tokenize(pre_token));
        }
        Ok(tokens)
    }
}
