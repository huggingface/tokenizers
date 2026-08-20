//! The `Sequence` pre-tokenizer. Lives here rather than in `tk-encode` because it is a
//! `Vec<PreTokenizerWrapper>`, and a `Vec` of a type can only be parameterised where that type is.
//!
//! Its runtime counterpart is `tk_encode::pre_tokenizers::sequence::PipelineSequence`, a `Vec` of
//! `PipelinePreTokenizer`; the `TryFrom` at the bottom is the lowering between the two.

use std::convert::{TryFrom, TryInto};

use tk_encode::pre_tokenizers::sequence::PipelineSequence;
use tk_encode::utils::macro_rules_attribute;
use tk_encode::{PreTokenizedString, PreTokenizer, Result};

use crate::macros::impl_serde_type;
use crate::pre_tokenizers::PreTokenizerWrapper;

#[derive(Clone, Debug, PartialEq)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct Sequence {
    pretokenizers: Vec<PreTokenizerWrapper>,
}

impl Sequence {
    pub fn new(pretokenizers: Vec<PreTokenizerWrapper>) -> Self {
        Self { pretokenizers }
    }
}

impl AsRef<[PreTokenizerWrapper]> for Sequence {
    fn as_ref(&self) -> &[PreTokenizerWrapper] {
        &self.pretokenizers
    }
}

impl AsMut<[PreTokenizerWrapper]> for Sequence {
    fn as_mut(&mut self) -> &mut [PreTokenizerWrapper] {
        &mut self.pretokenizers
    }
}

impl IntoIterator for Sequence {
    type Item = PreTokenizerWrapper;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.pretokenizers.into_iter()
    }
}

impl PreTokenizer for Sequence {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        for pretokenizer in &self.pretokenizers {
            pretokenizer.pre_tokenize(pretokenized)?;
        }
        Ok(())
    }
}

impl TryFrom<Sequence> for PipelineSequence {
    type Error = tk_encode::Error;
    fn try_from(value: Sequence) -> Result<Self> {
        Ok(PipelineSequence::new(
            value
                .pretokenizers
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<Vec<_>>>()?,
        ))
    }
}
