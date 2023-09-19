use crate::pre_tokenizers::PreTokenizerWrapper;
use crate::tokenizer::{PreTokenizedString, PreTokenizer, Result};
use crate::utils::macro_rules_attribute;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct Sequence {
    pre_tokenizers: Vec<PreTokenizerWrapper>,
}

impl Sequence {
    pub fn new(pre_tokenizers: Vec<PreTokenizerWrapper>) -> Self {
        Self { pre_tokenizers }
    }

    pub fn get_pre_tokenizers(&self) -> &[PreTokenizerWrapper] {
        &self.pre_tokenizers
    }

    pub fn get_pre_tokenizers_mut(&mut self) -> &mut [PreTokenizerWrapper] {
        &mut self.pre_tokenizers
    }
}

impl PreTokenizer for Sequence {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        for pre_tokenizer in &self.pre_tokenizers {
            pre_tokenizer.pre_tokenize(pretokenized)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pre_tokenizers::{punctuation::Punctuation, whitespace::WhitespaceSplit};
    use crate::{OffsetReferential, OffsetType};

    #[test]
    fn sequence_basic() {
        let pretokenizers = vec![
            PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit),
            PreTokenizerWrapper::Punctuation(Punctuation::default()),
        ];
        let pretok = Sequence::new(pretokenizers);
        let mut pretokenized: PreTokenizedString = "Hey friend!     How are you?!?".into();
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("Hey", (0, 3)),
                ("friend", (4, 10)),
                ("!", (10, 11)),
                ("How", (16, 19)),
                ("are", (20, 23)),
                ("you", (24, 27)),
                ("?", (27, 28)),
                ("!", (28, 29)),
                ("?", (29, 30)),
            ]
        );
    }
}
