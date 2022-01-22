use serde::{Deserialize, Serialize};

use crate::tokenizer::{PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior};
use crate::utils::macro_rules_attribute;

#[derive(Copy, Clone, Debug, PartialEq)]
#[non_exhaustive]
#[macro_rules_attribute(impl_serde_type!)]
pub struct CharDelimiterSplit {
    pub delimiter: char,
}

impl CharDelimiterSplit {
    pub fn new(delimiter: char) -> Self {
        CharDelimiterSplit { delimiter }
    }
}

impl PreTokenizer for CharDelimiterSplit {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        // TODO: Maybe add the option to specify the behavior
        pretokenized.split(|_, normalized| {
            normalized.split(self.delimiter, SplitDelimiterBehavior::Removed)
        })
    }
}
