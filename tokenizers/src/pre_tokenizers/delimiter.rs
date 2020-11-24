use serde::{Deserialize, Serialize};

use crate::tokenizer::{PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior};

#[derive(Copy, Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
#[non_exhaustive]
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
