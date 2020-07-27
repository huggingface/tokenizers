use crate::tokenizer::{PreTokenizedString, PreTokenizer, Result};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct CharDelimiterSplit {
    delimiter: char,
}

impl CharDelimiterSplit {
    pub fn new(delimiter: char) -> Self {
        CharDelimiterSplit { delimiter }
    }
}

#[typetag::serde]
impl PreTokenizer for CharDelimiterSplit {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, normalized| normalized.split(self.delimiter))
    }
}
