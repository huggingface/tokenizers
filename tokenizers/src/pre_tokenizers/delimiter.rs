use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize, Serializer};

use crate::tokenizer::{PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior};

#[derive(Copy, Clone, Debug, Deserialize)]
pub struct CharDelimiterSplit {
    delimiter: char,
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

impl Serialize for CharDelimiterSplit {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut m = serializer.serialize_struct("CharDelimiterSplit", 2)?;
        m.serialize_field("type", "CharDelimiterSplit")?;
        m.serialize_field("delimiter", &self.delimiter)?;
        m.end()
    }
}
