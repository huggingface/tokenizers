use serde::{Deserialize, Deserializer, Serialize};

use crate::tokenizer::{PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior};

#[derive(Copy, Clone, Debug, Serialize, PartialEq)]
#[serde(tag = "type")]
#[non_exhaustive]
pub struct CharDelimiterSplit {
    pub delimiter: char,
}

impl<'de> Deserialize<'de> for CharDelimiterSplit {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        enum Type {
            CharDelimiterSplit,
        }

        #[derive(Deserialize)]
        pub struct CharDelimiterSplitHelper {
            #[serde(rename = "type")]
            _type: Type,
            delimiter: char,
        }

        let helper = CharDelimiterSplitHelper::deserialize(deserializer)?;
        Ok(CharDelimiterSplit::new(helper.delimiter))
    }
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
