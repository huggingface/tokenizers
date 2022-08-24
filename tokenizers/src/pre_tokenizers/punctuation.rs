use serde::{Deserialize, Serialize};

use crate::tokenizer::{PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior};
use crate::utils::macro_rules_attribute;
use unicode_categories::UnicodeCategories;

fn is_punc(x: char) -> bool {
    char::is_ascii_punctuation(&x) || x.is_punctuation()
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct Punctuation {
    #[serde(default = "default_split")]
    behavior: SplitDelimiterBehavior,
}

fn default_split() -> SplitDelimiterBehavior {
    SplitDelimiterBehavior::Isolated
}

impl Punctuation {
    pub fn new(behavior: SplitDelimiterBehavior) -> Self {
        Self { behavior }
    }
}

impl Default for Punctuation {
    fn default() -> Self {
        Self::new(SplitDelimiterBehavior::Isolated)
    }
}

impl PreTokenizer for Punctuation {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, s| s.split(is_punc, self.behavior))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OffsetReferential, OffsetType};

    #[test]
    fn punctuation_basic() {
        let pretok = Punctuation::default();
        let mut pretokenized: PreTokenizedString = "Hey friend!     How are you?!?".into();
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("Hey friend", (0, 10)),
                ("!", (10, 11)),
                ("     How are you", (11, 27)),
                ("?", (27, 28)),
                ("!", (28, 29)),
                ("?", (29, 30)),
            ]
        );
    }

    #[test]
    fn deserialization() {
        let punctuation: Punctuation = serde_json::from_str(r#"{"type": "Punctuation"}"#).unwrap();
        assert_eq!(punctuation, Punctuation::default());
        assert_eq!(
            punctuation,
            Punctuation::new(SplitDelimiterBehavior::Isolated)
        );
    }

    #[test]
    #[should_panic]
    fn deserialization_erroneous() {
        let _punctuation: Punctuation =
            serde_json::from_str(r#"{"type": "WhitespaceSplit"}"#).unwrap();
    }
}
