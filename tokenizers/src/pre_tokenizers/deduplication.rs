use crate::tokenizer::{PreTokenizedString, PreTokenizer, Result, SplitDelimiterBehavior};

#[derive(Copy, Clone, Debug)]
pub struct Deduplication;
impl_serde_unit_struct!(DeduplicationVisitor, Deduplication);

impl PreTokenizer for Deduplication {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        pretokenized.split(|_, s| s.split(char::is_whitespace, SplitDelimiterBehavior::Removed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OffsetReferential;

    #[test]
    fn deduplication_basic() {
        let pretok = Deduplication;
        let mut pretokenized: PreTokenizedString = "Hey friend!     How are you?!?".into();
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        assert_eq!(
            pretokenized
                .get_splits(OffsetReferential::Original)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect::<Vec<_>>(),
            vec![
                ("Hey", (0, 3)),
                ("friend!", (4, 11)),
                ("How", (16, 19)),
                ("are", (20, 23)),
                ("you?!?", (24, 30)),
            ]
        );
    }
}
