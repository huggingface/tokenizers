use serde::{Deserialize, Serialize};

use crate::tokenizer::{
    normalizer::Range, OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, Result,
};
use crate::utils::macro_rules_attribute;
use crate::utils::truncation::{TruncationDirection, TruncationParams};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct Truncate {
    #[serde(flatten)]
    pub params: TruncationParams,
}

impl Truncate {
    pub fn new(params: TruncationParams) -> Self {
        Self { params }
    }
}

impl Default for Truncate {
    fn default() -> Self {
        Self {
            params: TruncationParams::default(),
        }
    }
}

impl PreTokenizer for Truncate {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        let max_len = self.params.max_length;
        let total_len: usize = pretokenized
            .get_splits(OffsetReferential::Normalized, OffsetType::Byte)
            .iter()
            .map(|(s, _, _)| s.len())
            .sum();
        if total_len <= max_len {
            return Ok(());
        }

        match self.params.direction {
            TruncationDirection::Right => {
                let mut remaining = max_len;
                pretokenized.split(|_, mut s| {
                    if remaining == 0 {
                        Ok(Vec::new())
                    } else {
                        let len = s.len();
                        if len <= remaining {
                            remaining -= len;
                            Ok(vec![s])
                        } else {
                            let slice = s
                                .slice(Range::Normalized(0..remaining))
                                .expect("NormalizedString bad slice");
                            remaining = 0;
                            Ok(vec![slice])
                        }
                    }
                })
            }
            TruncationDirection::Left => {
                let mut skip = total_len - max_len;
                pretokenized.split(|_, mut s| {
                    if skip >= s.len() {
                        skip -= s.len();
                        Ok(Vec::new())
                    } else {
                        if skip > 0 {
                            let len = s.len();
                            s = s
                                .slice(Range::Normalized(skip..len))
                                .expect("NormalizedString bad slice");
                            skip = 0;
                        }
                        Ok(vec![s])
                    }
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{OffsetReferential, OffsetType};

    #[test]
    fn truncate_right() {
        let params = TruncationParams { max_length: 4, ..Default::default() };
        let pretok = Truncate::new(params);
        let mut pretokenized = PreTokenizedString::from("Hello World");
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        let parts: Vec<_> = pretokenized
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, _o, _)| s)
            .collect();
        assert_eq!(parts.join(""), "Hell");
    }

    #[test]
    fn truncate_left() {
        let mut params = TruncationParams { max_length: 5, ..Default::default() };
        params.direction = TruncationDirection::Left;
        let pretok = Truncate::new(params);
        let mut pretokenized = PreTokenizedString::from("Hello World");
        pretok.pre_tokenize(&mut pretokenized).unwrap();
        let parts: Vec<_> = pretokenized
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, _o, _)| s)
            .collect();
        assert_eq!(parts.join(""), "World");
    }
}

