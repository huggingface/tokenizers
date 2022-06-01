use crate::processors::PostProcessorWrapper;
use crate::tokenizer::{Encoding, PostProcessor, Result};
use crate::utils::macro_rules_attribute;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct Sequence {
    processors: Vec<PostProcessorWrapper>,
}

impl Sequence {
    pub fn new(processors: Vec<PostProcessorWrapper>) -> Self {
        Self { processors }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum SequenceProcessorError {
    #[error("could not pop encodings vector")]
    EncodingsVecPop,
}

impl PostProcessor for Sequence {
    fn added_tokens(&self, is_pair: bool) -> usize {
        self.processors
            .iter()
            .map(|p| p.added_tokens(is_pair))
            .sum::<usize>()
    }

    fn process(
        &self,
        encoding: Encoding,
        pair_encoding: Option<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        let mut encodings = vec![encoding];
        if let Some(encoding) = pair_encoding {
            encodings.push(encoding);
        }

        for processor in &self.processors {
            encodings = processor.process_chain(encodings, add_special_tokens)?;
        }

        <dyn PostProcessor>::merge_encodings(encodings)
    }

    fn process_chain(
        &self,
        mut encodings: Vec<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        for processor in &self.processors {
            encodings = processor.process_chain(encodings, add_special_tokens)?;
        }
        let encoding_merged = Encoding::merge(encodings, false);

        Ok(vec![encoding_merged])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processors::{ByteLevel, PostProcessorWrapper};
    use crate::tokenizer::{Encoding, PostProcessor};
    use std::collections::HashMap;

    #[test]
    fn process_chain() {
        let start = Encoding::new(
            vec![0; 5],
            vec![],
            vec![
                "Ġ".into(),
                "ĠĠĠĠHelloĠĠ".into(),
                "ĠĠHello".into(),
                "HelloĠĠ".into(),
                "ĠĠĠĠ".into(),
            ],
            vec![],
            vec![(0, 1), (0, 11), (11, 18), (18, 25), (25, 29)],
            vec![],
            vec![],
            vec![],
            HashMap::new(),
        );

        let bytelevel = ByteLevel::default().trim_offsets(true);

        let sequence = Sequence::new(vec![PostProcessorWrapper::ByteLevel(bytelevel)]);
        let expected = Encoding::new(
            vec![0; 5],
            vec![],
            vec![
                "Ġ".into(),
                "ĠĠĠĠHelloĠĠ".into(),
                "ĠĠHello".into(),
                "HelloĠĠ".into(),
                "ĠĠĠĠ".into(),
            ],
            vec![],
            vec![(0, 0), (4, 9), (13, 18), (18, 23), (29, 29)],
            vec![],
            vec![],
            vec![],
            HashMap::new(),
        );
        let pair_expected = bytelevel
            .process(start.clone(), Some(start.clone()), false)
            .unwrap();

        assert_eq!(
            expected,
            sequence.process(start.clone(), None, false).unwrap()
        );

        assert_eq!(
            pair_expected,
            sequence
                .process(start.clone(), Some(start.clone()), false)
                .unwrap()
        );
    }
}
