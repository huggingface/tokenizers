use crate::processors::PostProcessorWrapper;
use crate::tokenizer::{Encoding, PostProcessor, Result};
use crate::utils::macro_rules_attribute;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct Sequence {
    processors: Vec<PostProcessorWrapper>,
}

impl Sequence {
    pub fn new(processors: Vec<PostProcessorWrapper>) -> Self {
        Self { processors }
    }

    pub fn get(&self, index: usize) -> Option<& PostProcessorWrapper> {
        self.processors.get(index as usize)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut PostProcessorWrapper> {
        self.processors.get_mut(index)
    }
}

impl PostProcessor for Sequence {
    fn added_tokens(&self, is_pair: bool) -> usize {
        self.processors
            .iter()
            .map(|p| p.added_tokens(is_pair))
            .sum::<usize>()
    }

    fn process_encodings(
        &self,
        mut encodings: Vec<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        for processor in &self.processors {
            encodings = processor.process_encodings(encodings, add_special_tokens)?;
        }
        Ok(encodings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processors::{ByteLevel, PostProcessorWrapper};
    use crate::tokenizer::{Encoding, PostProcessor};
    use std::collections::HashMap;
    use std::iter::FromIterator;

    #[test]
    fn process_chain() {
        let start = Encoding::new(
            vec![0; 5],
            vec![0; 5],
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
            vec![0; 5],
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
            HashMap::from_iter(vec![(0, 0..5)]),
        );

        assert_eq!(
            expected,
            bytelevel.process(start.clone(), None, false).unwrap()
        );
        assert_eq!(
            expected,
            sequence.process(start.clone(), None, false).unwrap()
        );

        let pair_expected = Encoding::new(
            vec![0; 10],
            vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            vec![
                "Ġ".into(),
                "ĠĠĠĠHelloĠĠ".into(),
                "ĠĠHello".into(),
                "HelloĠĠ".into(),
                "ĠĠĠĠ".into(),
                "Ġ".into(),
                "ĠĠĠĠHelloĠĠ".into(),
                "ĠĠHello".into(),
                "HelloĠĠ".into(),
                "ĠĠĠĠ".into(),
            ],
            vec![],
            vec![
                (0, 0),
                (4, 9),
                (13, 18),
                (18, 23),
                (29, 29),
                (0, 0),
                (4, 9),
                (13, 18),
                (18, 23),
                (29, 29),
            ],
            vec![],
            vec![],
            vec![],
            HashMap::from_iter(vec![(0, 0..5), (1, 5..10)]),
        );
        assert_eq!(
            pair_expected,
            bytelevel
                .process(start.clone(), Some(start.clone()), false)
                .unwrap()
        );
        assert_eq!(
            pair_expected,
            sequence.process(start.clone(), Some(start), false).unwrap()
        );
    }
}
