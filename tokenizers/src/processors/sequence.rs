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

        let encoding_merged = Encoding::merge(encodings, false);

        Ok(encoding_merged)
    }
}
