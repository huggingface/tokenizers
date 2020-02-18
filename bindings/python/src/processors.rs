extern crate tokenizers as tk;

use super::utils::Container;
use pyo3::prelude::*;

#[pyclass(dict)]
pub struct PostProcessor {
    pub processor: Container<dyn tk::tokenizer::PostProcessor + Sync>,
}

#[pymethods]
impl PostProcessor {
    fn num_special_tokens_to_add(&self, is_pair: bool) -> usize {
        self.processor.execute(|p| p.added_tokens(is_pair))
    }
}

#[pyclass(extends=PostProcessor)]
pub struct BertProcessing {}
#[pymethods]
impl BertProcessing {
    #[new]
    fn new(obj: &PyRawObject, sep: (String, u32), cls: (String, u32)) -> PyResult<()> {
        Ok(obj.init(PostProcessor {
            processor: Container::Owned(Box::new(tk::processors::bert::BertProcessing::new(
                sep, cls,
            ))),
        }))
    }
}

#[pyclass(extends=PostProcessor)]
pub struct RobertaProcessing {}
#[pymethods]
impl RobertaProcessing {
    #[new]
    fn new(obj: &PyRawObject, sep: (String, u32), cls: (String, u32)) -> PyResult<()> {
        Ok(obj.init(PostProcessor {
            processor: Container::Owned(Box::new(tk::processors::roberta::RobertaProcessing::new(
                sep, cls,
            ))),
        }))
    }
}
