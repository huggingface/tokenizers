extern crate tokenizers as tk;

use super::utils::Container;
use pyo3::prelude::*;

#[pyclass(dict)]
pub struct PostProcessor {
    pub processor: Container<dyn tk::tokenizer::PostProcessor + Sync>,
}

#[pyclass]
pub struct BertProcessing {}
#[pymethods]
impl BertProcessing {
    #[staticmethod]
    fn new(sep: (String, u32), cls: (String, u32)) -> PyResult<PostProcessor> {
        Ok(PostProcessor {
            processor: Container::Owned(Box::new(tk::processors::bert::BertProcessing::new(
                sep, cls,
            ))),
        })
    }
}
