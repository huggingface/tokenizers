extern crate tokenizers as tk;

use super::utils::Container;
use pyo3::prelude::*;

#[pyclass]
pub struct PreTokenizer {
    pub pretok: Container<dyn tk::tokenizer::PreTokenizer + Sync>,
}

#[pyclass]
pub struct ByteLevel {}
#[pymethods]
impl ByteLevel {
    #[staticmethod]
    fn new() -> PyResult<PreTokenizer> {
        Ok(PreTokenizer {
            pretok: Container::Owned(Box::new(tk::pre_tokenizers::byte_level::ByteLevel)),
        })
    }
}
