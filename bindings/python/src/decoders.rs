extern crate tokenizers as tk;

use super::utils::Container;
use pyo3::prelude::*;

#[pyclass]
pub struct Decoder {
    pub decoder: Container<dyn tk::tokenizer::Decoder + Sync>,
}

#[pyclass]
pub struct ByteLevel {}
#[pymethods]
impl ByteLevel {
    #[staticmethod]
    fn new() -> PyResult<Decoder> {
        Ok(Decoder {
            decoder: Container::Owned(Box::new(tk::decoders::byte_level::ByteLevel)),
        })
    }
}
