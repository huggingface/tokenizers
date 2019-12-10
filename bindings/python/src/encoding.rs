extern crate tokenizers as tk;

use pyo3::prelude::*;

#[pyclass]
#[repr(transparent)]
pub struct Encoding {
    encoding: tk::tokenizer::Encoding,
}

impl Encoding {
    pub fn new(encoding: tk::tokenizer::Encoding) -> Self {
        Encoding { encoding }
    }
}
