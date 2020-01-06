extern crate tokenizers as tk;

use pyo3::prelude::*;

#[pyclass]
#[repr(transparent)]
pub struct NormalizedString {
    s: tk::tokenizer::NormalizedString,
}
impl NormalizedString {
    pub fn new(s: tk::tokenizer::NormalizedString) -> NormalizedString {
        NormalizedString { s }
    }
}

#[pymethods]
impl NormalizedString {
    #[getter]
    fn get_original(&self) -> String {
        self.s.get_original().to_owned()
    }

    #[getter]
    fn get_normalized(&self) -> String {
        self.s.get().to_owned()
    }

    fn get_range(&self, start: usize, end: usize) -> Option<String> {
        self.s.get_range(start..end).map(|s| s.to_owned())
    }

    fn get_range_original(&self, start: usize, end: usize) -> Option<String> {
        self.s.get_range_original(start..end).map(|s| s.to_owned())
    }
}
