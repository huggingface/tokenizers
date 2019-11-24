extern crate tokenizers as tk;

use super::utils::Container;
use pyo3::prelude::*;

#[pyclass]
pub struct PreTokenizer {
    pub pretok: Container<dyn tk::tokenizer::PreTokenizer + Sync>,
}
#[pymethods]
impl PreTokenizer {
    #[staticmethod]
    fn from_python(pretok: PyObject) -> PyResult<Self> {
        let py_pretok = PyPreTokenizer::new(pretok)?;
        Ok(PreTokenizer {
            pretok: Container::Owned(Box::new(py_pretok)),
        })
    }
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

/// Attempt at providing Python the ability to give its own PreTokenizer
struct PyPreTokenizer {
    class: PyObject,
}

impl PyPreTokenizer {
    pub fn new(class: PyObject) -> PyResult<Self> {
        // test the given PyObject
        Ok(PyPreTokenizer { class })
    }
}

impl tk::tokenizer::PreTokenizer for PyPreTokenizer {
    fn pre_tokenize(&self, sentence: &str) -> Vec<String> {
        unimplemented!()
    }
}
