extern crate tokenizers as tk;

use super::utils::Container;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;

#[pyclass]
pub struct PreTokenizer {
    pub pretok: Container<dyn tk::tokenizer::PreTokenizer + Sync>,
}
#[pymethods]
impl PreTokenizer {
    #[staticmethod]
    fn custom(pretok: PyObject) -> PyResult<Self> {
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
        let pretok = PyPreTokenizer { class };

        // Quickly test the PyPreTokenizer
        pretok._pre_tokenize("This is a test sentence")?;

        Ok(pretok)
    }

    fn _pre_tokenize(&self, sentence: &str) -> PyResult<Vec<String>> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let args = PyTuple::new(py, &[sentence]);
        let res = self.class.call_method(py, "pre_tokenize", args, None)?;

        let tokens = res.cast_as::<PyList>(py).map_err(|_| {
            exceptions::TypeError::py_err("`pre_tokenize` is expected to return a List[str]`")
        })?;
        let tokens: Vec<String> = tokens.extract().map_err(|_| {
            exceptions::TypeError::py_err("`pre_tokenize` is expected to return a List[str]`")
        })?;

        Ok(tokens)
    }
}

impl tk::tokenizer::PreTokenizer for PyPreTokenizer {
    fn pre_tokenize(&self, sentence: &str) -> Vec<String> {
        match self._pre_tokenize(sentence) {
            Ok(res) => res,
            Err(e) => {
                let gil = Python::acquire_gil();
                let py = gil.python();
                e.print(py);

                // Return an empty Vec as fallback
                vec![]
            }
        }
    }
}
