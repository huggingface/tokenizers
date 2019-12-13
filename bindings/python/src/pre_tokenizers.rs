extern crate tokenizers as tk;

use super::error::{PyError, ToPyResult};
use super::utils::Container;
use pyo3::prelude::*;
use pyo3::types::*;
use std::collections::HashSet;
use tk::tokenizer::Result;

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

    fn pre_tokenize(&self, s: &str) -> PyResult<Vec<String>> {
        ToPyResult(self.pretok.execute(|pretok| pretok.pre_tokenize(s))).into()
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

#[pyclass]
pub struct BasicPreTokenizer {}
#[pymethods]
impl BasicPreTokenizer {
    #[staticmethod]
    fn new() -> PyResult<PreTokenizer> {
        // TODO: Parse kwargs for these
        let mut do_lower_case = true;
        let mut never_split = HashSet::new();
        let mut tokenize_chinese_chars = true;

        Ok(PreTokenizer {
            pretok: Container::Owned(Box::new(tk::pre_tokenizers::basic::BasicPreTokenizer::new(
                do_lower_case,
                never_split,
                tokenize_chinese_chars,
            ))),
        })
    }
}

/// Attempt at providing Python the ability to give its own PreTokenizer
struct PyPreTokenizer {
    class: PyObject,
}

impl PyPreTokenizer {
    pub fn new(class: PyObject) -> PyResult<Self> {
        Ok(PyPreTokenizer { class })
    }
}

impl tk::tokenizer::PreTokenizer for PyPreTokenizer {
    fn pre_tokenize(&self, sentence: &str) -> Result<Vec<String>> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let args = PyTuple::new(py, &[sentence]);
        match self.class.call_method(py, "pre_tokenize", args, None) {
            Ok(res) => Ok(res
                .cast_as::<PyList>(py)
                .map_err(|_| PyError::from("`pre_tokenize is expected to return a List[str]"))?
                .extract::<Vec<String>>()
                .map_err(|_| PyError::from("`pre_tokenize` is expected to return a List[str]"))?),
            Err(e) => {
                e.print(py);
                Err(Box::new(PyError::from(
                    "Error while calling `pre_tokenize`",
                )))
            }
        }
    }
}
