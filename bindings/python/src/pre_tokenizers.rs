extern crate tokenizers as tk;

use super::error::{PyError, ToPyResult};
use super::utils::Container;
use pyo3::prelude::*;
use pyo3::types::*;
use std::collections::HashSet;
use tk::tokenizer::Result;

#[pyclass(dict)]
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
    #[args(kwargs = "**")]
    fn new(kwargs: Option<&PyDict>) -> PyResult<PreTokenizer> {
        let mut add_prefix_space = true;

        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "add_prefix_space" => add_prefix_space = value.extract()?,
                    _ => println!("Ignored unknown kwargs option {}", key),
                }
            }
        }

        Ok(PreTokenizer {
            pretok: Container::Owned(Box::new(tk::pre_tokenizers::byte_level::ByteLevel::new(
                add_prefix_space,
            ))),
        })
    }
}

#[pyclass]
pub struct BertPreTokenizer {}
#[pymethods]
impl BertPreTokenizer {
    #[staticmethod]
    #[args(kwargs = "**")]
    fn new(kwargs: Option<&PyDict>) -> PyResult<PreTokenizer> {
        let mut do_basic_tokenize = true;
        let mut do_lower_case = true;
        let mut never_split = HashSet::new();
        let mut tokenize_chinese_chars = true;

        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "do_basic_tokenize" => do_basic_tokenize = val.extract()?,
                    "do_lower_case" => do_lower_case = val.extract()?,
                    "tokenize_chinese_chars" => tokenize_chinese_chars = val.extract()?,
                    "never_split" => {
                        let values: Vec<String> = val.extract()?;
                        never_split = values.into_iter().collect();
                    }
                    _ => println!("Ignored unknown kwargs option {}", key),
                }
            }
        }

        Ok(PreTokenizer {
            pretok: Container::Owned(Box::new(tk::pre_tokenizers::bert::BertPreTokenizer::new(
                do_basic_tokenize,
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
