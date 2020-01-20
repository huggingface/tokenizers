extern crate tokenizers as tk;

use super::error::{PyError, ToPyResult};
use super::utils::Container;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use tk::tokenizer::{Offsets, Result};

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

    fn pre_tokenize(&self, s: &str) -> PyResult<Vec<(String, Offsets)>> {
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

    #[staticmethod]
    fn alphabet() -> Vec<String> {
        tk::pre_tokenizers::byte_level::ByteLevel::alphabet()
            .into_iter()
            .map(|c| c.to_string())
            .collect()
    }
}

#[pyclass]
pub struct Whitespace {}
#[pymethods]
impl Whitespace {
    #[staticmethod]
    fn new() -> PyResult<PreTokenizer> {
        Ok(PreTokenizer {
            pretok: Container::Owned(Box::new(tk::pre_tokenizers::whitespace::Whitespace)),
        })
    }
}

#[pyclass]
pub struct WhitespaceSplit {}
#[pymethods]
impl WhitespaceSplit {
    #[staticmethod]
    fn new() -> PyResult<PreTokenizer> {
        Ok(PreTokenizer {
            pretok: Container::Owned(Box::new(tk::pre_tokenizers::whitespace::WhitespaceSplit)),
        })
    }
}

#[pyclass]
pub struct BertPreTokenizer {}
#[pymethods]
impl BertPreTokenizer {
    #[staticmethod]
    fn new() -> PyResult<PreTokenizer> {
        Ok(PreTokenizer {
            pretok: Container::Owned(Box::new(tk::pre_tokenizers::bert::BertPreTokenizer)),
        })
    }
}

#[pyclass]
pub struct Metaspace {}
#[pymethods]
impl Metaspace {
    #[staticmethod]
    #[args(kwargs = "**")]
    fn new(kwargs: Option<&PyDict>) -> PyResult<PreTokenizer> {
        let mut replacement = '▁';
        let mut add_prefix_space = true;

        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "replacement" => {
                        let s: &str = value.extract()?;
                        replacement = s.chars().nth(0).ok_or(exceptions::Exception::py_err(
                            "replacement must be a character",
                        ))?;
                    }
                    "add_prefix_space" => add_prefix_space = value.extract()?,
                    _ => println!("Ignored unknown kwarg option {}", key),
                }
            }
        }

        Ok(PreTokenizer {
            pretok: Container::Owned(Box::new(tk::pre_tokenizers::metaspace::Metaspace::new(
                replacement,
                add_prefix_space,
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
    fn pre_tokenize(&self, sentence: &str) -> Result<Vec<(String, Offsets)>> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let args = PyTuple::new(py, &[sentence]);
        match self.class.call_method(py, "pre_tokenize", args, None) {
            Ok(res) => Ok(res
                .cast_as::<PyList>(py)
                .map_err(|_| {
                    PyError::from("`pre_tokenize is expected to return a List[(str, (uint, uint))]")
                })?
                .extract::<Vec<(String, Offsets)>>()
                .map_err(|_| {
                    PyError::from(
                        "`pre_tokenize` is expected to return a List[(str, (uint, uint))]",
                    )
                })?),
            Err(e) => {
                e.print(py);
                Err(Box::new(PyError::from(
                    "Error while calling `pre_tokenize`",
                )))
            }
        }
    }
}
