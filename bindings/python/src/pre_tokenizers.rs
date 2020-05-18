extern crate tokenizers as tk;

use super::error::{PyError, ToPyResult};
use super::utils::Container;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tk::tokenizer::{Offsets, Result};

#[pyclass(dict, module = "tokenizers.pre_tokenizers")]
pub struct PreTokenizer {
    pub pretok: Container<dyn tk::tokenizer::PreTokenizer>,
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

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let data = self
            .pretok
            .execute(|pretok| serde_json::to_string(&pretok))
            .map_err(|e| {
                exceptions::Exception::py_err(format!(
                    "Error while attempting to pickle PreTokenizer: {}",
                    e.to_string()
                ))
            })?;
        Ok(PyBytes::new(py, data.as_bytes()).to_object(py))
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.pretok =
                    Container::Owned(serde_json::from_slice(s.as_bytes()).map_err(|e| {
                        exceptions::Exception::py_err(format!(
                            "Error while attempting to unpickle PreTokenizer: {}",
                            e.to_string()
                        ))
                    })?);
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn pre_tokenize(&self, s: &str) -> PyResult<Vec<(String, Offsets)>> {
        // TODO: Expose the NormalizedString
        let mut normalized = tk::tokenizer::NormalizedString::from(s);
        ToPyResult(
            self.pretok
                .execute(|pretok| pretok.pre_tokenize(&mut normalized)),
        )
        .into()
    }
}

#[pyclass(extends=PreTokenizer, module = "tokenizers.pre_tokenizers")]
pub struct ByteLevel {}
#[pymethods]
impl ByteLevel {
    #[new]
    #[args(kwargs = "**")]
    fn new(kwargs: Option<&PyDict>) -> PyResult<(Self, PreTokenizer)> {
        let mut byte_level = tk::pre_tokenizers::byte_level::ByteLevel::default();
        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "add_prefix_space" => {
                        byte_level = byte_level.add_prefix_space(value.extract()?)
                    }
                    _ => println!("Ignored unknown kwargs option {}", key),
                }
            }
        }

        Ok((
            ByteLevel {},
            PreTokenizer {
                pretok: Container::Owned(Box::new(byte_level)),
            },
        ))
    }

    #[staticmethod]
    fn alphabet() -> Vec<String> {
        tk::pre_tokenizers::byte_level::ByteLevel::alphabet()
            .into_iter()
            .map(|c| c.to_string())
            .collect()
    }
}

#[pyclass(extends=PreTokenizer, module = "tokenizers.pre_tokenizers")]
pub struct Whitespace {}
#[pymethods]
impl Whitespace {
    #[new]
    fn new() -> PyResult<(Self, PreTokenizer)> {
        Ok((
            Whitespace {},
            PreTokenizer {
                pretok: Container::Owned(Box::new(tk::pre_tokenizers::whitespace::Whitespace)),
            },
        ))
    }
}

#[pyclass(extends=PreTokenizer, module = "tokenizers.pre_tokenizers")]
pub struct WhitespaceSplit {}
#[pymethods]
impl WhitespaceSplit {
    #[new]
    fn new() -> PyResult<(Self, PreTokenizer)> {
        Ok((
            WhitespaceSplit {},
            PreTokenizer {
                pretok: Container::Owned(Box::new(tk::pre_tokenizers::whitespace::WhitespaceSplit)),
            },
        ))
    }
}

#[pyclass(extends=PreTokenizer, module = "tokenizers.pre_tokenizers")]
pub struct CharDelimiterSplit {}
#[pymethods]
impl CharDelimiterSplit {
    #[new]
    pub fn new(delimiter: &str) -> PyResult<(Self, PreTokenizer)> {
        let chr_delimiter = delimiter
            .chars()
            .nth(0)
            .ok_or(exceptions::Exception::py_err(
                "delimiter must be a single character",
            ))?;
        Ok((
            CharDelimiterSplit {},
            PreTokenizer {
                pretok: Container::Owned(Box::new(
                    tk::pre_tokenizers::delimiter::CharDelimiterSplit::new(chr_delimiter),
                )),
            },
        ))
    }

    fn __getnewargs__<'p>(&self, py: Python<'p>) -> PyResult<&'p PyTuple> {
        Ok(PyTuple::new(py, &[" "]))
    }
}

#[pyclass(extends=PreTokenizer, module = "tokenizers.pre_tokenizers")]
pub struct BertPreTokenizer {}
#[pymethods]
impl BertPreTokenizer {
    #[new]
    fn new() -> PyResult<(Self, PreTokenizer)> {
        Ok((
            BertPreTokenizer {},
            PreTokenizer {
                pretok: Container::Owned(Box::new(tk::pre_tokenizers::bert::BertPreTokenizer)),
            },
        ))
    }
}

#[pyclass(extends=PreTokenizer, module = "tokenizers.pre_tokenizers")]
pub struct Metaspace {}
#[pymethods]
impl Metaspace {
    #[new]
    #[args(kwargs = "**")]
    fn new(kwargs: Option<&PyDict>) -> PyResult<(Self, PreTokenizer)> {
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

        Ok((
            Metaspace {},
            PreTokenizer {
                pretok: Container::Owned(Box::new(tk::pre_tokenizers::metaspace::Metaspace::new(
                    replacement,
                    add_prefix_space,
                ))),
            },
        ))
    }
}

struct PyPreTokenizer {
    class: PyObject,
}

impl PyPreTokenizer {
    pub fn new(class: PyObject) -> PyResult<Self> {
        Ok(PyPreTokenizer { class })
    }
}

#[typetag::serde]
impl tk::tokenizer::PreTokenizer for PyPreTokenizer {
    fn pre_tokenize(
        &self,
        sentence: &mut tk::tokenizer::NormalizedString,
    ) -> Result<Vec<(String, Offsets)>> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let args = PyTuple::new(py, &[sentence.get()]);
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

impl Serialize for PyPreTokenizer {
    fn serialize<S>(&self, _serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Err(serde::ser::Error::custom(
            "Custom PyPreTokenizer cannot be serialized",
        ))
    }
}

impl<'de> Deserialize<'de> for PyPreTokenizer {
    fn deserialize<D>(_deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        unimplemented!("PyPreTokenizer cannot be deserialized")
    }
}
