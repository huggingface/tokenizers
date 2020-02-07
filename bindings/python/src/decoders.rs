extern crate tokenizers as tk;

use super::error::{PyError, ToPyResult};
use super::utils::Container;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use tk::tokenizer::Result;

#[pyclass(dict)]
pub struct Decoder {
    pub decoder: Container<dyn tk::tokenizer::Decoder + Sync>,
}
#[pymethods]
impl Decoder {
    #[staticmethod]
    fn custom(decoder: PyObject) -> PyResult<Self> {
        let decoder = PyDecoder::new(decoder)?;
        Ok(Decoder {
            decoder: Container::Owned(Box::new(decoder)),
        })
    }

    fn decode(&self, tokens: Vec<String>) -> PyResult<String> {
        ToPyResult(self.decoder.execute(|decoder| decoder.decode(tokens))).into()
    }
}

#[pyclass(extends=Decoder)]
pub struct ByteLevel {}
#[pymethods]
impl ByteLevel {
    #[new]
    fn new(obj: &PyRawObject) -> PyResult<()> {
        Ok(obj.init(Decoder {
            decoder: Container::Owned(Box::new(tk::decoders::byte_level::ByteLevel::new(false))),
        }))
    }
}

#[pyclass(extends=Decoder)]
pub struct WordPiece {}
#[pymethods]
impl WordPiece {
    #[new]
    #[args(kwargs="**")]
    fn new(obj: &PyRawObject, kwargs: Option<&PyDict>) -> PyResult<()> {
        let mut prefix = String::from("##");

        if let Some(kwargs) = kwargs {
            if let Some(p) = kwargs.get_item("prefix") {
                prefix = p.extract()?;
            }
        }

        Ok(obj.init(Decoder {
            decoder: Container::Owned(Box::new(tk::decoders::wordpiece::WordPiece::new(prefix))),
        }))
    }
}

#[pyclass(extends=Decoder)]
pub struct Metaspace {}
#[pymethods]
impl Metaspace {
    #[new]
    #[args(kwargs = "**")]
    fn new(obj: &PyRawObject, kwargs: Option<&PyDict>) -> PyResult<()> {
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

        Ok(obj.init(Decoder {
            decoder: Container::Owned(Box::new(tk::decoders::metaspace::Metaspace::new(
                replacement,
                add_prefix_space,
            ))),
        }))
    }
}

#[pyclass(extends=Decoder)]
pub struct BPEDecoder {}
#[pymethods]
impl BPEDecoder {
    #[new]
    #[args(kwargs = "**")]
    fn new(obj: &PyRawObject, kwargs: Option<&PyDict>) -> PyResult<()> {
        let mut suffix = String::from("</w");

        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "suffix" => suffix = value.extract()?,
                    _ => println!("Ignored unknown kwarg option {}", key),
                }
            }
        }

        Ok(obj.init(Decoder {
            decoder: Container::Owned(Box::new(tk::decoders::bpe::BPEDecoder::new(suffix))),
        }))
    }
}

struct PyDecoder {
    class: PyObject,
}

impl PyDecoder {
    pub fn new(class: PyObject) -> PyResult<Self> {
        Ok(PyDecoder { class })
    }
}

impl tk::tokenizer::Decoder for PyDecoder {
    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let args = PyTuple::new(py, &[tokens]);
        match self.class.call_method(py, "decode", args, None) {
            Ok(res) => Ok(res
                .cast_as::<PyString>(py)
                .map_err(|_| PyError::from("`decode` is expected to return a str"))?
                .to_string()
                .map_err(|_| PyError::from("`decode` is expected to return a str"))?
                .into_owned()),
            Err(e) => {
                e.print(py);
                Err(Box::new(PyError::from("Error while calling `decode`")))
            }
        }
    }
}
