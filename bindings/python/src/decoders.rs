extern crate tokenizers as tk;

use super::error::{PyError, ToPyResult};
use super::utils::Container;
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

#[pyclass]
pub struct ByteLevel {}
#[pymethods]
impl ByteLevel {
    #[staticmethod]
    fn new() -> PyResult<Decoder> {
        Ok(Decoder {
            decoder: Container::Owned(Box::new(tk::decoders::byte_level::ByteLevel::new(false))),
        })
    }
}

#[pyclass]
pub struct WordPiece {}
#[pymethods]
impl WordPiece {
    #[staticmethod]
    fn new() -> PyResult<Decoder> {
        Ok(Decoder {
            decoder: Container::Owned(Box::new(tk::decoders::wordpiece::WordPiece)),
        })
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
