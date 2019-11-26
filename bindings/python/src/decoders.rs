extern crate tokenizers as tk;

use super::utils::Container;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;

#[pyclass]
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
        Ok(self.decoder.execute(|decoder| decoder.decode(tokens)))
    }
}

#[pyclass]
pub struct ByteLevel {}
#[pymethods]
impl ByteLevel {
    #[staticmethod]
    fn new() -> PyResult<Decoder> {
        Ok(Decoder {
            decoder: Container::Owned(Box::new(tk::decoders::byte_level::ByteLevel)),
        })
    }
}

struct PyDecoder {
    class: PyObject,
}

impl PyDecoder {
    pub fn new(class: PyObject) -> PyResult<Self> {
        let decoder = PyDecoder { class };

        // Quickly test the PyDecoder
        decoder._decode(vec![
            "This".into(),
            "is".into(),
            "a".into(),
            "sentence".into(),
        ])?;

        Ok(decoder)
    }

    fn _decode(&self, tokens: Vec<String>) -> PyResult<String> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let args = PyTuple::new(py, &[tokens]);
        let res = self.class.call_method(py, "decode", args, None)?;

        let decoded = res
            .cast_as::<PyString>(py)
            .map_err(|_| exceptions::TypeError::py_err("`decode` is expected to return a str"))?;

        Ok(decoded.to_string()?.into_owned())
    }
}

impl tk::tokenizer::Decoder for PyDecoder {
    fn decode(&self, tokens: Vec<String>) -> String {
        match self._decode(tokens) {
            Ok(res) => res,
            Err(e) => {
                let gil = Python::acquire_gil();
                let py = gil.python();
                e.print(py);

                // Return an empty string as fallback
                String::from("")
            }
        }
    }
}
