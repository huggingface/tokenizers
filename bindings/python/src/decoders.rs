use std::sync::Arc;

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use serde::de::Error;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tk::decoders::bpe::BPEDecoder;
use tk::decoders::byte_level::ByteLevel;
use tk::decoders::metaspace::Metaspace;
use tk::decoders::wordpiece::WordPiece;
use tk::Decoder;
use tokenizers as tk;

use super::error::{PyError, ToPyResult};

#[pyclass(dict, module = "tokenizers.decoders", name=Decoder)]
#[derive(Clone)]
pub struct PyDecoder {
    pub decoder: Arc<dyn Decoder>,
}

impl PyDecoder {
    pub fn new(decoder: Arc<dyn Decoder>) -> Self {
        PyDecoder { decoder }
    }
}

#[typetag::serde]
impl Decoder for PyDecoder {
    fn decode(&self, tokens: Vec<String>) -> tk::Result<String> {
        self.decoder.decode(tokens)
    }
}

impl Serialize for PyDecoder {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.decoder.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyDecoder {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(PyDecoder {
            decoder: Arc::deserialize(deserializer)?,
        })
    }
}

#[pymethods]
impl PyDecoder {
    #[staticmethod]
    fn custom(decoder: PyObject) -> PyResult<Self> {
        let decoder = CustomDecoder::new(decoder).map(Arc::new)?;
        Ok(PyDecoder::new(decoder))
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let data = serde_json::to_string(&self.decoder).map_err(|e| {
            exceptions::Exception::py_err(format!(
                "Error while attempting to pickle Decoder: {}",
                e
            ))
        })?;
        Ok(PyBytes::new(py, data.as_bytes()).to_object(py))
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.decoder = serde_json::from_slice(s.as_bytes()).map_err(|e| {
                    exceptions::Exception::py_err(format!(
                        "Error while attempting to unpickle Decoder: {}",
                        e
                    ))
                })?;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn decode(&self, tokens: Vec<String>) -> PyResult<String> {
        ToPyResult(self.decoder.decode(tokens)).into()
    }
}

#[pyclass(extends=PyDecoder, module = "tokenizers.decoders", name=ByteLevel)]
pub struct PyByteLevelDec {}
#[pymethods]
impl PyByteLevelDec {
    #[new]
    fn new() -> PyResult<(Self, PyDecoder)> {
        Ok((
            PyByteLevelDec {},
            PyDecoder::new(Arc::new(ByteLevel::default())),
        ))
    }
}

#[pyclass(extends=PyDecoder, module = "tokenizers.decoders", name=WordPiece)]
pub struct PyWordPieceDec {}
#[pymethods]
impl PyWordPieceDec {
    #[new]
    #[args(kwargs = "**")]
    fn new(kwargs: Option<&PyDict>) -> PyResult<(Self, PyDecoder)> {
        let mut prefix = String::from("##");
        let mut cleanup = true;

        if let Some(kwargs) = kwargs {
            if let Some(p) = kwargs.get_item("prefix") {
                prefix = p.extract()?;
            }
            if let Some(c) = kwargs.get_item("cleanup") {
                cleanup = c.extract()?;
            }
        }

        Ok((
            PyWordPieceDec {},
            PyDecoder::new(Arc::new(WordPiece::new(prefix, cleanup))),
        ))
    }
}

#[pyclass(extends=PyDecoder, module = "tokenizers.decoders", name=Metaspace)]
pub struct PyMetaspaceDec {}
#[pymethods]
impl PyMetaspaceDec {
    #[new]
    #[args(kwargs = "**")]
    fn new(kwargs: Option<&PyDict>) -> PyResult<(Self, PyDecoder)> {
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
            PyMetaspaceDec {},
            PyDecoder::new(Arc::new(Metaspace::new(replacement, add_prefix_space))),
        ))
    }
}

#[pyclass(extends=PyDecoder, module = "tokenizers.decoders", name=BPEDecoder)]
pub struct PyBPEDecoder {}
#[pymethods]
impl PyBPEDecoder {
    #[new]
    #[args(kwargs = "**")]
    fn new(kwargs: Option<&PyDict>) -> PyResult<(Self, PyDecoder)> {
        let mut suffix = String::from("</w>");

        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "suffix" => suffix = value.extract()?,
                    _ => println!("Ignored unknown kwarg option {}", key),
                }
            }
        }

        Ok((
            PyBPEDecoder {},
            PyDecoder::new(Arc::new(BPEDecoder::new(suffix))),
        ))
    }
}

struct CustomDecoder {
    class: PyObject,
}

impl CustomDecoder {
    pub fn new(class: PyObject) -> PyResult<Self> {
        Ok(CustomDecoder { class })
    }
}

#[typetag::serde]
impl Decoder for CustomDecoder {
    fn decode(&self, tokens: Vec<String>) -> tk::Result<String> {
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

impl Serialize for CustomDecoder {
    fn serialize<S>(&self, _serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Err(serde::ser::Error::custom(
            "Custom PyDecoder cannot be serialized",
        ))
    }
}

impl<'de> Deserialize<'de> for CustomDecoder {
    fn deserialize<D>(_deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Err(D::Error::custom("PyDecoder cannot be deserialized"))
    }
}
