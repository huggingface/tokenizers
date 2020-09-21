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
use tk::decoders::DecoderWrapper;
use tk::Decoder;
use tokenizers as tk;

use super::error::ToPyResult;

#[pyclass(dict, module = "tokenizers.decoders", name=Decoder)]
#[derive(Clone, Deserialize, Serialize)]
pub struct PyDecoder {
    #[serde(flatten)]
    pub(crate) decoder: PyDecoderWrapper,
}

impl PyDecoder {
    pub(crate) fn new(decoder: PyDecoderWrapper) -> Self {
        PyDecoder { decoder }
    }

    pub(crate) fn get_as_subtype(&self) -> PyResult<PyObject> {
        let base = self.clone();
        let gil = Python::acquire_gil();
        let py = gil.python();
        Ok(match &self.decoder {
            PyDecoderWrapper::Custom(_) => Py::new(py, base)?.into_py(py),
            PyDecoderWrapper::Wrapped(inner) => match inner.as_ref() {
                DecoderWrapper::Metaspace(_) => Py::new(py, (PyMetaspaceDec {}, base))?.into_py(py),
                DecoderWrapper::WordPiece(_) => Py::new(py, (PyWordPieceDec {}, base))?.into_py(py),
                DecoderWrapper::ByteLevel(_) => Py::new(py, (PyByteLevelDec {}, base))?.into_py(py),
                DecoderWrapper::BPE(_) => Py::new(py, (PyBPEDecoder {}, base))?.into_py(py),
            },
        })
    }
}

impl Decoder for PyDecoder {
    fn decode(&self, tokens: Vec<String>) -> tk::Result<String> {
        self.decoder.decode(tokens)
    }
}

#[pymethods]
impl PyDecoder {
    #[staticmethod]
    fn custom(decoder: PyObject) -> PyResult<Self> {
        let decoder = PyDecoderWrapper::Custom(CustomDecoder::new(decoder).map(Arc::new)?);
        Ok(PyDecoder::new(decoder))
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let data = serde_json::to_string(&self.decoder).map_err(|e| {
            exceptions::PyException::new_err(format!(
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
                    exceptions::PyException::new_err(format!(
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
        Ok((PyByteLevelDec {}, ByteLevel::default().into()))
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

        Ok((PyWordPieceDec {}, WordPiece::new(prefix, cleanup).into()))
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
                        replacement = s.chars().next().ok_or_else(|| {
                            exceptions::PyValueError::new_err("replacement must be a character")
                        })?;
                    }
                    "add_prefix_space" => add_prefix_space = value.extract()?,
                    _ => println!("Ignored unknown kwarg option {}", key),
                }
            }
        }

        Ok((
            PyMetaspaceDec {},
            Metaspace::new(replacement, add_prefix_space).into(),
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

        Ok((PyBPEDecoder {}, BPEDecoder::new(suffix).into()))
    }
}

#[derive(Clone)]
pub(crate) struct CustomDecoder {
    inner: PyObject,
}

impl CustomDecoder {
    pub(crate) fn new(inner: PyObject) -> PyResult<Self> {
        Ok(CustomDecoder { inner })
    }
}

impl Decoder for CustomDecoder {
    fn decode(&self, tokens: Vec<String>) -> tk::Result<String> {
        Python::with_gil(|py| {
            let decoded = self
                .inner
                .call_method(py, "decode", (tokens,), None)?
                .extract::<String>(py)?;
            Ok(decoded)
        })
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

#[derive(Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub(crate) enum PyDecoderWrapper {
    Custom(Arc<CustomDecoder>),
    Wrapped(Arc<DecoderWrapper>),
}

impl<I> From<I> for PyDecoderWrapper
where
    I: Into<DecoderWrapper>,
{
    fn from(norm: I) -> Self {
        PyDecoderWrapper::Wrapped(Arc::new(norm.into()))
    }
}

impl<I> From<I> for PyDecoder
where
    I: Into<DecoderWrapper>,
{
    fn from(dec: I) -> Self {
        PyDecoder {
            decoder: dec.into().into(),
        }
    }
}

impl Decoder for PyDecoderWrapper {
    fn decode(&self, tokens: Vec<String>) -> tk::Result<String> {
        match self {
            PyDecoderWrapper::Wrapped(inner) => inner.decode(tokens),
            PyDecoderWrapper::Custom(inner) => inner.decode(tokens),
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use pyo3::prelude::*;
    use tk::decoders::metaspace::Metaspace;
    use tk::decoders::DecoderWrapper;

    use crate::decoders::{CustomDecoder, PyDecoder, PyDecoderWrapper};

    #[test]
    fn get_subtype() {
        let py_dec = PyDecoder::new(Metaspace::default().into());
        let py_meta = py_dec.get_as_subtype().unwrap();
        let gil = Python::acquire_gil();
        assert_eq!(
            "tokenizers.decoders.Metaspace",
            py_meta.as_ref(gil.python()).get_type().name()
        );
    }

    #[test]
    fn serialize() {
        let py_wrapped: PyDecoderWrapper = Metaspace::default().into();
        let py_ser = serde_json::to_string(&py_wrapped).unwrap();
        let rs_wrapped = DecoderWrapper::Metaspace(Metaspace::default());
        let rs_ser = serde_json::to_string(&rs_wrapped).unwrap();
        assert_eq!(py_ser, rs_ser);
        let py_dec: PyDecoder = serde_json::from_str(&rs_ser).unwrap();
        match py_dec.decoder {
            PyDecoderWrapper::Wrapped(msp) => match msp.as_ref() {
                DecoderWrapper::Metaspace(_) => {}
                _ => panic!("Expected Metaspace"),
            },
            _ => panic!("Expected wrapped, not custom."),
        }

        let obj = Python::with_gil(|py| {
            let py_msp = PyDecoder::new(Metaspace::default().into());
            let obj: PyObject = Py::new(py, py_msp).unwrap().into_py(py);
            obj
        });
        let py_seq = PyDecoderWrapper::Custom(Arc::new(CustomDecoder::new(obj).unwrap()));
        assert!(serde_json::to_string(&py_seq).is_err());
    }
}
