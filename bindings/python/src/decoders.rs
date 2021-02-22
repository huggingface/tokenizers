use std::sync::{Arc, RwLock};

use crate::utils::PyChar;
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

/// Base class for all decoders
///
/// This class is not supposed to be instantiated directly. Instead, any implementation of
/// a Decoder will return an instance of this class when instantiated.
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
            PyDecoderWrapper::Wrapped(inner) => match &*inner.as_ref().read().unwrap() {
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
        let decoder = PyDecoderWrapper::Custom(
            CustomDecoder::new(decoder).map(|d| Arc::new(RwLock::new(d)))?,
        );
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

    /// Decode the given list of tokens to a final string
    ///
    /// Args:
    ///     tokens (:obj:`List[str]`):
    ///         The list of tokens to decode
    ///
    /// Returns:
    ///     :obj:`str`: The decoded string
    #[text_signature = "(self, tokens)"]
    fn decode(&self, tokens: Vec<String>) -> PyResult<String> {
        ToPyResult(self.decoder.decode(tokens)).into()
    }
}

macro_rules! getter {
    ($self: ident, $variant: ident, $($name: tt)+) => {{
        let super_ = $self.as_ref();
        if let PyDecoderWrapper::Wrapped(ref wrap) = super_.decoder {
            if let DecoderWrapper::$variant(ref dec) = *wrap.read().unwrap() {
                dec.$($name)+
            } else {
                unreachable!()
            }
        } else {
            unreachable!()
        }
    }};
}

macro_rules! setter {
    ($self: ident, $variant: ident, $name: ident, $value: expr) => {{
        let super_ = $self.as_ref();
        if let PyDecoderWrapper::Wrapped(ref wrap) = super_.decoder {
            if let DecoderWrapper::$variant(ref mut dec) = *wrap.write().unwrap() {
                dec.$name = $value;
            }
        }
    }};
    ($self: ident, $variant: ident, @$name: ident, $value: expr) => {{
        let super_ = $self.as_ref();
        if let PyDecoderWrapper::Wrapped(ref wrap) = super_.decoder {
            if let DecoderWrapper::$variant(ref mut dec) = *wrap.write().unwrap() {
                dec.$name($value);
            }
        }
    }};
}

/// ByteLevel Decoder
///
/// This decoder is to be used in tandem with the :class:`~tokenizers.pre_tokenizers.ByteLevel`
/// :class:`~tokenizers.pre_tokenizers.PreTokenizer`.
#[pyclass(extends=PyDecoder, module = "tokenizers.decoders", name=ByteLevel)]
#[text_signature = "(self)"]
pub struct PyByteLevelDec {}
#[pymethods]
impl PyByteLevelDec {
    #[new]
    fn new() -> PyResult<(Self, PyDecoder)> {
        Ok((PyByteLevelDec {}, ByteLevel::default().into()))
    }
}

/// WordPiece Decoder
///
/// Args:
///     prefix (:obj:`str`, `optional`, defaults to :obj:`##`):
///         The prefix to use for subwords that are not a beginning-of-word
///
///     cleanup (:obj:`bool`, `optional`, defaults to :obj:`True`):
///         Whether to cleanup some tokenization artifacts. Mainly spaces before punctuation,
///         and some abbreviated english forms.
#[pyclass(extends=PyDecoder, module = "tokenizers.decoders", name=WordPiece)]
#[text_signature = "(self, prefix=\"##\", cleanup=True)"]
pub struct PyWordPieceDec {}
#[pymethods]
impl PyWordPieceDec {
    #[getter]
    fn get_prefix(self_: PyRef<Self>) -> String {
        getter!(self_, WordPiece, prefix.clone())
    }

    #[setter]
    fn set_prefix(self_: PyRef<Self>, prefix: String) {
        setter!(self_, WordPiece, prefix, prefix);
    }

    #[getter]
    fn get_cleanup(self_: PyRef<Self>) -> bool {
        getter!(self_, WordPiece, cleanup)
    }

    #[setter]
    fn set_cleanup(self_: PyRef<Self>, cleanup: bool) {
        setter!(self_, WordPiece, cleanup, cleanup);
    }

    #[new]
    #[args(prefix = "String::from(\"##\")", cleanup = "true")]
    fn new(prefix: String, cleanup: bool) -> PyResult<(Self, PyDecoder)> {
        Ok((PyWordPieceDec {}, WordPiece::new(prefix, cleanup).into()))
    }
}

/// Metaspace Decoder
///
/// Args:
///     replacement (:obj:`str`, `optional`, defaults to :obj:`▁`):
///         The replacement character. Must be exactly one character. By default we
///         use the `▁` (U+2581) meta symbol (Same as in SentencePiece).
///
///     add_prefix_space (:obj:`bool`, `optional`, defaults to :obj:`True`):
///         Whether to add a space to the first word if there isn't already one. This
///         lets us treat `hello` exactly like `say hello`.
#[pyclass(extends=PyDecoder, module = "tokenizers.decoders", name=Metaspace)]
#[text_signature = "(self, replacement = \"▁\", add_prefix_space = True)"]
pub struct PyMetaspaceDec {}
#[pymethods]
impl PyMetaspaceDec {
    #[getter]
    fn get_replacement(self_: PyRef<Self>) -> String {
        getter!(self_, Metaspace, get_replacement().to_string())
    }

    #[setter]
    fn set_replacement(self_: PyRef<Self>, replacement: PyChar) {
        setter!(self_, Metaspace, @set_replacement, replacement.0);
    }

    #[getter]
    fn get_add_prefix_space(self_: PyRef<Self>) -> bool {
        getter!(self_, Metaspace, add_prefix_space)
    }

    #[setter]
    fn set_add_prefix_space(self_: PyRef<Self>, add_prefix_space: bool) {
        setter!(self_, Metaspace, add_prefix_space, add_prefix_space);
    }

    #[new]
    #[args(replacement = "PyChar('▁')", add_prefix_space = "true")]
    fn new(replacement: PyChar, add_prefix_space: bool) -> PyResult<(Self, PyDecoder)> {
        Ok((
            PyMetaspaceDec {},
            Metaspace::new(replacement.0, add_prefix_space).into(),
        ))
    }
}

/// BPEDecoder Decoder
///
/// Args:
///     suffix (:obj:`str`, `optional`, defaults to :obj:`</w>`):
///         The suffix that was used to caracterize an end-of-word. This suffix will
///         be replaced by whitespaces during the decoding
#[pyclass(extends=PyDecoder, module = "tokenizers.decoders", name=BPEDecoder)]
#[text_signature = "(self, suffix=\"</w>\")"]
pub struct PyBPEDecoder {}
#[pymethods]
impl PyBPEDecoder {
    #[getter]
    fn get_suffix(self_: PyRef<Self>) -> String {
        getter!(self_, BPE, suffix.clone())
    }

    #[setter]
    fn set_suffix(self_: PyRef<Self>, suffix: String) {
        setter!(self_, BPE, suffix, suffix);
    }

    #[new]
    #[args(suffix = "String::from(\"</w>\")")]
    fn new(suffix: String) -> PyResult<(Self, PyDecoder)> {
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
    Custom(Arc<RwLock<CustomDecoder>>),
    Wrapped(Arc<RwLock<DecoderWrapper>>),
}

impl<I> From<I> for PyDecoderWrapper
where
    I: Into<DecoderWrapper>,
{
    fn from(norm: I) -> Self {
        PyDecoderWrapper::Wrapped(Arc::new(RwLock::new(norm.into())))
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
            PyDecoderWrapper::Wrapped(inner) => inner.read().unwrap().decode(tokens),
            PyDecoderWrapper::Custom(inner) => inner.read().unwrap().decode(tokens),
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::{Arc, RwLock};

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
            PyDecoderWrapper::Wrapped(msp) => match *msp.as_ref().read().unwrap() {
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
        let py_seq =
            PyDecoderWrapper::Custom(Arc::new(RwLock::new(CustomDecoder::new(obj).unwrap())));
        assert!(serde_json::to_string(&py_seq).is_err());
    }
}
