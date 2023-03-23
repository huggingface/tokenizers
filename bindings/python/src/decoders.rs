use std::sync::{Arc, RwLock};

use crate::utils::PyChar;
use crate::utils::PyPattern;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use serde::de::Error;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tk::decoders::bpe::BPEDecoder;
use tk::decoders::byte_fallback::ByteFallback;
use tk::decoders::byte_level::ByteLevel;
use tk::decoders::ctc::CTC;
use tk::decoders::fuse::Fuse;
use tk::decoders::metaspace::Metaspace;
use tk::decoders::sequence::Sequence;
use tk::decoders::strip::Strip;
use tk::decoders::wordpiece::WordPiece;
use tk::decoders::DecoderWrapper;
use tk::normalizers::replace::Replace;
use tk::Decoder;
use tokenizers as tk;

use super::error::ToPyResult;

/// Base class for all decoders
///
/// This class is not supposed to be instantiated directly. Instead, any implementation of
/// a Decoder will return an instance of this class when instantiated.
#[pyclass(dict, module = "tokenizers.decoders", name = "Decoder", subclass)]
#[derive(Clone, Deserialize, Serialize)]
pub struct PyDecoder {
    #[serde(flatten)]
    pub(crate) decoder: PyDecoderWrapper,
}

impl PyDecoder {
    pub(crate) fn new(decoder: PyDecoderWrapper) -> Self {
        PyDecoder { decoder }
    }

    pub(crate) fn get_as_subtype(&self, py: Python<'_>) -> PyResult<PyObject> {
        let base = self.clone();
        Ok(match &self.decoder {
            PyDecoderWrapper::Custom(_) => Py::new(py, base)?.into_py(py),
            PyDecoderWrapper::Wrapped(inner) => match &*inner.as_ref().read().unwrap() {
                DecoderWrapper::Metaspace(_) => Py::new(py, (PyMetaspaceDec {}, base))?.into_py(py),
                DecoderWrapper::WordPiece(_) => Py::new(py, (PyWordPieceDec {}, base))?.into_py(py),
                DecoderWrapper::ByteFallback(_) => {
                    Py::new(py, (PyByteFallbackDec {}, base))?.into_py(py)
                }
                DecoderWrapper::Strip(_) => Py::new(py, (PyStrip {}, base))?.into_py(py),
                DecoderWrapper::Fuse(_) => Py::new(py, (PyFuseDec {}, base))?.into_py(py),
                DecoderWrapper::ByteLevel(_) => Py::new(py, (PyByteLevelDec {}, base))?.into_py(py),
                DecoderWrapper::Replace(_) => Py::new(py, (PyReplaceDec {}, base))?.into_py(py),
                DecoderWrapper::BPE(_) => Py::new(py, (PyBPEDecoder {}, base))?.into_py(py),
                DecoderWrapper::CTC(_) => Py::new(py, (PyCTCDecoder {}, base))?.into_py(py),
                DecoderWrapper::Sequence(_) => {
                    Py::new(py, (PySequenceDecoder {}, base))?.into_py(py)
                }
            },
        })
    }
}

impl Decoder for PyDecoder {
    fn decode_chain(&self, tokens: Vec<String>) -> tk::Result<Vec<String>> {
        self.decoder.decode_chain(tokens)
    }
}

#[pymethods]
impl PyDecoder {
    #[staticmethod]
    fn custom(decoder: PyObject) -> Self {
        let decoder = PyDecoderWrapper::Custom(Arc::new(RwLock::new(CustomDecoder::new(decoder))));
        PyDecoder::new(decoder)
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
    #[pyo3(text_signature = "(self, tokens)")]
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
#[pyclass(extends=PyDecoder, module = "tokenizers.decoders", name = "ByteLevel")]
#[pyo3(text_signature = "(self)")]
pub struct PyByteLevelDec {}
#[pymethods]
impl PyByteLevelDec {
    #[new]
    #[pyo3(signature = (**_kwargs))]
    fn new(_kwargs: Option<&PyDict>) -> (Self, PyDecoder) {
        (PyByteLevelDec {}, ByteLevel::default().into())
    }
}

/// Replace Decoder
///
/// This decoder is to be used in tandem with the :class:`~tokenizers.pre_tokenizers.Replace`
/// :class:`~tokenizers.pre_tokenizers.PreTokenizer`.
#[pyclass(extends=PyDecoder, module = "tokenizers.decoders", name = "Replace")]
#[pyo3(text_signature = "(self, pattern, content)")]
pub struct PyReplaceDec {}
#[pymethods]
impl PyReplaceDec {
    #[new]
    fn new(pattern: PyPattern, content: String) -> PyResult<(Self, PyDecoder)> {
        Ok((
            PyReplaceDec {},
            ToPyResult(Replace::new(pattern, content)).into_py()?.into(),
        ))
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
#[pyclass(extends=PyDecoder, module = "tokenizers.decoders", name = "WordPiece")]
#[pyo3(text_signature = "(self, prefix=\"##\", cleanup=True)")]
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
    #[pyo3(signature = (prefix = String::from("##"), cleanup = true))]
    fn new(prefix: String, cleanup: bool) -> (Self, PyDecoder) {
        (PyWordPieceDec {}, WordPiece::new(prefix, cleanup).into())
    }
}

/// ByteFallback Decoder
/// ByteFallback is a simple trick which converts tokens looking like `<0x61>`
/// to pure bytes, and attempts to make them into a string. If the tokens
/// cannot be decoded you will get � instead for each inconvertable byte token
///
#[pyclass(extends=PyDecoder, module = "tokenizers.decoders", name = "ByteFallback")]
#[pyo3(text_signature = "(self)")]
pub struct PyByteFallbackDec {}
#[pymethods]
impl PyByteFallbackDec {
    #[new]
    #[pyo3(signature = ())]
    fn new() -> (Self, PyDecoder) {
        (PyByteFallbackDec {}, ByteFallback::new().into())
    }
}

/// Fuse Decoder
/// Fuse simply fuses every token into a single string.
/// This is the last step of decoding, this decoder exists only if
/// there is need to add other decoders *after* the fusion
#[pyclass(extends=PyDecoder, module = "tokenizers.decoders", name = "Fuse")]
#[pyo3(text_signature = "(self)")]
pub struct PyFuseDec {}
#[pymethods]
impl PyFuseDec {
    #[new]
    #[pyo3(signature = ())]
    fn new() -> (Self, PyDecoder) {
        (PyFuseDec {}, Fuse::new().into())
    }
}

/// Strip normalizer
/// Strips n left characters of each token, or n right characters of each token
#[pyclass(extends=PyDecoder, module = "tokenizers.decoders", name = "Strip")]
#[pyo3(text_signature = "(self, left=0, right=0)")]
pub struct PyStrip {}
#[pymethods]
impl PyStrip {
    #[getter]
    fn get_left(self_: PyRef<Self>) -> usize {
        getter!(self_, Strip, left)
    }

    #[setter]
    fn set_left(self_: PyRef<Self>, left: usize) {
        setter!(self_, Strip, left, left)
    }

    #[getter]
    fn get_right(self_: PyRef<Self>) -> usize {
        getter!(self_, Strip, right)
    }

    #[setter]
    fn set_right(self_: PyRef<Self>, right: usize) {
        setter!(self_, Strip, right, right)
    }

    #[new]
    #[pyo3(signature = (left=0, right=0))]
    fn new(left: usize, right: usize) -> (Self, PyDecoder) {
        (PyStrip {}, Strip::new(left, right).into())
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
#[pyclass(extends=PyDecoder, module = "tokenizers.decoders", name = "Metaspace")]
#[pyo3(text_signature = "(self, replacement = \"▁\", add_prefix_space = True)")]
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
    #[pyo3(signature = (replacement = PyChar('▁'), add_prefix_space = true))]
    fn new(replacement: PyChar, add_prefix_space: bool) -> (Self, PyDecoder) {
        (
            PyMetaspaceDec {},
            Metaspace::new(replacement.0, add_prefix_space).into(),
        )
    }
}

/// BPEDecoder Decoder
///
/// Args:
///     suffix (:obj:`str`, `optional`, defaults to :obj:`</w>`):
///         The suffix that was used to caracterize an end-of-word. This suffix will
///         be replaced by whitespaces during the decoding
#[pyclass(extends=PyDecoder, module = "tokenizers.decoders", name = "BPEDecoder")]
#[pyo3(text_signature = "(self, suffix=\"</w>\")")]
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
    #[pyo3(signature = (suffix = String::from("</w>")))]
    fn new(suffix: String) -> (Self, PyDecoder) {
        (PyBPEDecoder {}, BPEDecoder::new(suffix).into())
    }
}

/// CTC Decoder
///
/// Args:
///     pad_token (:obj:`str`, `optional`, defaults to :obj:`<pad>`):
///         The pad token used by CTC to delimit a new token.
///     word_delimiter_token (:obj:`str`, `optional`, defaults to :obj:`|`):
///         The word delimiter token. It will be replaced by a <space>
///     cleanup (:obj:`bool`, `optional`, defaults to :obj:`True`):
///         Whether to cleanup some tokenization artifacts.
///         Mainly spaces before punctuation, and some abbreviated english forms.
#[pyclass(extends=PyDecoder, module = "tokenizers.decoders", name = "CTC")]
#[pyo3(text_signature = "(self, pad_token=\"<pad>\", word_delimiter_token=\"|\", cleanup=True)")]
pub struct PyCTCDecoder {}
#[pymethods]
impl PyCTCDecoder {
    #[getter]
    fn get_pad_token(self_: PyRef<Self>) -> String {
        getter!(self_, CTC, pad_token.clone())
    }

    #[setter]
    fn set_pad_token(self_: PyRef<Self>, pad_token: String) {
        setter!(self_, CTC, pad_token, pad_token);
    }

    #[getter]
    fn get_word_delimiter_token(self_: PyRef<Self>) -> String {
        getter!(self_, CTC, word_delimiter_token.clone())
    }

    #[setter]
    fn set_word_delimiter_token(self_: PyRef<Self>, word_delimiter_token: String) {
        setter!(self_, CTC, word_delimiter_token, word_delimiter_token);
    }

    #[getter]
    fn get_cleanup(self_: PyRef<Self>) -> bool {
        getter!(self_, CTC, cleanup)
    }

    #[setter]
    fn set_cleanup(self_: PyRef<Self>, cleanup: bool) {
        setter!(self_, CTC, cleanup, cleanup);
    }

    #[new]
    #[pyo3(signature = (
        pad_token = String::from("<pad>"),
        word_delimiter_token = String::from("|"),
        cleanup = true
    ))]
    fn new(pad_token: String, word_delimiter_token: String, cleanup: bool) -> (Self, PyDecoder) {
        (
            PyCTCDecoder {},
            CTC::new(pad_token, word_delimiter_token, cleanup).into(),
        )
    }
}

/// Sequence Decoder
///
/// Args:
///     decoders (:obj:`List[Decoder]`)
///         The decoders that need to be chained
#[pyclass(extends=PyDecoder, module = "tokenizers.decoders", name="Sequence")]
#[pyo3(text_signature = "(self, decoders)")]
pub struct PySequenceDecoder {}
#[pymethods]
impl PySequenceDecoder {
    #[new]
    #[pyo3(signature = (decoders_py))]
    fn new(decoders_py: &PyList) -> PyResult<(Self, PyDecoder)> {
        let mut decoders: Vec<DecoderWrapper> = Vec::with_capacity(decoders_py.len());
        for decoder_py in decoders_py.iter() {
            let decoder: PyRef<PyDecoder> = decoder_py.extract()?;
            let decoder = match &decoder.decoder {
                PyDecoderWrapper::Wrapped(inner) => inner,
                PyDecoderWrapper::Custom(_) => unimplemented!(),
            };
            decoders.push(decoder.read().unwrap().clone());
        }
        Ok((PySequenceDecoder {}, Sequence::new(decoders).into()))
    }

    fn __getnewargs__<'p>(&self, py: Python<'p>) -> &'p PyTuple {
        PyTuple::new(py, [PyList::empty(py)])
    }
}

#[derive(Clone)]
pub(crate) struct CustomDecoder {
    inner: PyObject,
}

impl CustomDecoder {
    pub(crate) fn new(inner: PyObject) -> Self {
        CustomDecoder { inner }
    }
}

impl Decoder for CustomDecoder {
    fn decode(&self, tokens: Vec<String>) -> tk::Result<String> {
        Python::with_gil(|py| {
            let decoded = self
                .inner
                .call_method(py, "decode", (tokens,), None)?
                .extract(py)?;
            Ok(decoded)
        })
    }

    fn decode_chain(&self, tokens: Vec<String>) -> tk::Result<Vec<String>> {
        Python::with_gil(|py| {
            let decoded = self
                .inner
                .call_method(py, "decode_chain", (tokens,), None)?
                .extract(py)?;
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
    fn decode_chain(&self, tokens: Vec<String>) -> tk::Result<Vec<String>> {
        match self {
            PyDecoderWrapper::Wrapped(inner) => inner.read().unwrap().decode_chain(tokens),
            PyDecoderWrapper::Custom(inner) => inner.read().unwrap().decode_chain(tokens),
        }
    }
}

/// Decoders Module
#[pymodule]
pub fn decoders(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyDecoder>()?;
    m.add_class::<PyByteLevelDec>()?;
    m.add_class::<PyReplaceDec>()?;
    m.add_class::<PyWordPieceDec>()?;
    m.add_class::<PyByteFallbackDec>()?;
    m.add_class::<PyFuseDec>()?;
    m.add_class::<PyStrip>()?;
    m.add_class::<PyMetaspaceDec>()?;
    m.add_class::<PyBPEDecoder>()?;
    m.add_class::<PyCTCDecoder>()?;
    m.add_class::<PySequenceDecoder>()?;
    Ok(())
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
        Python::with_gil(|py| {
            let py_dec = PyDecoder::new(Metaspace::default().into());
            let py_meta = py_dec.get_as_subtype(py).unwrap();
            assert_eq!("Metaspace", py_meta.as_ref(py).get_type().name().unwrap());
        })
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
        let py_seq = PyDecoderWrapper::Custom(Arc::new(RwLock::new(CustomDecoder::new(obj))));
        assert!(serde_json::to_string(&py_seq).is_err());
    }
}
