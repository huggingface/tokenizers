use std::sync::{Arc, RwLock};

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use pyo3::PySequenceProtocol;

use crate::error::ToPyResult;
use crate::utils::{PyNormalizedString, PyNormalizedStringRefMut, PyPattern};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tk::normalizers::{
    BertNormalizer, Lowercase, Nmt, NormalizerWrapper, Precompiled, Replace, Strip, StripAccents,
    NFC, NFD, NFKC, NFKD,
};
use tk::{NormalizedString, Normalizer};
use tokenizers as tk;

/// Base class for all normalizers
///
/// This class is not supposed to be instantiated directly. Instead, any implementation of a
/// Normalizer will return an instance of this class when instantiated.
#[pyclass(dict, module = "tokenizers.normalizers", name=Normalizer)]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyNormalizer {
    #[serde(flatten)]
    pub(crate) normalizer: PyNormalizerTypeWrapper,
}

impl PyNormalizer {
    pub(crate) fn new(normalizer: PyNormalizerTypeWrapper) -> Self {
        PyNormalizer { normalizer }
    }
    pub(crate) fn get_as_subtype(&self) -> PyResult<PyObject> {
        let base = self.clone();
        let gil = Python::acquire_gil();
        let py = gil.python();
        Ok(match self.normalizer {
            PyNormalizerTypeWrapper::Sequence(_) => Py::new(py, (PySequence {}, base))?.into_py(py),
            PyNormalizerTypeWrapper::Single(ref inner) => match &*inner.as_ref().read().unwrap() {
                PyNormalizerWrapper::Custom(_) => Py::new(py, base)?.into_py(py),
                PyNormalizerWrapper::Wrapped(ref inner) => match inner {
                    NormalizerWrapper::Sequence(_) => {
                        Py::new(py, (PySequence {}, base))?.into_py(py)
                    }
                    NormalizerWrapper::BertNormalizer(_) => {
                        Py::new(py, (PyBertNormalizer {}, base))?.into_py(py)
                    }
                    NormalizerWrapper::StripNormalizer(_) => {
                        Py::new(py, (PyBertNormalizer {}, base))?.into_py(py)
                    }
                    NormalizerWrapper::StripAccents(_) => {
                        Py::new(py, (PyStripAccents {}, base))?.into_py(py)
                    }
                    NormalizerWrapper::NFC(_) => Py::new(py, (PyNFC {}, base))?.into_py(py),
                    NormalizerWrapper::NFD(_) => Py::new(py, (PyNFD {}, base))?.into_py(py),
                    NormalizerWrapper::NFKC(_) => Py::new(py, (PyNFKC {}, base))?.into_py(py),
                    NormalizerWrapper::NFKD(_) => Py::new(py, (PyNFKD {}, base))?.into_py(py),
                    NormalizerWrapper::Lowercase(_) => {
                        Py::new(py, (PyLowercase {}, base))?.into_py(py)
                    }
                    NormalizerWrapper::Precompiled(_) => {
                        Py::new(py, (PyPrecompiled {}, base))?.into_py(py)
                    }
                    NormalizerWrapper::Replace(_) => Py::new(py, (PyReplace {}, base))?.into_py(py),
                    NormalizerWrapper::Nmt(_) => Py::new(py, (PyNmt {}, base))?.into_py(py),
                },
            },
        })
    }
}

impl Normalizer for PyNormalizer {
    fn normalize(&self, normalized: &mut NormalizedString) -> tk::Result<()> {
        self.normalizer.normalize(normalized)
    }
}

#[pymethods]
impl PyNormalizer {
    #[staticmethod]
    fn custom(obj: PyObject) -> PyResult<Self> {
        Ok(Self {
            normalizer: PyNormalizerWrapper::Custom(CustomNormalizer::new(obj)).into(),
        })
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let data = serde_json::to_string(&self.normalizer).map_err(|e| {
            exceptions::PyException::new_err(format!(
                "Error while attempting to pickle Normalizer: {}",
                e
            ))
        })?;
        Ok(PyBytes::new(py, data.as_bytes()).to_object(py))
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.normalizer = serde_json::from_slice(s.as_bytes()).map_err(|e| {
                    exceptions::PyException::new_err(format!(
                        "Error while attempting to unpickle Normalizer: {}",
                        e
                    ))
                })?;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Normalize a :class:`~tokenizers.NormalizedString` in-place
    ///
    /// This method allows to modify a :class:`~tokenizers.NormalizedString` to
    /// keep track of the alignment information. If you just want to see the result
    /// of the normalization on a raw string, you can use
    /// :meth:`~tokenizers.normalizers.Normalizer.normalize_str`
    ///
    /// Args:
    ///     normalized (:class:`~tokenizers.NormalizedString`):
    ///         The normalized string on which to apply this
    ///         :class:`~tokenizers.normalizers.Normalizer`
    #[text_signature = "(self, normalized)"]
    fn normalize(&self, normalized: &mut PyNormalizedString) -> PyResult<()> {
        ToPyResult(self.normalizer.normalize(&mut normalized.normalized)).into()
    }

    /// Normalize the given string
    ///
    /// This method provides a way to visualize the effect of a
    /// :class:`~tokenizers.normalizers.Normalizer` but it does not keep track of the alignment
    /// information. If you need to get/convert offsets, you can use
    /// :meth:`~tokenizers.normalizers.Normalizer.normalize`
    ///
    /// Args:
    ///     sequence (:obj:`str`):
    ///         A string to normalize
    ///
    /// Returns:
    ///     :obj:`str`: A string after normalization
    #[text_signature = "(self, sequence)"]
    fn normalize_str(&self, sequence: &str) -> PyResult<String> {
        let mut normalized = NormalizedString::from(sequence);
        ToPyResult(self.normalizer.normalize(&mut normalized)).into_py()?;
        Ok(normalized.get().to_owned())
    }
}

macro_rules! getter {
    ($self: ident, $variant: ident, $name: ident) => {{
        let super_ = $self.as_ref();
        if let PyNormalizerTypeWrapper::Single(ref norm) = super_.normalizer {
            let wrapper = norm.read().unwrap();
            if let PyNormalizerWrapper::Wrapped(NormalizerWrapper::$variant(o)) = *wrapper {
                o.$name
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
        if let PyNormalizerTypeWrapper::Single(ref norm) = super_.normalizer {
            let mut wrapper = norm.write().unwrap();
            if let PyNormalizerWrapper::Wrapped(NormalizerWrapper::$variant(ref mut o)) = *wrapper {
                o.$name = $value;
            }
        }
    }};
}

/// BertNormalizer
///
/// Takes care of normalizing raw text before giving it to a Bert model.
/// This includes cleaning the text, handling accents, chinese chars and lowercasing
///
/// Args:
///     clean_text (:obj:`bool`, `optional`, defaults to :obj:`True`):
///         Whether to clean the text, by removing any control characters
///         and replacing all whitespaces by the classic one.
///
///     handle_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
///         Whether to handle chinese chars by putting spaces around them.
///
///     strip_accents (:obj:`bool`, `optional`):
///         Whether to strip all accents. If this option is not specified (ie == None),
///         then it will be determined by the value for `lowercase` (as in the original Bert).
///
///     lowercase (:obj:`bool`, `optional`, defaults to :obj:`True`):
///         Whether to lowercase.
#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=BertNormalizer)]
#[text_signature = "(self, clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=True)"]
pub struct PyBertNormalizer {}
#[pymethods]
impl PyBertNormalizer {
    #[getter]
    fn get_clean_text(self_: PyRef<Self>) -> bool {
        getter!(self_, BertNormalizer, clean_text)
    }

    #[setter]
    fn set_clean_text(self_: PyRef<Self>, clean_text: bool) {
        setter!(self_, BertNormalizer, clean_text, clean_text);
    }

    #[getter]
    fn get_handle_chinese_chars(self_: PyRef<Self>) -> bool {
        getter!(self_, BertNormalizer, handle_chinese_chars)
    }

    #[setter]
    fn set_handle_chinese_chars(self_: PyRef<Self>, handle_chinese_chars: bool) {
        setter!(
            self_,
            BertNormalizer,
            handle_chinese_chars,
            handle_chinese_chars
        );
    }

    #[getter]
    fn get_strip_accents(self_: PyRef<Self>) -> Option<bool> {
        getter!(self_, BertNormalizer, strip_accents)
    }

    #[setter]
    fn set_strip_accents(self_: PyRef<Self>, strip_accents: Option<bool>) {
        setter!(self_, BertNormalizer, strip_accents, strip_accents);
    }

    #[getter]
    fn get_lowercase(self_: PyRef<Self>) -> bool {
        getter!(self_, BertNormalizer, lowercase)
    }

    #[setter]
    fn set_lowercase(self_: PyRef<Self>, lowercase: bool) {
        setter!(self_, BertNormalizer, lowercase, lowercase)
    }

    #[new]
    #[args(
        clean_text = "true",
        handle_chinese_chars = "true",
        strip_accents = "None",
        lowercase = "true"
    )]
    fn new(
        clean_text: bool,
        handle_chinese_chars: bool,
        strip_accents: Option<bool>,
        lowercase: bool,
    ) -> PyResult<(Self, PyNormalizer)> {
        let normalizer =
            BertNormalizer::new(clean_text, handle_chinese_chars, strip_accents, lowercase);
        Ok((PyBertNormalizer {}, normalizer.into()))
    }
}

/// NFD Unicode Normalizer
#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=NFD)]
#[text_signature = "(self)"]
pub struct PyNFD {}
#[pymethods]
impl PyNFD {
    #[new]
    fn new() -> PyResult<(Self, PyNormalizer)> {
        Ok((PyNFD {}, PyNormalizer::new(NFD.into())))
    }
}

/// NFKD Unicode Normalizer
#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=NFKD)]
#[text_signature = "(self)"]
pub struct PyNFKD {}
#[pymethods]
impl PyNFKD {
    #[new]
    fn new() -> PyResult<(Self, PyNormalizer)> {
        Ok((PyNFKD {}, NFKD.into()))
    }
}

/// NFC Unicode Normalizer
#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=NFC)]
#[text_signature = "(self)"]
pub struct PyNFC {}
#[pymethods]
impl PyNFC {
    #[new]
    fn new() -> PyResult<(Self, PyNormalizer)> {
        Ok((PyNFC {}, NFC.into()))
    }
}

/// NFKC Unicode Normalizer
#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=NFKC)]
#[text_signature = "(self)"]
pub struct PyNFKC {}
#[pymethods]
impl PyNFKC {
    #[new]
    fn new() -> PyResult<(Self, PyNormalizer)> {
        Ok((PyNFKC {}, NFKC.into()))
    }
}

/// Allows concatenating multiple other Normalizer as a Sequence.
/// All the normalizers run in sequence in the given order
///
/// Args:
///     normalizers (:obj:`List[Normalizer]`):
///         A list of Normalizer to be run as a sequence
#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=Sequence)]
pub struct PySequence {}
#[pymethods]
impl PySequence {
    #[new]
    fn new(normalizers: &PyList) -> PyResult<(Self, PyNormalizer)> {
        let mut sequence = Vec::with_capacity(normalizers.len());
        for n in normalizers.iter() {
            let normalizer: PyRef<PyNormalizer> = n.extract()?;
            match &normalizer.normalizer {
                PyNormalizerTypeWrapper::Sequence(inner) => sequence.extend(inner.iter().cloned()),
                PyNormalizerTypeWrapper::Single(inner) => sequence.push(inner.clone()),
            }
        }
        Ok((
            PySequence {},
            PyNormalizer::new(PyNormalizerTypeWrapper::Sequence(sequence)),
        ))
    }

    fn __getnewargs__<'p>(&self, py: Python<'p>) -> PyResult<&'p PyTuple> {
        Ok(PyTuple::new(py, &[PyList::empty(py)]))
    }
}

#[pyproto]
impl PySequenceProtocol for PySequence {
    fn __len__(&self) -> usize {
        0
    }
}

/// Lowercase Normalizer
#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=Lowercase)]
#[text_signature = "(self)"]
pub struct PyLowercase {}
#[pymethods]
impl PyLowercase {
    #[new]
    fn new() -> PyResult<(Self, PyNormalizer)> {
        Ok((PyLowercase {}, Lowercase.into()))
    }
}

/// Strip normalizer
#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=Strip)]
#[text_signature = "(self, left=True, right=True)"]
pub struct PyStrip {}
#[pymethods]
impl PyStrip {
    #[getter]
    fn get_left(self_: PyRef<Self>) -> bool {
        getter!(self_, StripNormalizer, strip_left)
    }

    #[setter]
    fn set_left(self_: PyRef<Self>, left: bool) {
        setter!(self_, StripNormalizer, strip_left, left)
    }

    #[getter]
    fn get_right(self_: PyRef<Self>) -> bool {
        getter!(self_, StripNormalizer, strip_right)
    }

    #[setter]
    fn set_right(self_: PyRef<Self>, right: bool) {
        setter!(self_, StripNormalizer, strip_right, right)
    }

    #[new]
    #[args(left = "true", right = "true")]
    fn new(left: bool, right: bool) -> PyResult<(Self, PyNormalizer)> {
        Ok((PyStrip {}, Strip::new(left, right).into()))
    }
}

/// StripAccents normalizer
#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=StripAccents)]
#[text_signature = "(self)"]
pub struct PyStripAccents {}
#[pymethods]
impl PyStripAccents {
    #[new]
    fn new() -> PyResult<(Self, PyNormalizer)> {
        Ok((PyStripAccents {}, StripAccents.into()))
    }
}

/// Nmt normalizer
#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=Nmt)]
#[text_signature = "(self)"]
pub struct PyNmt {}
#[pymethods]
impl PyNmt {
    #[new]
    fn new() -> PyResult<(Self, PyNormalizer)> {
        Ok((PyNmt {}, Nmt.into()))
    }
}

/// Precompiled normalizer
/// Don't use manually it is used for compatiblity for SentencePiece.
#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=Precompiled)]
#[text_signature = "(self, precompiled_charsmap)"]
pub struct PyPrecompiled {}
#[pymethods]
impl PyPrecompiled {
    #[new]
    fn new(py_precompiled_charsmap: &PyBytes) -> PyResult<(Self, PyNormalizer)> {
        let precompiled_charsmap: &[u8] = FromPyObject::extract(py_precompiled_charsmap)?;
        Ok((
            PyPrecompiled {},
            Precompiled::from(precompiled_charsmap)
                .map_err(|e| {
                    exceptions::PyException::new_err(format!(
                        "Error while attempting to build Precompiled normalizer: {}",
                        e
                    ))
                })?
                .into(),
        ))
    }
}

/// Replace normalizer
#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name=Replace)]
#[text_signature = "(self, pattern, content)"]
pub struct PyReplace {}
#[pymethods]
impl PyReplace {
    #[new]
    fn new(pattern: PyPattern, content: String) -> PyResult<(Self, PyNormalizer)> {
        Ok((
            PyReplace {},
            ToPyResult(Replace::new(pattern, content)).into_py()?.into(),
        ))
    }
}

#[derive(Clone)]
pub(crate) struct CustomNormalizer {
    inner: PyObject,
}
impl CustomNormalizer {
    pub fn new(inner: PyObject) -> Self {
        Self { inner }
    }
}

impl tk::tokenizer::Normalizer for CustomNormalizer {
    fn normalize(&self, normalized: &mut NormalizedString) -> tk::Result<()> {
        Python::with_gil(|py| {
            let normalized = PyNormalizedStringRefMut::new(normalized);
            let py_normalized = self.inner.as_ref(py);
            py_normalized.call_method("normalize", (normalized.get(),), None)?;
            Ok(())
        })
    }
}

impl Serialize for CustomNormalizer {
    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Err(serde::ser::Error::custom(
            "Custom Normalizer cannot be serialized",
        ))
    }
}

impl<'de> Deserialize<'de> for CustomNormalizer {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Err(serde::de::Error::custom(
            "Custom Normalizer cannot be deserialized",
        ))
    }
}

#[derive(Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum PyNormalizerWrapper {
    Custom(CustomNormalizer),
    Wrapped(NormalizerWrapper),
}

impl Serialize for PyNormalizerWrapper {
    fn serialize<S>(&self, serializer: S) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        match self {
            PyNormalizerWrapper::Wrapped(inner) => inner.serialize(serializer),
            PyNormalizerWrapper::Custom(inner) => inner.serialize(serializer),
        }
    }
}

#[derive(Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum PyNormalizerTypeWrapper {
    Sequence(Vec<Arc<RwLock<PyNormalizerWrapper>>>),
    Single(Arc<RwLock<PyNormalizerWrapper>>),
}

impl Serialize for PyNormalizerTypeWrapper {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            PyNormalizerTypeWrapper::Sequence(seq) => {
                let mut ser = serializer.serialize_struct("Sequence", 2)?;
                ser.serialize_field("type", "Sequence")?;
                ser.serialize_field("normalizers", seq)?;
                ser.end()
            }
            PyNormalizerTypeWrapper::Single(inner) => inner.serialize(serializer),
        }
    }
}

impl<I> From<I> for PyNormalizerWrapper
where
    I: Into<NormalizerWrapper>,
{
    fn from(norm: I) -> Self {
        PyNormalizerWrapper::Wrapped(norm.into())
    }
}

impl<I> From<I> for PyNormalizerTypeWrapper
where
    I: Into<PyNormalizerWrapper>,
{
    fn from(norm: I) -> Self {
        PyNormalizerTypeWrapper::Single(Arc::new(RwLock::new(norm.into())))
    }
}

impl<I> From<I> for PyNormalizer
where
    I: Into<NormalizerWrapper>,
{
    fn from(norm: I) -> Self {
        PyNormalizer {
            normalizer: norm.into().into(),
        }
    }
}

impl Normalizer for PyNormalizerTypeWrapper {
    fn normalize(&self, normalized: &mut NormalizedString) -> tk::Result<()> {
        match self {
            PyNormalizerTypeWrapper::Single(inner) => inner.read().unwrap().normalize(normalized),
            PyNormalizerTypeWrapper::Sequence(inner) => inner
                .iter()
                .map(|n| n.read().unwrap().normalize(normalized))
                .collect(),
        }
    }
}

impl Normalizer for PyNormalizerWrapper {
    fn normalize(&self, normalized: &mut NormalizedString) -> tk::Result<()> {
        match self {
            PyNormalizerWrapper::Wrapped(inner) => inner.normalize(normalized),
            PyNormalizerWrapper::Custom(inner) => inner.normalize(normalized),
        }
    }
}

#[cfg(test)]
mod test {
    use pyo3::prelude::*;
    use tk::normalizers::unicode::{NFC, NFKC};
    use tk::normalizers::utils::Sequence;
    use tk::normalizers::NormalizerWrapper;

    use crate::normalizers::{PyNormalizer, PyNormalizerTypeWrapper, PyNormalizerWrapper};

    #[test]
    fn get_subtype() {
        let py_norm = PyNormalizer::new(NFC.into());
        let py_nfc = py_norm.get_as_subtype().unwrap();
        let gil = Python::acquire_gil();
        assert_eq!(
            "tokenizers.normalizers.NFC",
            py_nfc.as_ref(gil.python()).get_type().name()
        );
    }

    #[test]
    fn serialize() {
        let py_wrapped: PyNormalizerWrapper = NFKC.into();
        let py_ser = serde_json::to_string(&py_wrapped).unwrap();
        let rs_wrapped = NormalizerWrapper::NFKC(NFKC);
        let rs_ser = serde_json::to_string(&rs_wrapped).unwrap();
        assert_eq!(py_ser, rs_ser);
        let py_norm: PyNormalizer = serde_json::from_str(&rs_ser).unwrap();
        match py_norm.normalizer {
            PyNormalizerTypeWrapper::Single(inner) => match *inner.as_ref().read().unwrap() {
                PyNormalizerWrapper::Wrapped(NormalizerWrapper::NFKC(_)) => {}
                _ => panic!("Expected NFKC"),
            },
            _ => panic!("Expected wrapped, not sequence."),
        }

        let py_seq: PyNormalizerWrapper = Sequence::new(vec![NFC.into(), NFKC.into()]).into();
        let py_wrapper_ser = serde_json::to_string(&py_seq).unwrap();
        let rs_wrapped = NormalizerWrapper::Sequence(Sequence::new(vec![NFC.into(), NFKC.into()]));
        let rs_ser = serde_json::to_string(&rs_wrapped).unwrap();
        assert_eq!(py_wrapper_ser, rs_ser);

        let py_seq = PyNormalizer::new(py_seq.into());
        let py_ser = serde_json::to_string(&py_seq).unwrap();
        assert_eq!(py_wrapper_ser, py_ser);

        let rs_seq = Sequence::new(vec![NFC.into(), NFKC.into()]);
        let rs_ser = serde_json::to_string(&rs_seq).unwrap();
        assert_eq!(py_wrapper_ser, rs_ser);
    }

    #[test]
    fn deserialize_sequence() {
        let string = r#"{"type": "NFKC"}"#;
        let normalizer: PyNormalizer = serde_json::from_str(&string).unwrap();
        match normalizer.normalizer {
            PyNormalizerTypeWrapper::Single(inner) => match *inner.as_ref().read().unwrap() {
                PyNormalizerWrapper::Wrapped(NormalizerWrapper::NFKC(_)) => {}
                _ => panic!("Expected NFKC"),
            },
            _ => panic!("Expected wrapped, not sequence."),
        }

        let sequence_string = format!(r#"{{"type": "Sequence", "normalizers": [{}]}}"#, string);
        let normalizer: PyNormalizer = serde_json::from_str(&sequence_string).unwrap();

        match normalizer.normalizer {
            PyNormalizerTypeWrapper::Single(inner) => match &*inner.as_ref().read().unwrap() {
                PyNormalizerWrapper::Wrapped(NormalizerWrapper::Sequence(sequence)) => {
                    let normalizers = sequence.get_normalizers();
                    assert_eq!(normalizers.len(), 1);
                    match normalizers[0] {
                        NormalizerWrapper::NFKC(_) => {}
                        _ => panic!("Expected NFKC"),
                    }
                }
                _ => panic!("Expected sequence"),
            },
            _ => panic!("Expected single"),
        };
    }
}
