use pyo3::types::*;
use pyo3::{exceptions, prelude::*};
use std::sync::{Arc, RwLock};

use crate::error::ToPyResult;
use crate::utils::{PyNormalizedString, PyNormalizedStringRefMut, PyPattern};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tk::normalizers::{
    BertNormalizer, ByteLevel, Lowercase, Nmt, NormalizerWrapper, Precompiled, Prepend, Replace,
    Strip, StripAccents, NFC, NFD, NFKC, NFKD,
};
use tk::{NormalizedString, Normalizer};
use tokenizers as tk;

/// Represents the different kind of NormalizedString we can receive from Python:
///  - Owned: Created in Python and owned by Python
///  - RefMut: A mutable reference to a NormalizedString owned by Rust
#[derive(FromPyObject)]
enum PyNormalizedStringMut<'p> {
    Owned(PyRefMut<'p, PyNormalizedString>),
    RefMut(PyNormalizedStringRefMut),
}

impl PyNormalizedStringMut<'_> {
    /// Normalized the underlying `NormalizedString` using the provided normalizer
    pub fn normalize_with<N>(&mut self, normalizer: &N) -> PyResult<()>
    where
        N: Normalizer,
    {
        match self {
            PyNormalizedStringMut::Owned(ref mut n) => normalizer.normalize(&mut n.normalized),
            PyNormalizedStringMut::RefMut(n) => n.map_as_mut(|n| normalizer.normalize(n))?,
        }
        .map_err(|e| exceptions::PyException::new_err(format!("{}", e)))
    }
}

/// Base class for all normalizers
///
/// This class is not supposed to be instantiated directly. Instead, any implementation of a
/// Normalizer will return an instance of this class when instantiated.
#[pyclass(dict, module = "tokenizers.normalizers", name = "Normalizer", subclass)]
#[derive(Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PyNormalizer {
    pub(crate) normalizer: PyNormalizerTypeWrapper,
}

impl PyNormalizer {
    pub(crate) fn new(normalizer: PyNormalizerTypeWrapper) -> Self {
        PyNormalizer { normalizer }
    }
    pub(crate) fn get_as_subtype(&self, py: Python<'_>) -> PyResult<PyObject> {
        let base = self.clone();
        Ok(match self.normalizer {
            PyNormalizerTypeWrapper::Sequence(_) => Py::new(py, (PySequence {}, base))?
                .into_pyobject(py)?
                .into_any()
                .into(),
            PyNormalizerTypeWrapper::Single(ref inner) => match &*inner.as_ref().read().unwrap() {
                PyNormalizerWrapper::Custom(_) => {
                    Py::new(py, base)?.into_pyobject(py)?.into_any().into()
                }
                PyNormalizerWrapper::Wrapped(ref inner) => match inner {
                    NormalizerWrapper::Sequence(_) => Py::new(py, (PySequence {}, base))?
                        .into_pyobject(py)?
                        .into_any()
                        .into(),
                    NormalizerWrapper::BertNormalizer(_) => {
                        Py::new(py, (PyBertNormalizer {}, base))?
                            .into_pyobject(py)?
                            .into_any()
                            .into()
                    }
                    NormalizerWrapper::StripNormalizer(_) => Py::new(py, (PyStrip {}, base))?
                        .into_pyobject(py)?
                        .into_any()
                        .into(),
                    NormalizerWrapper::Prepend(_) => Py::new(py, (PyPrepend {}, base))?
                        .into_pyobject(py)?
                        .into_any()
                        .into(),
                    NormalizerWrapper::ByteLevel(_) => Py::new(py, (PyByteLevel {}, base))?
                        .into_pyobject(py)?
                        .into_any()
                        .into(),
                    NormalizerWrapper::StripAccents(_) => Py::new(py, (PyStripAccents {}, base))?
                        .into_pyobject(py)?
                        .into_any()
                        .into(),
                    NormalizerWrapper::NFC(_) => Py::new(py, (PyNFC {}, base))?
                        .into_pyobject(py)?
                        .into_any()
                        .into(),
                    NormalizerWrapper::NFD(_) => Py::new(py, (PyNFD {}, base))?
                        .into_pyobject(py)?
                        .into_any()
                        .into(),
                    NormalizerWrapper::NFKC(_) => Py::new(py, (PyNFKC {}, base))?
                        .into_pyobject(py)?
                        .into_any()
                        .into(),
                    NormalizerWrapper::NFKD(_) => Py::new(py, (PyNFKD {}, base))?
                        .into_pyobject(py)?
                        .into_any()
                        .into(),
                    NormalizerWrapper::Lowercase(_) => Py::new(py, (PyLowercase {}, base))?
                        .into_pyobject(py)?
                        .into_any()
                        .into(),
                    NormalizerWrapper::Precompiled(_) => Py::new(py, (PyPrecompiled {}, base))?
                        .into_pyobject(py)?
                        .into_any()
                        .into(),
                    NormalizerWrapper::Replace(_) => Py::new(py, (PyReplace {}, base))?
                        .into_pyobject(py)?
                        .into_any()
                        .into(),
                    NormalizerWrapper::Nmt(_) => Py::new(py, (PyNmt {}, base))?
                        .into_pyobject(py)?
                        .into_any()
                        .into(),
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
    fn custom(obj: PyObject) -> Self {
        Self {
            normalizer: PyNormalizerWrapper::Custom(CustomNormalizer::new(obj)).into(),
        }
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let data = serde_json::to_string(&self.normalizer).map_err(|e| {
            exceptions::PyException::new_err(format!(
                "Error while attempting to pickle Normalizer: {}",
                e
            ))
        })?;
        Ok(PyBytes::new(py, data.as_bytes()).into())
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&[u8]>(py) {
            Ok(s) => {
                self.normalizer = serde_json::from_slice(s).map_err(|e| {
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
    #[pyo3(text_signature = "(self, normalized)")]
    fn normalize(&self, mut normalized: PyNormalizedStringMut) -> PyResult<()> {
        normalized.normalize_with(&self.normalizer)
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
    #[pyo3(text_signature = "(self, sequence)")]
    fn normalize_str(&self, sequence: &str) -> PyResult<String> {
        let mut normalized = NormalizedString::from(sequence);
        ToPyResult(self.normalizer.normalize(&mut normalized)).into_py()?;
        Ok(normalized.get().to_owned())
    }

    fn __repr__(&self) -> PyResult<String> {
        crate::utils::serde_pyo3::repr(self)
            .map_err(|e| exceptions::PyException::new_err(e.to_string()))
    }

    fn __str__(&self) -> PyResult<String> {
        crate::utils::serde_pyo3::to_string(self)
            .map_err(|e| exceptions::PyException::new_err(e.to_string()))
    }
}

macro_rules! getter {
    ($self: ident, $variant: ident, $name: ident) => {{
        let super_ = $self.as_ref();
        if let PyNormalizerTypeWrapper::Single(ref norm) = super_.normalizer {
            let wrapper = norm.read().unwrap();
            if let PyNormalizerWrapper::Wrapped(NormalizerWrapper::$variant(o)) = (&*wrapper) {
                o.$name.clone()
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
#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name = "BertNormalizer")]
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
    #[pyo3(signature = (
        clean_text = true,
        handle_chinese_chars = true,
        strip_accents = None,
        lowercase = true
    ),
        text_signature = "(self, clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=True)")]
    fn new(
        clean_text: bool,
        handle_chinese_chars: bool,
        strip_accents: Option<bool>,
        lowercase: bool,
    ) -> (Self, PyNormalizer) {
        let normalizer =
            BertNormalizer::new(clean_text, handle_chinese_chars, strip_accents, lowercase);
        (PyBertNormalizer {}, normalizer.into())
    }
}

/// NFD Unicode Normalizer
#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name = "NFD")]
pub struct PyNFD {}
#[pymethods]
impl PyNFD {
    #[new]
    #[pyo3(text_signature = "(self)")]
    fn new() -> (Self, PyNormalizer) {
        (PyNFD {}, PyNormalizer::new(NFD.into()))
    }
}

/// NFKD Unicode Normalizer
#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name = "NFKD")]
pub struct PyNFKD {}
#[pymethods]
impl PyNFKD {
    #[new]
    #[pyo3(text_signature = "(self)")]
    fn new() -> (Self, PyNormalizer) {
        (PyNFKD {}, NFKD.into())
    }
}

/// NFC Unicode Normalizer
#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name = "NFC")]
pub struct PyNFC {}
#[pymethods]
impl PyNFC {
    #[new]
    #[pyo3(text_signature = "(self)")]
    fn new() -> (Self, PyNormalizer) {
        (PyNFC {}, NFC.into())
    }
}

/// NFKC Unicode Normalizer
#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name = "NFKC")]
pub struct PyNFKC {}
#[pymethods]
impl PyNFKC {
    #[new]
    #[pyo3(text_signature = "(self)")]
    fn new() -> (Self, PyNormalizer) {
        (PyNFKC {}, NFKC.into())
    }
}

/// Allows concatenating multiple other Normalizer as a Sequence.
/// All the normalizers run in sequence in the given order
///
/// Args:
///     normalizers (:obj:`List[Normalizer]`):
///         A list of Normalizer to be run as a sequence
#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name = "Sequence")]
pub struct PySequence {}

#[pymethods]
impl PySequence {
    #[new]
    #[pyo3(text_signature = None)]
    fn new(normalizers: &Bound<'_, PyList>) -> PyResult<(Self, PyNormalizer)> {
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

    fn __getnewargs__<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyTuple>> {
        PyTuple::new(py, [PyList::empty(py)])
    }

    fn __len__(&self) -> usize {
        0
    }

    fn __getitem__(self_: PyRef<'_, Self>, py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
        match &self_.as_ref().normalizer {
            PyNormalizerTypeWrapper::Sequence(inner) => match inner.get(index) {
                Some(item) => PyNormalizer::new(PyNormalizerTypeWrapper::Single(Arc::clone(item)))
                    .get_as_subtype(py),
                _ => Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    "Index not found",
                )),
            },
            PyNormalizerTypeWrapper::Single(inner) => {
                PyNormalizer::new(PyNormalizerTypeWrapper::Single(Arc::clone(inner)))
                    .get_as_subtype(py)
            }
        }
    }
}

/// Lowercase Normalizer
#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name = "Lowercase")]
pub struct PyLowercase {}
#[pymethods]
impl PyLowercase {
    #[new]
    #[pyo3(text_signature = "(self)")]
    fn new() -> (Self, PyNormalizer) {
        (PyLowercase {}, Lowercase.into())
    }
}

/// Strip normalizer
#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name = "Strip")]
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
    #[pyo3(signature = (left = true, right = true), text_signature = "(self, left=True, right=True)")]
    fn new(left: bool, right: bool) -> (Self, PyNormalizer) {
        (PyStrip {}, Strip::new(left, right).into())
    }
}

/// Prepend normalizer
#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name = "Prepend")]
pub struct PyPrepend {}
#[pymethods]
impl PyPrepend {
    #[getter]
    fn get_prepend(self_: PyRef<Self>) -> String {
        getter!(self_, Prepend, prepend)
    }

    #[setter]
    fn set_prepend(self_: PyRef<Self>, prepend: String) {
        setter!(self_, Prepend, prepend, prepend)
    }

    #[new]
    #[pyo3(signature = (prepend="▁".to_string()), text_signature = "(self, prepend)")]
    fn new(prepend: String) -> (Self, PyNormalizer) {
        (PyPrepend {}, Prepend::new(prepend).into())
    }
}

/// Bytelevel Normalizer
#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name = "ByteLevel")]
pub struct PyByteLevel {}
#[pymethods]
impl PyByteLevel {
    #[new]
    #[pyo3(text_signature = "(self)")]
    fn new() -> (Self, PyNormalizer) {
        (PyByteLevel {}, ByteLevel::new().into())
    }
}

/// StripAccents normalizer
#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name = "StripAccents")]
pub struct PyStripAccents {}
#[pymethods]
impl PyStripAccents {
    #[new]
    #[pyo3(text_signature = "(self)")]
    fn new() -> (Self, PyNormalizer) {
        (PyStripAccents {}, StripAccents.into())
    }
}

/// Nmt normalizer
#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name = "Nmt")]
pub struct PyNmt {}
#[pymethods]
impl PyNmt {
    #[new]
    #[pyo3(text_signature = "(self)")]
    fn new() -> (Self, PyNormalizer) {
        (PyNmt {}, Nmt.into())
    }
}

/// Precompiled normalizer
/// Don't use manually it is used for compatibility for SentencePiece.
#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name = "Precompiled")]
pub struct PyPrecompiled {}
#[pymethods]
impl PyPrecompiled {
    #[new]
    #[pyo3(text_signature = "(self, precompiled_charsmap)")]
    fn new(precompiled_charsmap: Vec<u8>) -> PyResult<(Self, PyNormalizer)> {
        // let precompiled_charsmap: Vec<u8> = FromPyObject::extract(py_precompiled_charsmap)?;
        Ok((
            PyPrecompiled {},
            Precompiled::from(&precompiled_charsmap)
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
#[pyclass(extends=PyNormalizer, module = "tokenizers.normalizers", name = "Replace")]
pub struct PyReplace {}
#[pymethods]
impl PyReplace {
    #[new]
    #[pyo3(text_signature = "(self, pattern, content)")]
    fn new(pattern: PyPattern, content: String) -> PyResult<(Self, PyNormalizer)> {
        Ok((
            PyReplace {},
            ToPyResult(Replace::new(pattern, content)).into_py()?.into(),
        ))
    }
}

#[derive(Debug)]
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
            let py_normalized = self.inner.bind(py);
            py_normalized.call_method("normalize", (normalized.get().clone(),), None)?;
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

#[derive(Debug, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
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
                .try_for_each(|n| n.read().unwrap().normalize(normalized)),
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

/// Normalizers Module
#[pymodule]
pub fn normalizers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNormalizer>()?;
    m.add_class::<PyBertNormalizer>()?;
    m.add_class::<PyNFD>()?;
    m.add_class::<PyNFKD>()?;
    m.add_class::<PyNFC>()?;
    m.add_class::<PyNFKC>()?;
    m.add_class::<PySequence>()?;
    m.add_class::<PyLowercase>()?;
    m.add_class::<PyStrip>()?;
    m.add_class::<PyStripAccents>()?;
    m.add_class::<PyPrepend>()?;
    m.add_class::<PyByteLevel>()?;
    m.add_class::<PyNmt>()?;
    m.add_class::<PyPrecompiled>()?;
    m.add_class::<PyReplace>()?;
    Ok(())
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
        Python::with_gil(|py| {
            let py_norm = PyNormalizer::new(NFC.into());
            let py_nfc = py_norm.get_as_subtype(py).unwrap();
            assert_eq!("NFC", py_nfc.bind(py).get_type().qualname().unwrap());
        })
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
        let normalizer: PyNormalizer = serde_json::from_str(string).unwrap();
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
