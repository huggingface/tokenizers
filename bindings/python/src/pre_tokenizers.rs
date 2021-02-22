use std::sync::{Arc, RwLock};

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use tk::pre_tokenizers::bert::BertPreTokenizer;
use tk::pre_tokenizers::byte_level::ByteLevel;
use tk::pre_tokenizers::delimiter::CharDelimiterSplit;
use tk::pre_tokenizers::digits::Digits;
use tk::pre_tokenizers::metaspace::Metaspace;
use tk::pre_tokenizers::punctuation::Punctuation;
use tk::pre_tokenizers::split::Split;
use tk::pre_tokenizers::unicode_scripts::UnicodeScripts;
use tk::pre_tokenizers::whitespace::{Whitespace, WhitespaceSplit};
use tk::pre_tokenizers::PreTokenizerWrapper;
use tk::tokenizer::Offsets;
use tk::{PreTokenizedString, PreTokenizer};
use tokenizers as tk;

use super::error::ToPyResult;
use super::utils::*;

/// Base class for all pre-tokenizers
///
/// This class is not supposed to be instantiated directly. Instead, any implementation of a
/// PreTokenizer will return an instance of this class when instantiated.
#[pyclass(dict, module = "tokenizers.pre_tokenizers", name=PreTokenizer)]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyPreTokenizer {
    #[serde(flatten)]
    pub(crate) pretok: PyPreTokenizerTypeWrapper,
}

impl PyPreTokenizer {
    #[allow(dead_code)]
    pub(crate) fn new(pretok: PyPreTokenizerTypeWrapper) -> Self {
        PyPreTokenizer { pretok }
    }

    pub(crate) fn get_as_subtype(&self) -> PyResult<PyObject> {
        let base = self.clone();
        let gil = Python::acquire_gil();
        let py = gil.python();
        Ok(match &self.pretok {
            PyPreTokenizerTypeWrapper::Sequence(_) => {
                Py::new(py, (PySequence {}, base))?.into_py(py)
            }
            PyPreTokenizerTypeWrapper::Single(ref inner) => {
                match &*inner.as_ref().read().unwrap() {
                    PyPreTokenizerWrapper::Custom(_) => Py::new(py, base)?.into_py(py),
                    PyPreTokenizerWrapper::Wrapped(inner) => match inner {
                        PreTokenizerWrapper::Whitespace(_) => {
                            Py::new(py, (PyWhitespace {}, base))?.into_py(py)
                        }
                        PreTokenizerWrapper::Split(_) => {
                            Py::new(py, (PySplit {}, base))?.into_py(py)
                        }
                        PreTokenizerWrapper::Punctuation(_) => {
                            Py::new(py, (PyPunctuation {}, base))?.into_py(py)
                        }
                        PreTokenizerWrapper::Sequence(_) => {
                            Py::new(py, (PySequence {}, base))?.into_py(py)
                        }
                        PreTokenizerWrapper::Metaspace(_) => {
                            Py::new(py, (PyMetaspace {}, base))?.into_py(py)
                        }
                        PreTokenizerWrapper::Delimiter(_) => {
                            Py::new(py, (PyCharDelimiterSplit {}, base))?.into_py(py)
                        }
                        PreTokenizerWrapper::WhitespaceSplit(_) => {
                            Py::new(py, (PyWhitespaceSplit {}, base))?.into_py(py)
                        }
                        PreTokenizerWrapper::ByteLevel(_) => {
                            Py::new(py, (PyByteLevel {}, base))?.into_py(py)
                        }
                        PreTokenizerWrapper::BertPreTokenizer(_) => {
                            Py::new(py, (PyBertPreTokenizer {}, base))?.into_py(py)
                        }
                        PreTokenizerWrapper::Digits(_) => {
                            Py::new(py, (PyDigits {}, base))?.into_py(py)
                        }
                        PreTokenizerWrapper::UnicodeScripts(_) => {
                            Py::new(py, (PyUnicodeScripts {}, base))?.into_py(py)
                        }
                    },
                }
            }
        })
    }
}

impl PreTokenizer for PyPreTokenizer {
    fn pre_tokenize(&self, normalized: &mut PreTokenizedString) -> tk::Result<()> {
        self.pretok.pre_tokenize(normalized)
    }
}

#[pymethods]
impl PyPreTokenizer {
    #[staticmethod]
    fn custom(pretok: PyObject) -> PyResult<Self> {
        Ok(PyPreTokenizer {
            pretok: PyPreTokenizerWrapper::Custom(CustomPreTokenizer::new(pretok)).into(),
        })
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let data = serde_json::to_string(&self.pretok).map_err(|e| {
            exceptions::PyException::new_err(format!(
                "Error while attempting to pickle PreTokenizer: {}",
                e.to_string()
            ))
        })?;
        Ok(PyBytes::new(py, data.as_bytes()).to_object(py))
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                let unpickled = serde_json::from_slice(s.as_bytes()).map_err(|e| {
                    exceptions::PyException::new_err(format!(
                        "Error while attempting to unpickle PreTokenizer: {}",
                        e
                    ))
                })?;
                self.pretok = unpickled;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Pre-tokenize a :class:`~tokenizers.PyPreTokenizedString` in-place
    ///
    /// This method allows to modify a :class:`~tokenizers.PreTokenizedString` to
    /// keep track of the pre-tokenization, and leverage the capabilities of the
    /// :class:`~tokenizers.PreTokenizedString`. If you just want to see the result of
    /// the pre-tokenization of a raw string, you can use
    /// :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize_str`
    ///
    /// Args:
    ///     pretok (:class:`~tokenizers.PreTokenizedString):
    ///         The pre-tokenized string on which to apply this
    ///         :class:`~tokenizers.pre_tokenizers.PreTokenizer`
    #[text_signature = "(self, pretok)"]
    fn pre_tokenize(&self, pretok: &mut PyPreTokenizedString) -> PyResult<()> {
        ToPyResult(self.pretok.pre_tokenize(&mut pretok.pretok)).into()
    }

    /// Pre tokenize the given string
    ///
    /// This method provides a way to visualize the effect of a
    /// :class:`~tokenizers.pre_tokenizers.PreTokenizer` but it does not keep track of the
    /// alignment, nor does it provide all the capabilities of the
    /// :class:`~tokenizers.PreTokenizedString`. If you need some of these, you can use
    /// :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize`
    ///
    /// Args:
    ///     sequence (:obj:`str`):
    ///         A string to pre-tokeize
    ///
    /// Returns:
    ///     :obj:`List[Tuple[str, Offsets]]`:
    ///         A list of tuple with the pre-tokenized parts and their offsets
    #[text_signature = "(self, sequence)"]
    fn pre_tokenize_str(&self, s: &str) -> PyResult<Vec<(String, Offsets)>> {
        let mut pretokenized = tk::tokenizer::PreTokenizedString::from(s);

        ToPyResult(self.pretok.pre_tokenize(&mut pretokenized)).into_py()?;

        Ok(pretokenized
            .get_splits(tk::OffsetReferential::Original, tk::OffsetType::Char)
            .into_iter()
            .map(|(s, o, _)| (s.to_owned(), o))
            .collect())
    }
}

macro_rules! getter {
    ($self: ident, $variant: ident, $($name: tt)+) => {{
        let super_ = $self.as_ref();
        if let PyPreTokenizerTypeWrapper::Single(ref single) = super_.pretok {
            if let PyPreTokenizerWrapper::Wrapped(PreTokenizerWrapper::$variant(ref pretok)) =
                *single.read().unwrap() {
                    pretok.$($name)+
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
        if let PyPreTokenizerTypeWrapper::Single(ref single) = super_.pretok {
            if let PyPreTokenizerWrapper::Wrapped(PreTokenizerWrapper::$variant(ref mut pretok)) =
                *single.write().unwrap()
            {
                pretok.$name = $value;
            }
        }
    }};
    ($self: ident, $variant: ident, @$name: ident, $value: expr) => {{
        let super_ = $self.as_ref();
        if let PyPreTokenizerTypeWrapper::Single(ref single) = super_.pretok {
            if let PyPreTokenizerWrapper::Wrapped(PreTokenizerWrapper::$variant(ref mut pretok)) =
                *single.write().unwrap()
            {
                pretok.$name($value);
            }
        }
    }};
}

/// ByteLevel PreTokenizer
///
/// This pre-tokenizer takes care of replacing all bytes of the given string
/// with a corresponding representation, as well as splitting into words.
///
/// Args:
///     add_prefix_space (:obj:`bool`, `optional`, defaults to :obj:`True`):
///         Whether to add a space to the first word if there isn't already one. This
///         lets us treat `hello` exactly like `say hello`.
#[pyclass(extends=PyPreTokenizer, module = "tokenizers.pre_tokenizers", name=ByteLevel)]
#[text_signature = "(self, add_prefix_space=True)"]
pub struct PyByteLevel {}
#[pymethods]
impl PyByteLevel {
    #[getter]
    fn get_add_prefix_space(self_: PyRef<Self>) -> bool {
        getter!(self_, ByteLevel, add_prefix_space)
    }

    #[setter]
    fn set_add_prefix_space(self_: PyRef<Self>, add_prefix_space: bool) {
        setter!(self_, ByteLevel, add_prefix_space, add_prefix_space);
    }

    #[new]
    #[args(add_prefix_space = "true")]
    fn new(add_prefix_space: bool) -> PyResult<(Self, PyPreTokenizer)> {
        Ok((
            PyByteLevel {},
            ByteLevel::default()
                .add_prefix_space(add_prefix_space)
                .into(),
        ))
    }

    /// Returns the alphabet used by this PreTokenizer.
    ///
    /// Since the ByteLevel works as its name suggests, at the byte level, it
    /// encodes each byte value to a unique visible character. This means that there is a
    /// total of 256 different characters composing this alphabet.
    ///
    /// Returns:
    ///     :obj:`List[str]`: A list of characters that compose the alphabet
    #[staticmethod]
    #[text_signature = "()"]
    fn alphabet() -> Vec<String> {
        ByteLevel::alphabet()
            .into_iter()
            .map(|c| c.to_string())
            .collect()
    }
}

/// This pre-tokenizer simply splits using the following regex: `\w+|[^\w\s]+`
#[pyclass(extends=PyPreTokenizer, module = "tokenizers.pre_tokenizers", name=Whitespace)]
#[text_signature = "(self)"]
pub struct PyWhitespace {}
#[pymethods]
impl PyWhitespace {
    #[new]
    fn new() -> PyResult<(Self, PyPreTokenizer)> {
        Ok((PyWhitespace {}, Whitespace::default().into()))
    }
}

/// This pre-tokenizer simply splits on the whitespace. Works like `.split()`
#[pyclass(extends=PyPreTokenizer, module = "tokenizers.pre_tokenizers", name=WhitespaceSplit)]
#[text_signature = "(self)"]
pub struct PyWhitespaceSplit {}
#[pymethods]
impl PyWhitespaceSplit {
    #[new]
    fn new() -> PyResult<(Self, PyPreTokenizer)> {
        Ok((PyWhitespaceSplit {}, WhitespaceSplit.into()))
    }
}

/// Split PreTokenizer
///
/// This versatile pre-tokenizer splits using the provided pattern and
/// according to the provided behavior. The pattern can be inverted by
/// making use of the invert flag.
///
/// Args:
///     pattern (:obj:`str` or :class:`~tokenizers.Regex`):
///         A pattern used to split the string. Usually a string or a Regex
///
///     behavior (:class:`~tokenizers.SplitDelimiterBehavior`):
///         The behavior to use when splitting.
///         Choices: "removed", "isolated", "merged_with_previous", "merged_with_next",
///         "contiguous"
///
///     invert (:obj:`bool`, `optional`, defaults to :obj:`False`):
///         Whether to invert the pattern.
#[pyclass(extends=PyPreTokenizer, module = "tokenizers.pre_tokenizers", name=Split)]
#[text_signature = "(self, pattern, behavior, invert=False)"]
pub struct PySplit {}
#[pymethods]
impl PySplit {
    #[new]
    #[args(invert = false)]
    fn new(
        pattern: PyPattern,
        behavior: PySplitDelimiterBehavior,
        invert: bool,
    ) -> PyResult<(Self, PyPreTokenizer)> {
        Ok((
            PySplit {},
            ToPyResult(Split::new(pattern, behavior.into(), invert))
                .into_py()?
                .into(),
        ))
    }

    fn __getnewargs__<'p>(&self, py: Python<'p>) -> PyResult<&'p PyTuple> {
        Ok(PyTuple::new(py, &[" ", "removed"]))
    }
}

/// This pre-tokenizer simply splits on the provided char. Works like `.split(delimiter)`
///
/// Args:
///     delimiter: str:
///         The delimiter char that will be used to split input
#[pyclass(extends=PyPreTokenizer, module = "tokenizers.pre_tokenizers", name=CharDelimiterSplit)]
pub struct PyCharDelimiterSplit {}
#[pymethods]
impl PyCharDelimiterSplit {
    #[getter]
    fn get_delimiter(self_: PyRef<Self>) -> String {
        getter!(self_, Delimiter, delimiter.to_string())
    }

    #[setter]
    fn set_delimiter(self_: PyRef<Self>, delimiter: PyChar) {
        setter!(self_, Delimiter, delimiter, delimiter.0);
    }

    #[new]
    pub fn new(delimiter: PyChar) -> PyResult<(Self, PyPreTokenizer)> {
        Ok((
            PyCharDelimiterSplit {},
            CharDelimiterSplit::new(delimiter.0).into(),
        ))
    }

    fn __getnewargs__<'p>(&self, py: Python<'p>) -> PyResult<&'p PyTuple> {
        Ok(PyTuple::new(py, &[" "]))
    }
}

/// BertPreTokenizer
///
/// This pre-tokenizer splits tokens on spaces, and also on punctuation.
/// Each occurence of a punctuation character will be treated separately.
#[pyclass(extends=PyPreTokenizer, module = "tokenizers.pre_tokenizers", name=BertPreTokenizer)]
#[text_signature = "(self)"]
pub struct PyBertPreTokenizer {}
#[pymethods]
impl PyBertPreTokenizer {
    #[new]
    fn new() -> PyResult<(Self, PyPreTokenizer)> {
        Ok((PyBertPreTokenizer {}, BertPreTokenizer.into()))
    }
}

/// This pre-tokenizer simply splits on punctuation as individual characters.`
#[pyclass(extends=PyPreTokenizer, module = "tokenizers.pre_tokenizers", name=Punctuation)]
#[text_signature = "(self)"]
pub struct PyPunctuation {}
#[pymethods]
impl PyPunctuation {
    #[new]
    fn new() -> PyResult<(Self, PyPreTokenizer)> {
        Ok((PyPunctuation {}, Punctuation.into()))
    }
}

/// This pre-tokenizer composes other pre_tokenizers and applies them in sequence
#[pyclass(extends=PyPreTokenizer, module = "tokenizers.pre_tokenizers", name=Sequence)]
#[text_signature = "(self, pretokenizers)"]
pub struct PySequence {}
#[pymethods]
impl PySequence {
    #[new]
    fn new(pre_tokenizers: &PyList) -> PyResult<(Self, PyPreTokenizer)> {
        let mut sequence = Vec::with_capacity(pre_tokenizers.len());
        for n in pre_tokenizers.iter() {
            let pretokenizer: PyRef<PyPreTokenizer> = n.extract()?;
            match &pretokenizer.pretok {
                PyPreTokenizerTypeWrapper::Sequence(inner) => {
                    sequence.extend(inner.iter().cloned())
                }
                PyPreTokenizerTypeWrapper::Single(inner) => sequence.push(inner.clone()),
            }
        }
        Ok((
            PySequence {},
            PyPreTokenizer::new(PyPreTokenizerTypeWrapper::Sequence(sequence)),
        ))
    }

    fn __getnewargs__<'p>(&self, py: Python<'p>) -> PyResult<&'p PyTuple> {
        Ok(PyTuple::new(py, &[PyList::empty(py)]))
    }
}

/// Metaspace pre-tokenizer
///
/// This pre-tokenizer replaces any whitespace by the provided replacement character.
/// It then tries to split on these spaces.
///
/// Args:
///     replacement (:obj:`str`, `optional`, defaults to :obj:`▁`):
///         The replacement character. Must be exactly one character. By default we
///         use the `▁` (U+2581) meta symbol (Same as in SentencePiece).
///
///     add_prefix_space (:obj:`bool`, `optional`, defaults to :obj:`True`):
///         Whether to add a space to the first word if there isn't already one. This
///         lets us treat `hello` exactly like `say hello`.
#[pyclass(extends=PyPreTokenizer, module = "tokenizers.pre_tokenizers", name=Metaspace)]
#[text_signature = "(self, replacement=\"_\", add_prefix_space=True)"]
pub struct PyMetaspace {}
#[pymethods]
impl PyMetaspace {
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
    fn new(replacement: PyChar, add_prefix_space: bool) -> PyResult<(Self, PyPreTokenizer)> {
        Ok((
            PyMetaspace {},
            Metaspace::new(replacement.0, add_prefix_space).into(),
        ))
    }
}

/// This pre-tokenizer simply splits using the digits in separate tokens
///
/// Args:
///     individual_digits (:obj:`bool`, `optional`, defaults to :obj:`False`):
///         If set to True, digits will each be separated as follows::
///
///             "Call 123 please" -> "Call ", "1", "2", "3", " please"
///
///         If set to False, digits will grouped as follows::
///
///             "Call 123 please" -> "Call ", "123", " please"
#[pyclass(extends=PyPreTokenizer, module = "tokenizers.pre_tokenizers", name=Digits)]
#[text_signature = "(self, individual_digits=False)"]
pub struct PyDigits {}
#[pymethods]
impl PyDigits {
    #[getter]
    fn get_individual_digits(self_: PyRef<Self>) -> bool {
        getter!(self_, Digits, individual_digits)
    }

    #[setter]
    fn set_individual_digits(self_: PyRef<Self>, individual_digits: bool) {
        setter!(self_, Digits, individual_digits, individual_digits);
    }

    #[new]
    #[args(individual_digits = false)]
    fn new(individual_digits: bool) -> PyResult<(Self, PyPreTokenizer)> {
        Ok((PyDigits {}, Digits::new(individual_digits).into()))
    }
}

/// This pre-tokenizer splits on characters that belong to different language family
/// It roughly follows https://github.com/google/sentencepiece/blob/master/data/Scripts.txt
/// Actually Hiragana and Katakana are fused with Han, and 0x30FC is Han too.
/// This mimicks SentencePiece Unigram implementation.
#[pyclass(extends=PyPreTokenizer, module = "tokenizers.pre_tokenizers", name=UnicodeScripts)]
#[text_signature = "(self)"]
pub struct PyUnicodeScripts {}
#[pymethods]
impl PyUnicodeScripts {
    #[new]
    fn new() -> PyResult<(Self, PyPreTokenizer)> {
        Ok((PyUnicodeScripts {}, UnicodeScripts::new().into()))
    }
}

#[derive(Clone)]
pub(crate) struct CustomPreTokenizer {
    inner: PyObject,
}

impl CustomPreTokenizer {
    pub fn new(inner: PyObject) -> Self {
        Self { inner }
    }
}

impl tk::tokenizer::PreTokenizer for CustomPreTokenizer {
    fn pre_tokenize(&self, sentence: &mut PreTokenizedString) -> tk::Result<()> {
        Python::with_gil(|py| {
            let pretok = PyPreTokenizedStringRefMut::new(sentence);
            let py_pretok = self.inner.as_ref(py);
            py_pretok.call_method("pre_tokenize", (pretok.get(),), None)?;
            Ok(())
        })
    }
}

impl Serialize for CustomPreTokenizer {
    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Err(serde::ser::Error::custom(
            "Custom PreTokenizer cannot be serialized",
        ))
    }
}

impl<'de> Deserialize<'de> for CustomPreTokenizer {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Err(serde::de::Error::custom(
            "Custom PreTokenizer cannot be deserialized",
        ))
    }
}

#[derive(Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum PyPreTokenizerWrapper {
    Custom(CustomPreTokenizer),
    Wrapped(PreTokenizerWrapper),
}

impl Serialize for PyPreTokenizerWrapper {
    fn serialize<S>(&self, serializer: S) -> Result<<S as Serializer>::Ok, <S as Serializer>::Error>
    where
        S: Serializer,
    {
        match self {
            PyPreTokenizerWrapper::Wrapped(inner) => inner.serialize(serializer),
            PyPreTokenizerWrapper::Custom(inner) => inner.serialize(serializer),
        }
    }
}

#[derive(Clone, Deserialize)]
#[serde(untagged)]
pub(crate) enum PyPreTokenizerTypeWrapper {
    Sequence(Vec<Arc<RwLock<PyPreTokenizerWrapper>>>),
    Single(Arc<RwLock<PyPreTokenizerWrapper>>),
}

impl Serialize for PyPreTokenizerTypeWrapper {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            PyPreTokenizerTypeWrapper::Sequence(seq) => {
                let mut ser = serializer.serialize_struct("Sequence", 2)?;
                ser.serialize_field("type", "Sequence")?;
                ser.serialize_field("pretokenizers", seq)?;
                ser.end()
            }
            PyPreTokenizerTypeWrapper::Single(inner) => inner.serialize(serializer),
        }
    }
}

impl<I> From<I> for PyPreTokenizerWrapper
where
    I: Into<PreTokenizerWrapper>,
{
    fn from(pretok: I) -> Self {
        PyPreTokenizerWrapper::Wrapped(pretok.into())
    }
}

impl<I> From<I> for PyPreTokenizerTypeWrapper
where
    I: Into<PyPreTokenizerWrapper>,
{
    fn from(pretok: I) -> Self {
        PyPreTokenizerTypeWrapper::Single(Arc::new(RwLock::new(pretok.into())))
    }
}

impl<I> From<I> for PyPreTokenizer
where
    I: Into<PreTokenizerWrapper>,
{
    fn from(pretok: I) -> Self {
        PyPreTokenizer {
            pretok: pretok.into().into(),
        }
    }
}

impl PreTokenizer for PyPreTokenizerTypeWrapper {
    fn pre_tokenize(&self, pretok: &mut PreTokenizedString) -> tk::Result<()> {
        match self {
            PyPreTokenizerTypeWrapper::Single(inner) => inner.read().unwrap().pre_tokenize(pretok),
            PyPreTokenizerTypeWrapper::Sequence(inner) => inner
                .iter()
                .map(|n| n.read().unwrap().pre_tokenize(pretok))
                .collect(),
        }
    }
}

impl PreTokenizer for PyPreTokenizerWrapper {
    fn pre_tokenize(&self, pretok: &mut PreTokenizedString) -> tk::Result<()> {
        match self {
            PyPreTokenizerWrapper::Wrapped(inner) => inner.pre_tokenize(pretok),
            PyPreTokenizerWrapper::Custom(inner) => inner.pre_tokenize(pretok),
        }
    }
}

#[cfg(test)]
mod test {
    use pyo3::prelude::*;
    use tk::pre_tokenizers::sequence::Sequence;
    use tk::pre_tokenizers::whitespace::{Whitespace, WhitespaceSplit};
    use tk::pre_tokenizers::PreTokenizerWrapper;

    use crate::pre_tokenizers::{
        CustomPreTokenizer, PyPreTokenizer, PyPreTokenizerTypeWrapper, PyPreTokenizerWrapper,
    };

    #[test]
    fn get_subtype() {
        let py_norm = PyPreTokenizer::new(Whitespace::default().into());
        let py_wsp = py_norm.get_as_subtype().unwrap();
        let gil = Python::acquire_gil();
        assert_eq!(
            "tokenizers.pre_tokenizers.Whitespace",
            py_wsp.as_ref(gil.python()).get_type().name()
        );
    }

    #[test]
    fn serialize() {
        let py_wrapped: PyPreTokenizerWrapper = Whitespace::default().into();
        let py_ser = serde_json::to_string(&py_wrapped).unwrap();
        let rs_wrapped = PreTokenizerWrapper::Whitespace(Whitespace::default());
        let rs_ser = serde_json::to_string(&rs_wrapped).unwrap();
        assert_eq!(py_ser, rs_ser);
        let py_pretok: PyPreTokenizer = serde_json::from_str(&rs_ser).unwrap();
        match py_pretok.pretok {
            PyPreTokenizerTypeWrapper::Single(inner) => match *inner.as_ref().read().unwrap() {
                PyPreTokenizerWrapper::Wrapped(PreTokenizerWrapper::Whitespace(_)) => {}
                _ => panic!("Expected Whitespace"),
            },
            _ => panic!("Expected wrapped, not custom."),
        }

        let py_seq: PyPreTokenizerWrapper =
            Sequence::new(vec![Whitespace::default().into(), WhitespaceSplit.into()]).into();
        let py_wrapper_ser = serde_json::to_string(&py_seq).unwrap();
        let rs_wrapped = PreTokenizerWrapper::Sequence(Sequence::new(vec![
            Whitespace::default().into(),
            WhitespaceSplit.into(),
        ]));
        let rs_ser = serde_json::to_string(&rs_wrapped).unwrap();
        assert_eq!(py_wrapper_ser, rs_ser);

        let py_seq = PyPreTokenizer::new(py_seq.into());
        let py_ser = serde_json::to_string(&py_seq).unwrap();
        assert_eq!(py_wrapper_ser, py_ser);

        let obj = Python::with_gil(|py| {
            let py_wsp = PyPreTokenizer::new(Whitespace::default().into());
            let obj: PyObject = Py::new(py, py_wsp).unwrap().into_py(py);
            obj
        });
        let py_seq: PyPreTokenizerWrapper =
            PyPreTokenizerWrapper::Custom(CustomPreTokenizer::new(obj));
        assert!(serde_json::to_string(&py_seq).is_err());
    }
}
