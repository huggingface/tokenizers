use std::sync::Arc;

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
use tk::pre_tokenizers::unicode_scripts::UnicodeScripts;
use tk::pre_tokenizers::whitespace::{Whitespace, WhitespaceSplit};
use tk::pre_tokenizers::PreTokenizerWrapper;
use tk::tokenizer::Offsets;
use tk::{PreTokenizedString, PreTokenizer};
use tokenizers as tk;

use super::error::ToPyResult;
use super::utils::*;

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
            PyPreTokenizerTypeWrapper::Single(ref inner) => match inner.as_ref() {
                PyPreTokenizerWrapper::Custom(_) => Py::new(py, base)?.into_py(py),
                PyPreTokenizerWrapper::Wrapped(inner) => match inner {
                    PreTokenizerWrapper::Whitespace(_) => {
                        Py::new(py, (PyWhitespace {}, base))?.into_py(py)
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
                    PreTokenizerWrapper::Digits(_) => Py::new(py, (PyDigits {}, base))?.into_py(py),
                    PreTokenizerWrapper::UnicodeScripts(_) => {
                        Py::new(py, (PyUnicodeScripts {}, base))?.into_py(py)
                    }
                },
            },
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

    fn pre_tokenize(&self, pretok: &mut PyPreTokenizedString) -> PyResult<()> {
        ToPyResult(self.pretok.pre_tokenize(&mut pretok.pretok)).into()
    }

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

#[pyclass(extends=PyPreTokenizer, module = "tokenizers.pre_tokenizers", name=ByteLevel)]
pub struct PyByteLevel {}
#[pymethods]
impl PyByteLevel {
    #[new]
    #[args(kwargs = "**")]
    fn new(kwargs: Option<&PyDict>) -> PyResult<(Self, PyPreTokenizer)> {
        let mut byte_level = ByteLevel::default();
        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "add_prefix_space" => {
                        byte_level = byte_level.add_prefix_space(value.extract()?)
                    }
                    _ => println!("Ignored unknown kwargs option {}", key),
                }
            }
        }

        Ok((PyByteLevel {}, byte_level.into()))
    }

    #[staticmethod]
    fn alphabet() -> Vec<String> {
        ByteLevel::alphabet()
            .into_iter()
            .map(|c| c.to_string())
            .collect()
    }
}

#[pyclass(extends=PyPreTokenizer, module = "tokenizers.pre_tokenizers", name=Whitespace)]
pub struct PyWhitespace {}
#[pymethods]
impl PyWhitespace {
    #[new]
    fn new() -> PyResult<(Self, PyPreTokenizer)> {
        Ok((PyWhitespace {}, Whitespace::default().into()))
    }
}

#[pyclass(extends=PyPreTokenizer, module = "tokenizers.pre_tokenizers", name=WhitespaceSplit)]
pub struct PyWhitespaceSplit {}
#[pymethods]
impl PyWhitespaceSplit {
    #[new]
    fn new() -> PyResult<(Self, PyPreTokenizer)> {
        Ok((PyWhitespaceSplit {}, WhitespaceSplit.into()))
    }
}

#[pyclass(extends=PyPreTokenizer, module = "tokenizers.pre_tokenizers", name=CharDelimiterSplit)]
pub struct PyCharDelimiterSplit {}
#[pymethods]
impl PyCharDelimiterSplit {
    #[new]
    pub fn new(delimiter: &str) -> PyResult<(Self, PyPreTokenizer)> {
        let chr_delimiter = delimiter.chars().next().ok_or_else(|| {
            exceptions::PyValueError::new_err("delimiter must be a single character")
        })?;
        Ok((
            PyCharDelimiterSplit {},
            CharDelimiterSplit::new(chr_delimiter).into(),
        ))
    }

    fn __getnewargs__<'p>(&self, py: Python<'p>) -> PyResult<&'p PyTuple> {
        Ok(PyTuple::new(py, &[" "]))
    }
}

#[pyclass(extends=PyPreTokenizer, module = "tokenizers.pre_tokenizers", name=BertPreTokenizer)]
pub struct PyBertPreTokenizer {}
#[pymethods]
impl PyBertPreTokenizer {
    #[new]
    fn new() -> PyResult<(Self, PyPreTokenizer)> {
        Ok((PyBertPreTokenizer {}, BertPreTokenizer.into()))
    }
}

#[pyclass(extends=PyPreTokenizer, module = "tokenizers.pre_tokenizers", name=Punctuation)]
pub struct PyPunctuation {}
#[pymethods]
impl PyPunctuation {
    #[new]
    fn new() -> PyResult<(Self, PyPreTokenizer)> {
        Ok((PyPunctuation {}, Punctuation.into()))
    }
}

#[pyclass(extends=PyPreTokenizer, module = "tokenizers.pre_tokenizers", name=Sequence)]
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

#[pyclass(extends=PyPreTokenizer, module = "tokenizers.pre_tokenizers", name=Metaspace)]
pub struct PyMetaspace {}
#[pymethods]
impl PyMetaspace {
    #[new]
    #[args(kwargs = "**")]
    fn new(kwargs: Option<&PyDict>) -> PyResult<(Self, PyPreTokenizer)> {
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
            PyMetaspace {},
            Metaspace::new(replacement, add_prefix_space).into(),
        ))
    }
}

#[pyclass(extends=PyPreTokenizer, module = "tokenizers.pre_tokenizers", name=Digits)]
pub struct PyDigits {}
#[pymethods]
impl PyDigits {
    #[new]
    #[args(individual_digits = false)]
    fn new(individual_digits: bool) -> PyResult<(Self, PyPreTokenizer)> {
        Ok((PyDigits {}, Digits::new(individual_digits).into()))
    }
}

#[pyclass(extends=PyPreTokenizer, module = "tokenizers.pre_tokenizers", name=UnicodeScripts)]
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
    Sequence(Vec<Arc<PyPreTokenizerWrapper>>),
    Single(Arc<PyPreTokenizerWrapper>),
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
        PyPreTokenizerTypeWrapper::Single(Arc::new(pretok.into()))
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
            PyPreTokenizerTypeWrapper::Single(inner) => inner.pre_tokenize(pretok),
            PyPreTokenizerTypeWrapper::Sequence(inner) => {
                inner.iter().map(|n| n.pre_tokenize(pretok)).collect()
            }
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
            PyPreTokenizerTypeWrapper::Single(inner) => match inner.as_ref() {
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
