use std::sync::Arc;

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use tk::pre_tokenizers::bert::BertPreTokenizer;
use tk::pre_tokenizers::byte_level::ByteLevel;
use tk::pre_tokenizers::delimiter::CharDelimiterSplit;
use tk::pre_tokenizers::metaspace::Metaspace;
use tk::pre_tokenizers::whitespace::{Whitespace, WhitespaceSplit};
use tk::tokenizer::Offsets;
use tk::{PreTokenizedString, PreTokenizer};
use tokenizers as tk;

use super::error::ToPyResult;

#[pyclass(dict, module = "tokenizers.pre_tokenizers", name=PreTokenizer)]
#[derive(Clone)]
pub struct PyPreTokenizer {
    pub pretok: Arc<dyn PreTokenizer>,
}

impl PyPreTokenizer {
    pub fn new(pretok: Arc<dyn PreTokenizer>) -> Self {
        PyPreTokenizer { pretok }
    }
}

impl Serialize for PyPreTokenizer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.pretok.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PyPreTokenizer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Arc::deserialize(deserializer).map(PyPreTokenizer::new)
    }
}

#[typetag::serde]
impl PreTokenizer for PyPreTokenizer {
    fn pre_tokenize(&self, normalized: &mut PreTokenizedString) -> tk::Result<()> {
        self.pretok.pre_tokenize(normalized)
    }
}

#[pymethods]
impl PyPreTokenizer {
    // #[staticmethod]
    // fn custom(pretok: PyObject) -> PyResult<Self> {
    //     let py_pretok = CustomPreTokenizer::new(pretok)?;
    //     Ok(PyPreTokenizer {
    //         pretok: Arc::new(py_pretok),
    //     })
    // }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let data = serde_json::to_string(&self.pretok.as_ref()).map_err(|e| {
            exceptions::Exception::py_err(format!(
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
                    exceptions::Exception::py_err(format!(
                        "Error while attempting to unpickle PreTokenizer: {}",
                        e.to_string()
                    ))
                })?;
                self.pretok = unpickled;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn pre_tokenize(&self, s: &str) -> PyResult<Vec<(String, Offsets)>> {
        // TODO: Expose the PreTokenizedString
        let mut pretokenized = tk::tokenizer::PreTokenizedString::from(s);

        ToPyResult(self.pretok.pre_tokenize(&mut pretokenized)).into_py()?;

        Ok(pretokenized
            .get_normalized(tk::OffsetReferential::Original)
            .into_iter()
            .map(|(s, o)| (s.to_owned(), o))
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

        Ok((PyByteLevel {}, PyPreTokenizer::new(Arc::new(byte_level))))
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
        Ok((
            PyWhitespace {},
            PyPreTokenizer::new(Arc::new(Whitespace::default())),
        ))
    }
}

#[pyclass(extends=PyPreTokenizer, module = "tokenizers.pre_tokenizers", name=WhitespaceSplit)]
pub struct PyWhitespaceSplit {}
#[pymethods]
impl PyWhitespaceSplit {
    #[new]
    fn new() -> PyResult<(Self, PyPreTokenizer)> {
        Ok((
            PyWhitespaceSplit {},
            PyPreTokenizer::new(Arc::new(WhitespaceSplit)),
        ))
    }
}

#[pyclass(extends=PyPreTokenizer, module = "tokenizers.pre_tokenizers", name=CharDelimiterSplit)]
pub struct PyCharDelimiterSplit {}
#[pymethods]
impl PyCharDelimiterSplit {
    #[new]
    pub fn new(delimiter: &str) -> PyResult<(Self, PyPreTokenizer)> {
        let chr_delimiter = delimiter
            .chars()
            .nth(0)
            .ok_or(exceptions::Exception::py_err(
                "delimiter must be a single character",
            ))?;
        Ok((
            PyCharDelimiterSplit {},
            PyPreTokenizer::new(Arc::new(CharDelimiterSplit::new(chr_delimiter))),
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
        Ok((
            PyBertPreTokenizer {},
            PyPreTokenizer::new(Arc::new(BertPreTokenizer)),
        ))
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
            PyMetaspace {},
            PyPreTokenizer::new(Arc::new(Metaspace::new(replacement, add_prefix_space))),
        ))
    }
}

// struct CustomPreTokenizer {
//     class: PyObject,
// }
//
// impl CustomPreTokenizer {
//     pub fn new(class: PyObject) -> PyResult<Self> {
//         Ok(CustomPreTokenizer { class })
//     }
// }
//
// #[typetag::serde]
// impl tk::tokenizer::PreTokenizer for CustomPreTokenizer {
//     fn pre_tokenize(&self, sentence: &mut NormalizedString) -> tk::Result<Vec<(String, Offsets)>> {
//         let gil = Python::acquire_gil();
//         let py = gil.python();
//
//         let args = PyTuple::new(py, &[sentence.get()]);
//         match self.class.call_method(py, "pre_tokenize", args, None) {
//             Ok(res) => Ok(res
//                 .cast_as::<PyList>(py)
//                 .map_err(|_| {
//                     PyError::from("`pre_tokenize is expected to return a List[(str, (uint, uint))]")
//                 })?
//                 .extract::<Vec<(String, Offsets)>>()
//                 .map_err(|_| {
//                     PyError::from(
//                         "`pre_tokenize` is expected to return a List[(str, (uint, uint))]",
//                     )
//                 })?),
//             Err(e) => {
//                 e.print(py);
//                 Err(Box::new(PyError::from(
//                     "Error while calling `pre_tokenize`",
//                 )))
//             }
//         }
//     }
// }
//
// impl Serialize for CustomPreTokenizer {
//     fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
//     where
//         S: Serializer,
//     {
//         Err(serde::ser::Error::custom(
//             "Custom PyPreTokenizer cannot be serialized",
//         ))
//     }
// }
//
// impl<'de> Deserialize<'de> for CustomPreTokenizer {
//     fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
//     where
//         D: Deserializer<'de>,
//     {
//         Err(D::Error::custom("PyDecoder cannot be deserialized"))
//     }
// }
