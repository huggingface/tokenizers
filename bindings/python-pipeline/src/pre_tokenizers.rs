use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tk_encode::pre_tokenizers::PreTokenizerWrapper;
use tk_encode::pre_tokenizers::bert::BertPreTokenizer;
use tk_encode::pre_tokenizers::byte_level::ByteLevel;
use tk_encode::pre_tokenizers::delimiter::CharDelimiterSplit;
use tk_encode::pre_tokenizers::digits::Digits;
use tk_encode::pre_tokenizers::fixed_length::FixedLength;
use tk_encode::pre_tokenizers::punctuation::Punctuation;
use tk_encode::pre_tokenizers::sequence::Sequence;
use tk_encode::pre_tokenizers::split::{Split, SplitPattern};
use tk_encode::pre_tokenizers::unicode_scripts::UnicodeScripts;
use tk_encode::pre_tokenizers::whitespace::{Whitespace, WhitespaceSplit};
use tk_encode::tokenizer::SplitDelimiterBehavior;

use crate::error::to_pyerr;

pub fn parse_behavior(s: &str) -> PyResult<SplitDelimiterBehavior> {
    match s {
        "removed" => Ok(SplitDelimiterBehavior::Removed),
        "isolated" => Ok(SplitDelimiterBehavior::Isolated),
        "merged_with_previous" => Ok(SplitDelimiterBehavior::MergedWithPrevious),
        "merged_with_next" => Ok(SplitDelimiterBehavior::MergedWithNext),
        "contiguous" => Ok(SplitDelimiterBehavior::Contiguous),
        other => Err(PyValueError::new_err(format!(
            "unknown behavior {other:?}; expected one of: removed, isolated, \
             merged_with_previous, merged_with_next, contiguous"
        ))),
    }
}

/// Base class for all pre-tokenizers. Immutable value: assigning it to a
/// Tokenizer copies the configuration, there is no shared state.
///
/// Only pre-tokenizers supported by the encode pipeline are constructible here;
/// notably `Metaspace` is not available yet.
#[pyclass(
    frozen,
    subclass,
    name = "PreTokenizer",
    module = "tokenizers_pipeline.pre_tokenizers"
)]
pub struct PyPreTokenizer {
    pub inner: PreTokenizerWrapper,
}

#[pymethods]
impl PyPreTokenizer {
    fn __repr__(&self) -> String {
        crate::component_repr(&self.inner)
    }
}

pub fn wrap_pre_tokenizer(
    py: Python<'_>,
    inner: PreTokenizerWrapper,
) -> PyResult<Py<PyPreTokenizer>> {
    let base = PyPreTokenizer {
        inner: inner.clone(),
    };
    let init = PyClassInitializer::from(base);
    let obj = match inner {
        PreTokenizerWrapper::BertPreTokenizer(_) => {
            Bound::new(py, init.add_subclass(PyBertPreTokenizer))?.into_super()
        }
        PreTokenizerWrapper::ByteLevel(_) => {
            Bound::new(py, init.add_subclass(PyByteLevel))?.into_super()
        }
        PreTokenizerWrapper::Delimiter(_) => {
            Bound::new(py, init.add_subclass(PyCharDelimiterSplit))?.into_super()
        }
        PreTokenizerWrapper::Whitespace(_) => {
            Bound::new(py, init.add_subclass(PyWhitespace))?.into_super()
        }
        PreTokenizerWrapper::WhitespaceSplit(_) => {
            Bound::new(py, init.add_subclass(PyWhitespaceSplit))?.into_super()
        }
        PreTokenizerWrapper::Sequence(_) => {
            Bound::new(py, init.add_subclass(PySequence))?.into_super()
        }
        PreTokenizerWrapper::Split(_) => Bound::new(py, init.add_subclass(PySplit))?.into_super(),
        PreTokenizerWrapper::Punctuation(_) => {
            Bound::new(py, init.add_subclass(PyPunctuation))?.into_super()
        }
        PreTokenizerWrapper::Digits(_) => Bound::new(py, init.add_subclass(PyDigits))?.into_super(),
        PreTokenizerWrapper::UnicodeScripts(_) => {
            Bound::new(py, init.add_subclass(PyUnicodeScripts))?.into_super()
        }
        PreTokenizerWrapper::FixedLength(_) => {
            Bound::new(py, init.add_subclass(PyFixedLength))?.into_super()
        }
        // Loadable from tokenizer.json but not constructible from Python (and
        // rejected by the pipeline at compile time): exposed as the base class.
        PreTokenizerWrapper::Metaspace(_) => Bound::new(py, init)?,
    };
    Ok(obj.unbind())
}

macro_rules! unit_pre_tokenizer {
    ($pyname:ident, $name:literal, $inner:expr) => {
        #[pyclass(frozen, extends = PyPreTokenizer, name = $name, module = "tokenizers_pipeline.pre_tokenizers")]
        pub struct $pyname;

        #[pymethods]
        impl $pyname {
            #[new]
            fn new() -> PyClassInitializer<Self> {
                PyClassInitializer::from(PyPreTokenizer { inner: $inner.into() }).add_subclass($pyname)
            }
        }
    };
}

unit_pre_tokenizer!(PyWhitespace, "Whitespace", Whitespace);
unit_pre_tokenizer!(PyWhitespaceSplit, "WhitespaceSplit", WhitespaceSplit);
unit_pre_tokenizer!(PyBertPreTokenizer, "BertPreTokenizer", BertPreTokenizer);
unit_pre_tokenizer!(PyUnicodeScripts, "UnicodeScripts", UnicodeScripts);

#[pyclass(frozen, extends = PyPreTokenizer, name = "ByteLevel", module = "tokenizers_pipeline.pre_tokenizers")]
pub struct PyByteLevel;

#[pymethods]
impl PyByteLevel {
    /// `add_prefix_space` is not supported by the pipeline and is always false.
    #[new]
    #[pyo3(signature = (*, use_regex = true))]
    fn new(use_regex: bool) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyPreTokenizer {
            inner: ByteLevel::new(false, true, use_regex).into(),
        })
        .add_subclass(PyByteLevel)
    }
}

#[pyclass(frozen, extends = PyPreTokenizer, name = "CharDelimiterSplit", module = "tokenizers_pipeline.pre_tokenizers")]
pub struct PyCharDelimiterSplit;

#[pymethods]
impl PyCharDelimiterSplit {
    #[new]
    fn new(delimiter: char) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyPreTokenizer {
            inner: CharDelimiterSplit::new(delimiter).into(),
        })
        .add_subclass(PyCharDelimiterSplit)
    }
}

#[pyclass(frozen, extends = PyPreTokenizer, name = "Digits", module = "tokenizers_pipeline.pre_tokenizers")]
pub struct PyDigits;

#[pymethods]
impl PyDigits {
    #[new]
    #[pyo3(signature = (*, individual_digits = false))]
    fn new(individual_digits: bool) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyPreTokenizer {
            inner: Digits::new(individual_digits).into(),
        })
        .add_subclass(PyDigits)
    }
}

#[pyclass(frozen, extends = PyPreTokenizer, name = "FixedLength", module = "tokenizers_pipeline.pre_tokenizers")]
pub struct PyFixedLength;

#[pymethods]
impl PyFixedLength {
    #[new]
    #[pyo3(signature = (*, length = 5))]
    fn new(length: usize) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyPreTokenizer {
            inner: FixedLength::new(length).into(),
        })
        .add_subclass(PyFixedLength)
    }
}

#[pyclass(frozen, extends = PyPreTokenizer, name = "Punctuation", module = "tokenizers_pipeline.pre_tokenizers")]
pub struct PyPunctuation;

#[pymethods]
impl PyPunctuation {
    #[new]
    #[pyo3(signature = (behavior = String::from("isolated")))]
    fn new(behavior: String) -> PyResult<PyClassInitializer<Self>> {
        Ok(PyClassInitializer::from(PyPreTokenizer {
            inner: Punctuation::new(parse_behavior(&behavior)?).into(),
        })
        .add_subclass(PyPunctuation))
    }
}

#[pyclass(frozen, extends = PyPreTokenizer, name = "Split", module = "tokenizers_pipeline.pre_tokenizers")]
pub struct PySplit;

#[pymethods]
impl PySplit {
    #[new]
    #[pyo3(signature = (pattern, behavior = String::from("isolated"), *, invert = false, regex = false))]
    fn new(
        pattern: &str,
        behavior: String,
        invert: bool,
        regex: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        let pattern = if regex {
            SplitPattern::Regex(pattern.to_owned())
        } else {
            SplitPattern::String(pattern.to_owned())
        };
        let split = Split::new(pattern, parse_behavior(&behavior)?, invert).map_err(to_pyerr)?;
        Ok(PyClassInitializer::from(PyPreTokenizer {
            inner: split.into(),
        })
        .add_subclass(PySplit))
    }
}

#[pyclass(frozen, extends = PyPreTokenizer, name = "Sequence", module = "tokenizers_pipeline.pre_tokenizers")]
pub struct PySequence;

#[pymethods]
impl PySequence {
    #[new]
    fn new(pre_tokenizers: Vec<PyRef<'_, PyPreTokenizer>>) -> PyClassInitializer<Self> {
        let inner: Vec<PreTokenizerWrapper> =
            pre_tokenizers.iter().map(|p| p.inner.clone()).collect();
        PyClassInitializer::from(PyPreTokenizer {
            inner: Sequence::new(inner).into(),
        })
        .add_subclass(PySequence)
    }
}

#[pymodule(gil_used = false)]
pub mod pre_tokenizers {
    #[pymodule_export]
    pub use super::{
        PyBertPreTokenizer, PyByteLevel, PyCharDelimiterSplit, PyDigits, PyFixedLength,
        PyPreTokenizer, PyPunctuation, PySequence, PySplit, PyUnicodeScripts, PyWhitespace,
        PyWhitespaceSplit,
    };
}
