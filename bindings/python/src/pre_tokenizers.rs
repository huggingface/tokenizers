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

/// Base class for all pre-tokenizers.
///
/// A pre-tokenizer cuts text into pieces (usually words); the model then turns
/// each piece into token ids. Pre-tokenizers are immutable values — assigning
/// one to a tokenizer copies it. Only pre-tokenizers the encode pipeline can
/// run are constructible here; `Metaspace` is not available yet.
#[pyclass(
    frozen,
    subclass,
    name = "PreTokenizer",
    module = "tokenizers.pre_tokenizers"
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
    ($pyname:ident, $name:literal, $inner:expr, $doc:literal) => {
        #[doc = $doc]
        #[pyclass(frozen, extends = PyPreTokenizer, name = $name, module = "tokenizers.pre_tokenizers")]
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

unit_pre_tokenizer!(
    PyWhitespace,
    "Whitespace",
    Whitespace,
    "Splits into runs of letters/digits/underscore or runs of other symbols (the pattern `\\w+|[^\\w\\s]+`)."
);
unit_pre_tokenizer!(
    PyWhitespaceSplit,
    "WhitespaceSplit",
    WhitespaceSplit,
    "Splits on whitespace only."
);
unit_pre_tokenizer!(
    PyBertPreTokenizer,
    "BertPreTokenizer",
    BertPreTokenizer,
    "The BERT split: on whitespace, and each punctuation character becomes its own piece."
);
unit_pre_tokenizer!(
    PyUnicodeScripts,
    "UnicodeScripts",
    UnicodeScripts,
    "Splits where the script changes (Latin to Han, for example), so a piece never mixes alphabets."
);

/// GPT-2 style byte-level splitting: cuts with the GPT-2 regex unless
/// `use_regex=False`. The pipeline does not support `add_prefix_space`, so it
/// is always off.
#[pyclass(frozen, extends = PyPreTokenizer, name = "ByteLevel", module = "tokenizers.pre_tokenizers")]
pub struct PyByteLevel;

#[pymethods]
impl PyByteLevel {
    #[new]
    #[pyo3(signature = (*, use_regex = true))]
    fn new(use_regex: bool) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyPreTokenizer {
            inner: ByteLevel::new(false, true, use_regex).into(),
        })
        .add_subclass(PyByteLevel)
    }

    /// The 256 characters byte-level tokens are spelled with, one per byte
    /// value. Pass it as a trainer's `initial_alphabet` so every byte gets a
    /// token even if it never appears in the training data.
    #[staticmethod]
    fn alphabet() -> Vec<String> {
        ByteLevel::alphabet().iter().map(char::to_string).collect()
    }
}

/// Splits on one fixed character, dropping it.
#[pyclass(frozen, extends = PyPreTokenizer, name = "CharDelimiterSplit", module = "tokenizers.pre_tokenizers")]
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

/// Separates digits from everything else. With `individual_digits=True`,
/// every digit becomes its own piece.
#[pyclass(frozen, extends = PyPreTokenizer, name = "Digits", module = "tokenizers.pre_tokenizers")]
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

/// Cuts the text into pieces of exactly `length` characters (the last one may
/// be shorter).
#[pyclass(frozen, extends = PyPreTokenizer, name = "FixedLength", module = "tokenizers.pre_tokenizers")]
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

/// Splits on punctuation. `behavior` says what happens to the punctuation
/// itself — see `Split` for the options; the default, "isolated", keeps each
/// punctuation character as its own piece.
#[pyclass(frozen, extends = PyPreTokenizer, name = "Punctuation", module = "tokenizers.pre_tokenizers")]
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

/// Splits on a pattern: a literal string, or a regular expression with
/// `regex=True`. `behavior` says what to do with each match — "removed" drops
/// it, "isolated" keeps it as its own piece, "merged_with_previous" /
/// "merged_with_next" glue it to a neighbor, "contiguous" merges runs of
/// matches. `invert=True` keeps the matches and splits everything else.
#[pyclass(frozen, extends = PyPreTokenizer, name = "Split", module = "tokenizers.pre_tokenizers")]
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

/// Runs several pre-tokenizers in order, each one further splitting the
/// pieces left by the previous.
#[pyclass(frozen, extends = PyPreTokenizer, name = "Sequence", module = "tokenizers.pre_tokenizers")]
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

/// How text is cut into pieces before the model runs.
#[pymodule(gil_used = false)]
pub mod pre_tokenizers {
    #[pymodule_export]
    pub use super::{
        PyBertPreTokenizer, PyByteLevel, PyCharDelimiterSplit, PyDigits, PyFixedLength,
        PyPreTokenizer, PyPunctuation, PySequence, PySplit, PyUnicodeScripts, PyWhitespace,
        PyWhitespaceSplit,
    };
}
