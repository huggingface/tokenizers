use pyo3::prelude::*;
use tk_encode::normalizers::{
    BertNormalizer, Lowercase, NFC, NFD, NFKC, NFKD, NormalizerWrapper, Prepend, Replace, Sequence,
    Strip, StripAccents,
};

use crate::error::to_pyerr;

/// Base class for all normalizers. Immutable value: assigning it to a
/// Tokenizer copies the configuration, there is no shared state.
#[pyclass(
    frozen,
    subclass,
    name = "Normalizer",
    module = "tokenizers_pipeline.normalizers"
)]
pub struct PyNormalizer {
    pub inner: NormalizerWrapper,
}

#[pymethods]
impl PyNormalizer {
    fn __repr__(&self) -> String {
        crate::component_repr(&self.inner)
    }
}

pub fn wrap_normalizer(py: Python<'_>, inner: NormalizerWrapper) -> PyResult<Py<PyNormalizer>> {
    let base = PyNormalizer {
        inner: inner.clone(),
    };
    let init = PyClassInitializer::from(base);
    let obj = match inner {
        NormalizerWrapper::BertNormalizer(_) => {
            Bound::new(py, init.add_subclass(PyBertNormalizer))?.into_super()
        }
        NormalizerWrapper::StripNormalizer(_) => {
            Bound::new(py, init.add_subclass(PyStrip))?.into_super()
        }
        NormalizerWrapper::StripAccents(_) => {
            Bound::new(py, init.add_subclass(PyStripAccents))?.into_super()
        }
        NormalizerWrapper::NFC(_) => Bound::new(py, init.add_subclass(PyNFC))?.into_super(),
        NormalizerWrapper::NFD(_) => Bound::new(py, init.add_subclass(PyNFD))?.into_super(),
        NormalizerWrapper::NFKC(_) => Bound::new(py, init.add_subclass(PyNFKC))?.into_super(),
        NormalizerWrapper::NFKD(_) => Bound::new(py, init.add_subclass(PyNFKD))?.into_super(),
        NormalizerWrapper::Sequence(_) => {
            Bound::new(py, init.add_subclass(PySequence))?.into_super()
        }
        NormalizerWrapper::Lowercase(_) => {
            Bound::new(py, init.add_subclass(PyLowercase))?.into_super()
        }
        NormalizerWrapper::Replace(_) => Bound::new(py, init.add_subclass(PyReplace))?.into_super(),
        NormalizerWrapper::Prepend(_) => Bound::new(py, init.add_subclass(PyPrepend))?.into_super(),
        // Loadable from tokenizer.json but not constructible from Python: exposed as the base class.
        NormalizerWrapper::Nmt(_)
        | NormalizerWrapper::Precompiled(_)
        | NormalizerWrapper::ByteLevel(_) => Bound::new(py, init)?,
    };
    Ok(obj.unbind())
}

macro_rules! unit_normalizer {
    ($pyname:ident, $name:literal, $inner:expr) => {
        #[pyclass(frozen, extends = PyNormalizer, name = $name, module = "tokenizers_pipeline.normalizers")]
        pub struct $pyname;

        #[pymethods]
        impl $pyname {
            #[new]
            fn new() -> PyClassInitializer<Self> {
                PyClassInitializer::from(PyNormalizer { inner: $inner.into() }).add_subclass($pyname)
            }
        }
    };
}

unit_normalizer!(PyNFC, "NFC", NFC);
unit_normalizer!(PyNFD, "NFD", NFD);
unit_normalizer!(PyNFKC, "NFKC", NFKC);
unit_normalizer!(PyNFKD, "NFKD", NFKD);
unit_normalizer!(PyLowercase, "Lowercase", Lowercase);
unit_normalizer!(PyStripAccents, "StripAccents", StripAccents);

#[pyclass(frozen, extends = PyNormalizer, name = "Strip", module = "tokenizers_pipeline.normalizers")]
pub struct PyStrip;

#[pymethods]
impl PyStrip {
    #[new]
    #[pyo3(signature = (*, left = true, right = true))]
    fn new(left: bool, right: bool) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyNormalizer {
            inner: Strip::new(left, right).into(),
        })
        .add_subclass(PyStrip)
    }
}

#[pyclass(frozen, extends = PyNormalizer, name = "Replace", module = "tokenizers_pipeline.normalizers")]
pub struct PyReplace;

#[pymethods]
impl PyReplace {
    #[new]
    #[pyo3(signature = (pattern, content, *, regex = false))]
    fn new(pattern: &str, content: &str, regex: bool) -> PyResult<PyClassInitializer<Self>> {
        use tk_encode::normalizers::replace::ReplacePattern;
        let pattern = if regex {
            ReplacePattern::Regex(pattern.to_owned())
        } else {
            ReplacePattern::String(pattern.to_owned())
        };
        let replace = Replace::new(pattern, content).map_err(to_pyerr)?;
        Ok(PyClassInitializer::from(PyNormalizer {
            inner: replace.into(),
        })
        .add_subclass(PyReplace))
    }
}

#[pyclass(frozen, extends = PyNormalizer, name = "Prepend", module = "tokenizers_pipeline.normalizers")]
pub struct PyPrepend;

#[pymethods]
impl PyPrepend {
    #[new]
    fn new(prepend: String) -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyNormalizer {
            inner: Prepend::new(prepend).into(),
        })
        .add_subclass(PyPrepend)
    }
}

#[pyclass(frozen, extends = PyNormalizer, name = "BertNormalizer", module = "tokenizers_pipeline.normalizers")]
pub struct PyBertNormalizer;

#[pymethods]
impl PyBertNormalizer {
    #[new]
    #[pyo3(signature = (*, clean_text = true, handle_chinese_chars = true, strip_accents = None, lowercase = true))]
    fn new(
        clean_text: bool,
        handle_chinese_chars: bool,
        strip_accents: Option<bool>,
        lowercase: bool,
    ) -> PyClassInitializer<Self> {
        let inner = BertNormalizer::new(clean_text, handle_chinese_chars, strip_accents, lowercase);
        PyClassInitializer::from(PyNormalizer {
            inner: inner.into(),
        })
        .add_subclass(PyBertNormalizer)
    }
}

#[pyclass(frozen, extends = PyNormalizer, name = "Sequence", module = "tokenizers_pipeline.normalizers")]
pub struct PySequence;

#[pymethods]
impl PySequence {
    #[new]
    fn new(normalizers: Vec<PyRef<'_, PyNormalizer>>) -> PyClassInitializer<Self> {
        let inner: Vec<NormalizerWrapper> = normalizers.iter().map(|n| n.inner.clone()).collect();
        PyClassInitializer::from(PyNormalizer {
            inner: Sequence::new(inner).into(),
        })
        .add_subclass(PySequence)
    }
}

#[pymodule(gil_used = false)]
pub mod normalizers {
    #[pymodule_export]
    pub use super::{
        PyBertNormalizer, PyLowercase, PyNFC, PyNFD, PyNFKC, PyNFKD, PyNormalizer, PyPrepend,
        PyReplace, PySequence, PyStrip, PyStripAccents,
    };
}
