use pyo3::prelude::*;
use tk_encode::models::ModelWrapper;
use tk_encode::models::bpe::BPE;
use tk_encode::models::unigram::Unigram;
use tk_encode::models::wordlevel::WordLevel;
use tk_encode::models::wordpiece::WordPiece;

use crate::error::to_pyerr;

/// Base class for all models. Not constructible from Python; holds the actual
/// Rust model by value (no sharing with the Tokenizer — assignment copies).
#[pyclass(
    frozen,
    subclass,
    name = "Model",
    module = "tokenizers_pipeline.models"
)]
pub struct PyModel {
    pub inner: ModelWrapper,
}

#[pymethods]
impl PyModel {
    fn __repr__(&self) -> String {
        crate::component_repr(&self.inner)
    }
}

pub fn wrap_model(py: Python<'_>, inner: ModelWrapper) -> PyResult<Py<PyModel>> {
    let base = PyModel {
        inner: inner.clone(),
    };
    let init = PyClassInitializer::from(base);
    let obj = match inner {
        ModelWrapper::BPE(_) => Bound::new(py, init.add_subclass(PyBPE))?.into_super(),
        ModelWrapper::WordPiece(_) => Bound::new(py, init.add_subclass(PyWordPiece))?.into_super(),
        ModelWrapper::WordLevel(_) => Bound::new(py, init.add_subclass(PyWordLevel))?.into_super(),
        ModelWrapper::Unigram(_) => Bound::new(py, init.add_subclass(PyUnigram))?.into_super(),
    };
    Ok(obj.unbind())
}

#[pyclass(frozen, extends = PyModel, name = "BPE", module = "tokenizers_pipeline.models")]
pub struct PyBPE;

#[pymethods]
impl PyBPE {
    #[new]
    #[pyo3(signature = (*, unk_token = None, dropout = None, fuse_unk = false, byte_fallback = false, ignore_merges = false))]
    fn new(
        unk_token: Option<String>,
        dropout: Option<f32>,
        fuse_unk: bool,
        byte_fallback: bool,
        ignore_merges: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        let mut builder = BPE::builder()
            .fuse_unk(fuse_unk)
            .byte_fallback(byte_fallback)
            .ignore_merges(ignore_merges);
        if let Some(unk) = unk_token {
            builder = builder.unk_token(unk);
        }
        if let Some(d) = dropout {
            builder = builder.dropout(d);
        }
        let bpe = builder.build().map_err(to_pyerr)?;
        Ok(PyClassInitializer::from(PyModel { inner: bpe.into() }).add_subclass(PyBPE))
    }

    /// Load a BPE from the legacy vocab.json + merges.txt format.
    #[staticmethod]
    #[pyo3(signature = (vocab, merges, *, unk_token = None) -> "BPE")]
    fn from_file(
        py: Python<'_>,
        vocab: &str,
        merges: &str,
        unk_token: Option<String>,
    ) -> PyResult<Py<PyModel>> {
        let mut builder = BPE::from_file(vocab, merges);
        if let Some(unk) = unk_token {
            builder = builder.unk_token(unk);
        }
        let bpe = py.detach(|| builder.build()).map_err(to_pyerr)?;
        wrap_model(py, bpe.into())
    }
}

#[pyclass(frozen, extends = PyModel, name = "WordPiece", module = "tokenizers_pipeline.models")]
pub struct PyWordPiece;

#[pymethods]
impl PyWordPiece {
    #[new]
    #[pyo3(signature = (*, unk_token = String::from("[UNK]"), continuing_subword_prefix = String::from("##"), max_input_chars_per_word = 100))]
    fn new(
        unk_token: String,
        continuing_subword_prefix: String,
        max_input_chars_per_word: usize,
    ) -> PyResult<PyClassInitializer<Self>> {
        let wp = WordPiece::builder()
            .unk_token(unk_token)
            .continuing_subword_prefix(continuing_subword_prefix)
            .max_input_chars_per_word(max_input_chars_per_word)
            .build()
            .map_err(to_pyerr)?;
        Ok(PyClassInitializer::from(PyModel { inner: wp.into() }).add_subclass(PyWordPiece))
    }
}

#[pyclass(frozen, extends = PyModel, name = "WordLevel", module = "tokenizers_pipeline.models")]
pub struct PyWordLevel;

#[pymethods]
impl PyWordLevel {
    #[new]
    #[pyo3(signature = (*, unk_token = String::from("[UNK]")))]
    fn new(unk_token: String) -> PyResult<PyClassInitializer<Self>> {
        let wl = WordLevel::builder()
            .unk_token(unk_token)
            .build()
            .map_err(to_pyerr)?;
        Ok(PyClassInitializer::from(PyModel { inner: wl.into() }).add_subclass(PyWordLevel))
    }
}

#[pyclass(frozen, extends = PyModel, name = "Unigram", module = "tokenizers_pipeline.models")]
pub struct PyUnigram;

#[pymethods]
impl PyUnigram {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyModel {
            inner: Unigram::default().into(),
        })
        .add_subclass(PyUnigram)
    }
}

#[pymodule(gil_used = false)]
pub mod models {
    #[pymodule_export]
    pub use super::{PyBPE, PyModel, PyUnigram, PyWordLevel, PyWordPiece};
}
