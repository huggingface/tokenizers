//! The pattern argument of the components that search the text, like `Split` and `Replace`.
//!
//! Python passes either a `str`, matched literally, or a [`Regex`]. [`Pattern`] takes both and
//! converts them to the JSON `tokenizer.json` stores.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use tk::utils::SysRegex;

use crate::error::err;

// `Deserialize` skips the compile check: it reads a pattern back from a spec, which `compile()`
// checks.
#[pyclass(frozen, from_py_object, module = "tokenizers")]
#[derive(Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Regex {
    pattern: String,
}

#[pymethods]
impl Regex {
    /// Fails on a pattern the rust regex engine cannot compile.
    #[new]
    fn new(pattern: String) -> PyResult<Self> {
        SysRegex::new(&pattern).map_err(err)?;
        Ok(Self { pattern })
    }
}

/// A pattern from `tokenizer.json`: `{"String": ...}` or `{"Regex": ...}`.
#[derive(FromPyObject, IntoPyObject, Serialize, Deserialize)]
pub(crate) enum Pattern {
    String(String),
    Regex(Regex),
}
