//! [`Regex`], and [`Pattern`]: the `str | Regex` argument of the components that search the text.

use pyo3::prelude::*;
use serde_json::{Value, json};
use tk::utils::SysRegex;

use crate::error::err;

/// A regular expression, for a component's pattern. A plain `str` pattern is a literal.
#[pyclass(frozen, module = "tokenizers")]
pub struct Regex {
    pattern: String,
}

#[pymethods]
impl Regex {
    /// Fails on a pattern the regex engine cannot compile.
    #[new]
    fn new(pattern: String) -> PyResult<Self> {
        SysRegex::new(&pattern).map_err(err)?;
        Ok(Self { pattern })
    }
}

/// A pattern as `tokenizer.json` writes it: `{"String": ...}` or `{"Regex": ...}`.
#[derive(FromPyObject)]
pub(crate) enum Pattern {
    Str(String),
    Regex(Py<Regex>),
}

impl From<Pattern> for Value {
    fn from(pattern: Pattern) -> Self {
        match pattern {
            Pattern::Str(literal) => json!({"String": literal}),
            Pattern::Regex(regex) => json!({"Regex": regex.get().pattern}),
        }
    }
}
