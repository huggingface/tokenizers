use pyo3::exceptions;
use pyo3::prelude::*;
use tk::utils::SysRegex;

/// Instantiate a new Regex with the given pattern
#[pyclass(module = "tokenizers", name = "Regex")]
pub struct PyRegex {
    pub inner: SysRegex,
    pub pattern: String,
}

#[pymethods]
impl PyRegex {
    #[new]
    #[pyo3(text_signature = "(self, pattern)")]
    fn new(s: &str) -> PyResult<Self> {
        Ok(Self {
            inner: SysRegex::new(s)
                .map_err(|e| exceptions::PyException::new_err(e.to_string().to_owned()))?,
            pattern: s.to_owned(),
        })
    }
}
