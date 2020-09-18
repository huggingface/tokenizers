use onig::Regex;
use pyo3::exceptions;
use pyo3::prelude::*;

#[pyclass(module = "tokenizers", name=Regex)]
pub struct PyRegex {
    pub inner: Regex,
}

#[pymethods]
impl PyRegex {
    #[new]
    fn new(s: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Regex::new(s)
                .map_err(|e| exceptions::PyException::new_err(e.description().to_owned()))?,
        })
    }
}
