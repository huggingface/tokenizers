use pyo3::exceptions;
use pyo3::prelude::*;
use std::fmt::{Display, Formatter, Result as FmtResult};
use tokenizers::tokenizer::Result;

#[derive(Debug)]
pub struct PyError(pub String);
impl PyError {
    pub fn from(s: &str) -> Self {
        PyError(String::from(s))
    }
    pub fn into_pyerr(self) -> PyErr {
        exceptions::Exception::py_err(format!("{}", self))
    }
}
impl Display for PyError {
    fn fmt(&self, fmt: &mut Formatter) -> FmtResult {
        write!(fmt, "{}", self.0)
    }
}
impl std::error::Error for PyError {}

pub struct ToPyResult<T>(pub Result<T>);
impl<T> std::convert::Into<PyResult<T>> for ToPyResult<T> {
    fn into(self) -> PyResult<T> {
        self.0.map_err(|e| { exceptions::Exception::py_err(format!("{}", e)) })
    }
}
