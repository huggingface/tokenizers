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
        match self.0 {
            Ok(o) => Ok(o),
            Err(e) => Err(exceptions::Exception::py_err(format!("{}", e))),
        }
    }
}
