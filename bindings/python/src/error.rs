use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::type_object::PyTypeObject;
use std::fmt::{Display, Formatter, Result as FmtResult};
use tokenizers::tokenizer::Result;

#[derive(Debug)]
pub struct PyError(pub String);
impl PyError {
    #[allow(dead_code)]
    pub fn from(s: &str) -> Self {
        PyError(String::from(s))
    }
    pub fn into_pyerr<T: PyTypeObject>(self) -> PyErr {
        PyErr::new::<T, _>(format!("{}", self))
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
        self.0
            .map_err(|e| exceptions::PyException::new_err(format!("{}", e)))
    }
}
impl<T> ToPyResult<T> {
    pub fn into_py(self) -> PyResult<T> {
        self.into()
    }
}

pub(crate) fn deprecation_warning(version: &str, message: &str) -> PyResult<()> {
    let gil = pyo3::Python::acquire_gil();
    let python = gil.python();
    let deprecation_warning = python.import("builtins")?.get("DeprecationWarning")?;
    let full_message = format!("Deprecated in {}: {}", version, message);
    pyo3::PyErr::warn(python, deprecation_warning, &full_message, 0)
}
