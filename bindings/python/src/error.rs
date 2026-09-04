use std::sync::PoisonError;

use pyo3::PyErr;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use tk_convert::ConvertError;

/// Convert a rust error to a Python ValueError
pub(crate) fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Convert a poisoned lock to a Python RuntimeError
pub(crate) fn poison_err<T>(_: PoisonError<T>) -> PyErr {
    PyRuntimeError::new_err("a previous access panicked while holding this lock")
}

/// Map IO error to the right Python types (FileNotFoundError, ...)
pub(crate) fn convert_err(e: ConvertError) -> PyErr {
    match &e {
        ConvertError::Io { source, .. } => std::io::Error::new(source.kind(), e.to_string()).into(),
        _ => err(e),
    }
}
