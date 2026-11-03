use pyo3::PyErr;
use pyo3::exceptions::PyValueError;

/// Convert a rust error to a Python ValueError
pub(crate) fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}
