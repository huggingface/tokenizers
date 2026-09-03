use pyo3::PyErr;
use pyo3::exceptions::PyValueError;

/// A `tk_encode`/`tk_convert`/`tk_serialize` error becomes a plain `ValueError`: no caller
/// distinguishes error kinds yet.
pub(crate) fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}
