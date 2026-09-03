use pyo3::PyErr;
use pyo3::exceptions::PyValueError;
use tk_convert::ConvertError;

/// Convert a rust error to a Python ValueError
pub(crate) fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Map IO error to the right Python types (FileNotFoundError, ...)
pub(crate) fn convert_err(e: ConvertError) -> PyErr {
    match &e {
        ConvertError::Io { source, .. } => std::io::Error::new(source.kind(), e.to_string()).into(),
        _ => err(e),
    }
}
