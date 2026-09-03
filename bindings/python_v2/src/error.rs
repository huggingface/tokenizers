use pyo3::PyErr;
use pyo3::exceptions::PyValueError;
use tk_convert::ConvertError;

/// A `tk_encode`/`tk_serialize` error becomes a plain `ValueError`: no caller distinguishes
/// error kinds yet.
pub(crate) fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A file that cannot be read raises what `open()` would have, `FileNotFoundError` for a
/// missing one; anything wrong with the file's contents is a `ValueError`.
pub(crate) fn convert_err(e: ConvertError) -> PyErr {
    match &e {
        ConvertError::Io { source, .. } => std::io::Error::new(source.kind(), e.to_string()).into(),
        _ => err(e),
    }
}
