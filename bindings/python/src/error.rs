use std::fmt;

use pyo3::PyErr;
use pyo3::exceptions::PyValueError;

/// Convert a rust error to a Python ValueError
pub(crate) fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A `Sequence` given another `Sequence` as an item.
#[derive(Debug)]
pub(crate) struct NestedSequenceError;

impl fmt::Display for NestedSequenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a Sequence item can't be a Sequence")
    }
}

impl From<NestedSequenceError> for PyErr {
    fn from(error: NestedSequenceError) -> Self {
        err(error)
    }
}
