use std::sync::{Mutex, PoisonError};

use pyo3::exceptions::{PyStopIteration, PyValueError};
use pyo3::prelude::*;
use tk::pipeline::HandleIter;

use crate::encoding::Encoding;

/// The encodings of a batch as they complete: `(index, Encoding)` pairs, `index` being the input's
/// position in the batch. Returned by `CompiledTokenizer.encode_iter`.
#[pyclass(frozen, module = "tokenizers")]
pub struct EncodeIter {
    // `HandleIter` is `Send` but not `Sync`
    inner: Mutex<HandleIter>,
}

impl EncodeIter {
    pub fn new(inner: HandleIter) -> Self {
        Self {
            inner: Mutex::new(inner),
        }
    }
}

#[pymethods]
impl EncodeIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Raises `StopIteration` itself rather than returning `None`, so the stub's item type is not
    /// `Optional`.
    fn __next__(&self, py: Python<'_>) -> PyResult<(usize, Encoding)> {
        // py.detach releases the GIL while waiting for the next completion. A panic in an earlier
        // call doesn't lose the encodings still pending: ignore the poison.
        let next = py.detach(|| {
            self.inner
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .next()
        });
        match next {
            None => Err(PyStopIteration::new_err(())),
            Some((index, Ok(encoding))) => Ok((index, Encoding::from(encoding))),
            Some((index, Err(e))) => Err(PyValueError::new_err(format!("input {index}: {e}"))),
        }
    }
}
