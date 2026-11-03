use std::sync::Mutex;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tk::pipeline::HandleIter;

use crate::encoding::Encoding;
use crate::error::err;

/// The encodings of a batch as they complete: `(index, Encoding)` pairs, `index` being the input's
/// position in the batch. Returned by `CompiledTokenizer.encode_iter`.
#[pyclass(module = "tokenizers")]
pub struct EncodeIter {
    // `HandleIter` is `Send` but not `Sync`; the lock is only ever taken by the iterating thread.
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

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<(usize, Encoding)>> {
        // py.detach releases the GIL while waiting for the next completion.
        let next = py.detach(|| {
            let mut inner = self
                .inner
                .lock()
                .map_err(|_| PyRuntimeError::new_err("a previous iteration panicked"))?;
            Ok::<_, PyErr>(inner.next())
        })?;
        match next {
            None => Ok(None),
            Some((index, result)) => Ok(Some((index, Encoding::from(result.map_err(err)?)))),
        }
    }
}
