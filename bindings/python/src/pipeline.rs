//! Pipeline python bindings

use std::sync::Mutex;

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::PyString;

use tk::pipeline::{EncodeHandle, Inputs, IntoInputs, PipelineToken, PipelineTokenizer};

use crate::error::ToPyResult;
use crate::tokenizer::PyTokenizer;

fn ids(tokens: Vec<PipelineToken>) -> Vec<u32> {
    tokens.iter().map(|t| t.id).collect()
}

struct PyStrView {
    ptr: *const u8,
    len: usize,
}

/// Keep alive mechanism for zero-copy access to utf8 python strings
struct PyStrBatch {
    /// the keep alive ref
    _owners: Vec<Py<PyString>>,
    /// (ptr, len)
    views: Vec<PyStrView>,
}

// SAFETY: the raw pointers reference CPython's cached UTF-8 buffers, which are
// immutable and outlive the owning `Py<PyString>`s in `_owners`; the batch
// never mutates or frees them, and pool workers only read through `get`.
unsafe impl Send for PyStrBatch {}
unsafe impl Sync for PyStrBatch {}

impl PyStrBatch {
    fn new(strings: Vec<Bound<'_, PyString>>) -> PyResult<Self> {
        let mut owners = Vec::with_capacity(strings.len());
        let mut views = Vec::with_capacity(strings.len());
        for s in strings {
            let view = s.to_str()?;
            views.push(PyStrView {
                ptr: view.as_ptr(),
                len: view.len(),
            });
            owners.push(s.unbind());
        }
        Ok(Self {
            _owners: owners,
            views,
        })
    }
}

impl Inputs for PyStrBatch {
    fn len(&self) -> usize {
        self.views.len()
    }

    fn get(&self, i: usize) -> &str {
        let (ptr, len) = (self.views[i].ptr, self.views[i].len);
        // SAFETY: ptr is kept alive by PyStrBatch::_owners, and the underlying buffer is immutable UTF-8
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len)) }
    }
}

impl IntoInputs for PyStrBatch {
    type Inputs = PyStrBatch;
    fn into_inputs(self) -> PyStrBatch {
        self
    }
}

#[pyclass(module = "tokenizers", name = "PipelineTokenizer")]
pub struct PyPipelineTokenizer {
    pipeline: PipelineTokenizer,
}

#[pymethods]
impl PyPipelineTokenizer {
    #[staticmethod]
    fn from_tokenizer(tokenizer: &PyTokenizer) -> PyResult<Self> {
        let json = {
            let guard = tokenizer.read_inner()?;
            serde_json::to_string(&*guard)
                .map_err(|e| exceptions::PyException::new_err(format!("{e}")))?
        };
        let tok: tk::Tokenizer = serde_json::from_str(&json)
            .map_err(|e| exceptions::PyException::new_err(format!("{e}")))?;
        let pipeline = PyResult::from(ToPyResult(PipelineTokenizer::try_from(&tok)))?;
        Ok(Self { pipeline })
    }

    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        let tok = PyResult::from(ToPyResult(tk::Tokenizer::from_file(path)))?;
        let pipeline = PyResult::from(ToPyResult(PipelineTokenizer::try_from(&tok)))?;
        Ok(Self { pipeline })
    }

    /// Encode a batch of `str`, returning an :class:`EncodeHandle` **immediately**:
    /// pool workers encode in the background while you hold the job, `wait()`
    /// for everything, or iterate results as they complete (input order).
    /// Zero-copy: the job reads the Python strings' UTF-8 buffers in place.
    fn encode_batch(
        &self,
        py: Python<'_>,
        input: Vec<Bound<'_, PyString>>,
    ) -> PyResult<PyEncodeHandle> {
        let batch = PyStrBatch::new(input)?;
        // Below the cost gate the job is computed inline — release the GIL.
        let job = py.detach(|| self.pipeline.encode(batch));
        Ok(PyEncodeHandle {
            job: Mutex::new(Some(job)),
        })
    }
}

/// An in-flight (or completed) batch encode. Results surface in input order:
/// `wait()` blocks for all of them; iterating yields each input's ids as it
/// becomes ready. The consuming thread *assists* the pool while it waits (all
/// with the GIL released). Dropping the job cancels unclaimed work.
#[pyclass(module = "tokenizers", name = "EncodeHandle")]
pub struct PyEncodeHandle {
    job: Mutex<Option<EncodeHandle>>,
}

impl PyEncodeHandle {
    fn take(&self) -> PyResult<EncodeHandle> {
        self.job
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .ok_or_else(|| exceptions::PyException::new_err("EncodeHandle already consumed"))
    }
}

#[pymethods]
impl PyEncodeHandle {
    /// Block until every input is encoded, returning ids lists in input order
    fn wait(&self, py: Python<'_>) -> PyResult<Vec<Vec<u32>>> {
        let job = self.take()?;
        let res = py.detach(|| job.wait_for_completion());
        let out = PyResult::from(ToPyResult(res))?;
        Ok(out.into_iter().map(ids).collect())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<Vec<u32>>> {
        let next = py.detach(|| {
            self.job
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_mut()
                .and_then(|j| j.next())
        });
        match next {
            None => Ok(None),
            Some(res) => Ok(Some(ids(PyResult::from(ToPyResult(res))?))),
        }
    }
}
