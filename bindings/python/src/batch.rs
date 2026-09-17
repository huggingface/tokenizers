use std::sync::Arc;

use numpy::ndarray::ArrayView2;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use tk_encode::pipeline::{Encoding as PipelineEncoding, PipelineToken};

use crate::encoding::Encoding;
use crate::type_hints::{U32Array, U32Array2};

/// A batch of encodings, sharing one contiguous id buffer.
///
/// Encoding a batch produces one buffer and the offsets saying where each document sits in it.
/// Whether the batch was padded only changes how it is worth reading back: a padded batch has one
/// row width, so it reads as a `(rows, stride)` array with no copy at all, while an unpadded one
/// is ragged and has to be read row by row. Either way the encode did the same work, and indexing
/// a row is a slice.
#[pyclass(frozen, sequence, module = "tokenizers")]
pub struct Batch {
    inner: Arc<PipelineEncoding>,
}

impl Batch {
    pub(crate) fn new(batch: PipelineEncoding) -> Self {
        Self {
            inner: Arc::new(batch),
        }
    }

    /// The whole id buffer, which is only meaningful row by row unless the batch is rectangular.
    fn ids_slice(&self) -> &[u32] {
        PipelineToken::ids_of(self.inner.ids())
    }

    /// The batch as one `Encoding` per document, for the cases that cannot keep the shared
    /// buffer: pickling it to another process, or comparing against a plain list.
    fn encodings(&self) -> Vec<Encoding> {
        (0..self.inner.rows())
            .map(|i| Encoding::row(Arc::clone(&self.inner), i))
            .collect()
    }

    fn rectangular(&self) -> PyResult<usize> {
        self.inner.stride().ok_or_else(|| {
            PyValueError::new_err(
                "batch rows have different lengths, so it has no 2D shape: pad the batch, or read \
                 its rows one at a time",
            )
        })
    }
}

#[pymethods]
impl Batch {
    /// The number of documents in the batch.
    fn __len__(&self) -> usize {
        self.inner.rows()
    }

    /// The encoding of document `i`, a view over the batch's buffer.
    fn __getitem__(&self, index: isize) -> PyResult<Encoding> {
        let rows = self.inner.rows() as isize;
        let row = if index < 0 { index + rows } else { index };
        if row < 0 || row >= rows {
            return Err(PyIndexError::new_err("batch index out of range"));
        }
        Ok(Encoding::row(Arc::clone(&self.inner), row as usize))
    }

    /// The row width when every document has the same one, else `None`.
    ///
    /// A padded batch is rectangular and can be read as a 2D array; an unpadded one is not.
    #[getter]
    fn stride(&self) -> Option<usize> {
        self.inner.stride()
    }

    /// The ids of every document, as a list of lists.
    #[getter]
    fn ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(
            py,
            (0..self.inner.rows()).map(|i| {
                self.inner
                    .row(i)
                    .map_or(&[][..], PipelineToken::ids_of)
                    .to_vec()
            }),
        )
    }

    /// Every document's ids as one read-only `(rows, stride)` `uint32` numpy array.
    ///
    /// A view over the batch's own buffer, not a copy: the batch built it at this exact shape.
    /// Raises if the batch is not rectangular.
    #[getter]
    fn ids_array<'py>(this: &Bound<'py, Self>) -> PyResult<U32Array2<'py>> {
        let batch = this.get();
        let stride = batch.rectangular()?;
        view2(this, batch.ids_slice(), batch.inner.rows(), stride)
    }

    /// The attention mask of every document as one read-only `(rows, stride)` `uint32` array.
    ///
    /// Unlike the ids this is widened from the batch's `u8` mask, so it does allocate. An
    /// unpadded batch has no mask and reads as all ones.
    #[getter]
    fn attention_mask_array<'py>(this: &Bound<'py, Self>) -> PyResult<U32Array2<'py>> {
        let batch = this.get();
        let stride = batch.rectangular()?;
        let rows = batch.inner.rows();
        let mask: Vec<u32> = match batch.inner.attention_mask() {
            Some(mask) => mask.iter().map(|&b| u32::from(b)).collect(),
            None => vec![1; rows * stride],
        };
        let array = PyArray1::from_vec(this.py(), mask)
            .reshape((rows, stride))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        array.readwrite().make_nonwriteable();
        Ok(U32Array2(array))
    }

    /// Where each document starts in the flat buffer, `len(batch) + 1` entries.
    #[getter]
    fn offsets<'py>(this: &Bound<'py, Self>) -> U32Array<'py> {
        let owned: Vec<u32> = this
            .get()
            .inner
            .offsets()
            .map_or_else(|| vec![0, this.get().inner.len() as u32], <[u32]>::to_vec);
        let array = PyArray1::from_vec(this.py(), owned);
        array.readwrite().make_nonwriteable();
        U32Array(array)
    }

    /// A batch equals any sequence holding the same encodings, so it compares against a list.
    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        PyList::new(py, self.encodings())?.as_any().eq(other)
    }

    /// Pickles as a plain list: the shared buffer cannot cross to another process, its rows can.
    fn __reduce__<'py>(&self, py: Python<'py>) -> (Bound<'py, PyAny>, (Vec<Encoding>,)) {
        (py.get_type::<PyList>().into_any(), (self.encodings(),))
    }

    fn __repr__(&self) -> String {
        match self.inner.stride() {
            Some(stride) => format!("Batch(rows={}, stride={stride})", self.inner.rows()),
            None => format!("Batch(rows={}, ragged)", self.inner.rows()),
        }
    }
}

/// A read-only `(rows, stride)` array over `data`, with `batch` as the array's base.
fn view2<'py>(
    batch: &Bound<'py, Batch>,
    data: &[u32],
    rows: usize,
    stride: usize,
) -> PyResult<U32Array2<'py>> {
    let view = ArrayView2::from_shape((rows, stride), data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    // SAFETY: `Batch` is frozen and never mutates its buffer, and it holds the `Arc` keeping that
    // buffer alive, so `data` outlives every array numpy hands out. Numpy keeps `batch` alive
    // through the array's base.
    let array = unsafe { PyArray2::borrow_from_array(&view, batch.clone().into_any()) };
    array.readwrite().make_nonwriteable();
    Ok(U32Array2(array))
}
