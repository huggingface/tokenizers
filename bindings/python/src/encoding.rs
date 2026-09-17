use std::sync::Arc;

use numpy::ndarray::ArrayView1;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use tk_encode::pipeline::{Encoding as PipelineEncoding, PipelineToken};

use crate::type_hints::U32Array;

/// How an [`Encoding`] gets at its ids.
///
/// `Row` is the batch case: every encoding in a batch shares one `Arc` over one contiguous id
/// buffer and remembers which document it is. The batch paths build exactly that buffer, so a
/// batch of 20k documents costs one allocation to hand back rather than one per document -- the
/// copy used to happen here, with the GIL held, after the threads had already finished.
enum Repr {
    Row {
        batch: Arc<PipelineEncoding>,
        row: usize,
    },
    /// An encoding that did not come from a batch buffer: unpickling, mostly. Carries the
    /// buffers outright, because a padded encoding's mask is not the all-ones default and there
    /// is no batch left to read it back from.
    Owned {
        ids: Vec<u32>,
        type_ids: Vec<u32>,
        attention_mask: Vec<u32>,
    },
}

/// Text encoded to token ids by a tokenizer.
#[pyclass(frozen, eq, module = "tokenizers")]
pub struct Encoding {
    repr: Repr,
}

impl Encoding {
    /// One document out of a shared batch. `row` must be in range.
    pub(crate) fn row(batch: Arc<PipelineEncoding>, row: usize) -> Self {
        debug_assert!(row < batch.rows(), "[BUG] row out of range");
        Self {
            repr: Repr::Row { batch, row },
        }
    }

    /// Every document of a batch, sharing its buffer.
    pub(crate) fn rows(batch: PipelineEncoding) -> Vec<Self> {
        let batch = Arc::new(batch);
        (0..batch.rows())
            .map(|row| Self::row(Arc::clone(&batch), row))
            .collect()
    }

    fn ids_slice(&self) -> &[u32] {
        match &self.repr {
            Repr::Row { batch, row } => batch
                .row(*row)
                .map_or(&[][..], |ids| PipelineToken::ids_of(ids)),
            Repr::Owned { ids, .. } => ids,
        }
    }

    /// A per-token `u8` buffer of the batch, narrowed to this document.
    ///
    /// `None` when the encoding carries no such buffer, which is the common case: a single
    /// sequence template has no type ids, and an unpadded encoding has no mask. Building the
    /// all-zero or all-one default eagerly cost a full-length allocation per document for
    /// something most callers never read, so it is left to the getter.
    fn row_bytes(&self, pick: fn(&PipelineEncoding) -> Option<&[u8]>) -> Option<&[u8]> {
        let Repr::Row { batch, row } = &self.repr else {
            return None;
        };
        let range = batch.row_range(*row)?;
        pick(batch)?.get(range)
    }
}

fn widen(bytes: &[u8]) -> Vec<u32> {
    bytes.iter().map(|&b| u32::from(b)).collect()
}

impl PartialEq for Encoding {
    /// Compares what the encoding says, not how it stores it: a row of a batch and the same ids
    /// unpickled are equal.
    fn eq(&self, other: &Self) -> bool {
        self.ids_slice() == other.ids_slice()
            && self.type_ids() == other.type_ids()
            && self.attention_mask() == other.attention_mask()
    }
}

/// What `Encoding.__reduce__` gives pickle: `ids`, `type_ids` and `attention_mask`.
type UnpickleArguments = (Vec<u32>, Vec<u32>, Vec<u32>);

#[pymethods]
impl Encoding {
    /// The id of each token, as a list of ints.
    #[getter]
    fn ids(&self) -> &[u32] {
        self.ids_slice()
    }

    /// The type id of each token, as a list of ints.
    #[getter]
    fn type_ids(&self) -> Vec<u32> {
        match &self.repr {
            Repr::Owned { type_ids, .. } => type_ids.clone(),
            Repr::Row { .. } => self
                .row_bytes(PipelineEncoding::type_ids)
                .map_or_else(|| vec![0; self.ids_slice().len()], widen),
        }
    }

    /// Attention mask when the encoding is padded: 1 for token ids, 0 for padding tokens.
    /// A list of ints.
    #[getter]
    fn attention_mask(&self) -> Vec<u32> {
        match &self.repr {
            Repr::Owned {
                attention_mask, ..
            } => attention_mask.clone(),
            Repr::Row { .. } => self
                .row_bytes(PipelineEncoding::attention_mask)
                .map_or_else(|| vec![1; self.ids_slice().len()], widen),
        }
    }

    /// The id of each token, as a read-only `uint32` numpy array.
    /// A view over the encoding, not a copy.
    #[getter]
    fn ids_array<'py>(this: &Bound<'py, Self>) -> U32Array<'py> {
        view(this, this.get().ids_slice())
    }

    /// The type id of each token, as a read-only `uint32` numpy array.
    #[getter]
    fn type_ids_array<'py>(this: &Bound<'py, Self>) -> U32Array<'py> {
        owned_array(this.py(), this.get().type_ids())
    }

    /// Attention mask when the encoding is padded: 1 for token ids, 0 for padding tokens.
    /// A read-only `uint32` numpy array.
    #[getter]
    fn attention_mask_array<'py>(this: &Bound<'py, Self>) -> U32Array<'py> {
        owned_array(this.py(), this.get().attention_mask())
    }

    /// The number of tokens in the encoding
    fn __len__(&self) -> usize {
        self.ids_slice().len()
    }

    /// Pickle rebuilds an `Encoding` by calling `_unpickle` with these arguments.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, UnpickleArguments)> {
        let arguments = (self.ids_slice().to_vec(), self.type_ids(), self.attention_mask());
        Ok((py.get_type::<Self>().getattr("_unpickle")?, arguments))
    }

    /// Unpickles an `Encoding`
    #[staticmethod]
    fn _unpickle(ids: Vec<u32>, type_ids: Vec<u32>, attention_mask: Vec<u32>) -> Self {
        Self {
            repr: Repr::Owned {
                ids,
                type_ids,
                attention_mask,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Encoding(ids={:?}, type_ids={:?}, attention_mask={:?})",
            self.ids_slice(),
            self.type_ids(),
            self.attention_mask()
        )
    }
}

/// A read-only numpy array over `data`, with `encoding` as the array's base.
fn view<'py>(encoding: &Bound<'py, Encoding>, data: &[u32]) -> U32Array<'py> {
    // SAFETY: `Encoding` is frozen and never mutates its ids, and when they are a row of a batch
    // it holds the `Arc` keeping that buffer alive, so `data` stays alive for as long as
    // `encoding` does. Numpy keeps `encoding` alive through the array's base.
    let array = unsafe {
        PyArray1::borrow_from_array(&ArrayView1::from(data), encoding.clone().into_any())
    };
    array.readwrite().make_nonwriteable();
    U32Array(array)
}

/// A read-only numpy array owning its data, for the buffers built on demand.
fn owned_array(py: Python<'_>, data: Vec<u32>) -> U32Array<'_> {
    let array = PyArray1::from_vec(py, data);
    array.readwrite().make_nonwriteable();
    U32Array(array)
}
