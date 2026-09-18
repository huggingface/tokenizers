use std::sync::Arc;

use numpy::ndarray::ArrayView1;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use tk_encode::pipeline::{Encoding as PipelineEncoding, PipelineToken};

use crate::type_hints::U32Array;

/// How an [`Encoding`] reaches its ids: a shared batch buffer, or its own.
enum Repr {
    InBatch {
        batch: Arc<PipelineEncoding>,
        document: usize,
    },
    /// Not from a batch buffer (unpickling): carries its own, a padded mask is not all ones.
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
    /// One document out of a shared batch. `document` must be in range.
    pub(crate) fn document(batch: Arc<PipelineEncoding>, document: usize) -> Self {
        debug_assert!(document < batch.n_documents(), "[BUG] document out of range");
        Self {
            repr: Repr::InBatch { batch, document },
        }
    }

    fn ids_slice(&self) -> &[u32] {
        match &self.repr {
            Repr::InBatch { batch, document } => batch
                .document(*document)
                .map_or(&[][..], |ids| PipelineToken::ids_of(ids)),
            Repr::Owned { ids, .. } => ids,
        }
    }

    /// A per-token `u8` buffer of the batch narrowed to this document, `None` when absent.
    fn row_bytes(&self, pick: fn(&PipelineEncoding) -> Option<&[u8]>) -> Option<&[u8]> {
        let Repr::InBatch { batch, document } = &self.repr else {
            return None;
        };
        let range = batch.document_range(*document)?;
        pick(batch)?.get(range)
    }
}

fn widen(bytes: &[u8]) -> Vec<u32> {
    bytes.iter().map(|&b| u32::from(b)).collect()
}

impl PartialEq for Encoding {
    /// Compares what the encoding says, not how it stores it: a document of a batch and the same ids
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
            Repr::InBatch { .. } => self
                .row_bytes(PipelineEncoding::type_ids)
                .map_or_else(|| vec![0; self.ids_slice().len()], widen),
        }
    }

    /// Attention mask when the encoding is padded: 1 for token ids, 0 for padding tokens.
    /// A list of ints.
    #[getter]
    fn attention_mask(&self) -> Vec<u32> {
        match &self.repr {
            Repr::Owned { attention_mask, .. } => attention_mask.clone(),
            Repr::InBatch { .. } => self
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
        let arguments = (
            self.ids_slice().to_vec(),
            self.type_ids(),
            self.attention_mask(),
        );
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
    // SAFETY: `Encoding` is frozen and holds the `Arc`, so `data` outlives every array.
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
