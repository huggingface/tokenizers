use numpy::ndarray::ArrayView1;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use tk_encode::pipeline::Encoding as PipelineEncoding;

use crate::pickle::{self, Reduced};
use crate::type_hints::U32Array;

/// Text encoded to token ids by a tokenizer.
#[pyclass(frozen, eq, module = "tokenizers")]
#[derive(PartialEq)]
pub struct Encoding {
    ids: Vec<u32>,
    type_ids: Vec<u32>,
    attention_mask: Vec<u32>,
}

impl From<&PipelineEncoding> for Encoding {
    fn from(e: &PipelineEncoding) -> Self {
        let ids: Vec<u32> = e.ids().iter().map(|t| t.id()).collect();
        let len = ids.len();
        // TODO: allocate defaults for attention_mask and type_ids lazily
        Self {
            type_ids: e.type_ids().map(widen).unwrap_or_else(|| vec![0; len]),
            attention_mask: e
                .attention_mask()
                .map(widen)
                .unwrap_or_else(|| vec![1; len]),
            ids,
        }
    }
}

fn widen(bytes: &[u8]) -> Vec<u32> {
    bytes.iter().map(|&b| u32::from(b)).collect()
}

/// What `Encoding.__reduce__` hands pickle: `ids`, `type_ids` and `attention_mask`.
type Arguments = (Vec<u32>, Vec<u32>, Vec<u32>);

#[pymethods]
impl Encoding {
    /// The id of each token, as a list of ints.
    #[getter]
    fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// The type id of each token, as a list of ints.
    #[getter]
    fn type_ids(&self) -> &[u32] {
        &self.type_ids
    }

    /// Attention mask when the encoding is padded: 1 for token ids, 0 for padding tokens.
    /// A list of ints.
    #[getter]
    fn attention_mask(&self) -> &[u32] {
        &self.attention_mask
    }

    /// The id of each token, as a read-only `uint32` numpy array.
    /// A view over the encoding, not a copy.
    #[getter]
    fn ids_array<'py>(this: &Bound<'py, Self>) -> U32Array<'py> {
        view(this, &this.get().ids)
    }

    /// The type id of each token, as a read-only `uint32` numpy array.
    /// A view over the encoding, not a copy.
    #[getter]
    fn type_ids_array<'py>(this: &Bound<'py, Self>) -> U32Array<'py> {
        view(this, &this.get().type_ids)
    }

    /// Attention mask when the encoding is padded: 1 for token ids, 0 for padding tokens.
    /// A read-only `uint32` numpy array, a view over the encoding, not a copy.
    #[getter]
    fn attention_mask_array<'py>(this: &Bound<'py, Self>) -> U32Array<'py> {
        view(this, &this.get().attention_mask)
    }

    /// The number of tokens in the encoding
    fn __len__(&self) -> usize {
        self.ids.len()
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<Reduced<Arguments>> {
        let arguments = (
            self.ids.clone(),
            self.type_ids.clone(),
            self.attention_mask.clone(),
        );
        Ok((pickle::rebuilder(py, "_unpickle_encoding")?, arguments))
    }

    fn __repr__(&self) -> String {
        format!(
            "Encoding(ids={:?}, type_ids={:?}, attention_mask={:?})",
            self.ids, self.type_ids, self.attention_mask
        )
    }
}

/// A read-only numpy array over `data`, with `encoding` as the array's base.
fn view<'py>(encoding: &Bound<'py, Encoding>, data: &[u32]) -> U32Array<'py> {
    // SAFETY: `Encoding` is frozen and never mutates its Vec<u32>, `data` stays alive for as
    // long as `encoding` is alive. Numpy keeps `encoding` alive through the array's base.
    let array = unsafe {
        PyArray1::borrow_from_array(&ArrayView1::from(data), encoding.clone().into_any())
    };
    array.readwrite().make_nonwriteable();
    U32Array(array)
}

/// Rebuilds the `Encoding` that `Encoding.__reduce__` took apart. Private, because an `Encoding`
/// comes out of `Tokenizer.encode` and is not built by hand.
#[pyfunction]
pub(crate) fn _unpickle_encoding(
    ids: Vec<u32>,
    type_ids: Vec<u32>,
    attention_mask: Vec<u32>,
) -> Encoding {
    Encoding {
        ids,
        type_ids,
        attention_mask,
    }
}
