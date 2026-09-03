use numpy::ndarray::ArrayView1;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use tk_encode::pipeline::Encoding as PipelineEncoding;

use crate::arrays::U32Array;

/// One encoded text: `ids`, `type_ids` and `attention_mask`, the fields the pipeline computes.
///
/// The pipeline only stores `type_ids` and `attention_mask` when a post-processor or padding
/// set them. When it did not, every token is of type 0 and attended to, so that is what the
/// two arrays report.
///
/// Each field reads as a read-only numpy array over the encoding's own memory. Nothing is
/// copied, and the array keeps the encoding alive for as long as the array exists.
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

#[pymethods]
impl Encoding {
    /// The id of each token.
    #[getter]
    fn ids<'py>(this: &Bound<'py, Self>) -> U32Array<'py> {
        view(this, &this.get().ids)
    }

    /// The type id of each token, 0 unless a post-processor or padding set it.
    #[getter]
    fn type_ids<'py>(this: &Bound<'py, Self>) -> U32Array<'py> {
        view(this, &this.get().type_ids)
    }

    /// 1 for every token, 0 for padding.
    #[getter]
    fn attention_mask<'py>(this: &Bound<'py, Self>) -> U32Array<'py> {
        view(this, &this.get().attention_mask)
    }

    /// The number of tokens, padding included.
    fn __len__(&self) -> usize {
        self.ids.len()
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
    // SAFETY: `Encoding` is frozen and never resizes its vectors, so `data` stays put for as
    // long as `encoding` lives, and numpy keeps `encoding` alive through the array's base.
    let array = unsafe {
        PyArray1::borrow_from_array(&ArrayView1::from(data), encoding.clone().into_any())
    };
    array.readwrite().make_nonwriteable();
    U32Array(array)
}
