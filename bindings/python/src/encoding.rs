use std::sync::OnceLock;

use numpy::ndarray::ArrayView1;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use tk_encode::pipeline::{Encoding as PipelineEncoding, PipelineToken};

use crate::type_hints::{Dtype, NDArray, U32Array};

/// Text encoded to token ids by a tokenizer.
#[pyclass(frozen, eq, module = "tokenizers")]
pub struct Encoding {
    inner: PipelineEncoding,
    type_ids: OnceLock<Vec<u32>>,
    attention_mask: OnceLock<Vec<u32>>,
}

impl From<PipelineEncoding> for Encoding {
    fn from(encoding: PipelineEncoding) -> Self {
        Self {
            inner: encoding,
            type_ids: OnceLock::new(),
            attention_mask: OnceLock::new(),
        }
    }
}

impl Encoding {
    pub(crate) fn get_ids(&self) -> &[u32] {
        PipelineToken::cast_slice(self.inner.ids())
    }
}

// The core stores "all zeros" type ids and an "all ones" mask as `None`, so equality compares
// the values Python sees, not the core's storage or which caches were filled.
impl PartialEq for Encoding {
    fn eq(&self, other: &Self) -> bool {
        self.ids() == other.ids()
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
    pub(crate) fn ids(&self) -> &[u32] {
        self.get_ids()
    }

    /// The type id of each token, as a list of ints.
    #[getter]
    fn type_ids(&self) -> &[u32] {
        self.type_ids.get_or_init(|| {
            if let Some(inner) = self.inner.type_ids() {
                bytes_slice_to_u32(inner)
            } else {
                vec![0; self.inner.len()]
            }
        })
    }

    /// Attention mask when the encoding is padded: 1 for token ids, 0 for padding tokens.
    /// A list of ints.
    #[getter]
    fn attention_mask(&self) -> &[u32] {
        self.attention_mask.get_or_init(|| {
            if let Some(inner) = self.inner.attention_mask() {
                bytes_slice_to_u32(inner)
            } else {
                vec![1; self.inner.len()]
            }
        })
    }

    /// The id of each token, as a read-only `uint32` numpy array.
    /// A view over the encoding, not a copy.
    #[getter]
    fn ids_array<'py>(this: &Bound<'py, Self>) -> U32Array<'py> {
        view(this, this.get().ids())
    }

    /// The type id of each token, as a read-only `uint32` numpy array.
    /// A view over the encoding, not a copy.
    #[getter]
    fn type_ids_array<'py>(this: &Bound<'py, Self>) -> U32Array<'py> {
        view(this, this.get().type_ids())
    }

    /// Attention mask when the encoding is padded: 1 for token ids, 0 for padding tokens.
    /// A read-only `uint32` numpy array, a view over the encoding, not a copy.
    #[getter]
    fn attention_mask_array<'py>(this: &Bound<'py, Self>) -> U32Array<'py> {
        view(this, this.get().attention_mask())
    }

    /// The number of tokens in the encoding
    fn __len__(&self) -> usize {
        self.ids().len()
    }

    /// Pickle rebuilds an `Encoding` by calling `_unpickle` with these arguments.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, UnpickleArguments)> {
        let arguments = (
            self.ids().to_vec(),
            self.type_ids().to_vec(),
            self.attention_mask().to_vec(),
        );
        Ok((py.get_type::<Self>().getattr("_unpickle")?, arguments))
    }

    /// Unpickles an `Encoding`
    #[staticmethod]
    fn _unpickle(ids: Vec<u32>, type_ids: Vec<u8>, attention_mask: Vec<u8>) -> Self {
        Self {
            inner: PipelineEncoding::new(
                ids.into_iter().map(PipelineToken::from).collect(),
                Some(type_ids),
                Some(attention_mask),
            ),
            type_ids: OnceLock::new(),
            attention_mask: OnceLock::new(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Encoding(ids={:?}, type_ids={:?}, attention_mask={:?})",
            self.ids(),
            self.type_ids(),
            self.attention_mask()
        )
    }
}

/// A read-only numpy array over `data`, with `encoding` as the array's base.
fn view<'py, T: Dtype>(encoding: &Bound<'py, Encoding>, data: &[T]) -> NDArray<'py, T> {
    // SAFETY: `Encoding` is frozen and never mutates its Vecs, `data` stays alive for as
    // long as `encoding` is alive. Numpy keeps `encoding` alive through the array's base.
    let array = unsafe {
        PyArray1::borrow_from_array(&ArrayView1::from(data), encoding.clone().into_any())
    };
    array.readwrite().make_nonwriteable();
    NDArray(array)
}

fn bytes_slice_to_u32(slice: &[u8]) -> Vec<u32> {
    slice.iter().map(|&byte| byte as u32).collect()
}
