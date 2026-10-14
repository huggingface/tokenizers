use numpy::PyArray1;
use pyo3::prelude::*;
use tk_encode::pipeline::Encoding as PipelineEncoding;

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

/// What `Encoding.__reduce__` gives pickle: `ids`, `type_ids` and `attention_mask`.
type UnpickleArguments = (Vec<u32>, Vec<u32>, Vec<u32>);

#[pymethods]
impl Encoding {
    /// The id of each token, as a list of ints.
    #[getter]
    pub(crate) fn ids(&self) -> &[u32] {
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

    /// Returns a copy of the id of each token, as a `uint32` numpy array.
    #[getter]
    fn ids_array<'py>(&self, py: Python<'py>) -> U32Array<'py> {
        U32Array(PyArray1::from_slice(py, &self.ids))
    }

    /// Returns a copy of the type id of each token, as a `uint32` numpy array.
    #[getter]
    fn type_ids_array<'py>(&self, py: Python<'py>) -> U32Array<'py> {
        U32Array(PyArray1::from_slice(py, &self.type_ids))
    }

    /// Returns a copy of the attention mask, as a `uint32` numpy array.
    /// When the encoding is padded: 1 for token ids, 0 for padding tokens.
    #[getter]
    fn attention_mask_array<'py>(&self, py: Python<'py>) -> U32Array<'py> {
        U32Array(PyArray1::from_slice(py, &self.attention_mask))
    }

    /// The number of tokens in the encoding
    fn __len__(&self) -> usize {
        self.ids.len()
    }

    /// Pickle rebuilds an `Encoding` by calling `_unpickle` with these arguments.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, UnpickleArguments)> {
        let arguments = (
            self.ids.clone(),
            self.type_ids.clone(),
            self.attention_mask.clone(),
        );
        Ok((py.get_type::<Self>().getattr("_unpickle")?, arguments))
    }

    /// Unpickles an `Encoding`
    #[staticmethod]
    fn _unpickle(ids: Vec<u32>, type_ids: Vec<u32>, attention_mask: Vec<u32>) -> Self {
        Self {
            ids,
            type_ids,
            attention_mask,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Encoding(ids={:?}, type_ids={:?}, attention_mask={:?})",
            self.ids, self.type_ids, self.attention_mask
        )
    }
}
