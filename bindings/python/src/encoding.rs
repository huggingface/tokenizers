use std::sync::OnceLock;

use numpy::PyArray1;
use pyo3::prelude::*;
use tk::pipeline::{Encoding as PipelineEncoding, PipelineToken};

use crate::type_hints::{NDArray, U32Array};

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
// the values Python sees, not the core's storage or which caches were filled. Overflowing
// windows are values Python sees too, and are compared the same way, recursively.
impl PartialEq for Encoding {
    fn eq(&self, other: &Self) -> bool {
        fn visible_type_ids(encoding: &PipelineEncoding) -> Vec<u32> {
            encoding
                .type_ids()
                .map_or_else(|| vec![0; encoding.len()], bytes_slice_to_u32)
        }
        fn visible_attention_mask(encoding: &PipelineEncoding) -> Vec<u32> {
            encoding
                .attention_mask()
                .map_or_else(|| vec![1; encoding.len()], bytes_slice_to_u32)
        }
        fn rows_eq(a: &PipelineEncoding, b: &PipelineEncoding) -> bool {
            PipelineToken::cast_slice(a.ids()) == PipelineToken::cast_slice(b.ids())
                && visible_type_ids(a) == visible_type_ids(b)
                && visible_attention_mask(a) == visible_attention_mask(b)
                && a.overflowing().len() == b.overflowing().len()
                && a.overflowing()
                    .iter()
                    .zip(b.overflowing())
                    .all(|(a, b)| rows_eq(a, b))
        }
        rows_eq(&self.inner, &other.inner)
    }
}

/// What `Encoding.__reduce__` gives pickle: `ids`, `type_ids`, `attention_mask` and the
/// `overflowing` encodings, which pickle through their own `__reduce__`.
type UnpickleArguments = (Vec<u32>, Vec<u32>, Vec<u32>, Vec<Encoding>);

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

    /// The overflowing encodings a truncation produced, as a list of `Encoding`.
    ///
    /// A truncated encoding keeps one window of the input -- from the start or the end, as its
    /// truncation `direction` says -- and returns the rest as these windows, which follow the
    /// kept one sharing `stride` tokens with the previous piece. Each is a complete row: same
    /// special tokens, same type ids, padded together with this encoding. Empty when the
    /// encoding was not truncated. With no room at all for sequence tokens, the encoding keeps
    /// only its special tokens and one window carries the sequence whole.
    #[getter]
    fn overflowing(&self) -> Vec<Encoding> {
        self.inner
            .overflowing()
            .iter()
            .cloned()
            .map(Encoding::from)
            .collect()
    }

    /// Returns a copy of the id of each token, as a `uint32` numpy array.
    #[getter]
    fn ids_array<'py>(&self, py: Python<'py>) -> U32Array<'py> {
        NDArray(PyArray1::from_slice(py, self.ids()))
    }

    /// Returns a copy of the type id of each token, as a `uint32` numpy array.
    #[getter]
    fn type_ids_array<'py>(&self, py: Python<'py>) -> U32Array<'py> {
        NDArray(PyArray1::from_slice(py, self.type_ids()))
    }

    /// Returns a copy of the attention mask, as a `uint32` numpy array.
    /// When the encoding is padded: 1 for token ids, 0 for padding tokens.
    #[getter]
    fn attention_mask_array<'py>(&self, py: Python<'py>) -> U32Array<'py> {
        NDArray(PyArray1::from_slice(py, self.attention_mask()))
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
            self.overflowing(),
        );
        Ok((py.get_type::<Self>().getattr("_unpickle")?, arguments))
    }

    /// Unpickles an `Encoding`. `overflowing` defaults to none for a pickle written before
    /// overflowing encodings were exposed.
    #[staticmethod]
    #[pyo3(signature = (ids, type_ids, attention_mask, overflowing=None))]
    fn _unpickle(
        ids: Vec<u32>,
        type_ids: Vec<u8>,
        attention_mask: Vec<u8>,
        overflowing: Option<Vec<PyRef<'_, Encoding>>>,
    ) -> Self {
        let mut encoding = Self {
            inner: PipelineEncoding::new(
                ids.into_iter().map(PipelineToken::from).collect(),
                Some(type_ids),
                Some(attention_mask),
            ),
            type_ids: OnceLock::new(),
            attention_mask: OnceLock::new(),
        };
        encoding.inner.set_overflowing(
            overflowing
                .unwrap_or_default()
                .into_iter()
                .map(|window| window.inner.clone())
                .collect(),
        );
        encoding
    }

    fn __repr__(&self) -> String {
        let overflowing: Vec<String> = self.overflowing().iter().map(Encoding::__repr__).collect();
        format!(
            "Encoding(ids={:?}, type_ids={:?}, attention_mask={:?}, overflowing=[{}])",
            self.ids(),
            self.type_ids(),
            self.attention_mask(),
            overflowing.join(", "),
        )
    }
}

fn bytes_slice_to_u32(slice: &[u8]) -> Vec<u32> {
    slice.iter().map(|&byte| byte as u32).collect()
}
