use pyo3::prelude::*;
use tk_encode::pipeline::Encoding as PipelineEncoding;

/// One encoded text: `ids`, `type_ids` and `attention_mask`, the fields the pipeline computes.
///
/// The pipeline only stores `type_ids` and `attention_mask` when a post-processor or padding
/// set them. When it did not, every token is of type 0 and attended to, so that is what the
/// two lists report.
///
/// Reading `ids`, `type_ids` or `attention_mask` builds a new Python list from the Rust vector
/// on every access, an O(n) copy. Read the attribute once rather than inside a loop.
#[pyclass(frozen, get_all)]
pub struct Encoding {
    /// The id of each token. A new list is built on every read.
    ids: Vec<u32>,
    /// The type id of each token, 0 unless a post-processor or padding set it. A new list is
    /// built on every read.
    type_ids: Vec<u32>,
    /// 1 for every token, 0 for padding. A new list is built on every read.
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
    fn __len__(&self) -> usize {
        self.ids.len()
    }
}
