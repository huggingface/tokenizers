use std::sync::Arc;

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyIndexError, PyNotImplementedError};
use pyo3::prelude::*;

use crate::tokenizer::PyTokenizer;

const NO_OFFSETS: &str = "character offsets are not tracked by the encode pipeline";
const NO_WORD_IDS: &str = "word ids are not emitted by the encode pipeline";
const NO_SPECIALS: &str = "the encode pipeline does not mark which tokens are special";

fn deferred(what: &str, why: &str) -> PyErr {
    PyNotImplementedError::new_err(format!("{what} is not available yet: {why}"))
}

/// The result of encoding one sequence: token ids plus the masks and metadata
/// a model consumes. The fields are derived from the ids on access, so an
/// `Encoding` costs the same to produce as a bare id array — `Tokenizer.encode`
/// runs exactly the work `encode_ids` does.
///
/// `encode` handles a single sequence, so `type_ids` and `attention_mask` are
/// constant (one segment, nothing padded). Fields that need per-token
/// provenance the pipeline does not track yet — which tokens are special
/// (`special_tokens_mask`, `sequence_ids`), word ids, and character offsets —
/// raise rather than returning a plausible-looking guess.
#[pyclass(frozen, name = "Encoding", module = "tokenizers")]
pub struct PyEncoding {
    ids: Arc<[u32]>,
    tokenizer: Py<PyTokenizer>,
}

impl PyEncoding {
    pub(crate) fn new(ids: Arc<[u32]>, tokenizer: Py<PyTokenizer>) -> Self {
        Self { ids, tokenizer }
    }
}

#[pymethods]
impl PyEncoding {
    fn __len__(&self) -> usize {
        self.ids.len()
    }

    fn __repr__(&self) -> String {
        format!("Encoding(length={})", self.ids.len())
    }

    /// The token ids, as a list.
    // FIXME: copies Rust -> Python on every access — a fresh list, and each
    // u32 boxed into an int object, with no caching. Ideally this would return
    // a numpy uint32 array viewing the Arc buffer (zero-copy), which stays
    // almost API-compatible: it indexes and iterates like a list, but isn't one.
    #[getter]
    fn ids(&self) -> Vec<u32> {
        self.ids.to_vec()
    }

    /// The token ids as a `numpy.uint32` array. This copies; for the copy-free
    /// array use `Tokenizer.encode_ids`, which hands ownership of the buffer
    /// straight to numpy.
    #[pyo3(signature = () -> "npt.NDArray[np.uint32]")]
    fn ids_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.ids.to_vec().into_pyarray(py)
    }

    /// The token strings behind the ids.
    #[getter]
    fn tokens(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.tokenizer.bind(py).get().ids_to_tokens(py, &self.ids)
    }

    /// Segment id per token: all 0 (single sequence).
    #[getter]
    fn type_ids(&self) -> Vec<u32> {
        vec![0; self.ids.len()]
    }

    /// Attention mask, one entry per token: all 1 (nothing padded).
    #[getter]
    fn attention_mask(&self) -> Vec<u32> {
        vec![1; self.ids.len()]
    }

    /// Special-tokens mask — not available: the pipeline does not mark which
    /// tokens are special yet (a backing structure for this is coming).
    #[getter]
    fn special_tokens_mask(&self) -> PyResult<Vec<u32>> {
        Err(deferred("special_tokens_mask", NO_SPECIALS))
    }

    /// The sequence each token belongs to — not available: it depends on which
    /// tokens are special, which the pipeline does not mark yet.
    #[getter]
    fn sequence_ids(&self) -> PyResult<Vec<Option<u32>>> {
        Err(deferred("sequence_ids", NO_SPECIALS))
    }

    /// Number of sequences in this encoding: always 1.
    #[getter]
    fn n_sequences(&self) -> usize {
        1
    }

    #[pyo3(signature = (token_index))]
    #[allow(unused_variables)]
    fn token_to_sequence(&self, token_index: usize) -> PyResult<Option<u32>> {
        Err(deferred("token_to_sequence", NO_SPECIALS))
    }

    /// Word id per token — not available: the pipeline does not emit word
    /// boundaries yet.
    #[getter]
    fn word_ids(&self) -> PyResult<Vec<Option<u32>>> {
        Err(deferred("word_ids", NO_WORD_IDS))
    }

    /// Character span per token — not available: the pipeline does not track
    /// offsets yet.
    #[getter]
    fn offsets(&self) -> PyResult<Vec<(usize, usize)>> {
        Err(deferred("offsets", NO_OFFSETS))
    }

    #[pyo3(signature = (token_index))]
    #[allow(unused_variables)]
    fn token_to_word(&self, token_index: usize) -> PyResult<Option<u32>> {
        Err(deferred("token_to_word", NO_WORD_IDS))
    }

    #[pyo3(signature = (word_index, sequence_index = 0))]
    #[allow(unused_variables)]
    fn word_to_tokens(
        &self,
        word_index: u32,
        sequence_index: usize,
    ) -> PyResult<Option<(usize, usize)>> {
        Err(deferred("word_to_tokens", NO_WORD_IDS))
    }

    #[pyo3(signature = (word_index, sequence_index = 0))]
    #[allow(unused_variables)]
    fn word_to_chars(
        &self,
        word_index: u32,
        sequence_index: usize,
    ) -> PyResult<Option<(usize, usize)>> {
        Err(deferred("word_to_chars", NO_OFFSETS))
    }

    #[pyo3(signature = (token_index))]
    #[allow(unused_variables)]
    fn token_to_chars(&self, token_index: usize) -> PyResult<Option<(usize, usize)>> {
        Err(deferred("token_to_chars", NO_OFFSETS))
    }

    #[pyo3(signature = (char_pos, sequence_index = 0))]
    #[allow(unused_variables)]
    fn char_to_token(&self, char_pos: usize, sequence_index: usize) -> PyResult<Option<usize>> {
        Err(deferred("char_to_token", NO_OFFSETS))
    }

    #[pyo3(signature = (char_pos, sequence_index = 0))]
    #[allow(unused_variables)]
    fn char_to_word(&self, char_pos: usize, sequence_index: usize) -> PyResult<Option<u32>> {
        Err(deferred("char_to_word", NO_OFFSETS))
    }
}

/// The result of encoding a batch: a sequence of `Encoding`s. Index it
/// (`batch[0]`) or iterate it.
#[pyclass(frozen, name = "EncodingBatch", module = "tokenizers")]
pub struct PyEncodingBatch {
    rows: Vec<Arc<[u32]>>,
    tokenizer: Py<PyTokenizer>,
}

impl PyEncodingBatch {
    pub(crate) fn new(rows: Vec<Arc<[u32]>>, tokenizer: Py<PyTokenizer>) -> Self {
        Self { rows, tokenizer }
    }
}

#[pymethods]
impl PyEncodingBatch {
    fn __len__(&self) -> usize {
        self.rows.len()
    }

    fn __repr__(&self) -> String {
        format!("EncodingBatch(size={})", self.rows.len())
    }

    fn __getitem__(&self, py: Python<'_>, index: isize) -> PyResult<PyEncoding> {
        let len = self.rows.len() as isize;
        let resolved = if index < 0 { index + len } else { index };
        if resolved < 0 || resolved >= len {
            return Err(PyIndexError::new_err("EncodingBatch index out of range"));
        }
        Ok(PyEncoding::new(
            self.rows[resolved as usize].clone(),
            self.tokenizer.clone_ref(py),
        ))
    }
}
