use pyo3::prelude::*;
use pyo3::types::PyDict;
use tk_encode::pipeline::DecodeStreamState;

use crate::error::err;
use crate::tokenizer::Tokenizer;
use crate::type_hints::StreamInput;

/// Incrementally decode token ids, buffering incomplete characters between steps.
///
/// Args:
///     ids: list[int], optional
///         Initial context ids. Complete context is not emitted; an incomplete
///         character is retained until subsequent ids complete it.
///     skip_special_tokens: bool, defaults to False
///         Whether to omit special tokens from the decoded text.
#[pyclass(module = "tokenizers.decoders", from_py_object)]
#[derive(Clone)]
pub struct DecodeStream {
    state: DecodeStreamState,
}

#[pymethods]
impl DecodeStream {
    #[new]
    #[pyo3(signature = (ids=None, skip_special_tokens=false))]
    fn new(ids: Option<Vec<u32>>, skip_special_tokens: Option<bool>) -> Self {
        Self {
            state: DecodeStreamState::new(
                ids.unwrap_or_default(),
                skip_special_tokens.unwrap_or(false),
            ),
        }
    }

    /// Append a token id or a sequence of ids and return the next decoded chunk.
    ///
    /// Returns None while a character is incomplete or no new text is available.
    /// The tokenizer's configured decoder and vocabulary are used for every step.
    fn step(
        &mut self,
        py: Python<'_>,
        tokenizer: &Tokenizer,
        id: StreamInput,
    ) -> PyResult<Option<String>> {
        py.detach(|| self.state.step(tokenizer.pipeline(), id.as_slice()))
            .map_err(err)
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyDict>) -> Self {
        self.clone()
    }
}
