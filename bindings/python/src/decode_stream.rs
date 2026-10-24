use pyo3::prelude::*;
use tk::pipeline::PipelineDecodeStream;

use crate::error::err;
use crate::tokenizer::Tokenizer;
use crate::type_hints::TokenIds;

/// Decodes a stream of token ids, for example a model generation.
///
/// A character can take several tokens, so `step` returns `None` if it needs more tokens to build a valid output.
///
/// Args:
///     ids:
///         Prompt ids that come before the first generated id. They give context to the first
///         steps and are never returned.
///     skip_special_tokens: bool
///         Whether special tokens should be omitted from the decoded text.
///
/// Example::
///
///     >>> from tokenizers import Tokenizer
///     >>> from tokenizers.decoders import DecodeStream
///     >>> tokenizer = Tokenizer.from_pretrained("gpt2")
///     >>> stream = DecodeStream()
///     >>> for id in tokenizer.encode("Hello, streaming world!").ids:
///     ...     print(stream.step(tokenizer, id) or "", end="")
#[pyclass(module = "tokenizers.decoders")]
pub struct DecodeStream(PipelineDecodeStream);

#[derive(FromPyObject)]
enum StepIds {
    One(u32),
    Many(Vec<u32>),
}

#[pymethods]
impl DecodeStream {
    #[new]
    #[pyo3(signature = (ids=None, skip_special_tokens=false))]
    fn new(ids: Option<TokenIds<'_>>, skip_special_tokens: bool) -> Self {
        let prompt = ids.map_or_else(Vec::new, |ids| ids.as_slice().to_vec());
        Self(PipelineDecodeStream::new(prompt, skip_special_tokens))
    }

    /// Process the next id(s) and returns the decoded text, or None if more tokens are required
    /// to decode to valid text.
    ///
    /// Args:
    ///     tokenizer: Tokenizer
    ///         The tokenizer that decodes the ids.
    ///     id: int | list[int]
    ///         The next id, or several ids.
    ///
    /// Returns:
    ///     str | None
    ///         The new text, or `None` while a character is still incomplete.
    fn step(&mut self, tokenizer: &Tokenizer, id: StepIds) -> PyResult<Option<String>> {
        let ids = match &id {
            StepIds::One(id) => std::slice::from_ref(id),
            StepIds::Many(ids) => ids,
        };
        self.0.step(tokenizer.pipeline(), ids).map_err(err)
    }
}
