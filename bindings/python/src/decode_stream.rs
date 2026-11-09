use std::slice;

use pyo3::prelude::*;
use tk::pipeline::PipelineDecodeStream;

use crate::compiled::CompiledTokenizer;
use crate::error::err;
use crate::type_hints::TokenIds;

/// Decodes a stream of token ids, such as a model's output, into text as the ids arrive.
///
/// One character can span several tokens, so `step` returns `None` until the character is
/// complete.
///
/// Args:
///     ids: list[int] | None
///         The prompt ids that come before the stream. `step` decodes them as context, but
///         never returns their text.
///     skip_special_tokens: bool
///         Whether to leave special tokens out of the text.
///
/// Example::
///
///     >>> from tokenizers import Tokenizer
///     >>> from tokenizers.decoders import DecodeStream
///     >>> tokenizer = Tokenizer.from_pretrained("gpt2").compile()
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
        let prompt: Vec<u32> = match ids {
            Some(ids) => ids.as_slice().to_vec(),
            None => Vec::new(),
        };
        Self(PipelineDecodeStream::new(prompt, skip_special_tokens))
    }

    /// Adds the next id, or several ids, to the stream and returns the new text.
    ///
    /// Args:
    ///     tokenizer: CompiledTokenizer
    ///         The tokenizer that decodes the ids.
    ///     id: int | list[int]
    ///         The next id, or a list of the next ids.
    ///
    /// Returns:
    ///     str | None
    ///         The new text, or `None` while the text ends with an incomplete character.
    fn step(&mut self, tokenizer: &CompiledTokenizer, id: StepIds) -> PyResult<Option<String>> {
        let ids = match &id {
            StepIds::One(id) => slice::from_ref(id),
            StepIds::Many(ids) => ids,
        };
        self.0.step(tokenizer.pipeline(), ids).map_err(err)
    }
}
