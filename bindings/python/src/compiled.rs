use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use tk::PaddingParams;
use tk::TruncationParams;
use tk::pipeline::EncodeOptions;
use tk::pipeline::Encoding as PipelineEncoding;
use tk::pipeline::Input;
use tk::pipeline::Inputs;
use tk::pipeline::PipelineToken;
use tk::pipeline::PipelineTokenizer;
use tk::{from_json, to_json};

use crate::encoding::Encoding;
use crate::error::err;
use crate::input::PyInput;
use crate::options::OverrideSentinel;
use crate::tokenizer::TokenizerSpec;
use crate::type_hints::TokenIds;

/// A tokenizer built by `Tokenizer.compile()`. Encodes text into token ids, and decodes token
/// ids back into text.
///
/// A compiled tokenizer is immutable and can be shared across threads.
/// Pickling serializes to the JSON config, and unpickling compiles the CompiledTokenizer
/// again from that JSON config.
#[pyclass(frozen, module = "tokenizers")]
pub struct CompiledTokenizer {
    pipeline: PipelineTokenizer,
}

impl CompiledTokenizer {
    pub(crate) fn new(py: Python<'_>, spec: &TokenizerSpec) -> PyResult<Self> {
        // py.detach releases the GIL while the Rust code runs.
        let pipeline = py.detach(|| spec.compile()).map_err(err)?;
        Ok(Self { pipeline })
    }

    fn encode_one(&self, input: PyInput, options: &EncodeOptions) -> tk::Result<PipelineEncoding> {
        let mut encodings = self.pipeline.encode(input, options).wait()?;
        Ok(encodings
            .pop()
            .expect("a single input encodes to one encoding"))
    }
}

fn encode_options(
    add_special_tokens: bool,
    encode_special_tokens: bool,
    padding: OverrideSentinel<PaddingParams>,
    truncation: OverrideSentinel<TruncationParams>,
) -> EncodeOptions {
    EncodeOptions {
        add_special_tokens,
        encode_special_tokens,
        padding: padding.into(),
        truncation: truncation.into(),
    }
}

#[pymethods]
impl CompiledTokenizer {
    /// Encodes the given text, or pair of texts, to token ids.
    ///
    /// Args:
    ///     input: str or tuple[str, str]
    ///         A text, or a pair of texts given as a two-item tuple or list.
    ///         The post-processor joins a pair, for example as `[CLS] A [SEP] B [SEP]`.
    ///     add_special_tokens: bool
    ///          Whether the post-processor adds its special tokens, such as `[CLS]` and `[SEP]`.
    ///     encode_special_tokens: bool
    ///         Whether special tokens should be encoded, ie go through the tokenizer model (`True`)
    ///         or be replaced by their id in the added vocabulary.
    ///     padding: `Padding` or `None`
    ///         Padding options. Pass `None` to disable padding.
    ///         When omitted, defaults to the padding of the spec this tokenizer was compiled from.
    ///     truncation: `Truncation` or `None`
    ///         Truncation options. Pass `None` to disable truncation.
    ///         When omitted, defaults to the truncation of the spec this tokenizer was compiled from.
    ///
    /// Returns:
    ///     Encoding
    #[pyo3(signature = (input, *, add_special_tokens=true, encode_special_tokens=false, padding=OverrideSentinel::<PaddingParams>::InheritConfig, truncation=OverrideSentinel::<TruncationParams>::InheritConfig))]
    fn encode(
        &self,
        py: Python<'_>,
        input: PyInput,
        add_special_tokens: bool,
        encode_special_tokens: bool,
        padding: OverrideSentinel<PaddingParams>,
        truncation: OverrideSentinel<TruncationParams>,
    ) -> PyResult<Encoding> {
        let options = encode_options(
            add_special_tokens,
            encode_special_tokens,
            padding,
            truncation,
        );
        // py.detach releases the GIL while the Rust code runs.
        let encoding = py
            .detach(|| self.encode_one(input, &options))
            .map_err(err)?;
        Ok(Encoding::from(encoding))
    }

    /// Encodes the given text, or pair of texts, and returns the string representation of each token.
    ///
    /// Shorthand for `decode_tokens(encode(input))`.
    /// If you also need to access token ids, use `encode(input)`.
    ///
    /// Args:
    ///     input: str or tuple[str, str]
    ///         A text, or a pair of texts given as a two-item tuple or list.
    ///         The post-processor joins a pair, for example as `[CLS] A [SEP] B [SEP]`.
    ///     add_special_tokens: bool
    ///          Whether the post-processor adds its special tokens, such as `[CLS]` and `[SEP]`.
    ///     encode_special_tokens: bool
    ///         Whether special tokens should be encoded, ie go through the tokenizer model (`True`)
    ///         or be replaced by their id in the added vocabulary.
    ///     padding: `Padding` or `None`
    ///         Padding options. Pass `None` to disable padding.
    ///         When omitted, defaults to the padding of the spec this tokenizer was compiled from.
    ///     truncation: `Truncation` or `None`
    ///         Truncation options. Pass `None` to disable truncation.
    ///         When omitted, defaults to the truncation of the spec this tokenizer was compiled from.
    ///
    /// Returns:
    ///     list[str]
    #[pyo3(signature = (input, *, add_special_tokens=true, encode_special_tokens=false, padding=OverrideSentinel::<PaddingParams>::InheritConfig, truncation=OverrideSentinel::<TruncationParams>::InheritConfig))]
    fn tokenize(
        &self,
        py: Python<'_>,
        input: PyInput,
        add_special_tokens: bool,
        encode_special_tokens: bool,
        padding: OverrideSentinel<PaddingParams>,
        truncation: OverrideSentinel<TruncationParams>,
    ) -> PyResult<Vec<String>> {
        let options = encode_options(
            add_special_tokens,
            encode_special_tokens,
            padding,
            truncation,
        );
        // py.detach releases the GIL while the Rust code runs.
        py.detach(|| -> tk::Result<Vec<String>> {
            let encoding = self.encode_one(input, &options)?;
            Ok(self
                .pipeline
                .decode_tokens(PipelineToken::cast_slice(encoding.ids()), false))
        })
        .map_err(err)
    }

    /// Encodes a batch of texts, or pairs of texts.
    /// The encodings come back in input order.
    ///
    /// Args:
    ///     inputs: list[str | tuple[str, str]]
    ///         The batch to encode. Each item is a text, or a pair of texts given as a two-item
    ///         tuple or list. A batch can mix both.
    ///     add_special_tokens: bool
    ///         Whether the post-processor adds its special tokens, such as `[CLS]` and `[SEP]`.
    ///     encode_special_tokens: bool
    ///         Whether special tokens should be encoded, ie go through the tokenizer model (`True`)
    ///         or be replaced by their id in the added vocabulary.
    ///     padding: `Padding` or `None`
    ///         Padding options. Pass `None` to disable padding.
    ///         When omitted, defaults to the padding of the spec this tokenizer was compiled from.
    ///     truncation: `Truncation` or `None`
    ///         Truncation options. Pass `None` to disable truncation.
    ///         When omitted, defaults to the truncation of the spec this tokenizer was compiled from.
    ///
    /// Returns:
    ///     List[Encoding]
    #[pyo3(signature = (inputs, *, add_special_tokens=true, encode_special_tokens=false, padding=OverrideSentinel::<PaddingParams>::InheritConfig, truncation=OverrideSentinel::<TruncationParams>::InheritConfig))]
    fn encode_batch(
        &self,
        py: Python<'_>,
        inputs: Vec<PyInput>,
        add_special_tokens: bool,
        encode_special_tokens: bool,
        padding: OverrideSentinel<PaddingParams>,
        truncation: OverrideSentinel<TruncationParams>,
    ) -> PyResult<Vec<Encoding>> {
        let options = encode_options(
            add_special_tokens,
            encode_special_tokens,
            padding,
            truncation,
        );
        let inputs = Inputs::Batch(inputs.into_iter().map(Input::from).collect());
        // py.detach releases the GIL while the Rust code runs.
        let encodings = py
            .detach(|| self.pipeline.encode(inputs, &options).wait())
            .map_err(err)?;
        Ok(encodings.into_iter().map(Encoding::from).collect())
    }

    /// Decodes token ids back into text
    ///
    /// Args:
    ///     ids:
    ///         The ids to decode, an `Encoding`, a numpy array or any sequence of ints.
    ///     skip_special_tokens: bool
    ///         Whether special tokens should not be added to the decoded text.
    ///
    /// Returns:
    ///     str
    #[pyo3(signature = (ids, skip_special_tokens=true))]
    fn decode(
        &self,
        py: Python<'_>,
        ids: TokenIds<'_>,
        skip_special_tokens: bool,
    ) -> PyResult<String> {
        let ids = ids.as_slice();
        // py.detach releases the GIL while the Rust code runs.
        py.detach(|| self.pipeline.decode(ids, skip_special_tokens))
            .map_err(err)
    }

    /// Converts token ids to their string representation. This does NOT apply the decoder.
    ///
    /// Args:
    ///     ids:
    ///         The ids to convert, an `Encoding`, a numpy array or any sequence of ints.
    ///     skip_special_tokens: bool
    ///         Whether to skip special tokens
    ///
    /// Returns:
    ///     list[str]
    #[pyo3(signature = (ids, skip_special_tokens=false))]
    fn decode_tokens(
        &self,
        py: Python<'_>,
        ids: TokenIds<'_>,
        skip_special_tokens: bool,
    ) -> PyResult<Vec<String>> {
        let ids = ids.as_slice();
        // py.detach releases the GIL while the Rust code runs.
        Ok(py.detach(|| self.pipeline.decode_tokens(ids, skip_special_tokens)))
    }

    /// Returns the vocabulary, as a dict from token to id.
    ///
    /// Args:
    ///     with_added_tokens: bool
    ///         Whether to include the added tokens, such as special tokens.
    ///
    /// Returns:
    ///     dict[str, int]
    #[pyo3(signature = (with_added_tokens=true))]
    fn get_vocab(&self, py: Python<'_>, with_added_tokens: bool) -> HashMap<String, u32> {
        // py.detach releases the GIL while the Rust code runs.
        py.detach(|| self.pipeline.get_vocab(with_added_tokens))
    }

    /// Returns the size of the vocabulary.
    ///
    /// Args:
    ///     with_added_tokens: bool
    ///         Whether to count the added tokens, such as special tokens.
    ///
    /// Returns:
    ///     int
    #[pyo3(signature = (with_added_tokens=true))]
    fn get_vocab_size(&self, py: Python<'_>, with_added_tokens: bool) -> usize {
        // py.detach releases the GIL while the Rust code runs.
        py.detach(|| self.pipeline.get_vocab_size(with_added_tokens))
    }

    /// Converts a token to its id.
    ///
    /// Takes the token as `get_vocab()` returns it, e.g. `"Ġhe"` for GPT-2's `" he"`.
    ///
    /// Args:
    ///     token: str
    ///         The token to convert.
    ///
    /// Returns:
    ///     int or None: `None` when the token is not in the vocabulary.
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.pipeline.token_to_id(token)
    }

    /// Converts an id to its token.
    ///
    /// Args:
    ///     id: int
    ///         The id to convert.
    ///
    /// Returns:
    ///     str or None: `None` when the id is not in the vocabulary.
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.pipeline.id_to_token(id)
    }

    /// Returns `self`: a compiled tokenizer cannot be changed, so a copy would be identical.
    fn __copy__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    /// Returns `self`: a compiled tokenizer cannot be changed, so a copy would be identical.
    fn __deepcopy__<'py>(slf: Bound<'py, Self>, _memo: &Bound<'py, PyDict>) -> Bound<'py, Self> {
        slf
    }

    /// Pickle rebuilds a `CompiledTokenizer` by calling `_unpickle` with these arguments.
    ///
    /// The pickle holds the whole `tokenizer.json`, written from the built tokenizer, and unpickling
    /// compiles it again.
    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        // py.detach releases the GIL while the Rust code runs.
        let json = py.detach(|| to_json(&self.pipeline)).map_err(err)?;
        // Bytes, not str, for the same reason as `Tokenizer.__reduce__`.
        let json = PyBytes::new(py, json.as_bytes());
        Ok((py.get_type::<Self>().getattr("_unpickle")?, (json,)))
    }

    #[staticmethod]
    fn _unpickle(py: Python<'_>, json: &[u8]) -> PyResult<Self> {
        let json = std::str::from_utf8(json).map_err(err)?;
        // py.detach releases the GIL while the Rust code runs.
        let pipeline = py.detach(|| from_json(json)).map_err(err)?;
        Ok(Self { pipeline })
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        // py.detach releases the GIL while the Rust code runs.
        let json = py.detach(|| to_json(&self.pipeline)).map_err(err)?;
        let spec = TokenizerSpec::from_json(&json).map_err(err)?;
        Ok(spec.repr("CompiledTokenizer"))
    }
}
