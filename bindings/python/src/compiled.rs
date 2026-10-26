use pyo3::prelude::*;
use pyo3::types::PyDict;
use tk::PaddingParams;
use tk::TruncationParams;
use tk::pipeline::EncodeOptions;
use tk::pipeline::Input;
use tk::pipeline::Inputs;
use tk::pipeline::PipelineToken;
use tk::pipeline::PipelineTokenizer;
use tk::to_json;

use crate::encoding::Encoding;
use crate::error::err;
use crate::input::PyInput;
use crate::options::OverrideSentinel;
use crate::repr;
use crate::tokenizer::TokenizerSpec;
use crate::type_hints::TokenIds;

/// A tokenizer built by `Tokenizer.compile()`. Encodes text into token ids, and decodes token
/// ids back into text.
///
/// A compiled tokenizer is immutable and can be shared across threads.
/// Pickling serializes the compiled tokenizer as its `tokenizer.json` JSON representation.
/// Unpickling builds the compiled tokenizer from that JSON spec.
#[pyclass(frozen, module = "tokenizers")]
pub struct CompiledTokenizer {
    pipeline: PipelineTokenizer,
}

impl CompiledTokenizer {
    pub(crate) fn new(py: Python<'_>, spec: &TokenizerSpec) -> PyResult<Self> {
        let pipeline = py.detach(|| spec.compile()).map_err(err)?;
        Ok(Self { pipeline })
    }
}

fn make_options(
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
        let options = make_options(
            add_special_tokens,
            encode_special_tokens,
            padding,
            truncation,
        );
        // py.detach releases the GIL while encode runs on Rust side
        let encoding = py
            .detach(|| self.pipeline.encode(input, &options).wait())
            .map_err(err)?
            .pop()
            .expect("encoding has exactly one element");
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
        let options = make_options(
            add_special_tokens,
            encode_special_tokens,
            padding,
            truncation,
        );
        py.detach(|| -> tk::Result<Vec<String>> {
            let encodings = self.pipeline.encode(input, &options).wait()?;
            Ok(self
                .pipeline
                .decode_tokens(PipelineToken::cast_slice(encodings[0].ids()), false))
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
        let options = make_options(
            add_special_tokens,
            encode_special_tokens,
            padding,
            truncation,
        );
        let inputs = Inputs::Batch(inputs.into_iter().map(Input::from).collect());
        // py.detach releases the GIL while encode runs on Rust side
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
        // py.detach releases the GIL
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
        Ok(py.detach(|| self.pipeline.decode_tokens(ids, skip_special_tokens)))
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
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (String,))> {
        let json = py.detach(|| to_json(&self.pipeline)).map_err(err)?;
        Ok((py.get_type::<Self>().getattr("_unpickle")?, (json,)))
    }

    /// Unpickles a `CompiledTokenizer`
    #[staticmethod]
    fn _unpickle(py: Python<'_>, json: &str) -> PyResult<Self> {
        Self::new(py, &TokenizerSpec::from_json(json).map_err(err)?)
    }

    fn __repr__(&self) -> PyResult<String> {
        repr::compiled_tokenizer(&self.pipeline).map_err(err)
    }
}
