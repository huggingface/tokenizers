use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, PoisonError};

use pyo3::prelude::*;
use pyo3::types::PyDict;
use tk::PaddingParams;
use tk::TruncationParams;
use tk::pipeline::Input;
use tk::pipeline::Inputs;
use tk::pipeline::PipelineToken;
use tk::pipeline::PipelineTokenizer;
use tk::pipeline::{EncodeOptions, Override};

use crate::components::ComponentKind;
use crate::components::decoders::Decoder;
use crate::components::models::Model;
use crate::components::normalizers::Normalizer;
use crate::components::pre_tokenizers::PreTokenizer;
use crate::components::processors::PostProcessor;
use crate::encoding::Encoding;
use crate::error::err;
use crate::input::PyInput;
use crate::options::{OverrideSentinel, Padding, Truncation};
use crate::repr;
use crate::spec::TokenizerSpec;
use crate::type_hints::{Token, TokenIds};

/// A tokenizer. Encodes text into token ids, and decodes token ids back into text.
#[pyclass(frozen, dict, weakref, module = "tokenizers")]
pub struct Tokenizer {
    // Needs a mutex so a concurrent thread can access the value while
    // encode is running.
    spec: Mutex<TokenizerSpec>,
}

impl Tokenizer {
    fn make_options(
        &self,
        add_special_tokens: bool,
        encode_special_tokens: bool,
        padding: OverrideSentinel<PaddingParams>,
        truncation: OverrideSentinel<TruncationParams>,
    ) -> PyResult<EncodeOptions> {
        let padding = match padding {
            OverrideSentinel::InheritConfig => self
                .spec()
                .padding
                .clone()
                .map_or(Override::Off, Override::With),
            OverrideSentinel::Off => Override::Off,
            OverrideSentinel::With(params) => Override::With(params),
        };
        let truncation = match truncation {
            OverrideSentinel::InheritConfig => self
                .spec()
                .truncation
                .clone()
                .map_or(Override::Off, Override::With),
            OverrideSentinel::Off => Override::Off,
            OverrideSentinel::With(params) => Override::With(params),
        };
        Ok(EncodeOptions {
            add_special_tokens,
            encode_special_tokens,
            padding,
            truncation,
        })
    }

    /// Returns with the lock released, so encodes can run concurrently.
    /// Calling `self.spec().compile()?.encode(..)` would hold the Mutex lock for the whole encode.
    fn pipeline(&self) -> tk::Result<PipelineTokenizer> {
        self.spec().compile()
    }

    fn from_spec(mut spec: TokenizerSpec) -> PyResult<Self> {
        // This triggers a compilation of the spec and validates the Json values
        spec.compile().map_err(err)?;
        Ok(Self {
            spec: Mutex::new(spec),
        })
    }

    /// Generic getter for a component kind
    fn get<K: ComponentKind>(slf: &Bound<'_, Self>) -> PyResult<Option<Py<K>>> {
        let Some(spec) = slf.get().spec().component(K::JSON_KEY).cloned() else {
            return Ok(None);
        };
        K::from_spec(slf.py(), spec).map(Some)
    }

    /// Generic setter for a component kind
    fn set<K: ComponentKind>(&self, component: Option<Bound<'_, K>>) {
        let spec = component.map(|c| K::spec(&c).clone());
        self.spec().set_component(K::JSON_KEY, spec);
    }

    pub(crate) fn spec(&self) -> MutexGuard<'_, TokenizerSpec> {
        // A panic during a compile leaves the previous compile in place, so the spec stays usable.
        self.spec.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[pymethods]
impl Tokenizer {
    /// Loads a `tokenizer.json`.
    ///
    /// Args:
    ///     path:
    ///         The file to read.
    #[staticmethod]
    #[pyo3(signature = (path))]
    fn from_file(path: PathBuf) -> PyResult<Self> {
        let text = std::fs::read_to_string(&path).map_err(|e| {
            std::io::Error::new(e.kind(), format!("cannot read {}: {e}", path.display()))
        })?;
        Self::from_spec(TokenizerSpec::from_json(&text).map_err(err)?)
    }

    /// Instantiate a new `Tokenizer` from an existing file on the Hugging Face Hub.
    ///
    /// Defers downloading and caching to `huggingface_hub.hf_hub_download`.
    ///
    /// Args:
    ///     identifier (`str`):
    ///         The *model id* of a repo hosted on huggingface.co that contains a `tokenizer.json` file,
    ///         e.g., `"openai-community/gpt2"`.
    ///     revision (`str`, *optional*, defaults to `"main"`):
    ///         The specific model version to use. It can be a branch name, a tag name, or a commit id,
    ///         since we use a git-based system for storing models and other artifacts on huggingface.co,
    ///         so `revision` can be any identifier allowed by git.
    ///     token (`str` or `bool`, *optional*):
    ///         The token to use as HTTP bearer authorization for remote files. If `True`, will use the
    ///         token generated when running `hf auth login`. If `False`, will send no token. If `None`,
    ///         will use the stored token when there is one.
    ///     cache_dir (`str` or `Path`, *optional*):
    ///         Path to a directory in which the downloaded file should be cached if the standard cache
    ///         should not be used.
    ///     force_download (`bool`, *optional*, defaults to `False`):
    ///         Whether or not to download the file again and override the cached version if it exists.
    ///     local_files_only (`bool`, *optional*, defaults to `False`):
    ///         Whether or not to only rely on the cache and not to attempt to download anything.
    ///     subfolder (`str`, *optional*):
    ///         In case `tokenizer.json` is located inside a subfolder of the model repo on huggingface.co,
    ///         specify it here.
    ///
    /// Returns:
    ///     `Tokenizer`: The tokenizer the file describes.
    ///
    /// Examples:
    ///
    /// ```python
    /// # Download tokenizer.json from huggingface.co and cache it.
    /// tokenizer = Tokenizer.from_pretrained("openai-community/gpt2")
    ///
    /// # Pin a revision: a branch name, a tag name or a commit id.
    /// tokenizer = Tokenizer.from_pretrained("openai-community/gpt2", revision="607a30d783dfa663caf39e06633721c8d4cfcd7e")
    ///
    /// # A private or gated repo, with the token `hf auth login` stored.
    /// tokenizer = Tokenizer.from_pretrained("my-org/my-model", token=True)
    ///
    /// # Read the cache without contacting the Hub.
    /// tokenizer = Tokenizer.from_pretrained("openai-community/gpt2", local_files_only=True)
    /// ```
    #[staticmethod]
    #[pyo3(signature = (identifier, revision="main", token=None, *, cache_dir=None, force_download=false, local_files_only=false, subfolder=None))]
    #[allow(clippy::too_many_arguments)]
    fn from_pretrained(
        py: Python<'_>,
        identifier: &str,
        revision: &str,
        token: Option<Token<'_>>,
        cache_dir: Option<PathBuf>,
        force_download: bool,
        local_files_only: bool,
        subfolder: Option<&str>,
    ) -> PyResult<Self> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("repo_id", identifier)?;
        kwargs.set_item("filename", "tokenizer.json")?;
        kwargs.set_item("revision", revision)?;
        kwargs.set_item("token", token.map(|token| token.0))?;
        kwargs.set_item("cache_dir", cache_dir)?;
        kwargs.set_item("force_download", force_download)?;
        kwargs.set_item("local_files_only", local_files_only)?;
        kwargs.set_item("subfolder", subfolder)?;
        kwargs.set_item("library_name", "tokenizers")?;
        kwargs.set_item("library_version", env!("CARGO_PKG_VERSION"))?;
        let path: PathBuf = PyModule::import(py, "huggingface_hub")?
            .getattr("hf_hub_download")?
            .call((), Some(&kwargs))?
            .extract()?;
        Self::from_file(path)
    }

    #[getter]
    fn model(slf: &Bound<'_, Self>) -> PyResult<Py<Model>> {
        let model = Self::get(slf)?;
        Ok(model.expect("a TokenizerSpec always holds a model"))
    }

    #[setter]
    fn set_model(&self, model: Bound<'_, Model>) {
        self.set(Some(model))
    }

    #[getter]
    fn normalizer(slf: &Bound<'_, Self>) -> PyResult<Option<Py<Normalizer>>> {
        Self::get(slf)
    }

    #[setter]
    fn set_normalizer(&self, normalizer: Option<Bound<'_, Normalizer>>) {
        self.set(normalizer)
    }

    #[getter]
    fn pre_tokenizer(slf: &Bound<'_, Self>) -> PyResult<Option<Py<PreTokenizer>>> {
        Self::get(slf)
    }

    #[setter]
    fn set_pre_tokenizer(&self, pre_tokenizer: Option<Bound<'_, PreTokenizer>>) {
        self.set(pre_tokenizer)
    }

    #[getter]
    fn post_processor(slf: &Bound<'_, Self>) -> PyResult<Option<Py<PostProcessor>>> {
        Self::get(slf)
    }

    #[setter]
    fn set_post_processor(&self, post_processor: Option<Bound<'_, PostProcessor>>) {
        self.set(post_processor)
    }

    #[getter]
    fn decoder(slf: &Bound<'_, Self>) -> PyResult<Option<Py<Decoder>>> {
        Self::get(slf)
    }

    #[setter]
    fn set_decoder(&self, decoder: Option<Bound<'_, Decoder>>) {
        self.set(decoder)
    }

    /// The padding applied to every encode, or `None`.
    /// Assign `None` to switch padding off.
    #[getter]
    fn padding(&self) -> Option<Padding> {
        self.spec().padding.clone().map(Padding::from)
    }

    #[setter]
    fn set_padding(&self, padding: Option<PyRef<'_, Padding>>) {
        self.spec().padding = padding.map(|padding| padding.params().clone());
    }

    /// The truncation applied to every encode, or `None`.
    /// Assign `None` to switch truncation off.
    #[getter]
    fn truncation(&self) -> Option<Truncation> {
        self.spec().truncation.clone().map(Truncation::from)
    }

    #[setter]
    fn set_truncation(&self, truncation: Option<PyRef<'_, Truncation>>) {
        self.spec().truncation = truncation.map(|truncation| truncation.params().clone());
    }

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
    ///         When omitted, defaults to the padding options configured on the tokenizer.
    ///     truncation: `Truncation` or `None`
    ///         Truncation options. Pass `None` to disable truncation.
    ///         When omitted, defaults to the truncation options configured on the tokenizer.
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
        let options = self.make_options(
            add_special_tokens,
            encode_special_tokens,
            padding,
            truncation,
        )?;
        // py.detach releases the GIL while encode runs on Rust side
        let encoding = py
            .detach(|| self.pipeline()?.encode(input, &options).wait())
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
    ///         When omitted, defaults to the padding options configured on the tokenizer.
    ///     truncation: `Truncation` or `None`
    ///         Truncation options. Pass `None` to disable truncation.
    ///         When omitted, defaults to the truncation options configured on the tokenizer.
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
        let options = self.make_options(
            add_special_tokens,
            encode_special_tokens,
            padding,
            truncation,
        )?;
        py.detach(|| -> tk::Result<Vec<String>> {
            let pipeline = self.pipeline()?;
            let encodings = pipeline.encode(input, &options).wait()?;
            let ids: Vec<u32> = PipelineToken::cast_slice(encodings[0].ids()).to_vec();
            Ok(pipeline.decode_tokens(&ids, false))
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
    ///         When omitted, defaults to the padding options configured on the tokenizer.
    ///     truncation: `Truncation` or `None`
    ///         Truncation options. Pass `None` to disable truncation.
    ///         When omitted, defaults to the truncation options configured on the tokenizer.
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
        let options = self.make_options(
            add_special_tokens,
            encode_special_tokens,
            padding,
            truncation,
        )?;
        let inputs = Inputs::Batch(inputs.into_iter().map(Input::from).collect());
        // py.detach releases the GIL while encode runs on Rust side
        let encodings = py
            .detach(|| self.pipeline()?.encode(inputs, &options).wait())
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
        py.detach(|| self.pipeline()?.decode(ids, skip_special_tokens))
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
        py.detach(|| -> tk::Result<Vec<String>> {
            let pipeline = self.pipeline()?;
            Ok(pipeline.decode_tokens(ids, skip_special_tokens))
        })
        .map_err(err)
    }

    /// Copies the tokenizer. The copy holds a reference to the same underlying compiled tokenizer.
    fn __copy__(&self) -> Self {
        Self {
            spec: Mutex::new(self.spec().copy()),
        }
    }

    /// Copies the tokenizer. The copy holds a reference to the same underlying compiled tokenizer.
    fn __deepcopy__(&self, _memo: &Bound<'_, PyDict>) -> Self {
        self.__copy__()
    }

    /// Pickle rebuilds a `Tokenizer` by calling `_unpickle` with these arguments.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (String,))> {
        let json = self.spec().to_json();
        Ok((py.get_type::<Self>().getattr("_unpickle")?, (json,)))
    }

    /// Unpickles a `Tokenizer`
    #[staticmethod]
    fn _unpickle(json: &str) -> PyResult<Self> {
        Self::from_spec(TokenizerSpec::from_json(json).map_err(err)?)
    }

    fn __repr__(&self) -> PyResult<String> {
        let file = self.spec().to_object();
        let padding = self
            .padding()
            .map_or_else(|| "None".to_owned(), |padding| padding.__repr__());
        let truncation = self
            .truncation()
            .map_or_else(|| "None".to_owned(), |truncation| truncation.__repr__());
        Ok(repr::tokenizer(&file, &padding, &truncation))
    }
}
