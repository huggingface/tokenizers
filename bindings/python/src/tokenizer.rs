use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use tk_encode::PaddingParams;
use tk_encode::TruncationParams;
use tk_encode::pipeline::{EncodeOptions, Override, PipelineTokenizer as Pipeline};

use crate::encoding::Encoding;
use crate::error::{convert_err, err, poison_err};
use crate::options::{OverrideSentinel, Padding, Truncation};
use crate::repr;
use crate::type_hints::{Token, TokenIds};

/// A tokenizer. Encodes text into token ids, and decodes token ids back into text.
#[pyclass(frozen, module = "tokenizers")]
pub struct Tokenizer {
    pipeline: Pipeline,
    // Needs a mutex so a concurrent thread can access the value while
    // encode is running.
    padding: Mutex<Option<PaddingParams>>,
    truncation: Mutex<Option<TruncationParams>>,
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
            OverrideSentinel::InheritConfig => {
                self.clone_padding()?.map_or(Override::Off, Override::With)
            }
            OverrideSentinel::Off => Override::Off,
            OverrideSentinel::With(params) => Override::With(params),
        };
        let truncation = match truncation {
            OverrideSentinel::InheritConfig => self
                .clone_truncation()?
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

    fn clone_padding(&self) -> PyResult<Option<PaddingParams>> {
        Ok(self.padding.lock().map_err(poison_err)?.clone())
    }

    fn clone_truncation(&self) -> PyResult<Option<TruncationParams>> {
        Ok(self.truncation.lock().map_err(poison_err)?.clone())
    }

    fn from_pipeline(pipeline: Pipeline) -> Self {
        let padding = pipeline.get_padding().cloned();
        let truncation = pipeline.get_truncation().cloned();
        Self {
            pipeline,
            padding: Mutex::new(padding),
            truncation: Mutex::new(truncation),
        }
    }
}

/// What `Tokenizer.__reduce__` gives to pickle
type UnpickleArguments = (String, Option<Padding>, Option<Truncation>);

#[pymethods]
impl Tokenizer {
    /// Loads a `tokenizer.json`.
    ///
    /// Args:
    ///     path:
    ///         The file to read.
    ///     role_to_token (`dict[str, str]`, *optional*):
    ///         Replaces the `role_to_token` map the file declares. When omitted, the file's map is kept.
    #[staticmethod]
    #[pyo3(signature = (path, *, role_to_token=None))]
    fn from_file(path: PathBuf, role_to_token: Option<BTreeMap<String, String>>) -> PyResult<Self> {
        let canonical = tk_convert::canonicalize_file(path).map_err(convert_err)?;
        let mut pipeline: Pipeline = tk_serialize::from_json(&canonical).map_err(err)?;
        if let Some(role_to_token) = role_to_token {
            pipeline = pipeline.with_role_to_token(role_to_token);
        }
        Ok(Self::from_pipeline(pipeline))
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
    ///     role_to_token (`dict[str, str]`, *optional*):
    ///         Overrides the `role_to_token` map defined by the tokenizer's config, if any
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
    #[pyo3(signature = (identifier, revision="main", token=None, *, cache_dir=None, force_download=false, local_files_only=false, subfolder=None, role_to_token=None))]
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
        role_to_token: Option<BTreeMap<String, String>>,
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
        Self::from_file(path, role_to_token)
    }

    /// A mapping of role (eg `eos_token`) to the corresponding token text.
    ///
    /// Returns a copy: editing does not change the tokenizer. Use `with_role_to_token` to mutate the tokenizer.
    #[getter]
    fn role_to_token(&self) -> BTreeMap<String, String> {
        self.pipeline.get_role_to_token().clone()
    }

    /// Returns a new copy of the `Tokenizer` instance with `role_to_token`.
    /// `self` is not mutated.
    ///
    /// Args:
    ///     role_to_token (`dict[str, str]`):
    ///         A mapping of role (eg `eos_token`) to the corresponding token text.
    ///
    /// Returns:
    ///     `Tokenizer`
    fn with_role_to_token(&self, role_to_token: BTreeMap<String, String>) -> PyResult<Self> {
        Ok(Self {
            pipeline: self.pipeline.with_role_to_token(role_to_token),
            padding: Mutex::new(self.clone_padding()?),
            truncation: Mutex::new(self.clone_truncation()?),
        })
    }

    /// The padding applied to every encode, or `None`.
    /// Assign `None` to switch padding off.
    #[getter]
    fn padding(&self) -> PyResult<Option<Padding>> {
        Ok(self.clone_padding()?.map(Padding::from))
    }

    #[setter]
    fn set_padding(&self, padding: Option<PyRef<'_, Padding>>) -> PyResult<()> {
        *self.padding.lock().map_err(poison_err)? = padding.map(|padding| padding.params().clone());
        Ok(())
    }

    /// The truncation applied to every encode, or `None`.
    /// Assign `None` to switch truncation off.
    #[getter]
    fn truncation(&self) -> PyResult<Option<Truncation>> {
        Ok(self.clone_truncation()?.map(Truncation::from))
    }

    #[setter]
    fn set_truncation(&self, truncation: Option<PyRef<'_, Truncation>>) -> PyResult<()> {
        *self.truncation.lock().map_err(poison_err)? =
            truncation.map(|truncation| truncation.params().clone());
        Ok(())
    }

    /// Encodes the given text to token ids.
    ///
    /// Args:
    ///     text: str
    ///         The text to encode.
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
    #[pyo3(signature = (text, *, add_special_tokens=true, encode_special_tokens=false, padding=OverrideSentinel::<PaddingParams>::InheritConfig, truncation=OverrideSentinel::<TruncationParams>::InheritConfig))]
    fn encode(
        &self,
        py: Python<'_>,
        text: String,
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
        let encodings = py
            .detach(|| self.pipeline.encode(text, &options).wait())
            .map_err(err)?;
        Ok(Encoding::from(&encodings[0]))
    }

    /// Encodes the given text and returns the string representation of each token.
    ///
    /// Shorthand for `decode_tokens(encode(text))`.
    /// If you also need to access token ids, use `encode(text)`.
    ///
    /// Args:
    ///     text: str
    ///         The text to tokenize.
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
    #[pyo3(signature = (text, *, add_special_tokens=true, encode_special_tokens=false, padding=OverrideSentinel::<PaddingParams>::InheritConfig, truncation=OverrideSentinel::<TruncationParams>::InheritConfig))]
    fn tokenize(
        &self,
        py: Python<'_>,
        text: String,
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
        py.detach(|| -> tk_encode::Result<Vec<String>> {
            let encodings = self.pipeline.encode(text, &options).wait()?;
            let ids: Vec<u32> = encodings[0].ids().iter().map(|token| token.id()).collect();
            Ok(self.pipeline.decode_tokens(&ids, false))
        })
        .map_err(err)
    }

    /// Encodes a batch of text.
    /// The encodings come back in input order.
    ///
    /// Args:
    ///     texts: List[str]
    ///         The batch of text to encode.
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
    #[pyo3(signature = (texts, *, add_special_tokens=true, encode_special_tokens=false, padding=OverrideSentinel::<PaddingParams>::InheritConfig, truncation=OverrideSentinel::<TruncationParams>::InheritConfig))]
    fn encode_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
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
        // py.detach releases the GIL while encode runs on Rust side
        let encodings = py
            .detach(|| self.pipeline.encode(texts, &options).wait())
            .map_err(err)?;
        Ok(encodings.iter().map(Encoding::from).collect())
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
    ) -> Vec<String> {
        let ids = ids.as_slice();
        py.detach(|| self.pipeline.decode_tokens(ids, skip_special_tokens))
    }

    /// Pickle rebuilds a `Tokenizer` by calling `_unpickle` with these arguments.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, UnpickleArguments)> {
        let json = tk_serialize::to_json(&self.pipeline).map_err(err)?;
        Ok((
            py.get_type::<Self>().getattr("_unpickle")?,
            (json, self.padding()?, self.truncation()?),
        ))
    }

    /// Unpickles a `Tokenizer`
    #[staticmethod]
    fn _unpickle(
        json: &str,
        padding: Option<PyRef<'_, Padding>>,
        truncation: Option<PyRef<'_, Truncation>>,
    ) -> PyResult<Self> {
        let pipeline: Pipeline = tk_serialize::from_json(json).map_err(err)?;
        Ok(Self {
            pipeline,
            padding: Mutex::new(padding.map(|padding| padding.params().clone())),
            truncation: Mutex::new(truncation.map(|trunc| trunc.params().clone())),
        })
    }

    fn __repr__(&self) -> PyResult<String> {
        let file = tk_serialize::to_json(&self.pipeline).map_err(err)?;
        let file = tk_serialize::json::Json::parse(&file).map_err(err)?;
        let padding = self
            .padding()?
            .map_or_else(|| "None".to_owned(), |padding| padding.__repr__());
        let truncation = self
            .truncation()?
            .map_or_else(|| "None".to_owned(), |truncation| truncation.__repr__());
        Ok(repr::tokenizer(&file, &padding, &truncation))
    }
}
