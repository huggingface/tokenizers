use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::{Map, Value};

use tk::pipeline::PipelineTokenizer;
use tk::{
    PaddingParams, Result, TokenizerJson, TruncationParams, canonicalize_value, from_json,
    padding_from_json, padding_to_json, truncation_from_json, truncation_to_json,
};

use crate::compiled::CompiledTokenizer;
use crate::error::err;
use crate::options::{Padding, Truncation};
use crate::repr;
use crate::type_hints::Token;

/// This class is used to spec and build a `CompiledTokenizer`
///
/// Call `compile()` to get a `CompiledTokenizer`, which encodes and decodes.
///
/// Example:
/// ```python
/// from tokenizers import TokenizerSpec
/// from tokenizers.normalizers import NFC
///
/// spec = TokenizerSpec.from_pretrained("deepseek-ai/DeepSeek-V4.1-Flash")
/// spec.normalizer = NFC()
/// tokenizer = spec.compile()
/// ids = tokenizer.encode("Hello, World!").ids
/// print(ids)
/// print(tokenizer.decode_tokens(ids))
/// ```
#[pyclass(name = "TokenizerSpec", frozen, dict, weakref, module = "tokenizers")]
pub struct PyTokenizerSpec {
    spec: Mutex<TokenizerSpec>,
}

impl PyTokenizerSpec {
    fn new(spec: TokenizerSpec) -> Self {
        Self {
            spec: Mutex::new(spec),
        }
    }

    fn spec(&self) -> MutexGuard<'_, TokenizerSpec> {
        self.spec.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[pymethods]
impl PyTokenizerSpec {
    /// Loads a `tokenizer.json`.
    ///
    /// The file is only read here. Errors in its components show up at `compile()`.
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
        Ok(Self::new(TokenizerSpec::from_json(&text).map_err(err)?))
    }

    /// Instantiate a new `TokenizerSpec` from an existing file on the Hugging Face Hub.
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
    ///     `TokenizerSpec`: The tokenizer the file describes.
    ///
    /// Examples:
    ///
    /// ```python
    /// # Download tokenizer.json from huggingface.co and cache it.
    /// tokenizer = TokenizerSpec.from_pretrained("openai-community/gpt2")
    ///
    /// # Pin a revision: a branch name, a tag name or a commit id.
    /// tokenizer = TokenizerSpec.from_pretrained("openai-community/gpt2", revision="607a30d783dfa663caf39e06633721c8d4cfcd7e")
    ///
    /// # A private or gated repo, with the token `hf auth login` stored.
    /// tokenizer = TokenizerSpec.from_pretrained("my-org/my-model", token=True)
    ///
    /// # Read the cache without contacting the Hub.
    /// tokenizer = TokenizerSpec.from_pretrained("openai-community/gpt2", local_files_only=True)
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

    /// The padding a tokenizer compiled from this spec applies to every encode, or `None`.
    /// Assign `None` to switch padding off.
    #[getter]
    fn padding(&self) -> Option<Padding> {
        self.spec().padding.clone().map(Padding::from)
    }

    #[setter]
    fn set_padding(&self, padding: Option<PyRef<'_, Padding>>) {
        self.spec().padding = padding.map(|padding| padding.params().clone());
    }

    /// The truncation a tokenizer compiled from this spec applies to every encode, or `None`.
    /// Assign `None` to switch truncation off.
    #[getter]
    fn truncation(&self) -> Option<Truncation> {
        self.spec().truncation.clone().map(Truncation::from)
    }

    #[setter]
    fn set_truncation(&self, truncation: Option<PyRef<'_, Truncation>>) {
        self.spec().truncation = truncation.map(|truncation| truncation.params().clone());
    }

    /// Builds a new `CompiledTokenizer` from the spec as it is now.
    ///
    /// Every call builds the whole tokenizer again. Later edits of the spec do not reach the
    /// compiled tokenizer: compile again to see them.
    fn compile(&self, py: Python<'_>) -> PyResult<CompiledTokenizer> {
        let spec = self.spec().copy();
        CompiledTokenizer::new(py, &spec)
    }

    /// An independent copy: editing one spec leaves the other unchanged.
    fn __copy__(&self) -> Self {
        Self::new(self.spec().copy())
    }

    /// An independent copy: editing one spec leaves the other unchanged.
    fn __deepcopy__(&self, _memo: &Bound<'_, PyDict>) -> Self {
        self.__copy__()
    }

    /// Pickle rebuilds a `TokenizerSpec` by calling `_unpickle` with these arguments.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (String,))> {
        let json = self.spec().to_json();
        Ok((py.get_type::<Self>().getattr("_unpickle")?, (json,)))
    }

    /// Unpickles a `TokenizerSpec`
    #[staticmethod]
    fn _unpickle(json: &str) -> PyResult<Self> {
        Ok(Self::new(TokenizerSpec::from_json(json).map_err(err)?))
    }

    fn __repr__(&self) -> String {
        let spec = self.spec();
        repr::tokenizer_spec(
            &spec.to_object(),
            spec.padding.as_ref(),
            spec.truncation.as_ref(),
        )
    }
}

type JsonObject = Map<String, Value>;

/// Holds the specification (json) of a [`PipelineTokenizer`].
pub(crate) struct TokenizerSpec {
    /// The tokenizers.json version field
    version: String,
    /// The truncation an encode applies when it doesn't override it.
    truncation: Option<TruncationParams>,
    /// The padding an encode applies when it doesn't override it.
    padding: Option<PaddingParams>,
    /// Only in "2.0" files
    role_to_token: Option<Arc<Value>>,
    added_tokens: Arc<Value>,
    normalizer: Option<Arc<ComponentSpec>>,
    pre_tokenizer: Option<Arc<ComponentSpec>>,
    post_processor: Option<Arc<ComponentSpec>>,
    decoder: Option<Arc<ComponentSpec>>,
    model: Arc<ComponentSpec>,
}

impl TokenizerSpec {
    /// Reads a tokenizers.json
    pub(crate) fn from_json(text: &str) -> Result<Self> {
        let file = TokenizerJson::from_value(serde_json::from_str(text)?)?;
        Ok(Self {
            version: file.version,
            truncation: truncation_from_json(&file.truncation.to_string())?,
            padding: padding_from_json(&file.padding.to_string())?,
            role_to_token: file.role_to_token.map(Arc::new),
            added_tokens: Arc::new(file.added_tokens),
            normalizer: file.normalizer.map(ComponentSpec::new),
            pre_tokenizer: file.pre_tokenizer.map(ComponentSpec::new),
            post_processor: file.post_processor.map(ComponentSpec::new),
            decoder: file.decoder.map(ComponentSpec::new),
            model: ComponentSpec::new(file.model),
        })
    }

    /// Builds a new [`PipelineTokenizer`] from the current spec, padding and truncation included.
    pub(crate) fn compile(&self) -> Result<PipelineTokenizer> {
        let mut tokenizer_json = Value::Object(self.to_object());
        canonicalize_value(&mut tokenizer_json)?;
        // XXX: the .to_string() copies JSON data a second time, could be improved
        from_json(&tokenizer_json.to_string())
    }

    /// An independent copy: editing components of the new object do not affect `self`'s components
    fn copy(&self) -> Self {
        let detached =
            |spec: &Option<Arc<ComponentSpec>>| spec.as_ref().map(|spec| spec.detached());
        Self {
            version: self.version.clone(),
            truncation: self.truncation.clone(),
            padding: self.padding.clone(),
            role_to_token: self.role_to_token.clone(),
            added_tokens: self.added_tokens.clone(),
            normalizer: detached(&self.normalizer),
            pre_tokenizer: detached(&self.pre_tokenizer),
            post_processor: detached(&self.post_processor),
            decoder: detached(&self.decoder),
            model: self.model.detached(),
        }
    }

    /// Returns a copy of the spec as a `JsonObject`
    fn to_object(&self) -> JsonObject {
        let json = |spec: &Option<Arc<ComponentSpec>>| {
            spec.as_ref().map(|spec| JsonObject::clone(&spec.json()))
        };
        TokenizerJson {
            version: self.version.clone(),
            truncation: params(truncation_to_json(self.truncation.as_ref())),
            padding: params(padding_to_json(self.padding.as_ref())),
            role_to_token: self.role_to_token.as_deref().cloned(),
            added_tokens: Value::clone(&self.added_tokens),
            normalizer: json(&self.normalizer),
            pre_tokenizer: json(&self.pre_tokenizer),
            post_processor: json(&self.post_processor),
            decoder: json(&self.decoder),
            model: JsonObject::clone(&self.model.json()),
        }
        .into_object()
    }

    fn to_json(&self) -> String {
        Value::Object(self.to_object()).to_string()
    }
}

/// A component's JSON spec, as a shared reference.
struct ComponentSpec(Mutex<Arc<JsonObject>>);

impl ComponentSpec {
    fn new(json: JsonObject) -> Arc<Self> {
        Arc::new(Self(Mutex::new(Arc::new(json))))
    }

    /// A new [`ComponentSpec`] over the same JSON. An edit of either one copies the JSON first.
    fn detached(&self) -> Arc<Self> {
        Arc::new(Self(Mutex::new(self.json())))
    }

    fn json(&self) -> Arc<JsonObject> {
        self.lock().clone()
    }

    fn lock(&self) -> MutexGuard<'_, Arc<JsonObject>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

fn params(json: String) -> Value {
    serde_json::from_str(&json).expect("tk-serialize writes valid JSON")
}
