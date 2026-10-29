use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use serde_json::Value;

use tk::pipeline::PipelineTokenizer;
use tk::{
    PaddingParams, Result, TruncationParams, canonicalize_value, from_json, padding_from_json,
    padding_to_json, truncation_from_json, truncation_to_json,
};

use crate::JsonObject;
use crate::compiled::CompiledTokenizer;
use crate::components::ComponentKind;
use crate::components::decoders::Decoder;
use crate::components::models::Model;
use crate::components::normalizers::Normalizer;
use crate::components::pre_tokenizers::PreTokenizer;
use crate::components::processors::PostProcessor;
use crate::error::err;
use crate::options::{Padding, Truncation};
use crate::repr;
use crate::type_hints::Token;

/// The contents of a `tokenizer.json`, which can be edited.
///
/// Call `compile()` to get a `CompiledTokenizer`, which encodes and decodes.
///
/// Example:
/// ```python
/// from tokenizers import Tokenizer, Truncation
///
/// tokenizer = Tokenizer.from_pretrained("deepseek-ai/DeepSeek-V4.1-Flash")
/// tokenizer.truncation = Truncation(max_length=512)
/// compiled = tokenizer.compile()
/// ids = compiled.encode("Hello, World!").ids
/// print(ids)
/// print(compiled.decode_tokens(ids))
/// ```
#[pyclass(frozen, dict, weakref, module = "tokenizers")]
pub struct Tokenizer {
    spec: Mutex<TokenizerSpec>,
}

impl Tokenizer {
    fn new(spec: TokenizerSpec) -> Self {
        Self {
            spec: Mutex::new(spec),
        }
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
        self.spec.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[pymethods]
impl Tokenizer {
    /// Loads a `tokenizer.json` and upgrades it to the 2.0 format.
    ///
    /// Errors in the file's components, such as the model or the normalizer, show up at `compile()`.
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
        Self::from_str(&text)
    }

    /// Loads a `tokenizer.json` from its content.
    ///
    /// Args:
    ///     json:
    ///         The content of a `tokenizer.json`.
    #[staticmethod]
    #[pyo3(signature = (json))]
    fn from_str(json: &str) -> PyResult<Self> {
        Ok(Self::new(TokenizerSpec::from_json(json).map_err(err)?))
    }

    /// Writes the tokenizer as the content of a `tokenizer.json`.
    ///
    /// Args:
    ///     pretty:
    ///         Whether to indent the JSON.
    #[pyo3(signature = (pretty=false))]
    fn to_str(&self, pretty: bool) -> String {
        let json = self.spec().to_json();
        if !pretty {
            return json;
        }
        let value: Value = serde_json::from_str(&json).expect("the spec writes valid JSON");
        serde_json::to_string_pretty(&value).expect("a serde_json::Value always serializes")
    }

    /// Writes the tokenizer to a `tokenizer.json` file.
    ///
    /// Args:
    ///     path:
    ///         The file to write.
    ///     pretty:
    ///         Whether to indent the JSON.
    #[pyo3(signature = (path, pretty=true))]
    fn save(&self, path: PathBuf, pretty: bool) -> PyResult<()> {
        std::fs::write(&path, self.to_str(pretty)).map_err(|e| {
            std::io::Error::new(e.kind(), format!("cannot write {}: {e}", path.display()))
        })?;
        Ok(())
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

    /// The tokenizer model, which turns text into token ids.
    #[getter]
    fn model(slf: &Bound<'_, Self>) -> PyResult<Py<Model>> {
        let model = Self::get(slf)?;
        Ok(model.expect("a TokenizerSpec always holds a model"))
    }

    #[setter]
    fn set_model(&self, model: Bound<'_, Model>) {
        self.set(Some(model))
    }

    /// The normalizer, which cleans up the text before it is split.
    /// `None` if the tokenizer has no normalizer.
    #[getter]
    fn normalizer(slf: &Bound<'_, Self>) -> PyResult<Option<Py<Normalizer>>> {
        Self::get(slf)
    }

    #[setter]
    fn set_normalizer(&self, normalizer: Option<Bound<'_, Normalizer>>) {
        self.set(normalizer)
    }

    /// The pre-tokenizer, which splits the text into words.
    /// `None` if the tokenizer has no pre-tokenizer.
    #[getter]
    fn pre_tokenizer(slf: &Bound<'_, Self>) -> PyResult<Option<Py<PreTokenizer>>> {
        Self::get(slf)
    }

    #[setter]
    fn set_pre_tokenizer(&self, pre_tokenizer: Option<Bound<'_, PreTokenizer>>) {
        self.set(pre_tokenizer)
    }

    /// The post-processor, which adds special tokens.
    /// `None` if the tokenizer has no post-processor.
    #[getter]
    fn post_processor(slf: &Bound<'_, Self>) -> PyResult<Option<Py<PostProcessor>>> {
        Self::get(slf)
    }

    #[setter]
    fn set_post_processor(&self, post_processor: Option<Bound<'_, PostProcessor>>) {
        self.set(post_processor)
    }

    /// The decoder, which turns token ids back into text.
    /// `None` if the tokenizer has no decoder.
    #[getter]
    fn decoder(slf: &Bound<'_, Self>) -> PyResult<Option<Py<Decoder>>> {
        Self::get(slf)
    }

    #[setter]
    fn set_decoder(&self, decoder: Option<Bound<'_, Decoder>>) {
        self.set(decoder)
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

    /// Pickle rebuilds a `Tokenizer` by calling `_unpickle` with these arguments.
    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        // Bytes, not str: a non-ASCII str isn't stored as UTF-8, so each pickle step re-encodes it.
        // https://docs.python.org/3.14/c-api/unicode.html#unicode-objects
        let json = PyBytes::new(py, self.spec().to_json().as_bytes());
        Ok((py.get_type::<Self>().getattr("_unpickle")?, (json,)))
    }

    #[staticmethod]
    fn _unpickle(json: &[u8]) -> PyResult<Self> {
        let json = std::str::from_utf8(json).map_err(err)?;
        Ok(Self::new(TokenizerSpec::from_json(json).map_err(err)?))
    }

    fn __repr__(&self) -> String {
        self.spec().repr("Tokenizer")
    }
}

/// A `tokenizer.json` in the 2.0 format, one field per top-level key.
pub(crate) struct TokenizerSpec {
    /// The truncation an encode applies when it doesn't override it.
    truncation: Option<TruncationParams>,
    /// The padding an encode applies when it doesn't override it.
    padding: Option<PaddingParams>,
    role_to_token: Arc<Value>,
    added_tokens: Arc<Value>,
    normalizer: Option<Arc<ComponentSpec>>,
    pre_tokenizer: Option<Arc<ComponentSpec>>,
    post_processor: Option<Arc<ComponentSpec>>,
    decoder: Option<Arc<ComponentSpec>>,
    model: Arc<ComponentSpec>,
}

impl TokenizerSpec {
    /// Reads a `tokenizer.json` of any version and upgrades it to the 2.0 format.
    pub(crate) fn from_json(text: &str) -> Result<Self> {
        let mut file = serde_json::from_str(text)?;
        canonicalize_value(&mut file)?;
        let Value::Object(mut file) = file else {
            unreachable!("canonicalize_value refuses anything but an object");
        };
        let mut take = |key| file.remove(key).unwrap_or(Value::Null);
        let component = |value| match value {
            Value::Object(json) => Some(ComponentSpec::new(json)),
            _ => None,
        };
        Ok(Self {
            truncation: truncation_from_json(&take("truncation").to_string())?,
            padding: padding_from_json(&take("padding").to_string())?,
            role_to_token: Arc::new(take("role_to_token")),
            added_tokens: Arc::new(take("added_tokens")),
            normalizer: component(take("normalizer")),
            pre_tokenizer: component(take("pre_tokenizer")),
            post_processor: component(take("post_processor")),
            decoder: component(take("decoder")),
            model: component(take("model"))
                .expect("canonicalize_value refuses a model that is not an object"),
        })
    }

    /// Builds a new [`PipelineTokenizer`] from the current spec, padding and truncation included.
    pub(crate) fn compile(&self) -> Result<PipelineTokenizer> {
        from_json(&self.to_json())
    }

    /// A deep copy: the resulting object is independent.
    fn copy(&self) -> Self {
        let copy = |spec: &Option<Arc<ComponentSpec>>| spec.as_ref().map(|spec| spec.copy());
        Self {
            truncation: self.truncation.clone(),
            padding: self.padding.clone(),
            role_to_token: self.role_to_token.clone(),
            added_tokens: self.added_tokens.clone(),
            normalizer: copy(&self.normalizer),
            pre_tokenizer: copy(&self.pre_tokenizer),
            post_processor: copy(&self.post_processor),
            decoder: copy(&self.decoder),
            model: self.model.copy(),
        }
    }

    pub(crate) fn component(&self, key: ComponentKey) -> Option<&Arc<ComponentSpec>> {
        match key {
            ComponentKey::Normalizer => self.normalizer.as_ref(),
            ComponentKey::PreTokenizer => self.pre_tokenizer.as_ref(),
            ComponentKey::PostProcessor => self.post_processor.as_ref(),
            ComponentKey::Decoder => self.decoder.as_ref(),
            ComponentKey::Model => Some(&self.model),
        }
    }

    pub(crate) fn set_component(&mut self, key: ComponentKey, spec: Option<Arc<ComponentSpec>>) {
        match key {
            ComponentKey::Normalizer => self.normalizer = spec,
            ComponentKey::PreTokenizer => self.pre_tokenizer = spec,
            ComponentKey::PostProcessor => self.post_processor = spec,
            ComponentKey::Decoder => self.decoder = spec,
            ComponentKey::Model => {
                self.model = spec.expect("calling set_component on model should never pass None")
            }
        }
    }

    /// The top-level keys and their values, in the order `to_json` writes them.
    fn fields(&self) -> [(&'static str, Field<'_>); 10] {
        let component = |spec: &Option<Arc<ComponentSpec>>| {
            Field::Component(spec.as_deref().map(ComponentSpec::json))
        };
        [
            ("version", Field::Json(Cow::Owned(Value::from("2.0")))),
            ("truncation", Field::Truncation(self.truncation.as_ref())),
            ("padding", Field::Padding(self.padding.as_ref())),
            (
                "role_to_token",
                Field::Json(Cow::Borrowed(&self.role_to_token)),
            ),
            (
                "added_tokens",
                Field::Json(Cow::Borrowed(&self.added_tokens)),
            ),
            ("normalizer", component(&self.normalizer)),
            ("pre_tokenizer", component(&self.pre_tokenizer)),
            ("post_processor", component(&self.post_processor)),
            ("decoder", component(&self.decoder)),
            ("model", Field::Component(Some(self.model.json()))),
        ]
    }

    /// Renders the spec as `name(...)`, one line per top-level key.
    pub(crate) fn repr(&self, name: &str) -> String {
        repr::tokenizer(name, self.fields())
    }

    /// Writes the spec as a `tokenizer.json` in the 2.0 format.
    fn to_json(&self) -> String {
        let mut out = String::from("{");
        for (key, field) in self.fields() {
            out.push('"');
            out.push_str(key);
            out.push_str("\":");
            out.push_str(&field.to_json());
            out.push(',');
        }
        out.pop();
        out.push('}');
        out
    }
}

/// The value under one top-level key of a `tokenizer.json`.
pub(crate) enum Field<'spec> {
    Json(Cow<'spec, Value>),
    Component(Option<Arc<JsonObject>>),
    Padding(Option<&'spec PaddingParams>),
    Truncation(Option<&'spec TruncationParams>),
}

impl Field<'_> {
    fn to_json(&self) -> String {
        match self {
            Field::Json(value) => value.to_string(),
            Field::Component(json) => serde_json::to_string(&json.as_deref())
                .expect("a JSON object with string keys always writes"),
            Field::Padding(padding) => padding_to_json(*padding),
            Field::Truncation(truncation) => truncation_to_json(*truncation),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ComponentKey {
    Normalizer,
    PreTokenizer,
    PostProcessor,
    Decoder,
    Model,
}

/// A component's JSON spec, as a shared reference.
pub(crate) struct ComponentSpec(Mutex<Arc<JsonObject>>);

impl ComponentSpec {
    pub(crate) fn new(json: JsonObject) -> Arc<Self> {
        Arc::new(Self(Mutex::new(Arc::new(json))))
    }

    /// A new [`ComponentSpec`] over the same JSON. An edit of either one copies the JSON first.
    fn copy(&self) -> Arc<Self> {
        Arc::new(Self(Mutex::new(self.json())))
    }

    pub(crate) fn json(&self) -> Arc<JsonObject> {
        self.lock().clone()
    }

    /// Runs `edit` on the JSON, and store a reference to the new value in `self`.
    /// Other pointers to the component (eg a compiled snapshot or a [`detached`](Self::detached) copy)
    /// are unaffected and point to the previous value.
    pub(crate) fn edit<R>(&self, edit: impl FnOnce(&mut JsonObject) -> R) -> R {
        edit(Arc::make_mut(&mut self.lock()))
    }

    fn lock(&self) -> MutexGuard<'_, Arc<JsonObject>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
