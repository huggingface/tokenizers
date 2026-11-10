//! The Python [`Tokenizer`]: holds an editable spec / configuration for a tokenizer.
//! It is then compiled into [`CompiledTokenizer`] to encode text into tokens and decode
//! tokens back to text.
//!
//! # Two tokenizer types
//!
//! The Python bindings has two objects to represent a tokenizer:
//!
//! - [`Tokenizer`] is the editable specification. It stores the settings as JSON.
//! - [`CompiledTokenizer`] is a immutable, optimized class for encoding and decoding. It never changes once built.
//!
//! `compile()` builds the second from the first:
//!
//! ```text
//! tokenizer.json ──from_str()──▶ Tokenizer ──compile()──▶ CompiledTokenizer
//!                                   ▲
//!                                   └── can be edited: tok.normalizer = ..., tok.padding = ...
//! ```
//!
//! [`CompiledTokenizer`]: crate::compiled::CompiledTokenizer
//!
//! # What `Tokenizer` holds
//!
//! The [`Tokenizer`] struct holds a [`TokenizerSpec`] behind a `Mutex`.
//! The `Mutex` lets setters work on a frozen Python class, from any thread.
//! The [`TokenizerSpec`] has one field per top-level key of `tokenizer.json`.
//!
//! - `truncation` and `padding` are default settings
//! - `added_tokens` and `role_to_token`
//! - `normalizer`, `pre_tokenizer`, `post_processor`, `decoder` and `model`
//!
//! `normalizer`, `pre_tokenizer`, `post_processor`, `decoder` and `model` are [`ComponentSpec`].
//! See [`components`](crate::components) module docs for more details about components.
//!
//! Copying a [`Tokenizer`] or its components is cheap:
//! see the section about [copy on write](crate::components::component_spec#arc-pointers-make-copies-cheap).
//!
//! # Loading from JSON
//!
//! `from_str`, `from_file` and `from_pretrained` accept a `tokenizer.json` of any version.
//! `canonicalize_value` first rewrites it into the 2.0 format. Then each top-level key is parsed to its
//! field.
//!
//! # Writing to JSON
//!
//! `to_str`, pickle, `repr` and `compile()` all write the spec as a JSON string.
//!
//! # Compiling
//!
//! `compile()` copies the spec and releases the mutex lock. immediately. It then builds from the copy
//! through `tk::from_json`, with the GIL released. So the build blocks no other thread, and later
//! edits never reach the compiled tokenizer.
//!
//! # Validation
//!
//! The JSON spec is validated at compilation time by `tk::from_json`, `.compile()` can fail if the
//! JSON spec stored in [`Tokenizer`] is not valid.

use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use serde_json::Value;

use tk::pipeline::PipelineTokenizer;
use tk::vocab::bucket_added_vocabulary::AddedToken;
use tk::{
    PaddingParams, Result, TruncationParams, canonicalize_value, from_json, padding_from_json,
    padding_to_json, truncation_from_json, truncation_to_json,
};

use crate::JsonObject;
use crate::added_token::TokenToAdd;
use crate::added_tokens_spec::AddedTokensSpec;
use crate::compiled::CompiledTokenizer;
use crate::components::ComponentKind;
use crate::components::component_spec::ComponentSpec;
use crate::components::decoders::Decoder;
use crate::components::models::Model;
use crate::components::normalizers::Normalizer;
use crate::components::pre_tokenizers::PreTokenizer;
use crate::components::processors::PostProcessor;
use crate::error::err;
use crate::options::{Padding, Truncation};
use crate::repr;
use crate::type_hints::Token;

/// An editable tokenizer, loaded from a `tokenizer.json`.
///
/// Edit its components and settings, then call `compile()` to get a `CompiledTokenizer`,
/// which encodes and decodes.
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
    fn from_spec(spec: TokenizerSpec) -> Self {
        Self {
            spec: Mutex::new(spec),
        }
    }

    /// Generic getter for a component kind, read from [`ComponentKind::tokenizer_field_mut`].
    fn get<K: ComponentKind>(slf: &Bound<'_, Self>) -> PyResult<Option<Py<K>>> {
        // Copy the field out, so the lock is released before from_spec builds Python objects.
        let spec = {
            let mut tokenizer_spec = slf.get().spec();
            K::tokenizer_field_mut(&mut tokenizer_spec).clone()
        };
        match spec {
            Some(spec) => K::from_spec(slf.py(), spec).map(Some),
            None => Ok(None),
        }
    }

    /// Generic setter for a component kind, written to [`ComponentKind::tokenizer_field_mut`].
    fn set<K: ComponentKind>(&self, component: Option<Bound<'_, K>>) {
        let spec = component.map(|component| K::spec(&component).clone());
        let mut tokenizer_spec = self.spec();
        *K::tokenizer_field_mut(&mut tokenizer_spec) = spec;
    }

    /// Locks the spec. Drop the guard before calling into Python, which can lock it again and
    /// deadlock: copy out what you need first, as [`Self::get`] does.
    fn spec(&self) -> MutexGuard<'_, TokenizerSpec> {
        self.spec.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[pymethods]
impl Tokenizer {
    /// A tokenizer made of `model` alone. Set the other components afterwards.
    ///
    /// The tokenizer shares `model`: an edit through either one reaches both.
    ///
    /// Args:
    ///     model:
    ///         The model that turns words into ids.
    #[new]
    #[pyo3(signature = (model))]
    fn new(model: Bound<'_, Model>) -> Self {
        Self::from_spec(TokenizerSpec::from_model(Model::spec(&model).clone()))
    }

    /// Loads a tokenizer from a `tokenizer.json` file.
    ///
    /// Components are not validated at this stage. A component with an unknown type or an invalid
    /// setting raises `ValueError` in `Tokenizer.compile()`.
    ///
    /// Args:
    ///     path: str or Path
    ///         The file to read.
    ///
    /// Raises:
    ///     ValueError: The JSON is malformed or is not a `tokenizer.json`.
    ///     OSError: The file cannot be read.
    #[staticmethod]
    #[pyo3(signature = (path))]
    fn from_file(path: PathBuf) -> PyResult<Self> {
        let text = std::fs::read_to_string(&path).map_err(|e| {
            std::io::Error::new(e.kind(), format!("cannot read {}: {e}", path.display()))
        })?;
        Self::from_str(&text)
    }

    /// Loads a tokenizer from a JSON string, in the `tokenizer.json` format.
    ///
    /// Components are not validated at this stage. A component with an unknown type or an invalid
    /// setting raises `ValueError` in `Tokenizer.compile()`.
    ///
    /// Args:
    ///     json: str
    ///         The JSON string to read.
    ///
    /// Raises:
    ///     ValueError: The JSON is malformed or is not a `tokenizer.json`.
    #[staticmethod]
    #[pyo3(signature = (json))]
    fn from_str(json: &str) -> PyResult<Self> {
        Ok(Self::from_spec(
            TokenizerSpec::from_json(json).map_err(err)?,
        ))
    }

    /// Returns the tokenizer as a JSON string, in the `tokenizer.json` format.
    ///
    /// Args:
    ///     pretty: bool
    ///         Whether to indent the JSON. Defaults to `False`.
    #[pyo3(signature = (pretty=false))]
    fn to_str(&self, pretty: bool) -> String {
        let json = self.spec().to_json();
        if !pretty {
            return json;
        }
        let value: Value = serde_json::from_str(&json).expect("the spec writes valid JSON");
        serde_json::to_string_pretty(&value).expect("a serde_json::Value always serializes")
    }

    /// Saves the tokenizer to a file as JSON, for example `tokenizer.json`
    ///
    /// Args:
    ///     path: str or Path
    ///         The file to write.
    ///     pretty: bool
    ///         Whether to indent the JSON. Defaults to `True`.
    #[pyo3(signature = (path, pretty=true))]
    fn save(&self, path: PathBuf, pretty: bool) -> PyResult<()> {
        std::fs::write(&path, self.to_str(pretty)).map_err(|e| {
            std::io::Error::new(e.kind(), format!("cannot write {}: {e}", path.display()))
        })?;
        Ok(())
    }

    /// Loads a tokenizer from the `tokenizer.json` of a model repo on the Hugging Face Hub.
    ///
    /// `huggingface_hub.hf_hub_download` downloads and caches the file.
    /// Components are not validated at this stage. A component with an unknown type or an invalid
    /// setting raises `ValueError` in `Tokenizer.compile()`.
    ///
    /// Args:
    ///     identifier: str
    ///         The id of a repo on huggingface.co, such as `"openai-community/gpt2"`.
    ///     revision: str
    ///         A branch name, a tag name or a commit id. Defaults to `"main"`.
    ///     token: str, bool or None
    ///         The token to send to the Hub. `True` uses the token stored by `hf auth login`.
    ///         `False` sends no token. `None` uses the stored token when there is one.
    ///     cache_dir: str, Path or None
    ///         A cache directory to use instead of the default one.
    ///     force_download: bool
    ///         Whether to download the file again, even when it is cached.
    ///     local_files_only: bool
    ///         Whether to read only from the cache, without contacting the Hub.
    ///     subfolder: str or None
    ///         The folder of the repo that holds `tokenizer.json`.
    ///
    /// Returns:
    ///     Tokenizer
    ///
    /// Raises:
    ///     ValueError: The downloaded JSON is malformed or is not a `tokenizer.json`.
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

    /// The model, which turns words into token ids.
    ///
    /// Returns an editable view of the tokenizer's model.
    #[getter]
    fn model(slf: &Bound<'_, Self>) -> PyResult<Py<Model>> {
        let model = Self::get(slf)?;
        Ok(model.expect("a TokenizerSpec always holds a model"))
    }

    #[setter]
    fn set_model(&self, model: Bound<'_, Model>) {
        self.set(Some(model))
    }

    /// The normalizer, which cleans up the text before it is split, or `None`.
    ///
    /// Returns an editable view of the tokenizer's normalizer.
    #[getter]
    fn normalizer(slf: &Bound<'_, Self>) -> PyResult<Option<Py<Normalizer>>> {
        Self::get(slf)
    }

    #[setter]
    fn set_normalizer(&self, normalizer: Option<Bound<'_, Normalizer>>) {
        self.set(normalizer)
    }

    /// The pre-tokenizer, which splits the text into words, or `None`.
    ///
    /// Returns an editable view of the tokenizer's pre-tokenizer.
    #[getter]
    fn pre_tokenizer(slf: &Bound<'_, Self>) -> PyResult<Option<Py<PreTokenizer>>> {
        Self::get(slf)
    }

    #[setter]
    fn set_pre_tokenizer(&self, pre_tokenizer: Option<Bound<'_, PreTokenizer>>) {
        self.set(pre_tokenizer)
    }

    /// The post-processor, which adds special tokens, or `None`.
    ///
    /// Returns an editable view of the tokenizer's post-processor.
    #[getter]
    fn post_processor(slf: &Bound<'_, Self>) -> PyResult<Option<Py<PostProcessor>>> {
        Self::get(slf)
    }

    #[setter]
    fn set_post_processor(&self, post_processor: Option<Bound<'_, PostProcessor>>) {
        self.set(post_processor)
    }

    /// The decoder, which turns token ids back into text, or `None`.
    ///
    /// Returns an editable view of the tokenizer's decoder.
    #[getter]
    fn decoder(slf: &Bound<'_, Self>) -> PyResult<Option<Py<Decoder>>> {
        Self::get(slf)
    }

    #[setter]
    fn set_decoder(&self, decoder: Option<Bound<'_, Decoder>>) {
        self.set(decoder)
    }

    /// The padding that `CompiledTokenizer.encode` applies by default, or `None` for no padding.
    #[getter]
    fn padding(&self) -> Option<Padding> {
        self.spec().padding.clone().map(Padding::from)
    }

    #[setter]
    fn set_padding(&self, padding: Option<PyRef<'_, Padding>>) {
        self.spec().padding = padding.map(|padding| padding.params().clone());
    }

    /// The truncation that `CompiledTokenizer.encode` applies by default, or `None` for no
    /// truncation.
    #[getter]
    fn truncation(&self) -> Option<Truncation> {
        self.spec().truncation.clone().map(Truncation::from)
    }

    #[setter]
    fn set_truncation(&self, truncation: Option<PyRef<'_, Truncation>>) {
        self.spec().truncation = truncation.map(|truncation| truncation.params().clone());
    }

    /// Adds tokens matched in the text before the model runs.
    ///
    /// A token already added with the same content gets the new flags. A token in the model's
    /// vocabulary keeps its id there; any other new token gets the next free id at `compile()`.
    ///
    /// Args:
    ///     tokens:
    ///         The tokens to add. A `str` is added as `AddedToken(content)`.
    ///
    /// Returns:
    ///     The number of tokens added or changed.
    fn add_tokens(&self, tokens: Vec<TokenToAdd<'_>>) -> usize {
        let tokens = tokens
            .into_iter()
            .map(|token| token.into_added_token(false));
        self.spec().add_tokens(tokens)
    }

    /// Adds special tokens: `decode` skips them with `skip_special_tokens`.
    ///
    /// Works as `add_tokens`, with every token marked special.
    ///
    /// Args:
    ///     tokens:
    ///         The tokens to add. A `str` is added as `AddedToken(content, special=True)`.
    ///
    /// Returns:
    ///     The number of tokens added or changed.
    fn add_special_tokens(&self, tokens: Vec<TokenToAdd<'_>>) -> usize {
        let tokens = tokens.into_iter().map(|token| token.into_added_token(true));
        self.spec().add_tokens(tokens)
    }

    /// Builds a `CompiledTokenizer` from the tokenizer's current settings.
    ///
    /// Edits made after the call don't reach the compiled tokenizer: call `compile()` again.
    ///
    /// Raises:
    ///     ValueError: A component has an unknown type or an invalid setting.
    fn compile(&self, py: Python<'_>) -> PyResult<CompiledTokenizer> {
        let spec = self.spec().deep_clone();
        CompiledTokenizer::new(py, &spec)
    }

    /// Returns an independent copy. Editing one tokenizer leaves the other unchanged.
    /// Copying is cheap, even with a large vocab.
    fn __copy__(&self) -> Self {
        Self::from_spec(self.spec().deep_clone())
    }

    /// Returns an independent copy. Editing one tokenizer leaves the other unchanged.
    /// Copying is cheap, even with a large vocab.
    fn __deepcopy__(&self, _memo: &Bound<'_, PyDict>) -> Self {
        self.__copy__()
    }

    /// Pickle rebuilds a `Tokenizer` by calling `_unpickle` with these arguments.
    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        let json = self.spec().to_json();
        // Bytes, not str: a non-ASCII str isn't stored as UTF-8, so each pickle step re-encodes it.
        // https://docs.python.org/3.14/c-api/unicode.html#unicode-objects
        let json = PyBytes::new(py, json.as_bytes());
        Ok((py.get_type::<Self>().getattr("_unpickle")?, (json,)))
    }

    #[staticmethod]
    fn _unpickle(json: &[u8]) -> PyResult<Self> {
        let json = std::str::from_utf8(json).map_err(err)?;
        Ok(Self::from_spec(
            TokenizerSpec::from_json(json).map_err(err)?,
        ))
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
    added_tokens: Arc<AddedTokensSpec>,
    pub(crate) normalizer: Option<ComponentSpec>,
    pub(crate) pre_tokenizer: Option<ComponentSpec>,
    pub(crate) post_processor: Option<ComponentSpec>,
    pub(crate) decoder: Option<ComponentSpec>,
    pub(crate) model: Option<ComponentSpec>,
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
        let component = |items_key, value| match value {
            Value::Object(json) => Some(ComponentSpec::from_json(items_key, json)),
            _ => None,
        };
        let added_tokens = match take("added_tokens") {
            Value::Null => AddedTokensSpec::new(Vec::new()),
            Value::Array(entries) => AddedTokensSpec::new(entries),
            _ => return Err("malformed tokenizer.json: `added_tokens` is not an array".into()),
        };
        Ok(Self {
            truncation: truncation_from_json(&take("truncation").to_string())?,
            padding: padding_from_json(&take("padding").to_string())?,
            role_to_token: Arc::new(take("role_to_token")),
            added_tokens: Arc::new(added_tokens),
            normalizer: component(Normalizer::ITEMS_KEY, take("normalizer")),
            pre_tokenizer: component(PreTokenizer::ITEMS_KEY, take("pre_tokenizer")),
            post_processor: component(PostProcessor::ITEMS_KEY, take("post_processor")),
            decoder: component(Decoder::ITEMS_KEY, take("decoder")),
            model: Some(
                component(Model::ITEMS_KEY, take("model"))
                    .expect("canonicalize_value refuses a model that is not an object"),
            ),
        })
    }

    /// A tokenizer made of `model` alone.
    pub(crate) fn from_model(model: ComponentSpec) -> Self {
        Self {
            truncation: None,
            padding: None,
            role_to_token: Arc::new(Value::Null),
            added_tokens: Arc::new(AddedTokensSpec::new(Vec::new())),
            normalizer: None,
            pre_tokenizer: None,
            post_processor: None,
            decoder: None,
            model: Some(model),
        }
    }

    /// Builds a new [`PipelineTokenizer`] from the current spec, padding and truncation included.
    pub(crate) fn compile(&self) -> Result<PipelineTokenizer> {
        from_json(&self.to_json())
    }

    /// Adds `tokens`, or updates the flags of those with the same content, and returns how many
    /// it added or updated.
    fn add_tokens(&mut self, tokens: impl IntoIterator<Item = AddedToken>) -> usize {
        Arc::make_mut(&mut self.added_tokens).add(tokens)
    }

    /// A deep copy: the resulting object is independent.
    fn deep_clone(&self) -> Self {
        let deep_clone =
            |spec: &Option<ComponentSpec>| spec.as_ref().map(ComponentSpec::deep_clone);
        Self {
            truncation: self.truncation.clone(),
            padding: self.padding.clone(),
            role_to_token: self.role_to_token.clone(),
            added_tokens: self.added_tokens.clone(),
            normalizer: deep_clone(&self.normalizer),
            pre_tokenizer: deep_clone(&self.pre_tokenizer),
            post_processor: deep_clone(&self.post_processor),
            decoder: deep_clone(&self.decoder),
            model: deep_clone(&self.model),
        }
    }

    /// The top-level keys and their values, in the order `to_json` writes them.
    fn fields(&self) -> [(&'static str, Field<'_>); 10] {
        let component = |spec: &Option<ComponentSpec>| match spec {
            Some(spec) => Field::JsonObject(spec.json()),
            None => Field::JsonValue(Cow::Owned(Value::Null)),
        };
        [
            ("version", Field::JsonValue(Cow::Owned(Value::from("2.0")))),
            ("truncation", Field::Truncation(self.truncation.as_ref())),
            ("padding", Field::Padding(self.padding.as_ref())),
            (
                "role_to_token",
                Field::JsonValue(Cow::Borrowed(&self.role_to_token)),
            ),
            (
                "added_tokens",
                Field::JsonValue(Cow::Borrowed(self.added_tokens.json())),
            ),
            ("normalizer", component(&self.normalizer)),
            ("pre_tokenizer", component(&self.pre_tokenizer)),
            ("post_processor", component(&self.post_processor)),
            ("decoder", component(&self.decoder)),
            ("model", component(&self.model)),
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
    JsonValue(Cow<'spec, Value>),
    JsonObject(Arc<JsonObject>),
    Padding(Option<&'spec PaddingParams>),
    Truncation(Option<&'spec TruncationParams>),
}

impl Field<'_> {
    fn to_json(&self) -> String {
        match self {
            Field::JsonValue(value) => value.to_string(),
            Field::JsonObject(json) => serde_json::to_string(&**json)
                .expect("a JSON object with string keys always writes"),
            Field::Padding(padding) => padding_to_json(*padding),
            Field::Truncation(truncation) => truncation_to_json(*truncation),
        }
    }
}
