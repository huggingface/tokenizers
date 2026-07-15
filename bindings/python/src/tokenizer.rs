use serde::{ser::Error as SerError, Serialize, Serializer};
use std::collections::{hash_map::DefaultHasher, BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::str::FromStr;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use pyo3::class::basic::CompareOp;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::*;
use pyo3::{exceptions, IntoPyObject};
use tk::models::bpe::BPE;
use tk::tokenizer::pipeline::PipelineTokenizer;
use tk::tokenizer::TokenizerImpl;
use tokenizers as tk;

use super::error::ToPyResult;
use super::models::PyModel;
use super::normalizers::PyNormalizer;
use super::pre_tokenizers::PyPreTokenizer;

/// Represents a token that can be be added to a :class:`~tokenizers.Tokenizer`.
/// It can have special options that defines the way it should behave.
///
/// Args:
///     content (:obj:`str`): The content of the token
///
///     single_word (:obj:`bool`, defaults to :obj:`False`):
///         Defines whether this token should only match single words. If :obj:`True`, this
///         token will never match inside of a word. For example the token ``ing`` would match
///         on ``tokenizing`` if this option is :obj:`False`, but not if it is :obj:`True`.
///         The notion of "`inside of a word`" is defined by the word boundaries pattern in
///         regular expressions (ie. the token should start and end with word boundaries).
///
///     lstrip (:obj:`bool`, defaults to :obj:`False`):
///         Defines whether this token should strip all potential whitespaces on its left side.
///         If :obj:`True`, this token will greedily match any whitespace on its left. For
///         example if we try to match the token ``[MASK]`` with ``lstrip=True``, in the text
///         ``"I saw a [MASK]"``, we would match on ``" [MASK]"``. (Note the space on the left).
///
///     rstrip (:obj:`bool`, defaults to :obj:`False`):
///         Defines whether this token should strip all potential whitespaces on its right
///         side. If :obj:`True`, this token will greedily match any whitespace on its right.
///         It works just like :obj:`lstrip` but on the right.
///
///     normalized (:obj:`bool`, defaults to :obj:`True` with :meth:`~tokenizers.Tokenizer.add_tokens` and :obj:`False` with :meth:`~tokenizers.Tokenizer.add_special_tokens`):
///         Defines whether this token should match against the normalized version of the input
///         text. For example, with the added token ``"yesterday"``, and a normalizer in charge of
///         lowercasing the text, the token could be extract from the input ``"I saw a lion
///         Yesterday"``.
///     special (:obj:`bool`, defaults to :obj:`False` with :meth:`~tokenizers.Tokenizer.add_tokens` and :obj:`False` with :meth:`~tokenizers.Tokenizer.add_special_tokens`):
///         Defines whether this token should be skipped when decoding.
///
#[pyclass(dict, module = "tokenizers", name = "AddedToken")]
pub struct PyAddedToken {
    pub content: String,
    pub special: bool,
    pub single_word: Option<bool>,
    pub lstrip: Option<bool>,
    pub rstrip: Option<bool>,
    pub normalized: Option<bool>,
}
impl PyAddedToken {
    pub fn from<S: Into<String>>(content: S, special: Option<bool>) -> Self {
        Self {
            content: content.into(),
            special: special.unwrap_or(false),
            single_word: None,
            lstrip: None,
            rstrip: None,
            normalized: None,
        }
    }

    pub fn get_token(&self) -> tk::tokenizer::AddedToken {
        let mut token = tk::AddedToken::from(&self.content, self.special);

        if let Some(sw) = self.single_word {
            token = token.single_word(sw);
        }
        if let Some(ls) = self.lstrip {
            token = token.lstrip(ls);
        }
        if let Some(rs) = self.rstrip {
            token = token.rstrip(rs);
        }
        if let Some(n) = self.normalized {
            token = token.normalized(n);
        }

        token
    }

    pub fn as_pydict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let token = self.get_token();

        dict.set_item("content", token.content)?;
        dict.set_item("single_word", token.single_word)?;
        dict.set_item("lstrip", token.lstrip)?;
        dict.set_item("rstrip", token.rstrip)?;
        dict.set_item("normalized", token.normalized)?;
        dict.set_item("special", token.special)?;

        Ok(dict)
    }
}

impl From<tk::AddedToken> for PyAddedToken {
    fn from(token: tk::AddedToken) -> Self {
        Self {
            content: token.content,
            single_word: Some(token.single_word),
            lstrip: Some(token.lstrip),
            rstrip: Some(token.rstrip),
            normalized: Some(token.normalized),
            special: token.special,
        }
    }
}

#[pymethods]
impl PyAddedToken {
    #[new]
    #[pyo3(
        signature = (content=None, **kwargs),
        text_signature = "(self, content=None, single_word=False, lstrip=False, rstrip=False, normalized=True, special=False)"
    )]
    fn __new__(content: Option<&str>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut token = PyAddedToken::from(content.unwrap_or(""), None);

        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: String = key.extract()?;
                match key.as_ref() {
                    "single_word" => token.single_word = Some(value.extract()?),
                    "lstrip" => token.lstrip = Some(value.extract()?),
                    "rstrip" => token.rstrip = Some(value.extract()?),
                    "normalized" => token.normalized = Some(value.extract()?),
                    "special" => token.special = value.extract()?,
                    _ => println!("Ignored unknown kwarg option {key}"),
                }
            }
        }

        Ok(token)
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.as_pydict(py)
    }

    fn __setstate__(&mut self, py: Python, state: Py<PyAny>) -> PyResult<()> {
        match state.cast_bound::<PyDict>(py) {
            Ok(state) => {
                for (key, value) in state {
                    let key: String = key.extract()?;
                    match key.as_ref() {
                        "content" => self.content = value.extract()?,
                        "single_word" => self.single_word = Some(value.extract()?),
                        "lstrip" => self.lstrip = Some(value.extract()?),
                        "rstrip" => self.rstrip = Some(value.extract()?),
                        "normalized" => self.normalized = Some(value.extract()?),
                        "special" => self.special = value.extract()?,
                        _ => {}
                    }
                }
                Ok(())
            }
            Err(e) => Err(e.into()),
        }
    }

    /// Get the content of this :obj:`AddedToken`
    #[getter]
    fn get_content(&self) -> &str {
        &self.content
    }

    /// Set the content of this :obj:`AddedToken`
    #[setter]
    fn set_content(&mut self, content: String) {
        self.content = content;
    }

    /// Get the value of the :obj:`rstrip` option
    #[getter]
    fn get_rstrip(&self) -> bool {
        self.get_token().rstrip
    }

    /// Get the value of the :obj:`lstrip` option
    #[getter]
    fn get_lstrip(&self) -> bool {
        self.get_token().lstrip
    }

    /// Get the value of the :obj:`single_word` option
    #[getter]
    fn get_single_word(&self) -> bool {
        self.get_token().single_word
    }

    /// Get the value of the :obj:`normalized` option
    #[getter]
    fn get_normalized(&self) -> bool {
        self.get_token().normalized
    }
    /// Get the value of the :obj:`special` option
    #[getter]
    fn get_special(&self) -> bool {
        self.get_token().special
    }

    /// Set the value of the :obj:`special` option
    #[setter]
    fn set_special(&mut self, special: bool) {
        self.special = special;
    }

    fn __str__(&self) -> PyResult<&str> {
        Ok(&self.content)
    }

    fn __repr__(&self) -> PyResult<String> {
        let bool_to_python = |p| match p {
            true => "True",
            false => "False",
        };

        let token = self.get_token();
        Ok(format!(
            "AddedToken(\"{}\", rstrip={}, lstrip={}, single_word={}, normalized={}, special={})",
            self.content,
            bool_to_python(token.rstrip),
            bool_to_python(token.lstrip),
            bool_to_python(token.single_word),
            bool_to_python(token.normalized),
            bool_to_python(token.special)
        ))
    }

    fn __richcmp__(&self, other: Py<PyAddedToken>, op: CompareOp) -> bool {
        use CompareOp::*;
        Python::attach(|py| match op {
            Lt | Le | Gt | Ge => false,
            Eq => self.get_token() == other.borrow(py).get_token(),
            Ne => self.get_token() != other.borrow(py).get_token(),
        })
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.get_token().hash(&mut hasher);
        hasher.finish()
    }
}

/// The inner tokenizer: pipeline-only, so it carries just the model, normalizer and
/// pre-tokenizer config (no post-processor / decoder / padding / truncation any more).
type Tokenizer = TokenizerImpl<PyModel, PyNormalizer, PyPreTokenizer>;

/// A :obj:`Tokenizer` works as a pipeline. It processes some raw text as input
/// and outputs a list of token ids.
///
/// The pipeline is: normalize -> pre-tokenize -> model (subword -> id). This build is
/// **inference-only** (the fast `PipelineTokenizer` engine); decoding, offsets, padding,
/// truncation and post-processing are not available.
///
/// Args:
///     model (:class:`~tokenizers.models.Model`):
///         The core algorithm that this :obj:`Tokenizer` should be using.
#[pyclass(dict, weakref, module = "tokenizers", name = "Tokenizer", from_py_object)]
pub struct PyTokenizer {
    /// `Arc` so cloning is a refcount bump; `RwLock` so concurrent setters and
    /// encoders don't race PyO3's per-pyclass borrow check on free-threaded Python.
    pub(crate) tokenizer: Arc<RwLock<Tokenizer>>,
}

impl Clone for PyTokenizer {
    fn clone(&self) -> Self {
        PyTokenizer {
            tokenizer: Arc::clone(&self.tokenizer),
        }
    }
}

impl Serialize for PyTokenizer {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let guard = self
            .tokenizer
            .read()
            .map_err(|_| S::Error::custom("Tokenizer RwLock is poisoned"))?;
        guard.serialize(serializer)
    }
}

impl PyTokenizer {
    fn new(tokenizer: Tokenizer) -> Self {
        PyTokenizer {
            tokenizer: Arc::new(RwLock::new(tokenizer)),
        }
    }

    pub(crate) fn read_inner(&self) -> PyResult<RwLockReadGuard<'_, Tokenizer>> {
        self.tokenizer
            .read()
            .map_err(|_| exceptions::PyException::new_err("Tokenizer RwLock is poisoned"))
    }

    pub(crate) fn write_inner(&self) -> PyResult<RwLockWriteGuard<'_, Tokenizer>> {
        self.tokenizer
            .write()
            .map_err(|_| exceptions::PyException::new_err("Tokenizer RwLock is poisoned"))
    }

    fn from_model(model: PyModel) -> Self {
        PyTokenizer::new(TokenizerImpl::new(model))
    }

    /// Build a fast `PipelineTokenizer` for the current config.
    ///
    /// The Python tokenizer carries `Py*` component wrappers; the pipeline needs the concrete
    /// `tk_encode` wrappers, so we round-trip through the (identical) tokenizer.json serde
    /// representation. ponytail: rebuilt per encode call — correct but not cached; add a
    /// mutation-invalidated cache if encode throughput matters.
    fn build_pipeline(&self) -> PyResult<PipelineTokenizer> {
        let json = {
            let guard = self.read_inner()?;
            serde_json::to_string(&*guard).map_err(|e| {
                exceptions::PyException::new_err(format!(
                    "Cannot serialize tokenizer for the pipeline: {e}"
                ))
            })?
        };
        let concrete = tk::Tokenizer::from_str(&json).map_err(|e| {
            exceptions::PyException::new_err(format!("Cannot rebuild tokenizer: {e}"))
        })?;
        PipelineTokenizer::try_from(&concrete).map_err(|e| {
            exceptions::PyException::new_err(format!(
                "This tokenizer is not supported by the pipeline encode path: {e}"
            ))
        })
    }
}

#[pymethods]
impl PyTokenizer {
    #[new]
    #[pyo3(text_signature = "(self, model)")]
    fn __new__(model: PyRef<PyModel>) -> Self {
        PyTokenizer::from_model(model.clone())
    }

    fn __getstate__(&self, py: Python) -> PyResult<Py<PyAny>> {
        let data = serde_json::to_string(&*self.read_inner()?).map_err(|e| {
            exceptions::PyException::new_err(format!(
                "Error while attempting to pickle Tokenizer: {e}"
            ))
        })?;
        Ok(PyBytes::new(py, data.as_bytes()).into())
    }

    fn __setstate__(&self, py: Python, state: Py<PyAny>) -> PyResult<()> {
        match state.extract::<&[u8]>(py) {
            Ok(s) => {
                *self.write_inner()? = serde_json::from_slice(s).map_err(|e| {
                    exceptions::PyException::new_err(format!(
                        "Error while attempting to unpickle Tokenizer: {e}"
                    ))
                })?;
                Ok(())
            }
            Err(e) => Err(e.into()),
        }
    }

    fn __getnewargs__<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyTuple>> {
        let model: Py<PyAny> = PyModel::from(BPE::default())
            .into_pyobject(py)?
            .into_any()
            .into();
        PyTuple::new(py, vec![model])
    }

    /// Instantiate a new :class:`~tokenizers.Tokenizer` from the given JSON string.
    #[staticmethod]
    #[pyo3(signature = (json) -> "Tokenizer")]
    #[pyo3(text_signature = "(json)")]
    fn from_str(json: &str) -> PyResult<Self> {
        let tokenizer: PyResult<_> = ToPyResult(json.parse()).into();
        Ok(Self::new(tokenizer?))
    }

    /// Instantiate a new :class:`~tokenizers.Tokenizer` from the file at the given path.
    #[staticmethod]
    #[pyo3(signature = (path) -> "Tokenizer")]
    #[pyo3(text_signature = "(path)")]
    fn from_file(path: &str) -> PyResult<Self> {
        let tokenizer: PyResult<_> = ToPyResult(Tokenizer::from_file(path)).into();
        Ok(Self::new(tokenizer?))
    }

    /// Instantiate a new :class:`~tokenizers.Tokenizer` from the given buffer.
    #[staticmethod]
    #[pyo3(signature = (buffer) -> "Tokenizer")]
    #[pyo3(text_signature = "(buffer)")]
    fn from_buffer(buffer: &Bound<'_, PyBytes>) -> PyResult<Self> {
        let tokenizer: Tokenizer = serde_json::from_slice(buffer.as_bytes()).map_err(|e| {
            exceptions::PyValueError::new_err(format!(
                "Cannot instantiate Tokenizer from buffer: {e}"
            ))
        })?;
        Ok(Self::new(tokenizer))
    }

    /// Instantiate a new :class:`~tokenizers.Tokenizer` from an existing file on the
    /// Hugging Face Hub.
    #[staticmethod]
    #[pyo3(signature = (identifier, revision = String::from("main"), token = None) -> "Tokenizer")]
    #[pyo3(text_signature = "(identifier, revision=\"main\", token=None)")]
    fn from_pretrained(
        identifier: &str,
        revision: String,
        token: Option<String>,
    ) -> PyResult<Self> {
        let path = Python::attach(|py| -> PyResult<String> {
            let huggingface_hub = PyModule::import(py, intern!(py, "huggingface_hub"))?;
            let hf_hub_download = huggingface_hub.getattr(intern!(py, "hf_hub_download"))?;
            let kwargs = [
                (intern!(py, "repo_id"), identifier),
                (intern!(py, "filename"), "tokenizer.json"),
                (intern!(py, "revision"), &revision),
            ]
            .into_py_dict(py)?;
            if let Some(token) = token {
                kwargs.set_item(intern!(py, "token"), token)?;
            }
            let path: String = hf_hub_download.call((), Some(&kwargs))?.extract()?;
            Ok(path)
        })?;

        let tokenizer: PyResult<_> = ToPyResult(Tokenizer::from_file(path)).into();
        Ok(Self::new(tokenizer?))
    }

    /// Gets a serialized string representing this :class:`~tokenizers.Tokenizer`.
    #[pyo3(signature = (pretty = false) -> "str")]
    #[pyo3(text_signature = "(self, pretty=False)")]
    fn to_str(&self, pretty: bool) -> PyResult<String> {
        ToPyResult(self.read_inner()?.to_string(pretty)).into()
    }

    /// Save the :class:`~tokenizers.Tokenizer` to the file at the given path.
    #[pyo3(signature = (path, pretty = true) -> "None")]
    #[pyo3(text_signature = "(self, path, pretty=True)")]
    fn save(&self, path: &str, pretty: bool) -> PyResult<()> {
        ToPyResult(self.read_inner()?.save(path, pretty)).into()
    }

    fn __repr__(&self) -> PyResult<String> {
        crate::utils::serde_pyo3::repr(self)
            .map_err(|e| exceptions::PyException::new_err(e.to_string()))
    }

    fn __str__(&self) -> PyResult<String> {
        crate::utils::serde_pyo3::to_string(self)
            .map_err(|e| exceptions::PyException::new_err(e.to_string()))
    }

    /// Get the underlying vocabulary
    #[pyo3(signature = (with_added_tokens = true) -> "dict[str, int]")]
    #[pyo3(text_signature = "(self, with_added_tokens=True)")]
    fn get_vocab(&self, with_added_tokens: bool) -> PyResult<HashMap<String, u32>> {
        Ok(self.read_inner()?.get_vocab(with_added_tokens))
    }

    /// Get the underlying added-token map, keyed by id.
    #[pyo3(signature = () -> "dict[int, AddedToken]")]
    #[pyo3(text_signature = "(self)")]
    fn get_added_tokens_decoder(&self) -> PyResult<BTreeMap<u32, PyAddedToken>> {
        let mut sorted_map = BTreeMap::new();
        for (key, value) in self.read_inner()?.get_added_tokens_decoder() {
            sorted_map.insert(key, value.into());
        }
        Ok(sorted_map)
    }

    /// Get the size of the underlying vocabulary
    #[pyo3(signature = (with_added_tokens = true) -> "int")]
    #[pyo3(text_signature = "(self, with_added_tokens=True)")]
    fn get_vocab_size(&self, with_added_tokens: bool) -> PyResult<usize> {
        Ok(self.read_inner()?.get_vocab_size(with_added_tokens))
    }

    /// Encode the given sequence into token ids.
    ///
    /// Args:
    ///     sequence (:obj:`str`): The sequence to encode.
    ///     add_special_tokens (:obj:`bool`, defaults to :obj:`True`): Whether to add the special
    ///         tokens.
    ///
    /// Returns:
    ///     :obj:`list[int]`: The token ids.
    #[pyo3(signature = (sequence, add_special_tokens = true) -> "list[int]")]
    #[pyo3(text_signature = "(self, sequence, add_special_tokens=True)")]
    fn encode(&self, sequence: &str, add_special_tokens: bool) -> PyResult<Vec<u32>> {
        let pipeline = self.build_pipeline()?;
        let tokens = pipeline
            .encode(sequence, add_special_tokens)
            .map_err(|e| exceptions::PyException::new_err(e.to_string()))?;
        Ok(tokens.into_iter().map(|t| t.id).collect())
    }

    /// Encode a batch of sequences into token ids.
    #[pyo3(signature = (input, add_special_tokens = true) -> "list[list[int]]")]
    #[pyo3(text_signature = "(self, input, add_special_tokens=True)")]
    fn encode_batch(
        &self,
        py: Python<'_>,
        input: Vec<String>,
        add_special_tokens: bool,
    ) -> PyResult<Vec<Vec<u32>>> {
        let pipeline = self.build_pipeline()?;
        py.detach(|| {
            input
                .iter()
                .map(|sequence| {
                    pipeline
                        .encode(sequence, add_special_tokens)
                        .map(|tokens| tokens.into_iter().map(|t| t.id).collect())
                        .map_err(|e| exceptions::PyException::new_err(e.to_string()))
                })
                .collect()
        })
    }

    /// Convert the given token to its corresponding id if it exists
    #[pyo3(signature = (token) -> "int | None", text_signature = "(self, token)")]
    fn token_to_id(&self, token: &str) -> PyResult<Option<u32>> {
        Ok(self.read_inner()?.token_to_id(token))
    }

    /// Convert the given id to its corresponding token if it exists
    #[pyo3(signature = (id) -> "str | None", text_signature = "(self, id)")]
    fn id_to_token(&self, id: u32) -> PyResult<Option<String>> {
        Ok(self.read_inner()?.id_to_token(id))
    }

    /// Modifies the tokenizer in order to use or not the special tokens during encoding.
    #[setter]
    fn set_encode_special_tokens(&self, value: bool) -> PyResult<()> {
        self.write_inner()?.set_encode_special_tokens(value);
        Ok(())
    }

    /// Get the value of the `encode_special_tokens` attribute
    #[getter]
    fn get_encode_special_tokens(&self) -> PyResult<bool> {
        Ok(self.read_inner()?.get_encode_special_tokens())
    }

    /// Add the given tokens to the vocabulary
    #[pyo3(text_signature = "(self, tokens)")]
    fn add_tokens(&self, tokens: &Bound<'_, PyList>) -> PyResult<usize> {
        let tokens = tokens
            .into_iter()
            .map(|token| {
                if let Ok(content) = token.extract::<String>() {
                    Ok(PyAddedToken::from(content, Some(false)).get_token())
                } else if let Ok(token) = token.extract::<PyRefMut<PyAddedToken>>() {
                    Ok(token.get_token())
                } else {
                    Err(exceptions::PyTypeError::new_err(
                        "Input must be a List[Union[str, AddedToken]]",
                    ))
                }
            })
            .collect::<PyResult<Vec<_>>>()?;

        ToPyResult(self.write_inner()?.add_tokens(tokens)).into()
    }

    /// Add the given special tokens to the Tokenizer.
    #[pyo3(text_signature = "(self, tokens)")]
    fn add_special_tokens(&self, tokens: &Bound<'_, PyList>) -> PyResult<usize> {
        let tokens = tokens
            .into_iter()
            .map(|token| {
                if let Ok(content) = token.extract::<String>() {
                    Ok(tk::tokenizer::AddedToken::from(content, true))
                } else if let Ok(mut token) = token.extract::<PyRefMut<PyAddedToken>>() {
                    token.special = true;
                    Ok(token.get_token())
                } else {
                    Err(exceptions::PyTypeError::new_err(
                        "Input must be a List[Union[str, AddedToken]]",
                    ))
                }
            })
            .collect::<PyResult<Vec<_>>>()?;

        ToPyResult(self.write_inner()?.add_special_tokens(tokens)).into()
    }

    /// The :class:`~tokenizers.models.Model` in use by the Tokenizer
    #[getter]
    fn get_model(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.read_inner()?.get_model().get_as_subtype(py)
    }

    /// Set the :class:`~tokenizers.models.Model`
    #[setter]
    fn set_model(&self, model: PyRef<PyModel>) -> PyResult<()> {
        self.write_inner()?.with_model(model.clone());
        Ok(())
    }

    /// The `optional` :class:`~tokenizers.normalizers.Normalizer` in use by the Tokenizer
    #[getter]
    fn get_normalizer(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if let Some(n) = self.read_inner()?.get_normalizer() {
            n.get_as_subtype(py)
        } else {
            Ok(py.None())
        }
    }

    /// Set the :class:`~tokenizers.normalizers.Normalizer`
    #[setter]
    fn set_normalizer(&self, normalizer: Option<PyRef<PyNormalizer>>) -> PyResult<()> {
        let normalizer_option = normalizer.map(|norm| norm.clone());
        ToPyResult(
            self.write_inner()?
                .with_normalizer(normalizer_option)
                .map(|_| ()),
        )
        .into()
    }

    /// The `optional` :class:`~tokenizers.pre_tokenizers.PreTokenizer` in use by the Tokenizer
    #[getter]
    fn get_pre_tokenizer(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if let Some(pt) = self.read_inner()?.get_pre_tokenizer() {
            pt.get_as_subtype(py)
        } else {
            Ok(py.None())
        }
    }

    /// Set the :class:`~tokenizers.pre_tokenizers.PreTokenizer`
    #[setter]
    fn set_pre_tokenizer(&self, pretok: Option<PyRef<PyPreTokenizer>>) -> PyResult<()> {
        self.write_inner()?
            .with_pre_tokenizer(pretok.map(|pre| pre.clone()));
        Ok(())
    }
}
