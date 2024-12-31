use serde::Serialize;
use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::{Hash, Hasher};

use numpy::{npyffi, PyArray1, PyArrayMethods};
use pyo3::class::basic::CompareOp;
use pyo3::exceptions;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::*;
use tk::models::bpe::BPE;
use tk::tokenizer::{
    Model, PaddingDirection, PaddingParams, PaddingStrategy, PostProcessor, TokenizerImpl,
    TruncationDirection, TruncationParams, TruncationStrategy,
};
use tk::utils::iter::ResultShunt;
use tokenizers as tk;

use super::decoders::PyDecoder;
use super::encoding::PyEncoding;
use super::error::{PyError, ToPyResult};
use super::models::PyModel;
use super::normalizers::PyNormalizer;
use super::pre_tokenizers::PyPreTokenizer;
use super::trainers::PyTrainer;
use crate::processors::PyPostProcessor;
use crate::utils::{MaybeSizedIterator, PyBufferedIterator};
use std::collections::BTreeMap;

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
    #[pyo3(signature = (content=None, **kwargs), text_signature = "(self, content, single_word=False, lstrip=False, rstrip=False, normalized=True, special=False)")]
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
                    _ => println!("Ignored unknown kwarg option {}", key),
                }
            }
        }

        Ok(token)
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.as_pydict(py)
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.downcast_bound::<PyDict>(py) {
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
        Python::with_gil(|py| match op {
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

struct TextInputSequence<'s>(tk::InputSequence<'s>);
impl<'s> FromPyObject<'s> for TextInputSequence<'s> {
    fn extract_bound(ob: &Bound<'s, PyAny>) -> PyResult<Self> {
        let err = exceptions::PyTypeError::new_err("TextInputSequence must be str");
        if let Ok(s) = ob.extract::<String>() {
            Ok(Self(s.into()))
        } else {
            Err(err)
        }
    }
}
impl<'s> From<TextInputSequence<'s>> for tk::InputSequence<'s> {
    fn from(s: TextInputSequence<'s>) -> Self {
        s.0
    }
}

struct PyArrayUnicode(Vec<String>);
impl FromPyObject<'_> for PyArrayUnicode {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        // SAFETY Making sure the pointer is a valid numpy array requires calling numpy C code
        if unsafe { npyffi::PyArray_Check(ob.py(), ob.as_ptr()) } == 0 {
            return Err(exceptions::PyTypeError::new_err("Expected an np.array"));
        }
        let arr = ob.as_ptr() as *mut npyffi::PyArrayObject;
        // SAFETY Getting all the metadata about the numpy array to check its sanity
        let (type_num, elsize, _alignment, data, nd, flags) = unsafe {
            let desc = (*arr).descr;
            (
                (*desc).type_num,
                npyffi::PyDataType_ELSIZE(ob.py(), desc) as usize,
                npyffi::PyDataType_ALIGNMENT(ob.py(), desc) as usize,
                (*arr).data,
                (*arr).nd,
                (*arr).flags,
            )
        };

        if nd != 1 {
            return Err(exceptions::PyTypeError::new_err(
                "Expected a 1 dimensional np.array",
            ));
        }
        if flags & (npyffi::NPY_ARRAY_C_CONTIGUOUS | npyffi::NPY_ARRAY_F_CONTIGUOUS) == 0 {
            return Err(exceptions::PyTypeError::new_err(
                "Expected a contiguous np.array",
            ));
        }
        if type_num != npyffi::types::NPY_TYPES::NPY_UNICODE as i32 {
            return Err(exceptions::PyTypeError::new_err(
                "Expected a np.array[dtype='U']",
            ));
        }

        // SAFETY Looking at the raw numpy data to create new owned Rust strings via copies (so it's safe afterwards).
        unsafe {
            let n_elem = *(*arr).dimensions as usize;
            let all_bytes = std::slice::from_raw_parts(data as *const u8, elsize * n_elem);

            let seq = (0..n_elem)
                .map(|i| {
                    let bytes = &all_bytes[i * elsize..(i + 1) * elsize];
                    Ok(std::str::from_utf8(bytes)?.to_owned())
                    // let unicode = pyo3::ffi::PyUnicode_FromKindAndData(
                    //     pyo3::ffi::PyUnicode_4BYTE_KIND as _,
                    //     bytes.as_ptr() as *const _,
                    //     elsize as isize / alignment as isize,
                    // );
                    // let py = ob.py();
                    // let obj = PyObject::from_owned_ptr(py, unicode);
                    // let s = obj.downcast_bound::<PyString>(py)?;
                    // Ok(s.to_string_lossy().trim_matches(char::from(0)).to_owned())
                })
                .collect::<PyResult<Vec<_>>>()?;

            Ok(Self(seq))
        }
    }
}
impl From<PyArrayUnicode> for tk::InputSequence<'_> {
    fn from(s: PyArrayUnicode) -> Self {
        s.0.into()
    }
}

struct PyArrayStr(Vec<String>);

impl FromPyObject<'_> for PyArrayStr {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let array = ob.downcast::<PyArray1<PyObject>>()?;
        let seq = array
            .readonly()
            .as_array()
            .iter()
            .map(|obj| {
                let s = obj.downcast_bound::<PyString>(ob.py())?;
                Ok(s.to_string_lossy().into_owned())
            })
            .collect::<PyResult<Vec<_>>>()?;

        Ok(Self(seq))
    }
}
impl From<PyArrayStr> for tk::InputSequence<'_> {
    fn from(s: PyArrayStr) -> Self {
        s.0.into()
    }
}

struct PreTokenizedInputSequence<'s>(tk::InputSequence<'s>);
impl<'s> FromPyObject<'s> for PreTokenizedInputSequence<'s> {
    fn extract_bound(ob: &Bound<'s, PyAny>) -> PyResult<Self> {
        if let Ok(seq) = ob.extract::<PyArrayUnicode>() {
            return Ok(Self(seq.into()));
        }
        if let Ok(seq) = ob.extract::<PyArrayStr>() {
            return Ok(Self(seq.into()));
        }
        if let Ok(s) = ob.downcast::<PyList>() {
            if let Ok(seq) = s.extract::<Vec<String>>() {
                return Ok(Self(seq.into()));
            }
        }
        if let Ok(s) = ob.downcast::<PyTuple>() {
            if let Ok(seq) = s.extract::<Vec<String>>() {
                return Ok(Self(seq.into()));
            }
        }
        Err(exceptions::PyTypeError::new_err(
            "PreTokenizedInputSequence must be Union[List[str], Tuple[str]]",
        ))
    }
}
impl<'s> From<PreTokenizedInputSequence<'s>> for tk::InputSequence<'s> {
    fn from(s: PreTokenizedInputSequence<'s>) -> Self {
        s.0
    }
}

struct TextEncodeInput<'s>(tk::EncodeInput<'s>);
impl<'s> FromPyObject<'s> for TextEncodeInput<'s> {
    fn extract_bound(ob: &Bound<'s, PyAny>) -> PyResult<Self> {
        if let Ok(i) = ob.extract::<TextInputSequence>() {
            return Ok(Self(i.into()));
        }
        if let Ok((i1, i2)) = ob.extract::<(TextInputSequence, TextInputSequence)>() {
            return Ok(Self((i1, i2).into()));
        }
        if let Ok(arr) = ob.extract::<Vec<Bound<PyAny>>>() {
            if arr.len() == 2 {
                let first = arr[0].extract::<TextInputSequence>()?;
                let second = arr[1].extract::<TextInputSequence>()?;
                return Ok(Self((first, second).into()));
            }
        }
        Err(exceptions::PyTypeError::new_err(
            "TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]",
        ))
    }
}
impl<'s> From<TextEncodeInput<'s>> for tk::tokenizer::EncodeInput<'s> {
    fn from(i: TextEncodeInput<'s>) -> Self {
        i.0
    }
}
struct PreTokenizedEncodeInput<'s>(tk::EncodeInput<'s>);
impl<'s> FromPyObject<'s> for PreTokenizedEncodeInput<'s> {
    fn extract_bound(ob: &Bound<'s, PyAny>) -> PyResult<Self> {
        if let Ok(i) = ob.extract::<PreTokenizedInputSequence>() {
            return Ok(Self(i.into()));
        }
        if let Ok((i1, i2)) = ob.extract::<(PreTokenizedInputSequence, PreTokenizedInputSequence)>()
        {
            return Ok(Self((i1, i2).into()));
        }
        if let Ok(arr) = ob.extract::<Vec<Bound<PyAny>>>() {
            if arr.len() == 2 {
                let first = arr[0].extract::<PreTokenizedInputSequence>()?;
                let second = arr[1].extract::<PreTokenizedInputSequence>()?;
                return Ok(Self((first, second).into()));
            }
        }
        Err(exceptions::PyTypeError::new_err(
            "PreTokenizedEncodeInput must be Union[PreTokenizedInputSequence, \
            Tuple[PreTokenizedInputSequence, PreTokenizedInputSequence]]",
        ))
    }
}
impl<'s> From<PreTokenizedEncodeInput<'s>> for tk::tokenizer::EncodeInput<'s> {
    fn from(i: PreTokenizedEncodeInput<'s>) -> Self {
        i.0
    }
}

type Tokenizer = TokenizerImpl<PyModel, PyNormalizer, PyPreTokenizer, PyPostProcessor, PyDecoder>;

/// A :obj:`Tokenizer` works as a pipeline. It processes some raw text as input
/// and outputs an :class:`~tokenizers.Encoding`.
///
/// Args:
///     model (:class:`~tokenizers.models.Model`):
///         The core algorithm that this :obj:`Tokenizer` should be using.
///
#[pyclass(dict, module = "tokenizers", name = "Tokenizer")]
#[derive(Clone, Serialize)]
#[serde(transparent)]
pub struct PyTokenizer {
    pub(crate) tokenizer: Tokenizer,
}

impl PyTokenizer {
    fn new(tokenizer: Tokenizer) -> Self {
        PyTokenizer { tokenizer }
    }

    fn from_model(model: PyModel) -> Self {
        PyTokenizer::new(TokenizerImpl::new(model))
    }
}

#[pymethods]
impl PyTokenizer {
    #[new]
    #[pyo3(text_signature = "(self, model)")]
    fn __new__(model: PyRef<PyModel>) -> Self {
        PyTokenizer::from_model(model.clone())
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let data = serde_json::to_string(&self.tokenizer).map_err(|e| {
            exceptions::PyException::new_err(format!(
                "Error while attempting to pickle Tokenizer: {}",
                e
            ))
        })?;
        Ok(PyBytes::new(py, data.as_bytes()).into())
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&[u8]>(py) {
            Ok(s) => {
                self.tokenizer = serde_json::from_slice(s).map_err(|e| {
                    exceptions::PyException::new_err(format!(
                        "Error while attempting to unpickle Tokenizer: {}",
                        e
                    ))
                })?;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn __getnewargs__<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyTuple>> {
        let model: PyObject = PyModel::from(BPE::default())
            .into_pyobject(py)?
            .into_any()
            .into();
        PyTuple::new(py, vec![model])
    }

    /// Instantiate a new :class:`~tokenizers.Tokenizer` from the given JSON string.
    ///
    /// Args:
    ///     json (:obj:`str`):
    ///         A valid JSON string representing a previously serialized
    ///         :class:`~tokenizers.Tokenizer`
    ///
    /// Returns:
    ///     :class:`~tokenizers.Tokenizer`: The new tokenizer
    #[staticmethod]
    #[pyo3(text_signature = "(json)")]
    fn from_str(json: &str) -> PyResult<Self> {
        let tokenizer: PyResult<_> = ToPyResult(json.parse()).into();
        Ok(Self::new(tokenizer?))
    }

    /// Instantiate a new :class:`~tokenizers.Tokenizer` from the file at the given path.
    ///
    /// Args:
    ///     path (:obj:`str`):
    ///         A path to a local JSON file representing a previously serialized
    ///         :class:`~tokenizers.Tokenizer`
    ///
    /// Returns:
    ///     :class:`~tokenizers.Tokenizer`: The new tokenizer
    #[staticmethod]
    #[pyo3(text_signature = "(path)")]
    fn from_file(path: &str) -> PyResult<Self> {
        let tokenizer: PyResult<_> = ToPyResult(Tokenizer::from_file(path)).into();
        Ok(Self::new(tokenizer?))
    }

    /// Instantiate a new :class:`~tokenizers.Tokenizer` from the given buffer.
    ///
    /// Args:
    ///     buffer (:obj:`bytes`):
    ///         A buffer containing a previously serialized :class:`~tokenizers.Tokenizer`
    ///
    /// Returns:
    ///     :class:`~tokenizers.Tokenizer`: The new tokenizer
    #[staticmethod]
    #[pyo3(text_signature = "(buffer)")]
    fn from_buffer(buffer: &Bound<'_, PyBytes>) -> PyResult<Self> {
        let tokenizer = serde_json::from_slice(buffer.as_bytes()).map_err(|e| {
            exceptions::PyValueError::new_err(format!(
                "Cannot instantiate Tokenizer from buffer: {}",
                e
            ))
        })?;
        Ok(Self { tokenizer })
    }

    /// Instantiate a new :class:`~tokenizers.Tokenizer` from an existing file on the
    /// Hugging Face Hub.
    ///
    /// Args:
    ///     identifier (:obj:`str`):
    ///         The identifier of a Model on the Hugging Face Hub, that contains
    ///         a tokenizer.json file
    ///     revision (:obj:`str`, defaults to `main`):
    ///         A branch or commit id
    ///     token (:obj:`str`, `optional`, defaults to `None`):
    ///         An optional auth token used to access private repositories on the
    ///         Hugging Face Hub
    ///
    /// Returns:
    ///     :class:`~tokenizers.Tokenizer`: The new tokenizer
    #[staticmethod]
    #[pyo3(signature = (identifier, revision = String::from("main"), token = None))]
    #[pyo3(text_signature = "(identifier, revision=\"main\", token=None)")]
    fn from_pretrained(
        identifier: &str,
        revision: String,
        token: Option<String>,
    ) -> PyResult<Self> {
        let path = Python::with_gil(|py| -> PyResult<String> {
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
    ///
    /// Args:
    ///     pretty (:obj:`bool`, defaults to :obj:`False`):
    ///         Whether the JSON string should be pretty formatted.
    ///
    /// Returns:
    ///     :obj:`str`: A string representing the serialized Tokenizer
    #[pyo3(signature = (pretty = false))]
    #[pyo3(text_signature = "(self, pretty=False)")]
    fn to_str(&self, pretty: bool) -> PyResult<String> {
        ToPyResult(self.tokenizer.to_string(pretty)).into()
    }

    /// Save the :class:`~tokenizers.Tokenizer` to the file at the given path.
    ///
    /// Args:
    ///     path (:obj:`str`):
    ///         A path to a file in which to save the serialized tokenizer.
    ///
    ///     pretty (:obj:`bool`, defaults to :obj:`True`):
    ///         Whether the JSON file should be pretty formatted.
    #[pyo3(signature = (path, pretty = true))]
    #[pyo3(text_signature = "(self, path, pretty=True)")]
    fn save(&self, path: &str, pretty: bool) -> PyResult<()> {
        ToPyResult(self.tokenizer.save(path, pretty)).into()
    }

    fn __repr__(&self) -> PyResult<String> {
        crate::utils::serde_pyo3::repr(self)
            .map_err(|e| exceptions::PyException::new_err(e.to_string()))
    }

    fn __str__(&self) -> PyResult<String> {
        crate::utils::serde_pyo3::to_string(self)
            .map_err(|e| exceptions::PyException::new_err(e.to_string()))
    }

    /// Return the number of special tokens that would be added for single/pair sentences.
    /// :param is_pair: Boolean indicating if the input would be a single sentence or a pair
    /// :return:
    #[pyo3(text_signature = "(self, is_pair)")]
    fn num_special_tokens_to_add(&self, is_pair: bool) -> usize {
        self.tokenizer
            .get_post_processor()
            .map_or(0, |p| p.added_tokens(is_pair))
    }

    /// Get the underlying vocabulary
    ///
    /// Args:
    ///     with_added_tokens (:obj:`bool`, defaults to :obj:`True`):
    ///         Whether to include the added tokens
    ///
    /// Returns:
    ///     :obj:`Dict[str, int]`: The vocabulary
    #[pyo3(signature = (with_added_tokens = true))]
    #[pyo3(text_signature = "(self, with_added_tokens=True)")]
    fn get_vocab(&self, with_added_tokens: bool) -> HashMap<String, u32> {
        self.tokenizer.get_vocab(with_added_tokens)
    }

    /// Get the underlying vocabulary
    ///
    /// Returns:
    ///     :obj:`Dict[int, AddedToken]`: The vocabulary
    #[pyo3(signature = ())]
    #[pyo3(text_signature = "(self)")]
    fn get_added_tokens_decoder(&self) -> BTreeMap<u32, PyAddedToken> {
        let mut sorted_map = BTreeMap::new();

        for (key, value) in self.tokenizer.get_added_tokens_decoder() {
            sorted_map.insert(key, value.into());
        }

        sorted_map
    }

    /// Get the size of the underlying vocabulary
    ///
    /// Args:
    ///     with_added_tokens (:obj:`bool`, defaults to :obj:`True`):
    ///         Whether to include the added tokens
    ///
    /// Returns:
    ///     :obj:`int`: The size of the vocabulary
    #[pyo3(signature = (with_added_tokens = true))]
    #[pyo3(text_signature = "(self, with_added_tokens=True)")]
    fn get_vocab_size(&self, with_added_tokens: bool) -> usize {
        self.tokenizer.get_vocab_size(with_added_tokens)
    }

    /// Enable truncation
    ///
    /// Args:
    ///     max_length (:obj:`int`):
    ///         The max length at which to truncate
    ///
    ///     stride (:obj:`int`, `optional`):
    ///         The length of the previous first sequence to be included in the overflowing
    ///         sequence
    ///
    ///     strategy (:obj:`str`, `optional`, defaults to :obj:`longest_first`):
    ///         The strategy used to truncation. Can be one of ``longest_first``, ``only_first`` or
    ///         ``only_second``.
    ///
    ///     direction (:obj:`str`, defaults to :obj:`right`):
    ///         Truncate direction
    #[pyo3(signature = (max_length, **kwargs))]
    #[pyo3(
        text_signature = "(self, max_length, stride=0, strategy='longest_first', direction='right')"
    )]
    fn enable_truncation(
        &mut self,
        max_length: usize,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let mut params = TruncationParams {
            max_length,
            ..Default::default()
        };

        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: String = key.extract()?;
                match key.as_ref() {
                    "stride" => params.stride = value.extract()?,
                    "strategy" => {
                        let value: String = value.extract()?;
                        params.strategy = match value.as_ref() {
                            "longest_first" => Ok(TruncationStrategy::LongestFirst),
                            "only_first" => Ok(TruncationStrategy::OnlyFirst),
                            "only_second" => Ok(TruncationStrategy::OnlySecond),
                            _ => Err(PyError(format!(
                                "Unknown `strategy`: `{}`. Use \
                                 one of `longest_first`, `only_first`, or `only_second`",
                                value
                            ))
                            .into_pyerr::<exceptions::PyValueError>()),
                        }?
                    }
                    "direction" => {
                        let value: String = value.extract()?;
                        params.direction = match value.as_ref() {
                            "left" => Ok(TruncationDirection::Left),
                            "right" => Ok(TruncationDirection::Right),
                            _ => Err(PyError(format!(
                                "Unknown `direction`: `{}`. Use \
                                 one of `left` or `right`.",
                                value
                            ))
                            .into_pyerr::<exceptions::PyValueError>()),
                        }?
                    }
                    _ => println!("Ignored unknown kwarg option {}", key),
                }
            }
        }

        if let Err(error_message) = self.tokenizer.with_truncation(Some(params)) {
            return Err(PyError(error_message.to_string()).into_pyerr::<exceptions::PyValueError>());
        }
        Ok(())
    }

    /// Disable truncation
    #[pyo3(text_signature = "(self)")]
    fn no_truncation(&mut self) {
        self.tokenizer
            .with_truncation(None)
            .expect("Failed to set truncation to `None`! This should never happen");
    }

    /// Get the currently set truncation parameters
    ///
    /// `Cannot set, use` :meth:`~tokenizers.Tokenizer.enable_truncation` `instead`
    ///
    /// Returns:
    ///     (:obj:`dict`, `optional`):
    ///         A dict with the current truncation parameters if truncation is enabled
    #[getter]
    fn get_truncation<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.tokenizer.get_truncation().map_or(Ok(None), |params| {
            let dict = PyDict::new(py);

            dict.set_item("max_length", params.max_length)?;
            dict.set_item("stride", params.stride)?;
            dict.set_item("strategy", params.strategy.as_ref())?;
            dict.set_item("direction", params.direction.as_ref())?;

            Ok(Some(dict))
        })
    }

    /// Enable the padding
    ///
    /// Args:
    ///     direction (:obj:`str`, `optional`, defaults to :obj:`right`):
    ///         The direction in which to pad. Can be either ``right`` or ``left``
    ///
    ///     pad_to_multiple_of (:obj:`int`, `optional`):
    ///         If specified, the padding length should always snap to the next multiple of the
    ///         given value. For example if we were going to pad witha length of 250 but
    ///         ``pad_to_multiple_of=8`` then we will pad to 256.
    ///
    ///     pad_id (:obj:`int`, defaults to 0):
    ///         The id to be used when padding
    ///
    ///     pad_type_id (:obj:`int`, defaults to 0):
    ///         The type id to be used when padding
    ///
    ///     pad_token (:obj:`str`, defaults to :obj:`[PAD]`):
    ///         The pad token to be used when padding
    ///
    ///     length (:obj:`int`, `optional`):
    ///         If specified, the length at which to pad. If not specified we pad using the size of
    ///         the longest sequence in a batch.
    #[pyo3(signature = (**kwargs))]
    #[pyo3(
        text_signature = "(self, direction='right', pad_id=0, pad_type_id=0, pad_token='[PAD]', length=None, pad_to_multiple_of=None)"
    )]
    fn enable_padding(&mut self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let mut params = PaddingParams::default();

        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: String = key.extract()?;
                match key.as_ref() {
                    "direction" => {
                        let value: String = value.extract()?;
                        params.direction = match value.as_ref() {
                            "left" => Ok(PaddingDirection::Left),
                            "right" => Ok(PaddingDirection::Right),
                            other => Err(PyError(format!(
                                "Unknown `direction`: `{}`. Use \
                                 one of `left` or `right`",
                                other
                            ))
                            .into_pyerr::<exceptions::PyValueError>()),
                        }?;
                    }
                    "pad_to_multiple_of" => {
                        if let Some(multiple) = value.extract()? {
                            params.pad_to_multiple_of = multiple;
                        }
                    }
                    "pad_id" => params.pad_id = value.extract()?,
                    "pad_type_id" => params.pad_type_id = value.extract()?,
                    "pad_token" => params.pad_token = value.extract()?,
                    "max_length" => {
                        println!(
                            "enable_padding(max_length=X) is deprecated, \
                                 use enable_padding(length=X) instead"
                        );
                        if let Some(l) = value.extract()? {
                            params.strategy = PaddingStrategy::Fixed(l);
                        } else {
                            params.strategy = PaddingStrategy::BatchLongest;
                        }
                    }
                    "length" => {
                        if let Some(l) = value.extract()? {
                            params.strategy = PaddingStrategy::Fixed(l);
                        } else {
                            params.strategy = PaddingStrategy::BatchLongest;
                        }
                    }
                    _ => println!("Ignored unknown kwarg option {}", key),
                }
            }
        }

        self.tokenizer.with_padding(Some(params));

        Ok(())
    }

    /// Disable padding
    #[pyo3(text_signature = "(self)")]
    fn no_padding(&mut self) {
        self.tokenizer.with_padding(None);
    }

    /// Get the current padding parameters
    ///
    /// `Cannot be set, use` :meth:`~tokenizers.Tokenizer.enable_padding` `instead`
    ///
    /// Returns:
    ///     (:obj:`dict`, `optional`):
    ///         A dict with the current padding parameters if padding is enabled
    #[getter]
    fn get_padding<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.tokenizer.get_padding().map_or(Ok(None), |params| {
            let dict = PyDict::new(py);

            dict.set_item(
                "length",
                match params.strategy {
                    tk::PaddingStrategy::BatchLongest => None,
                    tk::PaddingStrategy::Fixed(size) => Some(size),
                },
            )?;
            dict.set_item("pad_to_multiple_of", params.pad_to_multiple_of)?;
            dict.set_item("pad_id", params.pad_id)?;
            dict.set_item("pad_token", &params.pad_token)?;
            dict.set_item("pad_type_id", params.pad_type_id)?;
            dict.set_item("direction", params.direction.as_ref())?;

            Ok(Some(dict))
        })
    }

    /// Encode the given sequence and pair. This method can process raw text sequences
    /// as well as already pre-tokenized sequences.
    ///
    /// Example:
    ///     Here are some examples of the inputs that are accepted::
    ///
    ///         encode("A single sequence")`
    ///         encode("A sequence", "And its pair")`
    ///         encode([ "A", "pre", "tokenized", "sequence" ], is_pretokenized=True)`
    ///         encode(
    ///             [ "A", "pre", "tokenized", "sequence" ], [ "And", "its", "pair" ],
    ///             is_pretokenized=True
    ///         )
    ///
    /// Args:
    ///     sequence (:obj:`~tokenizers.InputSequence`):
    ///         The main input sequence we want to encode. This sequence can be either raw
    ///         text or pre-tokenized, according to the ``is_pretokenized`` argument:
    ///
    ///         - If ``is_pretokenized=False``: :class:`~tokenizers.TextInputSequence`
    ///         - If ``is_pretokenized=True``: :class:`~tokenizers.PreTokenizedInputSequence`
    ///
    ///     pair (:obj:`~tokenizers.InputSequence`, `optional`):
    ///         An optional input sequence. The expected format is the same that for ``sequence``.
    ///
    ///     is_pretokenized (:obj:`bool`, defaults to :obj:`False`):
    ///         Whether the input is already pre-tokenized
    ///
    ///     add_special_tokens (:obj:`bool`, defaults to :obj:`True`):
    ///         Whether to add the special tokens
    ///
    /// Returns:
    ///     :class:`~tokenizers.Encoding`: The encoded result
    ///
    #[pyo3(signature = (sequence, pair = None, is_pretokenized = false, add_special_tokens = true))]
    #[pyo3(
        text_signature = "(self, sequence, pair=None, is_pretokenized=False, add_special_tokens=True)"
    )]
    fn encode(
        &self,
        sequence: &Bound<'_, PyAny>,
        pair: Option<&Bound<'_, PyAny>>,
        is_pretokenized: bool,
        add_special_tokens: bool,
    ) -> PyResult<PyEncoding> {
        let sequence: tk::InputSequence = if is_pretokenized {
            sequence.extract::<PreTokenizedInputSequence>()?.into()
        } else {
            sequence.extract::<TextInputSequence>()?.into()
        };
        let input = match pair {
            Some(pair) => {
                let pair: tk::InputSequence = if is_pretokenized {
                    pair.extract::<PreTokenizedInputSequence>()?.into()
                } else {
                    pair.extract::<TextInputSequence>()?.into()
                };
                tk::EncodeInput::Dual(sequence, pair)
            }
            None => tk::EncodeInput::Single(sequence),
        };

        ToPyResult(
            self.tokenizer
                .encode_char_offsets(input, add_special_tokens)
                .map(|e| e.into()),
        )
        .into()
    }

    /// Encode the given batch of inputs. This method accept both raw text sequences
    /// as well as already pre-tokenized sequences. The reason we use `PySequence` is
    /// because it allows type checking with zero-cost (according to PyO3) as we don't
    /// have to convert to check.
    ///
    /// Example:
    ///     Here are some examples of the inputs that are accepted::
    ///
    ///         encode_batch([
    ///             "A single sequence",
    ///             ("A tuple with a sequence", "And its pair"),
    ///             [ "A", "pre", "tokenized", "sequence" ],
    ///             ([ "A", "pre", "tokenized", "sequence" ], "And its pair")
    ///         ])
    ///
    /// Args:
    ///     input (A :obj:`List`/:obj:`Tuple` of :obj:`~tokenizers.EncodeInput`):
    ///         A list of single sequences or pair sequences to encode. Each sequence
    ///         can be either raw text or pre-tokenized, according to the ``is_pretokenized``
    ///         argument:
    ///
    ///         - If ``is_pretokenized=False``: :class:`~tokenizers.TextEncodeInput`
    ///         - If ``is_pretokenized=True``: :class:`~tokenizers.PreTokenizedEncodeInput`
    ///
    ///     is_pretokenized (:obj:`bool`, defaults to :obj:`False`):
    ///         Whether the input is already pre-tokenized
    ///
    ///     add_special_tokens (:obj:`bool`, defaults to :obj:`True`):
    ///         Whether to add the special tokens
    ///
    /// Returns:
    ///     A :obj:`List` of :class:`~tokenizers.Encoding`: The encoded batch
    ///
    #[pyo3(signature = (input, is_pretokenized = false, add_special_tokens = true))]
    #[pyo3(text_signature = "(self, input, is_pretokenized=False, add_special_tokens=True)")]
    fn encode_batch(
        &self,
        py: Python<'_>,
        input: Vec<Bound<'_, PyAny>>,
        is_pretokenized: bool,
        add_special_tokens: bool,
    ) -> PyResult<Vec<PyEncoding>> {
        let mut items = Vec::<tk::EncodeInput>::with_capacity(input.len());
        for item in &input {
            let item: tk::EncodeInput = if is_pretokenized {
                item.extract::<PreTokenizedEncodeInput>()?.into()
            } else {
                item.extract::<TextEncodeInput>()?.into()
            };
            items.push(item);
        }
        py.allow_threads(|| {
            ToPyResult(
                self.tokenizer
                    .encode_batch_char_offsets(items, add_special_tokens)
                    .map(|encodings| encodings.into_iter().map(|e| e.into()).collect()),
            )
            .into()
        })
    }

    /// Encode the given batch of inputs. This method is faster than `encode_batch`
    /// because it doesn't keep track of offsets, they will be all zeros.
    ///
    /// Example:
    ///     Here are some examples of the inputs that are accepted::
    ///
    ///         encode_batch_fast([
    ///             "A single sequence",
    ///             ("A tuple with a sequence", "And its pair"),
    ///             [ "A", "pre", "tokenized", "sequence" ],
    ///             ([ "A", "pre", "tokenized", "sequence" ], "And its pair")
    ///         ])
    ///
    /// Args:
    ///     input (A :obj:`List`/:obj:`Tuple` of :obj:`~tokenizers.EncodeInput`):
    ///         A list of single sequences or pair sequences to encode. Each sequence
    ///         can be either raw text or pre-tokenized, according to the ``is_pretokenized``
    ///         argument:
    ///
    ///         - If ``is_pretokenized=False``: :class:`~tokenizers.TextEncodeInput`
    ///         - If ``is_pretokenized=True``: :class:`~tokenizers.PreTokenizedEncodeInput`
    ///
    ///     is_pretokenized (:obj:`bool`, defaults to :obj:`False`):
    ///         Whether the input is already pre-tokenized
    ///
    ///     add_special_tokens (:obj:`bool`, defaults to :obj:`True`):
    ///         Whether to add the special tokens
    ///
    /// Returns:
    ///     A :obj:`List` of :class:`~tokenizers.Encoding`: The encoded batch
    ///
    #[pyo3(signature = (input, is_pretokenized = false, add_special_tokens = true))]
    #[pyo3(text_signature = "(self, input, is_pretokenized=False, add_special_tokens=True)")]
    fn encode_batch_fast(
        &self,
        py: Python<'_>,
        input: Vec<Bound<'_, PyAny>>,
        is_pretokenized: bool,
        add_special_tokens: bool,
    ) -> PyResult<Vec<PyEncoding>> {
        let mut items = Vec::<tk::EncodeInput>::with_capacity(input.len());
        for item in &input {
            let item: tk::EncodeInput = if is_pretokenized {
                item.extract::<PreTokenizedEncodeInput>()?.into()
            } else {
                item.extract::<TextEncodeInput>()?.into()
            };
            items.push(item);
        }
        py.allow_threads(|| {
            ToPyResult(
                self.tokenizer
                    .encode_batch_fast(items, add_special_tokens)
                    .map(|encodings| encodings.into_iter().map(|e| e.into()).collect()),
            )
            .into()
        })
    }

    /// Decode the given list of ids back to a string
    ///
    /// This is used to decode anything coming back from a Language Model
    ///
    /// Args:
    ///     ids (A :obj:`List/Tuple` of :obj:`int`):
    ///         The list of ids that we want to decode
    ///
    ///     skip_special_tokens (:obj:`bool`, defaults to :obj:`True`):
    ///         Whether the special tokens should be removed from the decoded string
    ///
    /// Returns:
    ///     :obj:`str`: The decoded string
    #[pyo3(signature = (ids, skip_special_tokens = true))]
    #[pyo3(text_signature = "(self, ids, skip_special_tokens=True)")]
    fn decode(&self, ids: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        ToPyResult(self.tokenizer.decode(&ids, skip_special_tokens)).into()
    }

    /// Decode a batch of ids back to their corresponding string
    ///
    /// Args:
    ///     sequences (:obj:`List` of :obj:`List[int]`):
    ///         The batch of sequences we want to decode
    ///
    ///     skip_special_tokens (:obj:`bool`, defaults to :obj:`True`):
    ///         Whether the special tokens should be removed from the decoded strings
    ///
    /// Returns:
    ///     :obj:`List[str]`: A list of decoded strings
    #[pyo3(signature = (sequences, skip_special_tokens = true))]
    #[pyo3(text_signature = "(self, sequences, skip_special_tokens=True)")]
    fn decode_batch(
        &self,
        py: Python<'_>,
        sequences: Vec<Vec<u32>>,
        skip_special_tokens: bool,
    ) -> PyResult<Vec<String>> {
        py.allow_threads(|| {
            let slices = sequences.iter().map(|v| &v[..]).collect::<Vec<&[u32]>>();
            ToPyResult(self.tokenizer.decode_batch(&slices, skip_special_tokens)).into()
        })
    }

    /// Convert the given token to its corresponding id if it exists
    ///
    /// Args:
    ///     token (:obj:`str`):
    ///         The token to convert
    ///
    /// Returns:
    ///     :obj:`Optional[int]`: An optional id, :obj:`None` if out of vocabulary
    #[pyo3(text_signature = "(self, token)")]
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    /// Convert the given id to its corresponding token if it exists
    ///
    /// Args:
    ///     id (:obj:`int`):
    ///         The id to convert
    ///
    /// Returns:
    ///     :obj:`Optional[str]`: An optional token, :obj:`None` if out of vocabulary
    #[pyo3(text_signature = "(self, id)")]
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }

    /// Modifies the tokenizer in order to use or not the special tokens
    /// during encoding.
    ///
    /// Args:
    ///     value (:obj:`bool`):
    ///         Whether to use the special tokens or not
    ///
    #[setter]
    fn set_encode_special_tokens(&mut self, value: bool) {
        self.tokenizer.set_encode_special_tokens(value);
    }
    /// Get the value of the `encode_special_tokens` attribute
    ///
    /// Returns:
    ///     :obj:`bool`: the tokenizer's encode_special_tokens attribute
    #[getter]
    fn get_encode_special_tokens(&self) -> bool {
        self.tokenizer.get_encode_special_tokens()
    }
    /// Add the given tokens to the vocabulary
    ///
    /// The given tokens are added only if they don't already exist in the vocabulary.
    /// Each token then gets a new attributed id.
    ///
    /// Args:
    ///     tokens (A :obj:`List` of :class:`~tokenizers.AddedToken` or :obj:`str`):
    ///         The list of tokens we want to add to the vocabulary. Each token can be either a
    ///         string or an instance of :class:`~tokenizers.AddedToken` for more customization.
    ///
    /// Returns:
    ///     :obj:`int`: The number of tokens that were created in the vocabulary
    #[pyo3(text_signature = "(self, tokens)")]
    fn add_tokens(&mut self, tokens: &Bound<'_, PyList>) -> PyResult<usize> {
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

        Ok(self.tokenizer.add_tokens(&tokens))
    }

    /// Add the given special tokens to the Tokenizer.
    ///
    /// If these tokens are already part of the vocabulary, it just let the Tokenizer know about
    /// them. If they don't exist, the Tokenizer creates them, giving them a new id.
    ///
    /// These special tokens will never be processed by the model (ie won't be split into
    /// multiple tokens), and they can be removed from the output when decoding.
    ///
    /// Args:
    ///     tokens (A :obj:`List` of :class:`~tokenizers.AddedToken` or :obj:`str`):
    ///         The list of special tokens we want to add to the vocabulary. Each token can either
    ///         be a string or an instance of :class:`~tokenizers.AddedToken` for more
    ///         customization.
    ///
    /// Returns:
    ///     :obj:`int`: The number of tokens that were created in the vocabulary
    #[pyo3(text_signature = "(self, tokens)")]
    fn add_special_tokens(&mut self, tokens: &Bound<'_, PyList>) -> PyResult<usize> {
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

        Ok(self.tokenizer.add_special_tokens(&tokens))
    }

    /// Train the Tokenizer using the given files.
    ///
    /// Reads the files line by line, while keeping all the whitespace, even new lines.
    /// If you want to train from data store in-memory, you can check
    /// :meth:`~tokenizers.Tokenizer.train_from_iterator`
    ///
    /// Args:
    ///     files (:obj:`List[str]`):
    ///         A list of path to the files that we should use for training
    ///
    ///     trainer (:obj:`~tokenizers.trainers.Trainer`, `optional`):
    ///         An optional trainer that should be used to train our Model
    #[pyo3(signature = (files, trainer = None))]
    #[pyo3(text_signature = "(self, files, trainer = None)")]
    fn train(&mut self, files: Vec<String>, trainer: Option<&mut PyTrainer>) -> PyResult<()> {
        let mut trainer =
            trainer.map_or_else(|| self.tokenizer.get_model().get_trainer(), |t| t.clone());
        Python::with_gil(|py| {
            py.allow_threads(|| {
                ToPyResult(
                    self.tokenizer
                        .train_from_files(&mut trainer, files)
                        .map(|_| {}),
                )
                .into()
            })
        })
    }

    /// Train the Tokenizer using the provided iterator.
    ///
    /// You can provide anything that is a Python Iterator
    ///
    ///     * A list of sequences :obj:`List[str]`
    ///     * A generator that yields :obj:`str` or :obj:`List[str]`
    ///     * A Numpy array of strings
    ///     * ...
    ///
    /// Args:
    ///     iterator (:obj:`Iterator`):
    ///         Any iterator over strings or list of strings
    ///
    ///     trainer (:obj:`~tokenizers.trainers.Trainer`, `optional`):
    ///         An optional trainer that should be used to train our Model
    ///
    ///     length (:obj:`int`, `optional`):
    ///         The total number of sequences in the iterator. This is used to
    ///         provide meaningful progress tracking
    #[pyo3(signature = (iterator, trainer = None, length = None))]
    #[pyo3(text_signature = "(self, iterator, trainer=None, length=None)")]
    fn train_from_iterator(
        &mut self,
        py: Python,
        iterator: &Bound<'_, PyAny>,
        trainer: Option<&mut PyTrainer>,
        length: Option<usize>,
    ) -> PyResult<()> {
        let mut trainer =
            trainer.map_or_else(|| self.tokenizer.get_model().get_trainer(), |t| t.clone());

        let buffered_iter = PyBufferedIterator::new(
            iterator,
            |element| {
                // Each element of the iterator can either be:
                //  - An iterator, to allow batching
                //  - A string
                if let Ok(s) = element.downcast::<PyString>() {
                    itertools::Either::Right(std::iter::once(s.to_cow().map(|s| s.into_owned())))
                } else {
                    match element.try_iter() {
                        Ok(iter) => itertools::Either::Left(
                            iter.map(|i| i?.extract::<String>())
                                .collect::<Vec<_>>()
                                .into_iter(),
                        ),
                        Err(e) => itertools::Either::Right(std::iter::once(Err(e))),
                    }
                }
            },
            256,
        )?;

        py.allow_threads(|| {
            ResultShunt::process(buffered_iter, |iter| {
                self.tokenizer
                    .train(&mut trainer, MaybeSizedIterator::new(iter, length))
                    .map(|_| {})
                    .map_err(|e| exceptions::PyException::new_err(e.to_string()))
            })?
        })
    }

    /// Apply all the post-processing steps to the given encodings.
    ///
    /// The various steps are:
    ///
    ///     1. Truncate according to the set truncation params (provided with
    ///        :meth:`~tokenizers.Tokenizer.enable_truncation`)
    ///     2. Apply the :class:`~tokenizers.processors.PostProcessor`
    ///     3. Pad according to the set padding params (provided with
    ///        :meth:`~tokenizers.Tokenizer.enable_padding`)
    ///
    /// Args:
    ///     encoding (:class:`~tokenizers.Encoding`):
    ///         The :class:`~tokenizers.Encoding` corresponding to the main sequence.
    ///
    ///     pair (:class:`~tokenizers.Encoding`, `optional`):
    ///         An optional :class:`~tokenizers.Encoding` corresponding to the pair sequence.
    ///
    ///     add_special_tokens (:obj:`bool`):
    ///         Whether to add the special tokens
    ///
    /// Returns:
    ///     :class:`~tokenizers.Encoding`: The final post-processed encoding
    #[pyo3(signature = (encoding, pair = None, add_special_tokens = true))]
    #[pyo3(text_signature = "(self, encoding, pair=None, add_special_tokens=True)")]
    fn post_process(
        &self,
        encoding: &PyEncoding,
        pair: Option<&PyEncoding>,
        add_special_tokens: bool,
    ) -> PyResult<PyEncoding> {
        ToPyResult(
            self.tokenizer
                .post_process(
                    encoding.encoding.clone(),
                    pair.map(|p| p.encoding.clone()),
                    add_special_tokens,
                )
                .map(|e| e.into()),
        )
        .into()
    }

    /// The :class:`~tokenizers.models.Model` in use by the Tokenizer
    #[getter]
    fn get_model(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.tokenizer.get_model().get_as_subtype(py)
    }

    /// Set the :class:`~tokenizers.models.Model`
    #[setter]
    fn set_model(&mut self, model: PyRef<PyModel>) {
        self.tokenizer.with_model(model.clone());
    }

    /// The `optional` :class:`~tokenizers.normalizers.Normalizer` in use by the Tokenizer
    #[getter]
    fn get_normalizer(&self, py: Python<'_>) -> PyResult<PyObject> {
        if let Some(n) = self.tokenizer.get_normalizer() {
            n.get_as_subtype(py)
        } else {
            Ok(py.None())
        }
    }

    /// Set the :class:`~tokenizers.normalizers.Normalizer`
    #[setter]
    fn set_normalizer(&mut self, normalizer: Option<PyRef<PyNormalizer>>) {
        let normalizer_option = normalizer.map(|norm| norm.clone());
        self.tokenizer.with_normalizer(normalizer_option);
    }

    /// The `optional` :class:`~tokenizers.pre_tokenizers.PreTokenizer` in use by the Tokenizer
    #[getter]
    fn get_pre_tokenizer(&self, py: Python<'_>) -> PyResult<PyObject> {
        if let Some(pt) = self.tokenizer.get_pre_tokenizer() {
            pt.get_as_subtype(py)
        } else {
            Ok(py.None())
        }
    }

    /// Set the :class:`~tokenizers.normalizers.Normalizer`
    #[setter]
    fn set_pre_tokenizer(&mut self, pretok: Option<PyRef<PyPreTokenizer>>) {
        self.tokenizer
            .with_pre_tokenizer(pretok.map(|pre| pre.clone()));
    }

    /// The `optional` :class:`~tokenizers.processors.PostProcessor` in use by the Tokenizer
    #[getter]
    fn get_post_processor(&self, py: Python<'_>) -> PyResult<PyObject> {
        if let Some(n) = self.tokenizer.get_post_processor() {
            n.get_as_subtype(py)
        } else {
            Ok(py.None())
        }
    }

    /// Set the :class:`~tokenizers.processors.PostProcessor`
    #[setter]
    fn set_post_processor(&mut self, processor: Option<PyRef<PyPostProcessor>>) {
        self.tokenizer
            .with_post_processor(processor.map(|p| p.clone()));
    }

    /// The `optional` :class:`~tokenizers.decoders.Decoder` in use by the Tokenizer
    #[getter]
    fn get_decoder(&self, py: Python<'_>) -> PyResult<PyObject> {
        if let Some(dec) = self.tokenizer.get_decoder() {
            dec.get_as_subtype(py)
        } else {
            Ok(py.None())
        }
    }

    /// Set the :class:`~tokenizers.decoders.Decoder`
    #[setter]
    fn set_decoder(&mut self, decoder: Option<PyRef<PyDecoder>>) {
        self.tokenizer.with_decoder(decoder.map(|d| d.clone()));
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::models::PyModel;
    use crate::normalizers::{PyNormalizer, PyNormalizerTypeWrapper};
    use std::sync::{Arc, RwLock};
    use tempfile::NamedTempFile;
    use tk::normalizers::{Lowercase, NFKC};

    #[test]
    fn serialize() {
        let mut tokenizer = Tokenizer::new(PyModel::from(BPE::default()));
        tokenizer.with_normalizer(Some(PyNormalizer::new(PyNormalizerTypeWrapper::Sequence(
            vec![
                Arc::new(RwLock::new(NFKC.into())),
                Arc::new(RwLock::new(Lowercase.into())),
            ],
        ))));

        let tmp = NamedTempFile::new().unwrap().into_temp_path();
        tokenizer.save(&tmp, false).unwrap();

        Tokenizer::from_file(&tmp).unwrap();
    }

    #[test]
    fn serde_pyo3() {
        let mut tokenizer = Tokenizer::new(PyModel::from(BPE::default()));
        tokenizer.with_normalizer(Some(PyNormalizer::new(PyNormalizerTypeWrapper::Sequence(
            vec![
                Arc::new(RwLock::new(NFKC.into())),
                Arc::new(RwLock::new(Lowercase.into())),
            ],
        ))));

        let output = crate::utils::serde_pyo3::to_string(&tokenizer).unwrap();
        assert_eq!(output, "Tokenizer(version=\"1.0\", truncation=None, padding=None, added_tokens=[], normalizer=Sequence(normalizers=[NFKC(), Lowercase()]), pre_tokenizer=None, post_processor=None, decoder=None, model=BPE(dropout=None, unk_token=None, continuing_subword_prefix=None, end_of_word_suffix=None, fuse_unk=False, byte_fallback=False, ignore_merges=False, vocab={}, merges=[]))");
    }
}
