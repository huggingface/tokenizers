use std::collections::{hash_map::DefaultHasher, HashMap};
use std::hash::{Hash, Hasher};

use numpy::PyArray1;
use pyo3::class::basic::CompareOp;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use pyo3::PyObjectProtocol;
use tk::models::bpe::BPE;
use tk::tokenizer::{
    Model, PaddingDirection, PaddingParams, PaddingStrategy, PostProcessor, TokenizerImpl,
    TruncationParams, TruncationStrategy,
};
use tokenizers as tk;

use super::decoders::PyDecoder;
use super::encoding::PyEncoding;
use super::error::{PyError, ToPyResult};
use super::models::PyModel;
use super::normalizers::PyNormalizer;
use super::pre_tokenizers::PyPreTokenizer;
use super::trainers::PyTrainer;
use crate::processors::PyPostProcessor;

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
///
#[pyclass(dict, module = "tokenizers", name=AddedToken)]
#[text_signature = "(self, content, single_word=False, lstrip=False, rstrip=False, normalized=True)"]
pub struct PyAddedToken {
    pub content: String,
    pub is_special_token: bool,
    pub single_word: Option<bool>,
    pub lstrip: Option<bool>,
    pub rstrip: Option<bool>,
    pub normalized: Option<bool>,
}
impl PyAddedToken {
    pub fn from<S: Into<String>>(content: S, is_special_token: Option<bool>) -> Self {
        Self {
            content: content.into(),
            is_special_token: is_special_token.unwrap_or(false),
            single_word: None,
            lstrip: None,
            rstrip: None,
            normalized: None,
        }
    }

    pub fn get_token(&self) -> tk::tokenizer::AddedToken {
        let mut token = tk::AddedToken::from(&self.content, self.is_special_token);

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

    pub fn as_pydict<'py>(&self, py: Python<'py>) -> PyResult<&'py PyDict> {
        let dict = PyDict::new(py);
        let token = self.get_token();

        dict.set_item("content", token.content)?;
        dict.set_item("single_word", token.single_word)?;
        dict.set_item("lstrip", token.lstrip)?;
        dict.set_item("rstrip", token.rstrip)?;
        dict.set_item("normalized", token.normalized)?;

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
            is_special_token: !token.normalized,
        }
    }
}

#[pymethods]
impl PyAddedToken {
    #[new]
    #[args(kwargs = "**")]
    fn __new__(content: Option<&str>, kwargs: Option<&PyDict>) -> PyResult<Self> {
        let mut token = PyAddedToken::from(content.unwrap_or(""), None);

        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "single_word" => token.single_word = Some(value.extract()?),
                    "lstrip" => token.lstrip = Some(value.extract()?),
                    "rstrip" => token.rstrip = Some(value.extract()?),
                    "normalized" => token.normalized = Some(value.extract()?),
                    _ => println!("Ignored unknown kwarg option {}", key),
                }
            }
        }

        Ok(token)
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<&'py PyDict> {
        self.as_pydict(py)
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyDict>(py) {
            Ok(state) => {
                for (key, value) in state {
                    let key: &str = key.extract()?;
                    match key {
                        "content" => self.content = value.extract()?,
                        "single_word" => self.single_word = Some(value.extract()?),
                        "lstrip" => self.lstrip = Some(value.extract()?),
                        "rstrip" => self.rstrip = Some(value.extract()?),
                        "normalized" => self.normalized = Some(value.extract()?),
                        _ => {}
                    }
                }
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Get the content of this :obj:`AddedToken`
    #[getter]
    fn get_content(&self) -> &str {
        &self.content
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
}
#[pyproto]
impl PyObjectProtocol for PyAddedToken {
    fn __str__(&'p self) -> PyResult<&'p str> {
        Ok(&self.content)
    }

    fn __repr__(&self) -> PyResult<String> {
        let bool_to_python = |p| match p {
            true => "True",
            false => "False",
        };

        let token = self.get_token();
        Ok(format!(
            "AddedToken(\"{}\", rstrip={}, lstrip={}, single_word={}, normalized={})",
            self.content,
            bool_to_python(token.rstrip),
            bool_to_python(token.lstrip),
            bool_to_python(token.single_word),
            bool_to_python(token.normalized)
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
    fn extract(ob: &'s PyAny) -> PyResult<Self> {
        let err = exceptions::PyTypeError::new_err("TextInputSequence must be str");
        if let Ok(s) = ob.downcast::<PyString>() {
            Ok(Self(s.to_string_lossy().into()))
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
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let array = ob.downcast::<PyArray1<u8>>()?;
        let arr = array.as_array_ptr();
        let (type_num, elsize, alignment, data) = unsafe {
            let desc = (*arr).descr;
            (
                (*desc).type_num,
                (*desc).elsize as usize,
                (*desc).alignment as usize,
                (*arr).data,
            )
        };
        let n_elem = array.shape()[0];

        // type_num == 19 => Unicode
        if type_num != 19 {
            return Err(exceptions::PyTypeError::new_err(
                "Expected a np.array[dtype='U']",
            ));
        }

        unsafe {
            let all_bytes = std::slice::from_raw_parts(data as *const u8, elsize * n_elem);

            let seq = (0..n_elem)
                .map(|i| {
                    let bytes = &all_bytes[i * elsize..(i + 1) * elsize];
                    let unicode = pyo3::ffi::PyUnicode_FromUnicode(
                        bytes.as_ptr() as *const _,
                        elsize as isize / alignment as isize,
                    );
                    let gil = Python::acquire_gil();
                    let py = gil.python();
                    let obj = PyObject::from_owned_ptr(py, unicode);
                    let s = obj.cast_as::<PyString>(py)?;
                    Ok(s.to_string_lossy().trim_matches(char::from(0)).to_owned())
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
    fn extract(ob: &PyAny) -> PyResult<Self> {
        let array = ob.downcast::<PyArray1<u8>>()?;
        let arr = array.as_array_ptr();
        let (type_num, data) = unsafe { ((*(*arr).descr).type_num, (*arr).data) };
        let n_elem = array.shape()[0];

        if type_num != 17 {
            return Err(exceptions::PyTypeError::new_err(
                "Expected a np.array[dtype='O']",
            ));
        }

        unsafe {
            let objects = std::slice::from_raw_parts(data as *const PyObject, n_elem);

            let seq = objects
                .iter()
                .map(|obj| {
                    let gil = Python::acquire_gil();
                    let py = gil.python();
                    let s = obj.cast_as::<PyString>(py)?;
                    Ok(s.to_string_lossy().into_owned())
                })
                .collect::<PyResult<Vec<_>>>()?;

            Ok(Self(seq))
        }
    }
}
impl From<PyArrayStr> for tk::InputSequence<'_> {
    fn from(s: PyArrayStr) -> Self {
        s.0.into()
    }
}

struct PreTokenizedInputSequence<'s>(tk::InputSequence<'s>);
impl<'s> FromPyObject<'s> for PreTokenizedInputSequence<'s> {
    fn extract(ob: &'s PyAny) -> PyResult<Self> {
        if let Ok(seq) = ob.extract::<PyArrayUnicode>() {
            return Ok(Self(seq.into()));
        }
        if let Ok(seq) = ob.extract::<PyArrayStr>() {
            return Ok(Self(seq.into()));
        }
        if let Ok(s) = ob.downcast::<PyList>() {
            if let Ok(seq) = s.extract::<Vec<&str>>() {
                return Ok(Self(seq.into()));
            }
        }
        if let Ok(s) = ob.downcast::<PyTuple>() {
            if let Ok(seq) = s.extract::<Vec<&str>>() {
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
    fn extract(ob: &'s PyAny) -> PyResult<Self> {
        if let Ok(i) = ob.extract::<TextInputSequence>() {
            return Ok(Self(i.into()));
        }
        if let Ok((i1, i2)) = ob.extract::<(TextInputSequence, TextInputSequence)>() {
            return Ok(Self((i1, i2).into()));
        }
        if let Ok(arr) = ob.extract::<Vec<&PyAny>>() {
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
    fn extract(ob: &'s PyAny) -> PyResult<Self> {
        if let Ok(i) = ob.extract::<PreTokenizedInputSequence>() {
            return Ok(Self(i.into()));
        }
        if let Ok((i1, i2)) = ob.extract::<(PreTokenizedInputSequence, PreTokenizedInputSequence)>()
        {
            return Ok(Self((i1, i2).into()));
        }
        if let Ok(arr) = ob.extract::<Vec<&PyAny>>() {
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
#[pyclass(dict, module = "tokenizers", name=Tokenizer)]
#[text_signature = "(self, model)"]
#[derive(Clone)]
pub struct PyTokenizer {
    tokenizer: Tokenizer,
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
        Ok(PyBytes::new(py, data.as_bytes()).to_object(py))
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&PyBytes>(py) {
            Ok(s) => {
                self.tokenizer = serde_json::from_slice(s.as_bytes()).map_err(|e| {
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

    fn __getnewargs__<'p>(&self, py: Python<'p>) -> PyResult<&'p PyTuple> {
        let model = PyModel::from(BPE::default()).into_py(py);
        let args = PyTuple::new(py, vec![model]);
        Ok(args)
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
    #[text_signature = "(json)"]
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
    #[text_signature = "(path)"]
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
    #[text_signature = "(buffer)"]
    fn from_buffer(buffer: &PyBytes) -> PyResult<Self> {
        let tokenizer = serde_json::from_slice(buffer.as_bytes()).map_err(|e| {
            exceptions::PyValueError::new_err(format!(
                "Cannot instantiate Tokenizer from buffer: {}",
                e
            ))
        })?;
        Ok(Self { tokenizer })
    }

    /// Gets a serialized string representing this :class:`~tokenizers.Tokenizer`.
    ///
    /// Args:
    ///     pretty (:obj:`bool`, defaults to :obj:`False`):
    ///         Whether the JSON string should be pretty formatted.
    ///
    /// Returns:
    ///     :obj:`str`: A string representing the serialized Tokenizer
    #[args(pretty = false)]
    #[text_signature = "(self, pretty=False)"]
    fn to_str(&self, pretty: bool) -> PyResult<String> {
        ToPyResult(self.tokenizer.to_string(pretty)).into()
    }

    /// Save the :class:`~tokenizers.Tokenizer` to the file at the given path.
    ///
    /// Args:
    ///     path (:obj:`str`):
    ///         A path to a file in which to save the serialized tokenizer.
    ///
    ///     pretty (:obj:`bool`, defaults to :obj:`False`):
    ///         Whether the JSON file should be pretty formatted.
    #[args(pretty = false)]
    #[text_signature = "(self, pretty=False)"]
    fn save(&self, path: &str, pretty: bool) -> PyResult<()> {
        ToPyResult(self.tokenizer.save(path, pretty)).into()
    }

    /// Return the number of special tokens that would be added for single/pair sentences.
    /// :param is_pair: Boolean indicating if the input would be a single sentence or a pair
    /// :return:
    #[text_signature = "(self, is_pair)"]
    fn num_special_tokens_to_add(&self, is_pair: bool) -> PyResult<usize> {
        Ok(self
            .tokenizer
            .get_post_processor()
            .map_or(0, |p| p.added_tokens(is_pair)))
    }

    /// Get the underlying vocabulary
    ///
    /// Args:
    ///     with_added_tokens (:obj:`bool`, defaults to :obj:`True`):
    ///         Whether to include the added tokens
    ///
    /// Returns:
    ///     :obj:`Dict[str, int]`: The vocabulary
    #[args(with_added_tokens = true)]
    #[text_signature = "(self, with_added_tokens=True)"]
    fn get_vocab(&self, with_added_tokens: bool) -> PyResult<HashMap<String, u32>> {
        Ok(self.tokenizer.get_vocab(with_added_tokens))
    }

    /// Get the size of the underlying vocabulary
    ///
    /// Args:
    ///     with_added_tokens (:obj:`bool`, defaults to :obj:`True`):
    ///         Whether to include the added tokens
    ///
    /// Returns:
    ///     :obj:`int`: The size of the vocabulary
    #[args(with_added_tokens = true)]
    #[text_signature = "(self, with_added_tokens=True)"]
    fn get_vocab_size(&self, with_added_tokens: bool) -> PyResult<usize> {
        Ok(self.tokenizer.get_vocab_size(with_added_tokens))
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
    #[args(kwargs = "**")]
    #[text_signature = "(self, max_length, stride=0, strategy='longest_first')"]
    fn enable_truncation(&mut self, max_length: usize, kwargs: Option<&PyDict>) -> PyResult<()> {
        let mut params = TruncationParams::default();
        params.max_length = max_length;

        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "stride" => params.stride = value.extract()?,
                    "strategy" => {
                        let value: &str = value.extract()?;
                        params.strategy = match value {
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
                    _ => println!("Ignored unknown kwarg option {}", key),
                }
            }
        }

        self.tokenizer.with_truncation(Some(params));

        Ok(())
    }

    /// Disable truncation
    #[text_signature = "(self)"]
    fn no_truncation(&mut self) {
        self.tokenizer.with_truncation(None);
    }

    /// Get the currently set truncation parameters
    ///
    /// `Cannot set, use` :meth:`~tokenizers.Tokenizer.enable_truncation` `instead`
    ///
    /// Returns:
    ///     (:obj:`dict`, `optional`):
    ///         A dict with the current truncation parameters if truncation is enabled
    #[getter]
    fn get_truncation<'py>(&self, py: Python<'py>) -> PyResult<Option<&'py PyDict>> {
        self.tokenizer.get_truncation().map_or(Ok(None), |params| {
            let dict = PyDict::new(py);

            dict.set_item("max_length", params.max_length)?;
            dict.set_item("stride", params.stride)?;
            dict.set_item("strategy", params.strategy.as_ref())?;

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
    #[args(kwargs = "**")]
    #[text_signature = "(self, direction='right', pad_id=0, pad_type_id=0, pad_token='[PAD]', length=None, pad_to_multiple_of=None)"]
    fn enable_padding(&mut self, kwargs: Option<&PyDict>) -> PyResult<()> {
        let mut params = PaddingParams::default();

        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: &str = key.extract()?;
                match key {
                    "direction" => {
                        let value: &str = value.extract()?;
                        params.direction = match value {
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
    #[text_signature = "(self)"]
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
    fn get_padding<'py>(&self, py: Python<'py>) -> PyResult<Option<&'py PyDict>> {
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
    #[args(pair = "None", is_pretokenized = "false", add_special_tokens = "true")]
    #[text_signature = "(self, sequence, pair=None, is_pretokenized=False, add_special_tokens=True)"]
    fn encode(
        &self,
        sequence: &PyAny,
        pair: Option<&PyAny>,
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
    /// as well as already pre-tokenized sequences.
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
    #[args(is_pretokenized = "false", add_special_tokens = "true")]
    #[text_signature = "(self, input, is_pretokenized=False, add_special_tokens=True)"]
    fn encode_batch(
        &self,
        input: Vec<&PyAny>,
        is_pretokenized: bool,
        add_special_tokens: bool,
    ) -> PyResult<Vec<PyEncoding>> {
        let input: Vec<tk::EncodeInput> = input
            .into_iter()
            .map(|o| {
                let input: tk::EncodeInput = if is_pretokenized {
                    o.extract::<PreTokenizedEncodeInput>()?.into()
                } else {
                    o.extract::<TextEncodeInput>()?.into()
                };
                Ok(input)
            })
            .collect::<PyResult<Vec<tk::EncodeInput>>>()?;
        let gil = Python::acquire_gil();
        gil.python().allow_threads(|| {
            ToPyResult(
                self.tokenizer
                    .encode_batch_char_offsets(input, add_special_tokens)
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
    #[args(skip_special_tokens = true)]
    #[text_signature = "(self, ids, skip_special_tokens=True)"]
    fn decode(&self, ids: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        ToPyResult(self.tokenizer.decode(ids, skip_special_tokens)).into()
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
    #[args(skip_special_tokens = true)]
    #[text_signature = "(self, sequences, skip_special_tokens=True)"]
    fn decode_batch(
        &self,
        sequences: Vec<Vec<u32>>,
        skip_special_tokens: bool,
    ) -> PyResult<Vec<String>> {
        let gil = Python::acquire_gil();
        gil.python().allow_threads(|| {
            ToPyResult(self.tokenizer.decode_batch(sequences, skip_special_tokens)).into()
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
    #[text_signature = "(self, token)"]
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
    #[text_signature = "(self, id)"]
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
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
    #[text_signature = "(self, tokens)"]
    fn add_tokens(&mut self, tokens: &PyList) -> PyResult<usize> {
        let tokens = tokens
            .into_iter()
            .map(|token| {
                if let Ok(content) = token.extract::<String>() {
                    Ok(PyAddedToken::from(content, Some(false)).get_token())
                } else if let Ok(mut token) = token.extract::<PyRefMut<PyAddedToken>>() {
                    token.is_special_token = false;
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
    #[text_signature = "(self, tokens)"]
    fn add_special_tokens(&mut self, tokens: &PyList) -> PyResult<usize> {
        let tokens = tokens
            .into_iter()
            .map(|token| {
                if let Ok(content) = token.extract::<String>() {
                    Ok(tk::tokenizer::AddedToken::from(content, true))
                } else if let Ok(mut token) = token.extract::<PyRefMut<PyAddedToken>>() {
                    token.is_special_token = true;
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
    #[args(trainer = "None")]
    #[text_signature = "(self, files, trainer = None)"]
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
    #[args(trainer = "None", length = "None")]
    #[text_signature = "(self, iterator, trainer=None, length=None)"]
    fn train_from_iterator(
        &mut self,
        iterator: &PyAny,
        trainer: Option<&mut PyTrainer>,
        length: Option<usize>,
    ) -> PyResult<()> {
        use crate::utils::PySendIterator;

        let mut trainer =
            trainer.map_or_else(|| self.tokenizer.get_model().get_trainer(), |t| t.clone());

        let py_send = PySendIterator::new(
            // Each element of the iterator can either be:
            //  - An iterator, to allow batching
            //  - A string
            iterator.iter()?.flat_map(|seq| match seq {
                Ok(s) => {
                    if let Ok(s) = s.downcast::<PyString>() {
                        itertools::Either::Right(std::iter::once(s.to_str()))
                    } else {
                        match s.iter() {
                            Ok(iter) => itertools::Either::Left(iter.map(|i| i?.extract::<&str>())),
                            Err(e) => itertools::Either::Right(std::iter::once(Err(e))),
                        }
                    }
                }
                Err(e) => itertools::Either::Right(std::iter::once(Err(e))),
            }),
            length,
        );

        py_send.execute(|iter| {
            self.tokenizer
                .train(&mut trainer, iter)
                .map(|_| {})
                .map_err(|e| exceptions::PyException::new_err(e.to_string()))
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
    #[args(pair = "None", add_special_tokens = true)]
    #[text_signature = "(self, encoding, pair=None, add_special_tokens=True)"]
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
    fn get_model(&self) -> PyResult<PyObject> {
        self.tokenizer.get_model().get_as_subtype()
    }

    /// Set the :class:`~tokenizers.models.Model`
    #[setter]
    fn set_model(&mut self, model: PyRef<PyModel>) {
        self.tokenizer.with_model(model.clone());
    }

    /// The `optional` :class:`~tokenizers.normalizers.Normalizer` in use by the Tokenizer
    #[getter]
    fn get_normalizer(&self) -> PyResult<PyObject> {
        if let Some(n) = self.tokenizer.get_normalizer() {
            n.get_as_subtype()
        } else {
            Ok(Python::acquire_gil().python().None())
        }
    }

    /// Set the :class:`~tokenizers.normalizers.Normalizer`
    #[setter]
    fn set_normalizer(&mut self, normalizer: PyRef<PyNormalizer>) {
        self.tokenizer.with_normalizer(normalizer.clone());
    }

    /// The `optional` :class:`~tokenizers.pre_tokenizers.PreTokenizer` in use by the Tokenizer
    #[getter]
    fn get_pre_tokenizer(&self) -> PyResult<PyObject> {
        if let Some(pt) = self.tokenizer.get_pre_tokenizer() {
            pt.get_as_subtype()
        } else {
            Ok(Python::acquire_gil().python().None())
        }
    }

    /// Set the :class:`~tokenizers.normalizers.Normalizer`
    #[setter]
    fn set_pre_tokenizer(&mut self, pretok: PyRef<PyPreTokenizer>) {
        self.tokenizer.with_pre_tokenizer(pretok.clone());
    }

    /// The `optional` :class:`~tokenizers.processors.PostProcessor` in use by the Tokenizer
    #[getter]
    fn get_post_processor(&self) -> PyResult<PyObject> {
        if let Some(n) = self.tokenizer.get_post_processor() {
            n.get_as_subtype()
        } else {
            Ok(Python::acquire_gil().python().None())
        }
    }

    /// Set the :class:`~tokenizers.processors.PostProcessor`
    #[setter]
    fn set_post_processor(&mut self, processor: PyRef<PyPostProcessor>) {
        self.tokenizer.with_post_processor(processor.clone());
    }

    /// The `optional` :class:`~tokenizers.decoders.Decoder` in use by the Tokenizer
    #[getter]
    fn get_decoder(&self) -> PyResult<PyObject> {
        if let Some(dec) = self.tokenizer.get_decoder() {
            dec.get_as_subtype()
        } else {
            Ok(Python::acquire_gil().python().None())
        }
    }

    /// Set the :class:`~tokenizers.decoders.Decoder`
    #[setter]
    fn set_decoder(&mut self, decoder: PyRef<PyDecoder>) {
        self.tokenizer.with_decoder(decoder.clone());
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
        tokenizer.with_normalizer(PyNormalizer::new(PyNormalizerTypeWrapper::Sequence(vec![
            Arc::new(RwLock::new(NFKC.into())),
            Arc::new(RwLock::new(Lowercase.into())),
        ])));

        let tmp = NamedTempFile::new().unwrap().into_temp_path();
        tokenizer.save(&tmp, false).unwrap();

        Tokenizer::from_file(&tmp).unwrap();
    }
}
