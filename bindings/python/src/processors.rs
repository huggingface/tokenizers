use std::convert::TryInto;
use std::sync::Arc;
use std::sync::RwLock;

use crate::encoding::PyEncoding;
use crate::error::ToPyResult;
use pyo3::exceptions;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::*;
use serde::ser::SerializeStruct;
use serde::Deserializer;
use serde::Serializer;
use serde::{Deserialize, Serialize};
use tk::processors::bert::BertProcessing;
use tk::processors::byte_level::ByteLevel;
use tk::processors::roberta::RobertaProcessing;
use tk::processors::template::{SpecialToken, Template};
use tk::processors::PostProcessorWrapper;
use tk::{Encoding, PostProcessor};
use tokenizers as tk;

/// Base class for all post-processors
///
/// This class is not supposed to be instantiated directly. Instead, any implementation of
/// a PostProcessor will return an instance of this class when instantiated.
#[pyclass(
    dict,
    module = "tokenizers.processors",
    name = "PostProcessor",
    subclass
)]
#[derive(Clone, Deserialize, Serialize)]
#[serde(transparent)]
pub struct PyPostProcessor {
    processor: PyPostProcessorTypeWrapper,
}

impl<I> From<I> for PyPostProcessor
where
    I: Into<PostProcessorWrapper>,
{
    fn from(processor: I) -> Self {
        PyPostProcessor {
            processor: processor.into().into(),
        }
    }
}

impl PyPostProcessor {
    pub(crate) fn new(processor: PyPostProcessorTypeWrapper) -> Self {
        PyPostProcessor { processor }
    }

    pub(crate) fn get_as_subtype(&self, py: Python<'_>) -> PyResult<PyObject> {
        let base = self.clone();
        Ok(
            match self.processor {
                PyPostProcessorTypeWrapper::Sequence(_) => Py::new(py, (PySequence {}, base))?
                .into_pyobject(py)?
                .into_any()
                .into(),
                PyPostProcessorTypeWrapper::Single(ref inner) => {

            match &*inner.read().map_err(|_| {
                PyException::new_err("RwLock synchronisation primitive is poisoned, cannot get subtype of PyPostProcessor")
            })? {
                PostProcessorWrapper::ByteLevel(_) => Py::new(py, (PyByteLevel {}, base))?
                    .into_pyobject(py)?
                    .into_any()
                    .into(),
                PostProcessorWrapper::Bert(_) => Py::new(py, (PyBertProcessing {}, base))?
                    .into_pyobject(py)?
                    .into_any()
                    .into(),
                PostProcessorWrapper::Roberta(_) => Py::new(py, (PyRobertaProcessing {}, base))?
                    .into_pyobject(py)?
                    .into_any()
                    .into(),
                PostProcessorWrapper::Template(_) => Py::new(py, (PyTemplateProcessing {}, base))?
                    .into_pyobject(py)?
                    .into_any()
                    .into(),
                PostProcessorWrapper::Sequence(_) => Py::new(py, (PySequence {}, base))?
                    .into_pyobject(py)?
                    .into_any()
                    .into(),
            }
                }
            }
        )
    }
}

impl PostProcessor for PyPostProcessor {
    // TODO: update signature to `tk::Result<usize>`
    fn added_tokens(&self, is_pair: bool) -> usize {
        self.processor.added_tokens(is_pair)
    }

    fn process_encodings(
        &self,
        encodings: Vec<Encoding>,
        add_special_tokens: bool,
    ) -> tk::Result<Vec<Encoding>> {
        self.processor
            .process_encodings(encodings, add_special_tokens)
    }
}

#[pymethods]
impl PyPostProcessor {
    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let data = serde_json::to_string(&self.processor).map_err(|e| {
            exceptions::PyException::new_err(format!(
                "Error while attempting to pickle PostProcessor: {e}"
            ))
        })?;
        Ok(PyBytes::new(py, data.as_bytes()).into())
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&[u8]>(py) {
            Ok(s) => {
                self.processor = serde_json::from_slice(s).map_err(|e| {
                    exceptions::PyException::new_err(format!(
                        "Error while attempting to unpickle PostProcessor: {e}"
                    ))
                })?;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Return the number of special tokens that would be added for single/pair sentences.
    ///
    /// Args:
    ///     is_pair (:obj:`bool`):
    ///         Whether the input would be a pair of sequences
    ///
    /// Returns:
    ///     :obj:`int`: The number of tokens to add
    #[pyo3(text_signature = "(self, is_pair)")]
    fn num_special_tokens_to_add(&self, is_pair: bool) -> PyResult<usize> {
        Ok(self.processor.added_tokens(is_pair))
    }

    /// Post-process the given encodings, generating the final one
    ///
    /// Args:
    ///     encoding (:class:`~tokenizers.Encoding`):
    ///         The encoding for the first sequence
    ///
    ///     pair (:class:`~tokenizers.Encoding`, `optional`):
    ///         The encoding for the pair sequence
    ///
    ///     add_special_tokens (:obj:`bool`):
    ///         Whether to add the special tokens
    ///
    /// Return:
    ///     :class:`~tokenizers.Encoding`: The final encoding
    #[pyo3(signature = (encoding, pair = None, add_special_tokens = true))]
    #[pyo3(text_signature = "(self, encoding, pair=None, add_special_tokens=True)")]
    fn process(
        &self,
        encoding: &PyEncoding,
        pair: Option<&PyEncoding>,
        add_special_tokens: bool,
    ) -> PyResult<PyEncoding> {
        let final_encoding = ToPyResult(self.processor.process(
            encoding.encoding.clone(),
            pair.map(|e| e.encoding.clone()),
            add_special_tokens,
        ))
        .into_py()?;
        Ok(final_encoding.into())
    }

    fn __repr__(&self) -> PyResult<String> {
        crate::utils::serde_pyo3::repr(self)
            .map_err(|e| exceptions::PyException::new_err(e.to_string()))
    }

    fn __str__(&self) -> PyResult<String> {
        crate::utils::serde_pyo3::to_string(self)
            .map_err(|e| exceptions::PyException::new_err(e.to_string()))
    }
}

macro_rules! getter {
    ($self: ident, $variant: ident, $($name: tt)+) => {{
        let super_ = $self.as_ref();
        if let PyPostProcessorTypeWrapper::Single(ref single) = super_.processor {
            if let PostProcessorWrapper::$variant(ref post) = *single.read().expect(
                "RwLock synchronisation primitive is poisoned, cannot get subtype of PyPostProcessor"
            ) {
                post.$($name)+
            } else {
                unreachable!()
            }
        } else {
            unreachable!()
        }
    }};
}

macro_rules! setter {
    ($self: ident, $variant: ident, $name: ident, $value: expr) => {{
        let super_ = $self.as_ref();
        if let PyPostProcessorTypeWrapper::Single(ref single) = super_.processor {
        if let PostProcessorWrapper::$variant(ref mut post) = *single.write().expect(
            "RwLock synchronisation primitive is poisoned, cannot get subtype of PyPostProcessor",
        ) {
            post.$name = $value;
        }
        }
    }};
    ($self: ident, $variant: ident, @$name: ident, $value: expr) => {{
        let super_ = $self.as_ref();
        if let PyPostProcessorTypeWrapper::Single(ref single) = super_.processor {
        if let PostProcessorWrapper::$variant(ref mut post) = *single.write().expect(
            "RwLock synchronisation primitive is poisoned, cannot get subtype of PyPostProcessor",
        ) {
            post.$name($value);
        }
        }
    };};
}

#[derive(Clone)]
pub(crate) enum PyPostProcessorTypeWrapper {
    Sequence(Vec<Arc<RwLock<PostProcessorWrapper>>>),
    Single(Arc<RwLock<PostProcessorWrapper>>),
}

impl PostProcessor for PyPostProcessorTypeWrapper {
    fn added_tokens(&self, is_pair: bool) -> usize {
        match self {
            PyPostProcessorTypeWrapper::Single(inner) => inner
                .read()
                .expect("RwLock synchronisation primitive is poisoned, cannot get subtype of PyPostProcessor")
                .added_tokens(is_pair),
            PyPostProcessorTypeWrapper::Sequence(inner) => inner.iter().map(|p| {
                p.read()
                    .expect("RwLock synchronisation primitive is poisoned, cannot get subtype of PyPostProcessor")
                    .added_tokens(is_pair)
            }).sum::<usize>(),
        }
    }

    fn process_encodings(
        &self,
        mut encodings: Vec<Encoding>,
        add_special_tokens: bool,
    ) -> tk::Result<Vec<Encoding>> {
        match self {
            PyPostProcessorTypeWrapper::Single(inner) => inner
                .read()
                .map_err(|_| PyException::new_err("RwLock synchronisation primitive is poisoned, cannot get subtype of PyPreTokenizer"))?
                .process_encodings(encodings, add_special_tokens),
            PyPostProcessorTypeWrapper::Sequence(inner) => {
                for processor in inner.iter() {
                    encodings = processor
                        .read()
                        .map_err(|_| PyException::new_err("RwLock synchronisation primitive is poisoned, cannot get subtype of PyPreTokenizer"))?
                        .process_encodings(encodings, add_special_tokens)?;
                }
        Ok(encodings)
            },
        }
    }
}

impl<'de> Deserialize<'de> for PyPostProcessorTypeWrapper {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wrapper = PostProcessorWrapper::deserialize(deserializer)?;
        Ok(wrapper.into())
    }
}

impl Serialize for PyPostProcessorTypeWrapper {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            PyPostProcessorTypeWrapper::Sequence(seq) => {
                let mut ser = serializer.serialize_struct("Sequence", 2)?;
                ser.serialize_field("type", "Sequence")?;
                ser.serialize_field("processors", seq)?;
                ser.end()
            }
            PyPostProcessorTypeWrapper::Single(inner) => inner.serialize(serializer),
        }
    }
}

impl<I> From<I> for PyPostProcessorTypeWrapper
where
    I: Into<PostProcessorWrapper>,
{
    fn from(processor: I) -> Self {
        let processor = processor.into();
        match processor {
            PostProcessorWrapper::Sequence(seq) => PyPostProcessorTypeWrapper::Sequence(
                seq.into_iter().map(|p| Arc::new(RwLock::new(p))).collect(),
            ),
            _ => PyPostProcessorTypeWrapper::Single(Arc::new(RwLock::new(processor.clone()))),
        }
    }
}

/// This post-processor takes care of adding the special tokens needed by
/// a Bert model:
///
///     - a SEP token
///     - a CLS token
///
/// Args:
///     sep (:obj:`Tuple[str, int]`):
///         A tuple with the string representation of the SEP token, and its id
///
///     cls (:obj:`Tuple[str, int]`):
///         A tuple with the string representation of the CLS token, and its id
#[pyclass(extends=PyPostProcessor, module = "tokenizers.processors", name = "BertProcessing")]
pub struct PyBertProcessing {}
#[pymethods]
impl PyBertProcessing {
    #[new]
    #[pyo3(text_signature = "(self, sep, cls)")]
    fn new(sep: (String, u32), cls: (String, u32)) -> (Self, PyPostProcessor) {
        (PyBertProcessing {}, BertProcessing::new(sep, cls).into())
    }

    fn __getnewargs__<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyTuple>> {
        PyTuple::new(py, [("", 0), ("", 0)])
    }

    #[getter]
    fn get_sep(self_: PyRef<Self>) -> Result<Bound<'_, PyTuple>, PyErr> {
        let py = self_.py();
        let (tok, id) = getter!(self_, Bert, get_sep_copy());
        PyTuple::new(
            py,
            Vec::<PyObject>::from([tok.into_pyobject(py)?.into(), id.into_pyobject(py)?.into()]),
        )
    }

    #[setter]
    fn set_sep(self_: PyRef<Self>, sep: Bound<'_, PyTuple>) -> PyResult<()> {
        let sep = sep.extract()?;
        setter!(self_, Bert, sep, sep);
        Ok(())
    }

    #[getter]
    fn get_cls(self_: PyRef<Self>) -> Result<Bound<'_, PyTuple>, PyErr> {
        let py = self_.py();
        let (tok, id) = getter!(self_, Bert, get_cls_copy());
        PyTuple::new(
            py,
            Vec::<PyObject>::from([tok.into_pyobject(py)?.into(), id.into_pyobject(py)?.into()]),
        )
    }

    #[setter]
    fn set_cls(self_: PyRef<Self>, cls: Bound<'_, PyTuple>) -> PyResult<()> {
        let cls = cls.extract()?;
        setter!(self_, Bert, cls, cls);
        Ok(())
    }
}

/// This post-processor takes care of adding the special tokens needed by
/// a Roberta model:
///
///     - a SEP token
///     - a CLS token
///
/// It also takes care of trimming the offsets.
/// By default, the ByteLevel BPE might include whitespaces in the produced tokens. If you don't
/// want the offsets to include these whitespaces, then this PostProcessor should be initialized
/// with :obj:`trim_offsets=True`
///
/// Args:
///     sep (:obj:`Tuple[str, int]`):
///         A tuple with the string representation of the SEP token, and its id
///
///     cls (:obj:`Tuple[str, int]`):
///         A tuple with the string representation of the CLS token, and its id
///
///     trim_offsets (:obj:`bool`, `optional`, defaults to :obj:`True`):
///         Whether to trim the whitespaces from the produced offsets.
///
///     add_prefix_space (:obj:`bool`, `optional`, defaults to :obj:`True`):
///         Whether the add_prefix_space option was enabled during pre-tokenization. This
///         is relevant because it defines the way the offsets are trimmed out.
#[pyclass(extends=PyPostProcessor, module = "tokenizers.processors", name = "RobertaProcessing")]
pub struct PyRobertaProcessing {}
#[pymethods]
impl PyRobertaProcessing {
    #[new]
    #[pyo3(signature = (sep, cls, trim_offsets = true, add_prefix_space = true), text_signature = "(self, sep, cls, trim_offsets=True, add_prefix_space=True)")]
    fn new(
        sep: (String, u32),
        cls: (String, u32),
        trim_offsets: bool,
        add_prefix_space: bool,
    ) -> (Self, PyPostProcessor) {
        let proc = RobertaProcessing::new(sep, cls)
            .trim_offsets(trim_offsets)
            .add_prefix_space(add_prefix_space);
        (PyRobertaProcessing {}, proc.into())
    }

    fn __getnewargs__<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyTuple>> {
        PyTuple::new(py, [("", 0), ("", 0)])
    }

    #[getter]
    fn get_sep(self_: PyRef<Self>) -> Result<Bound<'_, PyTuple>, PyErr> {
        let py = self_.py();
        let (tok, id) = getter!(self_, Roberta, get_sep_copy());
        PyTuple::new(
            py,
            Vec::<PyObject>::from([tok.into_pyobject(py)?.into(), id.into_pyobject(py)?.into()]),
        )
    }

    #[setter]
    fn set_sep(self_: PyRef<Self>, sep: Bound<'_, PyTuple>) -> PyResult<()> {
        let sep = sep.extract()?;
        setter!(self_, Roberta, sep, sep);
        Ok(())
    }

    #[getter]
    fn get_cls(self_: PyRef<Self>) -> Result<Bound<'_, PyTuple>, PyErr> {
        let py = self_.py();
        let (tok, id) = getter!(self_, Roberta, get_cls_copy());
        PyTuple::new(
            py,
            Vec::<PyObject>::from([tok.into_pyobject(py)?.into(), id.into_pyobject(py)?.into()]),
        )
    }

    #[setter]
    fn set_cls(self_: PyRef<Self>, cls: Bound<'_, PyTuple>) -> PyResult<()> {
        let cls = cls.extract()?;
        setter!(self_, Roberta, cls, cls);
        Ok(())
    }

    #[getter]
    fn get_trim_offsets(self_: PyRef<Self>) -> bool {
        getter!(self_, Roberta, trim_offsets)
    }

    #[setter]
    fn set_trim_offsets(self_: PyRef<Self>, trim_offsets: bool) {
        setter!(self_, Roberta, trim_offsets, trim_offsets)
    }

    #[getter]
    fn get_add_prefix_space(self_: PyRef<Self>) -> bool {
        getter!(self_, Roberta, add_prefix_space)
    }

    #[setter]
    fn set_add_prefix_space(self_: PyRef<Self>, add_prefix_space: bool) {
        setter!(self_, Roberta, add_prefix_space, add_prefix_space)
    }
}

/// This post-processor takes care of trimming the offsets.
///
/// By default, the ByteLevel BPE might include whitespaces in the produced tokens. If you don't
/// want the offsets to include these whitespaces, then this PostProcessor must be used.
///
/// Args:
///     trim_offsets (:obj:`bool`):
///         Whether to trim the whitespaces from the produced offsets.
#[pyclass(extends=PyPostProcessor, module = "tokenizers.processors", name = "ByteLevel")]
pub struct PyByteLevel {}
#[pymethods]
impl PyByteLevel {
    #[new]
    #[pyo3(signature = (add_prefix_space = None, trim_offsets = None, use_regex = None, **_kwargs), text_signature = "(self, trim_offsets=True)")]
    fn new(
        add_prefix_space: Option<bool>,
        trim_offsets: Option<bool>,
        use_regex: Option<bool>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> (Self, PyPostProcessor) {
        let mut byte_level = ByteLevel::default();

        if let Some(aps) = add_prefix_space {
            byte_level = byte_level.add_prefix_space(aps);
        }

        if let Some(to) = trim_offsets {
            byte_level = byte_level.trim_offsets(to);
        }

        if let Some(ur) = use_regex {
            byte_level = byte_level.use_regex(ur);
        }

        (PyByteLevel {}, byte_level.into())
    }

    #[getter]
    fn get_add_prefix_space(self_: PyRef<Self>) -> bool {
        getter!(self_, ByteLevel, add_prefix_space)
    }

    #[setter]
    fn set_add_prefix_space(self_: PyRef<Self>, add_prefix_space: bool) {
        setter!(self_, ByteLevel, add_prefix_space, add_prefix_space)
    }

    #[getter]
    fn get_trim_offsets(self_: PyRef<Self>) -> bool {
        getter!(self_, ByteLevel, trim_offsets)
    }

    #[setter]
    fn set_trim_offsets(self_: PyRef<Self>, trim_offsets: bool) {
        setter!(self_, ByteLevel, trim_offsets, trim_offsets)
    }

    #[getter]
    fn get_use_regex(self_: PyRef<Self>) -> bool {
        getter!(self_, ByteLevel, use_regex)
    }

    #[setter]
    fn set_use_regex(self_: PyRef<Self>, use_regex: bool) {
        setter!(self_, ByteLevel, use_regex, use_regex)
    }
}

#[derive(Clone, Debug)]
pub struct PySpecialToken(SpecialToken);

impl From<PySpecialToken> for SpecialToken {
    fn from(v: PySpecialToken) -> Self {
        v.0
    }
}

impl FromPyObject<'_> for PySpecialToken {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(v) = ob.extract::<(String, u32)>() {
            Ok(Self(v.into()))
        } else if let Ok(v) = ob.extract::<(u32, String)>() {
            Ok(Self(v.into()))
        } else if let Ok(d) = ob.downcast::<PyDict>() {
            let id = d
                .get_item("id")?
                .ok_or_else(|| exceptions::PyValueError::new_err("`id` must be specified"))?
                .extract::<String>()?;
            let ids = d
                .get_item("ids")?
                .ok_or_else(|| exceptions::PyValueError::new_err("`ids` must be specified"))?
                .extract::<Vec<u32>>()?;
            let tokens = d
                .get_item("tokens")?
                .ok_or_else(|| exceptions::PyValueError::new_err("`tokens` must be specified"))?
                .extract::<Vec<String>>()?;

            Ok(Self(
                ToPyResult(SpecialToken::new(id, ids, tokens)).into_py()?,
            ))
        } else {
            Err(exceptions::PyTypeError::new_err(
                "Expected Union[Tuple[str, int], Tuple[int, str], dict]",
            ))
        }
    }
}

#[derive(Clone, Debug)]
pub struct PyTemplate(Template);

impl From<PyTemplate> for Template {
    fn from(v: PyTemplate) -> Self {
        v.0
    }
}

impl FromPyObject<'_> for PyTemplate {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(s) = ob.extract::<String>() {
            Ok(Self(
                s.try_into().map_err(exceptions::PyValueError::new_err)?,
            ))
        } else if let Ok(s) = ob.extract::<Vec<String>>() {
            Ok(Self(
                s.try_into().map_err(exceptions::PyValueError::new_err)?,
            ))
        } else {
            Err(exceptions::PyTypeError::new_err(
                "Expected Union[str, List[str]]",
            ))
        }
    }
}

/// Provides a way to specify templates in order to add the special tokens to each
/// input sequence as relevant.
///
/// Let's take :obj:`BERT` tokenizer as an example. It uses two special tokens, used to
/// delimitate each sequence. :obj:`[CLS]` is always used at the beginning of the first
/// sequence, and :obj:`[SEP]` is added at the end of both the first, and the pair
/// sequences. The final result looks like this:
///
///     - Single sequence: :obj:`[CLS] Hello there [SEP]`
///     - Pair sequences: :obj:`[CLS] My name is Anthony [SEP] What is my name? [SEP]`
///
/// With the type ids as following::
///
///     [CLS]   ...   [SEP]   ...   [SEP]
///       0      0      0      1      1
///
/// You can achieve such behavior using a TemplateProcessing::
///
///     TemplateProcessing(
///         single="[CLS] $0 [SEP]",
///         pair="[CLS] $A [SEP] $B:1 [SEP]:1",
///         special_tokens=[("[CLS]", 1), ("[SEP]", 0)],
///     )
///
/// In this example, each input sequence is identified using a ``$`` construct. This identifier
/// lets us specify each input sequence, and the type_id to use. When nothing is specified,
/// it uses the default values. Here are the different ways to specify it:
///
///     - Specifying the sequence, with default ``type_id == 0``: ``$A`` or ``$B``
///     - Specifying the `type_id` with default ``sequence == A``: ``$0``, ``$1``, ``$2``, ...
///     - Specifying both: ``$A:0``, ``$B:1``, ...
///
/// The same construct is used for special tokens: ``<identifier>(:<type_id>)?``.
///
/// **Warning**: You must ensure that you are giving the correct tokens/ids as these
/// will be added to the Encoding without any further check. If the given ids correspond
/// to something totally different in a `Tokenizer` using this `PostProcessor`, it
/// might lead to unexpected results.
///
/// Args:
///     single (:obj:`Template`):
///         The template used for single sequences
///
///     pair (:obj:`Template`):
///         The template used when both sequences are specified
///
///     special_tokens (:obj:`Tokens`):
///         The list of special tokens used in each sequences
///
/// Types:
///
///     Template (:obj:`str` or :obj:`List`):
///         - If a :obj:`str` is provided, the whitespace is used as delimiter between tokens
///         - If a :obj:`List[str]` is provided, a list of tokens
///
///     Tokens (:obj:`List[Union[Tuple[int, str], Tuple[str, int], dict]]`):
///         - A :obj:`Tuple` with both a token and its associated ID, in any order
///         - A :obj:`dict` with the following keys:
///             - "id": :obj:`str` => The special token id, as specified in the Template
///             - "ids": :obj:`List[int]` => The associated IDs
///             - "tokens": :obj:`List[str]` => The associated tokens
///
///          The given dict expects the provided :obj:`ids` and :obj:`tokens` lists to have
///          the same length.
#[pyclass(extends=PyPostProcessor, module = "tokenizers.processors", name = "TemplateProcessing")]
pub struct PyTemplateProcessing {}
#[pymethods]
impl PyTemplateProcessing {
    #[new]
    #[pyo3(signature = (single = None, pair = None, special_tokens = None), text_signature = "(self, single, pair, special_tokens)")]
    fn new(
        single: Option<PyTemplate>,
        pair: Option<PyTemplate>,
        special_tokens: Option<Vec<PySpecialToken>>,
    ) -> PyResult<(Self, PyPostProcessor)> {
        let mut builder = tk::processors::template::TemplateProcessing::builder();

        if let Some(seq) = single {
            builder.single(seq.into());
        }
        if let Some(seq) = pair {
            builder.pair(seq.into());
        }
        if let Some(sp) = special_tokens {
            builder.special_tokens(sp);
        }
        let processor = builder
            .build()
            .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;

        Ok((PyTemplateProcessing {}, processor.into()))
    }

    #[getter]
    fn get_single(self_: PyRef<Self>) -> String {
        getter!(self_, Template, get_single())
    }

    #[setter]
    fn set_single(self_: PyRef<Self>, single: PyTemplate) -> PyResult<()> {
        let template: Template = Template::from(single);
        let super_ = self_.as_ref();
        if let PyPostProcessorTypeWrapper::Single(ref inner) = super_.processor {
            if let PostProcessorWrapper::Template(ref mut post) = *inner
                .write()
                .map_err(|_| PyException::new_err("RwLock synchronisation primitive is poisoned, cannot get subtype of PyPostProcessor"))? {
                post.set_single(template);
            }
        }
        Ok(())
    }
}

/// Sequence Processor
///
/// Args:
///     processors (:obj:`List[PostProcessor]`)
///         The processors that need to be chained
#[pyclass(extends=PyPostProcessor, module = "tokenizers.processors", name = "Sequence")]
pub struct PySequence {}

#[pymethods]
impl PySequence {
    #[new]
    #[pyo3(signature = (processors_py), text_signature = "(self, processors)")]
    fn new(processors_py: &Bound<'_, PyList>) -> PyResult<(Self, PyPostProcessor)> {
        let mut processors = Vec::with_capacity(processors_py.len());
        for n in processors_py.iter() {
            let processor: PyRef<PyPostProcessor> = n.extract()?;
            match &processor.processor {
                PyPostProcessorTypeWrapper::Sequence(inner) => {
                    processors.extend(inner.iter().cloned())
                }
                PyPostProcessorTypeWrapper::Single(inner) => processors.push(inner.clone()),
            }
        }
        Ok((
            PySequence {},
            PyPostProcessor::new(PyPostProcessorTypeWrapper::Sequence(processors)),
        ))
    }

    fn __getnewargs__<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyTuple>> {
        PyTuple::new(py, [PyList::empty(py)])
    }

    fn __getitem__(self_: PyRef<'_, Self>, py: Python<'_>, index: usize) -> PyResult<Py<PyAny>> {
        match &self_.as_ref().processor {
            PyPostProcessorTypeWrapper::Sequence(ref inner) => match inner.get(index) {
                Some(item) => {
                    PyPostProcessor::new(PyPostProcessorTypeWrapper::Single(item.clone()))
                        .get_as_subtype(py)
                }
                _ => Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                    "Index not found",
                )),
            },
            _ => Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "This processor is not a Sequence, it does not support __getitem__",
            )),
        }
    }

    fn __setitem__(self_: PyRef<'_, Self>, index: usize, value: Bound<'_, PyAny>) -> PyResult<()> {
        let processor: PyPostProcessor = value.extract()?;
        let PyPostProcessorTypeWrapper::Single(processor) = processor.processor else {
            return Err(PyException::new_err("processor should not be a sequence"));
        };

        match &self_.as_ref().processor {
            PyPostProcessorTypeWrapper::Sequence(inner) => match inner.get(index) {
                Some(item) => {
                    *item
                        .write()
                        .map_err(|_| PyException::new_err("RwLock synchronisation primitive is poisoned, cannot get subtype of PyPostProcessor"))? = processor
                        .read()
                        .map_err(|_| PyException::new_err("RwLock synchronisation primitive is poisoned, cannot get subtype of PyPostProcessor"))?
                        .clone();
                }
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                        "Index not found",
                    ))
                }
            },
            _ => {
                return Err(PyException::new_err(
                    "This processor is not a Sequence, it does not support __setitem__",
                ))
            }
        };
        Ok(())
    }
}

/// Processors Module
#[pymodule]
pub fn processors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPostProcessor>()?;
    m.add_class::<PyBertProcessing>()?;
    m.add_class::<PyRobertaProcessing>()?;
    m.add_class::<PyByteLevel>()?;
    m.add_class::<PyTemplateProcessing>()?;
    m.add_class::<PySequence>()?;
    Ok(())
}

#[cfg(test)]
mod test {
    use std::sync::{Arc, RwLock};

    use pyo3::prelude::*;
    use tk::processors::bert::BertProcessing;
    use tk::processors::PostProcessorWrapper;

    use crate::processors::{PyPostProcessor, PyPostProcessorTypeWrapper};

    #[test]
    fn get_subtype() {
        Python::with_gil(|py| {
            let py_proc = PyPostProcessor::new(PyPostProcessorTypeWrapper::Single(Arc::new(
                RwLock::new(BertProcessing::new(("SEP".into(), 0), ("CLS".into(), 1)).into()),
            )));
            let py_bert = py_proc.get_as_subtype(py).unwrap();
            assert_eq!(
                "BertProcessing",
                py_bert.bind(py).get_type().qualname().unwrap()
            );
        })
    }

    #[test]
    fn serialize() {
        let rs_processing = BertProcessing::new(("SEP".into(), 0), ("CLS".into(), 1));
        let rs_wrapper: PostProcessorWrapper = rs_processing.clone().into();
        let rs_processing_ser = serde_json::to_string(&rs_processing).unwrap();
        let rs_wrapper_ser = serde_json::to_string(&rs_wrapper).unwrap();

        let py_processing = PyPostProcessor::new(PyPostProcessorTypeWrapper::Single(Arc::new(
            RwLock::new(rs_wrapper),
        )));
        let py_ser = serde_json::to_string(&py_processing).unwrap();
        assert_eq!(py_ser, rs_processing_ser);
        assert_eq!(py_ser, rs_wrapper_ser);

        let py_processing: PyPostProcessor = serde_json::from_str(&rs_processing_ser).unwrap();
        match py_processing.processor {
            PyPostProcessorTypeWrapper::Single(inner) => match *inner.as_ref().read().unwrap() {
                PostProcessorWrapper::Bert(_) => (),
                _ => panic!("Expected Bert postprocessor."),
            },
            _ => panic!("Expected a single processor, got a sequence"),
        }

        let py_processing: PyPostProcessor = serde_json::from_str(&rs_wrapper_ser).unwrap();
        match py_processing.processor {
            PyPostProcessorTypeWrapper::Single(inner) => match *inner.as_ref().read().unwrap() {
                PostProcessorWrapper::Bert(_) => (),
                _ => panic!("Expected Bert postprocessor."),
            },
            _ => panic!("Expected a single processor, got a sequence"),
        };
    }
}
