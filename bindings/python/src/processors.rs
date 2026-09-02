//! Post-processors, expressed over the pipeline's canonical [`Template`] IR.
//!
//! The engine keeps one shape -- `prefix? $A infix? ($B suffix?)?` -- for both the single and the
//! pair case, and remembers only token *ids*. Two consequences show through to Python:
//!
//! - `BertProcessing`, `RobertaProcessing` and `ByteLevel` are constructors, not identities. They
//!   build the template their name implies, and a processor read back out of a tokenizer comes
//!   back as `TemplateProcessing` -- which is also the only `type` the writer emits.
//! - `single` and `pair` read back as the canonical piece list, not the string they were written
//!   as. Resolving a name to an id is one-way; the table is not kept.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::encoding::PyEncoding;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use tk::pipeline::{PipelinePostProcessor, PipelineToken, Template};
use tokenizers as tk;

fn poisoned() -> PyErr {
    exceptions::PyException::new_err(
        "RwLock synchronisation primitive is poisoned, cannot access PostProcessor",
    )
}

/// Base class for all post-processors
///
/// This class is not supposed to be instantiated directly. Instead, any implementation of
/// a PostProcessor will return an instance of this class when instantiated.
#[pyclass(
    dict,
    module = "tokenizers.processors",
    name = "PostProcessor",
    subclass,
    from_py_object
)]
#[derive(Clone)]
pub struct PyPostProcessor {
    processor: Arc<RwLock<PipelinePostProcessor>>,
}

impl From<PipelinePostProcessor> for PyPostProcessor {
    fn from(processor: PipelinePostProcessor) -> Self {
        PyPostProcessor {
            processor: Arc::new(RwLock::new(processor)),
        }
    }
}

impl PyPostProcessor {
    pub(crate) fn new(processor: Arc<RwLock<PipelinePostProcessor>>) -> Self {
        PyPostProcessor { processor }
    }

    /// Every canonical post-processor is a template, so this is the only subtype there is to
    /// hand back. It stays a method because the tokenizer reaches for it generically.
    pub(crate) fn get_as_subtype(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let base = self.clone();
        Ok(Py::new(py, (PyTemplateProcessing {}, base))?.into_any())
    }

    fn read(&self) -> PyResult<std::sync::RwLockReadGuard<'_, PipelinePostProcessor>> {
        self.processor.read().map_err(|_| poisoned())
    }
}

#[pymethods]
impl PyPostProcessor {
    fn __getstate__(&self, py: Python) -> PyResult<Py<PyAny>> {
        let json = to_json(&*self.read()?).to_string();
        Ok(PyBytes::new(py, json.as_bytes()).into())
    }

    fn __setstate__(&mut self, py: Python, state: Py<PyAny>) -> PyResult<()> {
        let bytes = state.extract::<&[u8]>(py)?;
        let value: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| {
            exceptions::PyException::new_err(format!(
                "Error while attempting to unpickle PostProcessor: {e}"
            ))
        })?;
        self.processor = Arc::new(RwLock::new(from_json(&value)?));
        Ok(())
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
        let processor = self.read()?;
        let template = if is_pair {
            &processor.pair
        } else {
            &processor.single
        };
        Ok(template.n_special())
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
    #[pyo3(
        signature = (encoding, pair = None, add_special_tokens = true) -> "Encoding"
    )]
    #[pyo3(text_signature = "(self, encoding, pair=None, add_special_tokens=True)")]
    fn process(
        &self,
        encoding: &PyEncoding,
        pair: Option<&PyEncoding>,
        add_special_tokens: bool,
    ) -> PyResult<PyEncoding> {
        let processor = self.read()?;
        let template = if pair.is_some() {
            &processor.pair
        } else {
            &processor.single
        };
        let s1 = encoding.encoding.ids().to_vec();
        let s2 = pair.map(|e| e.encoding.ids().to_vec());
        let out = if add_special_tokens {
            template.post_process::<true>(s1, s2)
        } else {
            template.post_process::<false>(s1, s2)
        };
        Ok(out.into())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("TemplateProcessing({})", to_json(&*self.read()?)))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(to_json(&*self.read()?).to_string())
    }
}

// ---------------------------------------------------------------------------
// The canonical JSON shape, which is what pickling and `str()` round-trip through.
// It mirrors `tk_serialize`'s reader and writer; those are crate-private there.
// ---------------------------------------------------------------------------

fn to_json(processor: &PipelinePostProcessor) -> serde_json::Value {
    serde_json::json!({
        "type": "TemplateProcessing",
        "single": pieces_to_json(&processor.single),
        "pair": pieces_to_json(&processor.pair),
    })
}

/// One template as an array of pieces: `{"seq": "A"}` for a sequence, `{"ids": [...]}` for a run
/// of special tokens, and `type_id` only when it is not the `0` a piece defaults to.
fn pieces_to_json(template: &Template) -> Vec<serde_json::Value> {
    let mut out = Vec::new();
    push_specials(&mut out, &template.prefix);
    out.push(seq_piece("A", template.a_type_id));
    push_specials(&mut out, &template.infix);
    if let Some(type_id) = template.b_type_id {
        out.push(seq_piece("B", type_id));
    }
    push_specials(&mut out, &template.suffix);
    out
}

fn seq_piece(seq: &str, type_id: u8) -> serde_json::Value {
    let mut piece = serde_json::Map::new();
    piece.insert("seq".into(), seq.into());
    if type_id != 0 {
        piece.insert("type_id".into(), type_id.into());
    }
    piece.into()
}

/// Consecutive specials sharing a `type_id` go back out as the one `{"ids": [...]}` piece they
/// most likely came in as.
fn push_specials(out: &mut Vec<serde_json::Value>, specials: &[(PipelineToken, u8)]) {
    let mut rest = specials;
    while let Some(&(_, type_id)) = rest.first() {
        let n = rest.iter().take_while(|&&(_, id)| id == type_id).count();
        let ids: Vec<u32> = rest[..n].iter().map(|&(token, _)| token.id()).collect();
        let mut piece = serde_json::Map::new();
        piece.insert("ids".into(), ids.into());
        if type_id != 0 {
            piece.insert("type_id".into(), type_id.into());
        }
        out.push(piece.into());
        rest = &rest[n..];
    }
}

fn from_json(value: &serde_json::Value) -> PyResult<PipelinePostProcessor> {
    let template_for = |key: &str| -> PyResult<Template> {
        let array = value
            .get(key)
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| {
                exceptions::PyValueError::new_err(format!("TemplateProcessing is missing `{key}`"))
            })?;
        let mut pieces = Vec::with_capacity(array.len());
        for piece in array {
            let type_id = piece
                .get("type_id")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as u8;
            if let Some(seq) = piece.get("seq").and_then(serde_json::Value::as_str) {
                pieces.push(Piece::Seq {
                    b: seq == "B",
                    type_id,
                });
            } else if let Some(ids) = piece.get("ids").and_then(serde_json::Value::as_array) {
                let ids = ids
                    .iter()
                    .map(|id| {
                        id.as_u64().map(|id| id as u32).ok_or_else(|| {
                            exceptions::PyValueError::new_err("a template piece has a bad id")
                        })
                    })
                    .collect::<PyResult<Vec<u32>>>()?;
                pieces.push(Piece::Ids { ids, type_id });
            } else {
                return Err(exceptions::PyValueError::new_err(
                    "a template piece has neither `seq` nor `ids`",
                ));
            }
        }
        build_template(&pieces, key == "pair")
    };

    Ok(PipelinePostProcessor {
        single: template_for("single")?,
        pair: template_for("pair")?,
    })
}

// ---------------------------------------------------------------------------
// Parsing the Python-facing template spelling into the canonical shape.
// ---------------------------------------------------------------------------

/// One parsed piece: a reference to an input sequence, or a run of special-token ids.
enum Piece {
    Seq { b: bool, type_id: u8 },
    Ids { ids: Vec<u32>, type_id: u8 },
}

/// `<identifier>(:<type_id>)?`, where the identifier is `$A`/`$B`/`$<type_id>` for a sequence and
/// a special token's name otherwise.
fn parse_piece(token: &str, specials: &HashMap<String, Vec<u32>>) -> PyResult<Piece> {
    let (identifier, type_id) = match token.rsplit_once(':') {
        Some((identifier, id)) => {
            let id = id.parse::<u8>().map_err(|_| {
                exceptions::PyValueError::new_err(format!("bad type_id in template piece {token:?}"))
            })?;
            (identifier, Some(id))
        }
        None => (token, None),
    };

    if let Some(rest) = identifier.strip_prefix('$') {
        // `$A`/`$B` name the sequence; `$0`/`$1` name a type id on sequence A; bare `$` is A.
        return match rest {
            "" | "A" | "a" => Ok(Piece::Seq {
                b: false,
                type_id: type_id.unwrap_or(0),
            }),
            "B" | "b" => Ok(Piece::Seq {
                b: true,
                type_id: type_id.unwrap_or(0),
            }),
            digits => {
                let parsed = digits.parse::<u8>().map_err(|_| {
                    exceptions::PyValueError::new_err(format!(
                        "unknown template sequence {identifier:?}"
                    ))
                })?;
                Ok(Piece::Seq {
                    b: false,
                    type_id: type_id.unwrap_or(parsed),
                })
            }
        };
    }

    let ids = specials.get(identifier).ok_or_else(|| {
        exceptions::PyValueError::new_err(format!(
            "the template uses the special token {identifier:?}, which is not in `special_tokens`"
        ))
    })?;
    Ok(Piece::Ids {
        ids: ids.clone(),
        type_id: type_id.unwrap_or(0),
    })
}

/// Place the pieces into the one shape the engine holds: specials land before A, between A and B,
/// or after the last sequence -- the only three places a piece can be.
fn build_template(pieces: &[Piece], is_pair: bool) -> PyResult<Template> {
    let mut template = Template::default();
    let (mut prefix, mut infix, mut suffix) = (Vec::new(), Vec::new(), Vec::new());
    let mut seen_a = false;

    for piece in pieces {
        match piece {
            Piece::Seq { b: false, type_id } if !seen_a => {
                seen_a = true;
                template.a_type_id = *type_id;
            }
            Piece::Seq { b: true, type_id } if seen_a && template.b_type_id.is_none() => {
                template.b_type_id = Some(*type_id);
            }
            Piece::Seq { b, .. } => {
                let seq = if *b { "B" } else { "A" };
                return Err(exceptions::PyValueError::new_err(format!(
                    "not supported: template references sequence {seq} out of order or more than once"
                )));
            }
            Piece::Ids { ids, type_id } => {
                let dst = match (seen_a, template.b_type_id.is_some(), is_pair) {
                    (false, ..) => &mut prefix,
                    (true, false, true) => &mut infix,
                    _ => &mut suffix,
                };
                dst.extend(ids.iter().map(|&id| (PipelineToken::from(id), *type_id)));
            }
        }
    }

    if !seen_a {
        return Err(exceptions::PyValueError::new_err(
            "not supported: template does not reference sequence A",
        ));
    }
    if is_pair != template.b_type_id.is_some() {
        let key = if is_pair { "pair" } else { "single" };
        return Err(exceptions::PyValueError::new_err(format!(
            "not supported: `{key}` template references the wrong sequences"
        )));
    }
    (template.prefix, template.infix, template.suffix) =
        (prefix.into(), infix.into(), suffix.into());
    Ok(template)
}

/// A template as Python spells it: a whitespace-delimited string, or a list of pieces.
struct PyTemplateSpec(Vec<String>);

impl<'a, 'py> FromPyObject<'a, 'py> for PyTemplateSpec {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(s) = ob.extract::<String>() {
            Ok(Self(s.split_whitespace().map(String::from).collect()))
        } else if let Ok(v) = ob.extract::<Vec<String>>() {
            Ok(Self(v))
        } else {
            Err(exceptions::PyTypeError::new_err(
                "Expected Union[str, List[str]]",
            ))
        }
    }
}

/// The `special_tokens` table: a name, and the ids it stands for.
struct PySpecialTokens(HashMap<String, Vec<u32>>);

impl<'a, 'py> FromPyObject<'a, 'py> for PySpecialTokens {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let mut table = HashMap::new();
        for item in ob.try_iter()? {
            let item = item?;
            if let Ok((name, id)) = item.extract::<(String, u32)>() {
                table.insert(name, vec![id]);
            } else if let Ok((id, name)) = item.extract::<(u32, String)>() {
                table.insert(name, vec![id]);
            } else if let Ok(dict) = item.cast::<PyDict>() {
                let name = dict
                    .get_item("id")?
                    .ok_or_else(|| exceptions::PyValueError::new_err("`id` must be specified"))?
                    .extract::<String>()?;
                let ids = dict
                    .get_item("ids")?
                    .ok_or_else(|| exceptions::PyValueError::new_err("`ids` must be specified"))?
                    .extract::<Vec<u32>>()?;
                // `tokens` is accepted for compatibility; only the ids reach the engine.
                if let Some(tokens) = dict.get_item("tokens")? {
                    let tokens = tokens.extract::<Vec<String>>()?;
                    if tokens.len() != ids.len() {
                        return Err(exceptions::PyValueError::new_err(
                            "`ids` and `tokens` must have the same length",
                        ));
                    }
                }
                table.insert(name, ids);
            } else {
                return Err(exceptions::PyTypeError::new_err(
                    "Expected Union[Tuple[str, int], Tuple[int, str], dict]",
                ));
            }
        }
        Ok(Self(table))
    }
}

// ---------------------------------------------------------------------------
// The constructors.
// ---------------------------------------------------------------------------

/// A run of ids at one type id, as the engine holds it.
fn run(ids: &[u32], type_id: u8) -> Box<[(PipelineToken, u8)]> {
    ids.iter()
        .map(|&id| (PipelineToken::from(id), type_id))
        .collect()
}

/// This post-processor takes care of adding the special tokens needed by
/// a Bert model:
///
///     - a SEP token
///     - a CLS token
///
/// It builds the templates :obj:`[CLS] $A [SEP]` and :obj:`[CLS] $A [SEP] $B:1 [SEP]:1`. The
/// result is a :class:`~tokenizers.processors.TemplateProcessing`; only the string
/// representation of the tokens is dropped, since the engine keeps ids.
///
/// Args:
///     sep (:obj:`Tuple[str, int]`):
///         A tuple with the string representation of the SEP token, and its id
///
///     cls (:obj:`Tuple[str, int]`):
///         A tuple with the string representation of the CLS token, and its id
///
/// Example::
///
///     >>> from tokenizers.processors import BertProcessing
///     >>> processor = BertProcessing(("[SEP]", 102), ("[CLS]", 101))
///     >>> processor.process(encoding)
///     # Encoding with [CLS] at start and [SEP] at end
///
#[pyclass(extends=PyPostProcessor, module = "tokenizers.processors", name = "BertProcessing")]
pub struct PyBertProcessing {}

#[pymethods]
impl PyBertProcessing {
    #[new]
    #[pyo3(text_signature = "(self, sep, cls)")]
    fn new(sep: (String, u32), cls: (String, u32)) -> PyClassInitializer<Self> {
        let (sep, cls) = (sep.1, cls.1);
        let processor = PipelinePostProcessor {
            // `[CLS] $A [SEP]`
            single: Template {
                prefix: run(&[cls], 0),
                suffix: run(&[sep], 0),
                ..Template::default()
            },
            // `[CLS] $A [SEP] $B@1 [SEP]@1`
            pair: Template {
                prefix: run(&[cls], 0),
                infix: run(&[sep], 0),
                suffix: run(&[sep], 1),
                a_type_id: 0,
                b_type_id: Some(1),
            },
        };
        PyClassInitializer::<PyPostProcessor>::from(PyPostProcessor::from(processor))
            .add_subclass(PyBertProcessing {})
    }

    fn __getnewargs__<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyTuple>> {
        PyTuple::new(py, [("", 0), ("", 0)])
    }
}

/// This post-processor takes care of adding the special tokens needed by
/// a Roberta model:
///
///     - a SEP token
///     - a CLS token
///
/// It builds the templates :obj:`<s> $A </s>` and :obj:`<s> $A </s></s> $B </s>`, all at type
/// id 0. The result is a :class:`~tokenizers.processors.TemplateProcessing`.
///
/// Args:
///     sep (:obj:`Tuple[str, int]`):
///         A tuple with the string representation of the SEP token, and its id
///
///     cls (:obj:`Tuple[str, int]`):
///         A tuple with the string representation of the CLS token, and its id
///
///     trim_offsets (:obj:`bool`, `optional`, defaults to :obj:`True`):
///         Accepted for compatibility and ignored. The pipeline does not keep offsets, so
///         there is nothing to trim.
///
///     add_prefix_space (:obj:`bool`, `optional`, defaults to :obj:`True`):
///         Accepted for compatibility and ignored, for the same reason.
///
/// Example::
///
///     >>> from tokenizers.processors import RobertaProcessing
///     >>> processor = RobertaProcessing(("</s>", 2), ("<s>", 0))
///     >>> processor.process(encoding)
///     # Encoding with <s> at start and </s> at end
///
#[pyclass(extends=PyPostProcessor, module = "tokenizers.processors", name = "RobertaProcessing")]
pub struct PyRobertaProcessing {}

#[pymethods]
impl PyRobertaProcessing {
    #[new]
    #[pyo3(
        signature = (sep, cls, trim_offsets = true, add_prefix_space = true),
        text_signature = "(self, sep, cls, trim_offsets=True, add_prefix_space=True)"
    )]
    fn new(
        sep: (String, u32),
        cls: (String, u32),
        trim_offsets: bool,
        add_prefix_space: bool,
    ) -> PyClassInitializer<Self> {
        let _ = (trim_offsets, add_prefix_space);
        let (sep, cls) = (sep.1, cls.1);
        let processor = PipelinePostProcessor {
            // `<s> $A </s>`
            single: Template {
                prefix: run(&[cls], 0),
                suffix: run(&[sep], 0),
                ..Template::default()
            },
            // `<s> $A </s></s> $B </s>`, all at type id 0
            pair: Template {
                prefix: run(&[cls], 0),
                infix: run(&[sep, sep], 0),
                suffix: run(&[sep], 0),
                a_type_id: 0,
                b_type_id: Some(0),
            },
        };
        PyClassInitializer::<PyPostProcessor>::from(PyPostProcessor::from(processor))
            .add_subclass(PyRobertaProcessing {})
    }

    fn __getnewargs__<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyTuple>> {
        PyTuple::new(py, [("", 0), ("", 0)])
    }
}

/// A post-processor that weaves nothing.
///
/// It exists because a `ByteLevel` post-processor only ever re-tagged offsets, and the pipeline
/// does not keep offsets. Constructing one gives the frame that adds no tokens and no type ids,
/// which is also what a missing `post_processor` gives.
///
/// Args:
///     trim_offsets (:obj:`bool`, `optional`):
///         Accepted for compatibility and ignored.
///
///     add_prefix_space (:obj:`bool`, `optional`):
///         Accepted for compatibility and ignored.
///
///     use_regex (:obj:`bool`, `optional`):
///         Accepted for compatibility and ignored.
///
/// Example::
///
///     >>> from tokenizers.processors import ByteLevel
///     >>> processor = ByteLevel()
///
#[pyclass(extends=PyPostProcessor, module = "tokenizers.processors", name = "ByteLevel")]
pub struct PyByteLevel {}

#[pymethods]
impl PyByteLevel {
    #[new]
    #[pyo3(
        signature = (add_prefix_space = None, trim_offsets = None, use_regex = None, **_kwargs),
        text_signature = "(self, add_prefix_space=None, trim_offsets=None, use_regex=None)"
    )]
    fn new(
        add_prefix_space: Option<bool>,
        trim_offsets: Option<bool>,
        use_regex: Option<bool>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyClassInitializer<Self> {
        let _ = (add_prefix_space, trim_offsets, use_regex);
        PyClassInitializer::<PyPostProcessor>::from(PyPostProcessor::from(
            PipelinePostProcessor::default(),
        ))
        .add_subclass(PyByteLevel {})
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
/// The engine holds one shape, ``prefix? $A infix? ($B suffix?)?``, so a template must
/// reference ``$A`` exactly once, reference ``$B`` at most once and after ``$A``, and place its
/// special tokens only before ``$A``, between the two sequences, or after the last one. The
/// ``single`` template must not reference ``$B``, and the ``pair`` template must. Anything else
/// raises a :obj:`ValueError` rather than being silently reshaped.
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
    #[pyo3(
        signature = (single = None, pair = None, special_tokens = None),
        text_signature = "(self, single=None, pair=None, special_tokens=None)"
    )]
    fn new(
        single: Option<PyTemplateSpec>,
        pair: Option<PyTemplateSpec>,
        special_tokens: Option<PySpecialTokens>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let specials = special_tokens.map(|t| t.0).unwrap_or_default();

        let build = |spec: Option<PyTemplateSpec>, is_pair: bool| -> PyResult<Template> {
            let Some(spec) = spec else {
                // The template that reproduces the sequence, which is what an unset one means.
                return Ok(Template {
                    b_type_id: is_pair.then_some(1),
                    ..Template::default()
                });
            };
            let pieces = spec
                .0
                .iter()
                .map(|token| parse_piece(token, &specials))
                .collect::<PyResult<Vec<Piece>>>()?;
            build_template(&pieces, is_pair)
        };

        let processor = PipelinePostProcessor {
            single: build(single, false)?,
            pair: build(pair, true)?,
        };

        Ok(
            PyClassInitializer::<PyPostProcessor>::from(PyPostProcessor::from(processor))
                .add_subclass(PyTemplateProcessing {}),
        )
    }

    /// The single-sequence template, as the canonical list of pieces.
    #[getter]
    fn get_single(self_: PyRef<Self>) -> PyResult<String> {
        let processor = self_.as_ref().read()?;
        Ok(serde_json::Value::from(pieces_to_json(&processor.single)).to_string())
    }

    /// The pair template, as the canonical list of pieces.
    #[getter]
    fn get_pair(self_: PyRef<Self>) -> PyResult<String> {
        let processor = self_.as_ref().read()?;
        Ok(serde_json::Value::from(pieces_to_json(&processor.pair)).to_string())
    }
}

/// Processors Module
#[pymodule(gil_used = false)]
pub mod processors {
    #[pymodule_export]
    pub use super::PyBertProcessing;
    #[pymodule_export]
    pub use super::PyByteLevel;
    #[pymodule_export]
    pub use super::PyPostProcessor;
    #[pymodule_export]
    pub use super::PyRobertaProcessing;
    #[pymodule_export]
    pub use super::PyTemplateProcessing;
}

#[cfg(test)]
mod test {
    use super::*;

    /// `[CLS] $A [SEP]` / `[CLS] $A [SEP] $B:1 [SEP]:1`, the shape the docstring advertises.
    fn bert() -> PipelinePostProcessor {
        PipelinePostProcessor {
            single: Template {
                prefix: run(&[1], 0),
                suffix: run(&[0], 0),
                ..Template::default()
            },
            pair: Template {
                prefix: run(&[1], 0),
                infix: run(&[0], 0),
                suffix: run(&[0], 1),
                a_type_id: 0,
                b_type_id: Some(1),
            },
        }
    }

    fn parse(single: &str, pair: &str, specials: &[(&str, u32)]) -> PyResult<PipelinePostProcessor> {
        let table: HashMap<String, Vec<u32>> = specials
            .iter()
            .map(|&(name, id)| (name.to_string(), vec![id]))
            .collect();
        let build = |spec: &str, is_pair: bool| -> PyResult<Template> {
            let pieces = spec
                .split_whitespace()
                .map(|token| parse_piece(token, &table))
                .collect::<PyResult<Vec<Piece>>>()?;
            build_template(&pieces, is_pair)
        };
        Ok(PipelinePostProcessor {
            single: build(single, false)?,
            pair: build(pair, true)?,
        })
    }

    #[test]
    fn the_docstring_template_lowers_to_the_documented_shape() {
        let parsed = parse(
            "[CLS] $0 [SEP]",
            "[CLS] $A [SEP] $B:1 [SEP]:1",
            &[("[CLS]", 1), ("[SEP]", 0)],
        )
        .unwrap();
        let expected = bert();
        assert_eq!(parsed.single.prefix, expected.single.prefix);
        assert_eq!(parsed.single.suffix, expected.single.suffix);
        assert_eq!(parsed.pair.prefix, expected.pair.prefix);
        assert_eq!(parsed.pair.infix, expected.pair.infix);
        assert_eq!(parsed.pair.suffix, expected.pair.suffix);
        assert_eq!(parsed.pair.b_type_id, Some(1));
    }

    #[test]
    fn json_round_trips() {
        let json = to_json(&bert());
        let back = from_json(&json).unwrap();
        assert_eq!(to_json(&back), json);
        assert_eq!(json["type"], "TemplateProcessing");
    }

    #[test]
    fn shapes_the_engine_cannot_hold_are_rejected() {
        // A special token in the middle of a sequence, a `$B` before `$A`, `$A` twice, a `single`
        // that references `$B`, and a `pair` that does not.
        for (single, pair) in [
            ("$A [SEP] $A", "$A $B"),
            ("$A", "$B $A"),
            ("$A $A", "$A $B"),
            ("$A $B", "$A $B"),
            ("$A", "$A"),
        ] {
            assert!(
                parse(single, pair, &[("[SEP]", 0)]).is_err(),
                "expected {single:?} / {pair:?} to be rejected"
            );
        }
    }

    #[test]
    fn an_unknown_special_token_names_itself() {
        let err = parse("[CLS] $A", "$A $B", &[]).unwrap_err();
        assert!(
            format!("{err}").contains("[CLS]"),
            "error should name the token: {err}"
        );
    }
}
