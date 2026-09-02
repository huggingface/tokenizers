//! Post-processors. One class, because the engine holds one shape.
//!
//! `BertProcessing`, `RobertaProcessing`, `ByteLevel` and a `Sequence` wrapper were spellings of a
//! template, so they are tk-convert's to rewrite, not classes here. Everything below builds the
//! legacy JSON its arguments describe and hands it to `canonicalize_post_processor` and
//! `post_processor_from_json` -- so the bindings own no parser, no validator and no writer.

use std::sync::{Arc, RwLock};

use crate::encoding::PyEncoding;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::*;
use serde_json::{Map, Value, json};
use tk::pipeline::PipelinePostProcessor;
use tokenizers as tk;

fn py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
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

    /// Every canonical post-processor is a template, so that is the only subtype to hand back.
    pub(crate) fn get_as_subtype(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(Py::new(py, (PyTemplateProcessing {}, self.clone()))?.into_any())
    }

    fn read(&self) -> PyResult<std::sync::RwLockReadGuard<'_, PipelinePostProcessor>> {
        self.processor
            .read()
            .map_err(|_| PyException::new_err("PostProcessor lock is poisoned"))
    }

    fn json(&self) -> PyResult<String> {
        tk::post_processor_to_json(&self.read()?).map_err(py_err)
    }
}

#[pymethods]
impl PyPostProcessor {
    fn __getstate__(&self, py: Python) -> PyResult<Py<PyAny>> {
        Ok(PyBytes::new(py, self.json()?.as_bytes()).into())
    }

    fn __setstate__(&mut self, py: Python, state: Py<PyAny>) -> PyResult<()> {
        let json = std::str::from_utf8(state.extract::<&[u8]>(py)?).map_err(py_err)?;
        self.processor = Arc::new(RwLock::new(
            tk::post_processor_from_json(json).map_err(py_err)?,
        ));
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
        Ok(if is_pair {
            processor.pair.n_special()
        } else {
            processor.single.n_special()
        })
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
        let a = encoding.encoding.ids().to_vec();
        let b = pair.map(|e| e.encoding.ids().to_vec());
        Ok(if add_special_tokens {
            template.post_process::<true>(a, b)
        } else {
            template.post_process::<false>(a, b)
        }
        .into())
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("TemplateProcessing({})", self.json()?))
    }

    fn __str__(&self) -> PyResult<String> {
        self.json()
    }
}

/// One template piece, in the legacy spelling tk-convert resolves: `$A`/`$B`/`$0` name a sequence,
/// anything else names a special token to look up in `special_tokens`, and either may carry
/// `:<type_id>`.
fn piece(token: &str) -> Value {
    // Only a suffix that parses is a `type_id`; a token that merely contains a colon keeps it.
    let (name, type_id) = match token.rsplit_once(':') {
        Some((name, id)) => match id.parse::<u64>() {
            Ok(id) => (name, Some(id)),
            Err(_) => (token, None),
        },
        None => (token, None),
    };
    let (key, id, default) = match name.strip_prefix('$') {
        // `$0`/`$1` name a type id on sequence A; `$`, `$A` and `$B` name the sequence itself.
        Some(seq) => match seq {
            "" | "A" | "a" => ("Sequence", "A".to_string(), 0),
            "B" | "b" => ("Sequence", "B".to_string(), 0),
            digits => ("Sequence", "A".to_string(), digits.parse().unwrap_or(0)),
        },
        None => ("SpecialToken", name.to_string(), 0),
    };
    json!({key: {"id": id, "type_id": type_id.unwrap_or(default)}})
}

/// A template as Python spells it: a whitespace-delimited string, or a list of pieces.
fn pieces(spec: Option<Vec<String>>, default: &str) -> Value {
    let spec = spec.unwrap_or_else(|| default.split(' ').map(String::from).collect());
    Value::Array(spec.iter().map(|t| piece(t)).collect())
}

fn template_spec(ob: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<String>>> {
    let Some(ob) = ob else { return Ok(None) };
    if let Ok(s) = ob.extract::<String>() {
        Ok(Some(s.split_whitespace().map(String::from).collect()))
    } else {
        Ok(Some(ob.extract::<Vec<String>>()?))
    }
}

/// The `special_tokens` table, in the legacy shape: `name -> {"ids": [...]}`.
fn special_tokens(ob: Option<&Bound<'_, PyAny>>) -> PyResult<Value> {
    let mut table = Map::new();
    let Some(ob) = ob else {
        return Ok(Value::Object(table));
    };
    for item in ob.try_iter()? {
        let item = item?;
        let (name, ids) = if let Ok((name, id)) = item.extract::<(String, u32)>() {
            (name, vec![id])
        } else if let Ok((id, name)) = item.extract::<(u32, String)>() {
            (name, vec![id])
        } else if let Ok(d) = item.cast::<PyDict>() {
            let get = |key: &str| {
                d.get_item(key)?
                    .ok_or_else(|| PyValueError::new_err(format!("`{key}` must be specified")))
            };
            (
                get("id")?.extract::<String>()?,
                get("ids")?.extract::<Vec<u32>>()?,
            )
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Expected Union[Tuple[str, int], Tuple[int, str], dict]",
            ));
        };
        table.insert(name.clone(), json!({"id": name, "ids": ids}));
    }
    Ok(Value::Object(table))
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
/// The engine holds one shape, ``prefix? $A infix? ($B suffix?)?``, so ``single`` must reference
/// ``$A`` once and not ``$B``, ``pair`` must reference both in that order, and special tokens go
/// only before ``$A``, between the sequences, or after the last one. Anything else raises a
/// :obj:`ValueError` rather than being silently reshaped.
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
        single: Option<&Bound<'_, PyAny>>,
        pair: Option<&Bound<'_, PyAny>>,
        special_tokens: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        // An unset template is the one that reproduces the sequence.
        let mut node = json!({
            "type": "TemplateProcessing",
            "single": pieces(template_spec(single)?, "$A"),
            "pair": pieces(template_spec(pair)?, "$A $B:1"),
            "special_tokens": special_tokens(special_tokens)?,
        });
        tk::canonicalize_post_processor(&mut node).map_err(py_err)?;
        let processor = tk::post_processor_from_json(&node.to_string()).map_err(py_err)?;
        Ok(
            PyClassInitializer::<PyPostProcessor>::from(PyPostProcessor::from(processor))
                .add_subclass(PyTemplateProcessing {}),
        )
    }
}

/// Processors Module
#[pymodule(gil_used = false)]
pub mod processors {
    #[pymodule_export]
    pub use super::PyPostProcessor;
    #[pymodule_export]
    pub use super::PyTemplateProcessing;
}

#[cfg(test)]
mod test {
    use super::*;

    /// The docstring's own example, down to the ids the engine ends up holding.
    #[test]
    fn the_documented_template_lowers_to_the_documented_frame() {
        let mut node = json!({
            "type": "TemplateProcessing",
            "single": pieces(Some(vec!["[CLS]".into(), "$0".into(), "[SEP]".into()]), "$A"),
            "pair": pieces(Some("[CLS] $A [SEP] $B:1 [SEP]:1".split(' ').map(String::from).collect()), ""),
            "special_tokens": json!({
                "[CLS]": {"id": "[CLS]", "ids": [1]},
                "[SEP]": {"id": "[SEP]", "ids": [0]},
            }),
        });
        tk::canonicalize_post_processor(&mut node).unwrap();
        let p = tk::post_processor_from_json(&node.to_string()).unwrap();

        assert_eq!(p.single.n_special(), 2);
        assert_eq!(p.pair.b_type_id, Some(1));
        assert_eq!(p.pair.suffix.len(), 1);
        // `[SEP]:1` in the suffix is the only thing tagged in the single-sequence direction.
        assert!(!p.single.has_type_ids());
        assert!(p.pair.has_type_ids());
    }

    /// The shapes the engine cannot hold are refused by the reader, not reshaped here.
    #[test]
    fn a_shape_the_engine_cannot_hold_is_refused() {
        for (single, pair) in [("$A [SEP] $A", "$A $B"), ("$A", "$B $A"), ("$A $B", "$A $B")] {
            let mut node = json!({
                "type": "TemplateProcessing",
                "single": pieces(Some(single.split(' ').map(String::from).collect()), ""),
                "pair": pieces(Some(pair.split(' ').map(String::from).collect()), ""),
                "special_tokens": json!({"[SEP]": {"id": "[SEP]", "ids": [0]}}),
            });
            let read = tk::canonicalize_post_processor(&mut node)
                .map_err(py_err)
                .and_then(|()| tk::post_processor_from_json(&node.to_string()).map_err(py_err));
            assert!(read.is_err(), "expected {single:?} / {pair:?} to be refused");
        }
    }

    /// A name with no entry in the table cannot become an id, and says which name.
    #[test]
    fn an_unknown_special_token_names_itself() {
        let mut node = json!({
            "type": "TemplateProcessing",
            "single": pieces(Some(vec!["[CLS]".into(), "$A".into()]), ""),
            "pair": pieces(None, "$A $B:1"),
            "special_tokens": json!({}),
        });
        let err = tk::canonicalize_post_processor(&mut node).unwrap_err();
        assert!(format!("{err}").contains("[CLS]"), "{err}");
    }
}
