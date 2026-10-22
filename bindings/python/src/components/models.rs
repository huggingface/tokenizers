use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use pyo3::prelude::*;
use serde_json::{Value, json};

use crate::error::err;
use crate::spec::{ComponentKey, JsonObject};

use super::{Component, ComponentKind, ComponentSpec, JsonFields};

/// Base class for all tokenizer models
#[pyclass(frozen, subclass, extends = Component, module = "tokenizers.models")]
#[derive(Default)]
pub struct Model;

#[pymethods]
impl Model {
    /// Rebuilds a component from its `__getstate__`
    #[staticmethod]
    fn _unpickle(py: Python<'_>, state: &[u8]) -> PyResult<Py<Self>> {
        Self::from_state(py, state)
    }
}

impl ComponentKind for Model {
    const JSON_KEY: ComponentKey = ComponentKey::Model;

    fn from_spec(py: Python<'_>, spec: Arc<ComponentSpec>) -> PyResult<Py<Self>> {
        let json = spec.json();
        let model_type = json.get("type").unwrap_or(&Value::Null);
        match model_type.as_str() {
            Some("BPE") => Self::subclass_from_spec(py, spec, Bpe),
            Some("WordPiece") => Self::subclass_from_spec(py, spec, WordPiece),
            Some("Unigram") => Self::subclass_from_spec(py, spec, Unigram),
            Some("WordLevel") => Self::subclass_from_spec(py, spec, WordLevel),
            _ => Err(err(format!("Unknown model \"type\": {model_type}"))),
        }
    }
}

/// A vocabulary: `{token: id}`, or the path of a file holding one.
#[derive(FromPyObject)]
enum Vocab {
    Dict(HashMap<String, u32>),
    Path(PathBuf),
}

impl Vocab {
    /// A path names a `vocab.json`.
    fn read_json(self) -> PyResult<JsonObject> {
        let vocab = match self {
            Self::Dict(vocab) => vocab,
            Self::Path(path) => {
                serde_json::from_str(&std::fs::read_to_string(path)?).map_err(err)?
            }
        };
        Ok(by_id(vocab))
    }

    /// A path names a `vocab.txt`: one token per line, its id being the line number.
    fn read_lines(self) -> PyResult<JsonObject> {
        let vocab = match self {
            Self::Dict(vocab) => vocab,
            Self::Path(path) => {
                let text = std::fs::read_to_string(path)?;
                let lines = text.lines().zip(0..);
                lines
                    .map(|(line, id)| (line.trim_end().to_owned(), id))
                    .collect()
            }
        };
        Ok(by_id(vocab))
    }
}

/// `{token: id}` in id order, the order 0.23 writes.
fn by_id(vocab: HashMap<String, u32>) -> JsonObject {
    let mut entries: Vec<_> = vocab.into_iter().collect();
    entries.sort_unstable_by(|(a, a_id), (b, b_id)| (a_id, a).cmp(&(b_id, b)));
    entries
        .into_iter()
        .map(|(token, id)| (token, id.into()))
        .collect()
}

/// BPE merges: `(left, right)` pairs, or the path of a `merges.txt`.
#[derive(FromPyObject)]
enum Merges {
    Pairs(Vec<(String, String)>),
    Path(PathBuf),
}

impl Merges {
    /// A `merges.txt` holds one `left right` pair per line, after a `#version` line.
    fn read(self) -> PyResult<Vec<(String, String)>> {
        let path = match self {
            Self::Pairs(pairs) => return Ok(pairs),
            Self::Path(path) => path,
        };
        let text = std::fs::read_to_string(&path)?;
        let lines = text
            .lines()
            .zip(1..)
            .filter(|(line, _)| !line.starts_with("#version"));
        lines
            .map(
                |(line, number)| match line.split(' ').collect::<Vec<_>>()[..] {
                    [left, right] => Ok((left.to_owned(), right.to_owned())),
                    _ => Err(err(format!(
                        "{}: line {number} is not a `left right` pair",
                        path.display()
                    ))),
                },
            )
            .collect()
    }
}

/// A byte-pair encoding model.
#[pyclass(frozen, extends = Model, name = "BPE", module = "tokenizers.models")]
pub struct Bpe;

#[pymethods]
impl Bpe {
    #[new]
    #[pyo3(signature = (vocab=None, merges=None, *, dropout=None, unk_token=None, continuing_subword_prefix=None, end_of_word_suffix=None, fuse_unk=false, byte_fallback=false, ignore_merges=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        vocab: Option<Vocab>,
        merges: Option<Merges>,
        dropout: Option<f64>,
        unk_token: Option<String>,
        continuing_subword_prefix: Option<String>,
        end_of_word_suffix: Option<String>,
        fuse_unk: bool,
        byte_fallback: bool,
        ignore_merges: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        let json = json!({
            "type": "BPE",
            "dropout": dropout,
            "unk_token": unk_token,
            "continuing_subword_prefix": continuing_subword_prefix,
            "end_of_word_suffix": end_of_word_suffix,
            "fuse_unk": fuse_unk,
            "byte_fallback": byte_fallback,
            "ignore_merges": ignore_merges,
            "vocab": vocab.map(Vocab::read_json).transpose()?.unwrap_or_default(),
            "merges": merges.map(Merges::read).transpose()?.unwrap_or_default(),
        });
        Ok(Model::init(json, Self))
    }

    #[getter]
    fn unk_token(slf: &Bound<'_, Self>) -> PyResult<Option<String>> {
        slf.field("unk_token")
    }

    #[getter]
    fn continuing_subword_prefix(slf: &Bound<'_, Self>) -> PyResult<Option<String>> {
        slf.field("continuing_subword_prefix")
    }

    #[getter]
    fn end_of_word_suffix(slf: &Bound<'_, Self>) -> PyResult<Option<String>> {
        slf.field("end_of_word_suffix")
    }

    /// Whether a word found whole in the vocab skips the merges.
    #[getter]
    fn ignore_merges(slf: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(slf.field::<Option<bool>>("ignore_merges")?.unwrap_or(false))
    }

    #[setter]
    fn set_ignore_merges(slf: &Bound<'_, Self>, ignore_merges: bool) {
        slf.set_field("ignore_merges", ignore_merges);
    }
}

/// BERT's model: a word becomes its longest prefix in the vocab, then the longest continuations,
/// each written with `continuing_subword_prefix`.
#[pyclass(frozen, extends = Model, module = "tokenizers.models")]
pub struct WordPiece;

#[pymethods]
impl WordPiece {
    #[new]
    #[pyo3(signature = (vocab=None, *, unk_token="[UNK]"))]
    fn new(vocab: Option<Vocab>, unk_token: &str) -> PyResult<PyClassInitializer<Self>> {
        let json = json!({
            "type": "WordPiece",
            "unk_token": unk_token,
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
            "vocab": vocab.map(Vocab::read_lines).transpose()?.unwrap_or_default(),
        });
        Ok(Model::init(json, Self))
    }

    #[getter]
    fn continuing_subword_prefix(slf: &Bound<'_, Self>) -> PyResult<Option<String>> {
        slf.field("continuing_subword_prefix")
    }
}

/// SentencePiece's unigram model: a list of `(token, score)` pairs. The text is split into the
/// tokens with the best total score.
#[pyclass(frozen, extends = Model, module = "tokenizers.models")]
pub struct Unigram;

#[pymethods]
impl Unigram {
    #[new]
    #[pyo3(signature = (vocab, unk_id=None, byte_fallback=false))]
    fn new(
        vocab: Vec<(String, f64)>,
        unk_id: Option<usize>,
        byte_fallback: bool,
    ) -> PyClassInitializer<Self> {
        let json = json!({
            "type": "Unigram",
            "unk_id": unk_id,
            "vocab": vocab,
            "byte_fallback": byte_fallback,
        });
        Model::init(json, Self)
    }
}

/// Maps each word to its id, or to `unk_token`.
#[pyclass(frozen, extends = Model, module = "tokenizers.models")]
pub struct WordLevel;

#[pymethods]
impl WordLevel {
    #[new]
    #[pyo3(signature = (vocab=None, unk_token=None))]
    fn new(vocab: Option<Vocab>, unk_token: Option<&str>) -> PyResult<PyClassInitializer<Self>> {
        let json = json!({
            "type": "WordLevel",
            "vocab": vocab.map(Vocab::read_json).transpose()?.unwrap_or_default(),
            "unk_token": unk_token.unwrap_or("<unk>"),
        });
        Ok(Model::init(json, Self))
    }
}

#[pymodule]
pub mod models {
    #[pymodule_export]
    use super::{Bpe, Model, Unigram, WordLevel, WordPiece};
}
