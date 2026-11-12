use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use pyo3::prelude::*;
use serde_json::json;

use crate::components::component_spec::ComponentSpec;
use crate::error::err;
use crate::type_hints::VocabDict;

use super::macros::component_kind;
use super::{ComponentKind, JsonFieldsExt};

component_kind! {
    kind: Model,
    python_module: "tokenizers.models",
    tokenizer_field: model,
    items_key: None,
    classes: {
        "BPE" => Bpe,
        "WordPiece" => WordPiece,
        "Unigram" => Unigram,
        "WordLevel" => WordLevel,
    },
}

/// A vocabulary: `{token: id}`, or the path of a file holding one.
#[derive(FromPyObject)]
enum Vocab {
    Dict(VocabDict),
    Path(PathBuf),
}

impl Vocab {
    /// A path names a `vocab.json`. No vocabulary reads as an empty one.
    fn read_json(vocab: Option<Self>) -> PyResult<HashMap<String, u32>> {
        match vocab {
            None => Ok(HashMap::new()),
            Some(Self::Dict(vocab)) => Ok(vocab.0),
            Some(Self::Path(path)) => serde_json::from_str(&fs::read_to_string(path)?).map_err(err),
        }
    }

    /// A path names a `vocab.txt`: one token per line, its id being the line number. No
    /// vocabulary reads as an empty one.
    fn read_lines(vocab: Option<Self>) -> PyResult<HashMap<String, u32>> {
        match vocab {
            None => Ok(HashMap::new()),
            Some(Self::Dict(vocab)) => Ok(vocab.0),
            Some(Self::Path(path)) => {
                let text = fs::read_to_string(path)?;
                let lines = text.lines().zip(0..);
                Ok(lines
                    .map(|(line, id)| (line.trim_end().to_owned(), id))
                    .collect())
            }
        }
    }
}

/// BPE merges: `(left, right)` pairs, or the path of a `merges.txt`.
#[derive(FromPyObject)]
enum Merges {
    Pairs(Vec<(String, String)>),
    Path(PathBuf),
}

impl Merges {
    /// A `merges.txt` holds one `left right` pair per line, after a `#version` line. No merges
    /// read as an empty list.
    fn read(merges: Option<Self>) -> PyResult<Vec<(String, String)>> {
        let path = match merges {
            None => return Ok(Vec::new()),
            Some(Self::Pairs(pairs)) => return Ok(pairs),
            Some(Self::Path(path)) => path,
        };
        let text = fs::read_to_string(&path)?;
        let mut pairs = Vec::new();
        for (line, number) in text.lines().zip(1..) {
            if line.starts_with("#version") {
                continue;
            }
            match line.split_once(' ') {
                Some((left, right)) if !right.contains(' ') => {
                    pairs.push((left.to_owned(), right.to_owned()));
                }
                _ => {
                    let path = path.display();
                    return Err(err(format!(
                        "{path}: line {number} is not a `left right` pair"
                    )));
                }
            }
        }
        Ok(pairs)
    }
}

/// A byte-pair encoding model.
#[pyclass(frozen, extends = Model, name = "BPE", module = "tokenizers.models")]
pub struct Bpe;

#[pymethods]
impl Bpe {
    #[new]
    #[pyo3(signature = (
        vocab=None,
        merges=None,
        *,
        dropout=None,
        unk_token=None,
        continuing_subword_prefix=None,
        end_of_word_suffix=None,
        fuse_unk=false,
        byte_fallback=false,
        ignore_merges=false,
        byte_level=false,
    ))]
    #[allow(
        clippy::too_many_arguments,
        reason = "one argument per keyword of the Python `BPE` constructor"
    )]
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
        byte_level: bool,
    ) -> PyResult<PyClassInitializer<Self>> {
        Ok(Model::initializer(
            ComponentSpec::from_literal(json!({
                "type": "BPE",
                "dropout": dropout,
                "unk_token": unk_token,
                "continuing_subword_prefix": continuing_subword_prefix,
                "end_of_word_suffix": end_of_word_suffix,
                "fuse_unk": fuse_unk,
                "byte_fallback": byte_fallback,
                "ignore_merges": ignore_merges,
                "byte_level": byte_level,
                "vocab": Vocab::read_json(vocab)?,
                "merges": Merges::read(merges)?,
            })),
            Self,
        ))
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
        // Files written before the field existed leave it out.
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
    #[pyo3(signature = (
        vocab=None,
        *,
        unk_token="[UNK]",
        continuing_subword_prefix="##",
        max_input_chars_per_word=100,
    ))]
    fn new(
        vocab: Option<Vocab>,
        unk_token: &str,
        continuing_subword_prefix: &str,
        max_input_chars_per_word: usize,
    ) -> PyResult<PyClassInitializer<Self>> {
        Ok(Model::initializer(
            ComponentSpec::from_literal(json!({
                "type": "WordPiece",
                "unk_token": unk_token,
                "continuing_subword_prefix": continuing_subword_prefix,
                "max_input_chars_per_word": max_input_chars_per_word,
                "vocab": Vocab::read_lines(vocab)?,
            })),
            Self,
        ))
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
        Model::initializer(
            ComponentSpec::from_literal(json!({
                "type": "Unigram",
                "unk_id": unk_id,
                "vocab": vocab,
                "byte_fallback": byte_fallback,
            })),
            Self,
        )
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
        Ok(Model::initializer(
            ComponentSpec::from_literal(json!({
                "type": "WordLevel",
                "vocab": Vocab::read_json(vocab)?,
                "unk_token": unk_token.unwrap_or("<unk>"),
            })),
            Self,
        ))
    }
}

#[pymodule]
pub mod models {
    #[pymodule_export]
    use super::{Bpe, Model, Unigram, WordLevel, WordPiece};
}
