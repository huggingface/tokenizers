//! The JSON→JSON pass that turns an old `tokenizer.json` into a canonical one, so that
//! `tk-serialize` needs no backwards-compatibility branch of its own. It edits a
//! `serde_json::Value`; it builds no component and names no wrapper enum.

use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error("the config is not JSON: {0}")]
    Json(#[from] serde_json::Error),

    #[error("cannot read {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("a tokenizer.json must be a JSON object, found {found}")]
    NotAnObject { found: &'static str },

    #[error("the config has no `model`")]
    MissingModel,

    #[error("`model` must be a JSON object, found {found}")]
    ModelNotObject { found: &'static str },

    #[error("legacy merge {merge:?} does not split into exactly two tokens on a space")]
    BadMerge { merge: String },

    #[error("a merge is neither a `[\"a\", \"b\"]` pair nor an `\"a b\"` string")]
    BadMergeShape,

    #[error("`merges` must be an array, found {found}")]
    MergesNotArray { found: &'static str },

    /// Spelled exactly as the two read paths spell it; a caller may be matching on the text.
    #[error("add_prefix_space does not match declared prepend_scheme")]
    PrefixSpaceMismatch,

    #[error("unknown metaspace prepend_scheme {scheme:?}")]
    UnknownPrependScheme { scheme: String },

    #[error("a `Metaspace` has no `replacement`")]
    MetaspaceNoReplacement,

    #[error("a `Metaspace` `replacement` must be exactly one character, got {got:?}")]
    MetaspaceBadReplacement { got: String },

    #[error(
        "a `Metaspace` pre-tokenizer with `prepend_scheme: first` has no canonical spelling yet"
    )]
    MetaspacePrependSchemeFirst,

    #[error("a `Metaspace` pre-tokenizer with `split: false` has no canonical spelling")]
    MetaspaceNoSplit,

    #[error("a `ByteLevel` pre-tokenizer with `add_prefix_space: true` is not supported")]
    ByteLevelAddPrefixSpace,

    #[error("a `ByteLevel` pre-tokenizer must be the last member of its `Sequence`")]
    ByteLevelNotLast,

    #[error("a `ByteLevel` pre-tokenizer needs a BPE model, found {model}")]
    ByteLevelOnNonBpeModel { model: String },

    #[error("a template piece is neither a `Sequence` nor a `SpecialToken`")]
    BadTemplatePiece,

    #[error("the template names the special token {name:?}, which its `special_tokens` does not")]
    UnknownTemplateSpecial { name: String },
}

/// Canonicalise a `tokenizer.json` held as a string. Pretty-printed, because a human usually reads
/// the result next.
pub fn canonicalize_str(json: &str) -> Result<String, ConvertError> {
    let mut value: Value = serde_json::from_str(json)?;
    canonicalize_value(&mut value)?;
    Ok(serde_json::to_string_pretty(&value)?)
}

/// [`canonicalize_str`] without the pretty-printing.
///
/// The pretty variant exists for a human reading the result. A *reader* does not read it -- it
/// re-parses it -- so the indentation is added only to be walked again, and on the configs in
/// `../data` that measured 5.7% of load time (up to 7.6% on llama-3). Use this when the output is
/// going straight into [`tk_serialize::from_json`](https://docs.rs/tk-serialize); use
/// [`canonicalize_str`] when a person is going to look at it.
pub fn canonicalize_str_compact(json: &str) -> Result<String, ConvertError> {
    let mut value: Value = serde_json::from_str(json)?;
    canonicalize_value(&mut value)?;
    Ok(serde_json::to_string(&value)?)
}

/// [`canonicalize_file`] without the pretty-printing; see [`canonicalize_str_compact`].
pub fn canonicalize_file_compact(path: impl AsRef<Path>) -> Result<String, ConvertError> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| ConvertError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    canonicalize_str_compact(&text)
}

/// Canonicalise a `tokenizer.json` read from a file.
pub fn canonicalize_file(path: impl AsRef<Path>) -> Result<String, ConvertError> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| ConvertError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    canonicalize_str(&text)
}

/// Idempotent: every step recognises the canonical shape and returns, or rewrites the legacy one
/// into it. Safe to run unconditionally in front of a reader.
pub fn canonicalize_value(value: &mut Value) -> Result<(), ConvertError> {
    let found = kind_of(value);
    let root = value
        .as_object_mut()
        .ok_or(ConvertError::NotAnObject { found })?;

    root.insert("version".to_string(), Value::String(VERSION.to_string()));

    {
        let model = root.get_mut("model").ok_or(ConvertError::MissingModel)?;
        let found = kind_of(model);
        let model = model
            .as_object_mut()
            .ok_or(ConvertError::ModelNotObject { found })?;
        fill_model_type(model);
        canonicalize_merges(model)?;
    }

    // Both read the `pre_tokenizer` slot and write elsewhere -- the normalizer chain, the model --
    // so they run before the per-component fill below sees either.
    lower_metaspace_pre_tokenizer(root)?;
    lower_byte_level_pre_tokenizer(root)?;
    lower_template_processing(root)?;

    fill_model_defaults(root)?;

    // `Metaspace` survives as a *decoder*, so all four slots are still walked.
    for slot in ["normalizer", "pre_tokenizer", "post_processor", "decoder"] {
        if let Some(node) = root.get_mut(slot) {
            canonicalize_component(node)?;
        }
    }
    Ok(())
}

/// The canonical format this pass emits.
const VERSION: &str = "2.0";

/// `{"type": tag, ...fields}`.
fn tagged(tag: &str, fields: &[(&str, Value)]) -> Value {
    let mut obj = Map::new();
    obj.insert("type".to_string(), Value::String(tag.to_string()));
    for (k, v) in fields {
        obj.insert((*k).to_string(), v.clone());
    }
    Value::Object(obj)
}

/// Append `normalizer` to the chain, after whatever the config already declared.
///
/// The order matters and is the one the old reader used: a `Metaspace` pre-tokenizer rewrote the
/// text *after* the declared normalizer had run.
fn append_normalizer(root: &mut Map<String, Value>, normalizer: Value) {
    let existing = root.get("normalizer").cloned().unwrap_or(Value::Null);
    let chain = match existing {
        Value::Null => normalizer,
        Value::Object(ref o) if o.get("type").and_then(Value::as_str) == Some("Sequence") => {
            let mut members = o
                .get("normalizers")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            members.push(normalizer);
            tagged("Sequence", &[("normalizers", Value::Array(members))])
        }
        declared => tagged(
            "Sequence",
            &[("normalizers", Value::Array(vec![declared, normalizer]))],
        ),
    };
    root.insert("normalizer".to_string(), chain);
}

/// Give the model a `"type"` if it has none, in the order both read paths use.
///
/// `merges` **must** be tested before `continuing_subword_prefix`: a serialized BPE writes
/// `"continuing_subword_prefix": null`, so gpt2 would otherwise read as a WordPiece with no merges.
/// Key presence, not non-null, for the same reason.
fn fill_model_type(model: &mut Map<String, Value>) {
    // An unknown tag is not this pass's business to refuse; a newer reader may know it.
    if model.get("type").and_then(Value::as_str).is_some() {
        return;
    }
    let kind = if model.contains_key("merges") {
        "BPE"
    } else if model.contains_key("continuing_subword_prefix") {
        "WordPiece"
    } else if model.get("vocab").is_some_and(Value::is_array) {
        // Only Unigram's vocab is an array of [token, score] pairs.
        "Unigram"
    } else {
        "WordLevel"
    };
    model.insert("type".to_string(), Value::String(kind.to_string()));
}

/// Rewrite `merges` from the `merges.txt` spelling into pairs.
fn canonicalize_merges(model: &mut Map<String, Value>) -> Result<(), ConvertError> {
    let Some(merges) = model.get_mut("merges") else {
        return Ok(());
    };
    // An explicit null is how some writers spell "not a BPE".
    if merges.is_null() {
        return Ok(());
    }
    let found = kind_of(merges);
    let items = merges
        .as_array_mut()
        .ok_or(ConvertError::MergesNotArray { found })?;

    let mut out = Vec::with_capacity(items.len());
    for entry in items.drain(..) {
        match entry {
            Value::Array(ref pair) if pair.len() == 2 && pair.iter().all(|p| p.is_string()) => {
                out.push(entry);
            }
            Value::Array(_) => return Err(ConvertError::BadMergeShape),
            Value::String(s) => {
                // Dropped, not rewritten: the config path filters it before numbering the ranks.
                if s.starts_with("#version") {
                    continue;
                }
                // Exactly two parts, never `split_once`: a merge whose token contains a space has
                // never loaded, and guessing would produce a different tokenizer rather than an error.
                let parts: Vec<&str> = s.split(' ').collect();
                if parts.len() != 2 {
                    return Err(ConvertError::BadMerge { merge: s });
                }
                out.push(Value::Array(vec![
                    Value::String(parts[0].to_string()),
                    Value::String(parts[1].to_string()),
                ]));
            }
            _ => return Err(ConvertError::BadMergeShape),
        }
    }
    *items = out;
    Ok(())
}

const SEQUENCE_CHILDREN: [&str; 4] = ["normalizers", "pretokenizers", "decoders", "processors"];

/// Only `Sequence` children are descended into: a blind walk would eventually meet a data object
/// with a `"type"` key (a post-processor's `special_tokens`) and rewrite it.
fn canonicalize_component(node: &mut Value) -> Result<(), ConvertError> {
    match node {
        Value::Object(obj) => {
            if obj.get("type").and_then(Value::as_str) == Some("Metaspace") {
                canonicalize_metaspace(obj)?;
            }
            for key in SEQUENCE_CHILDREN {
                if let Some(children) = obj.get_mut(key) {
                    canonicalize_component(children)?;
                }
            }
            Ok(())
        }
        Value::Array(items) => {
            for item in items {
                canonicalize_component(item)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// `Metaspace`, the one spelling change that is not a rename. Reproduced from the two read paths,
/// quirks included, because token ids depend on them:
///
/// - an absent `prepend_scheme` is `"always"`, not `"never"` (the old `add_prefix_space` defaulted
///   to `true`);
/// - `add_prefix_space: true` is ignored outright, so an explicit `prepend_scheme` always wins;
/// - `add_prefix_space: false` is checked against the *defaulted* scheme, so it is an error unless
///   `prepend_scheme: "never"` is spelled beside it;
/// - `str_rep` is thrown away.
fn canonicalize_metaspace(obj: &mut Map<String, Value>) -> Result<(), ConvertError> {
    // Filter null, not plain `get`: an explicit null means "not spelled", as `get_some` does.
    let declared = obj
        .get("prepend_scheme")
        .filter(|v| !v.is_null())
        .map(|v| match v.as_str() {
            Some(s @ ("always" | "first" | "never")) => Ok(s.to_string()),
            _ => Err(ConvertError::UnknownPrependScheme {
                scheme: match v.as_str() {
                    Some(s) => s.to_string(),
                    None => v.to_string(),
                },
            }),
        })
        .transpose()?;
    let scheme = declared.unwrap_or_else(|| "always".to_string());

    let add_prefix_space = obj
        .get("add_prefix_space")
        .filter(|v| !v.is_null())
        .and_then(Value::as_bool);
    if add_prefix_space == Some(false) && scheme != "never" {
        return Err(ConvertError::PrefixSpaceMismatch);
    }

    // Checked here so a two-character `replacement` is refused rather than silently truncated.
    match obj.get("replacement").and_then(Value::as_str) {
        None => return Err(ConvertError::MetaspaceNoReplacement),
        Some(s) => {
            let mut chars = s.chars();
            if !(chars.next().is_some() && chars.next().is_none()) {
                return Err(ConvertError::MetaspaceBadReplacement { got: s.to_string() });
            }
        }
    }

    obj.insert("prepend_scheme".to_string(), Value::String(scheme));
    obj.remove("add_prefix_space");
    obj.remove("str_rep");
    // Only the pre-tokenizer form had `split`, and that form is lowered away. What is left is a
    // decoder, which the canonical writer spells as `{type, replacement, prepend_scheme}`.
    obj.remove("split");
    Ok(())
}

/// `serde_json` has no public name for a `Value`'s variant.
fn kind_of(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// A `Metaspace` pre-tokenizer is two components, so it does not survive as itself.
///
/// It becomes a `MetaspaceNormalizer` in the `normalizer` slot (which rewrites the text) plus a
/// `Split` on the delimiter here (which cuts it). `Sequence[WhitespaceSplit, Metaspace]` -- the
/// shape t5 and albert ship -- is the same thing with `drop_whitespace`, and the whole `Sequence`
/// collapses to the lone `Split`.
fn lower_metaspace_pre_tokenizer(root: &mut Map<String, Value>) -> Result<(), ConvertError> {
    let Some(pretok) = root.get("pre_tokenizer") else {
        return Ok(());
    };
    let (metaspace, drop_whitespace) = match pretok {
        Value::Object(o) if o.get("type").and_then(Value::as_str) == Some("Metaspace") => {
            (o.clone(), false)
        }
        Value::Object(o) if o.get("type").and_then(Value::as_str) == Some("Sequence") => {
            match o
                .get("pretokenizers")
                .and_then(Value::as_array)
                .map(|m| &m[..])
            {
                Some([first, second])
                    if first.get("type").and_then(Value::as_str) == Some("WhitespaceSplit")
                        && second.get("type").and_then(Value::as_str) == Some("Metaspace") =>
                {
                    (
                        second.as_object().expect("matched as an object").clone(),
                        true,
                    )
                }
                _ => return Ok(()),
            }
        }
        _ => return Ok(()),
    };

    // `split` defaulted to true, and is read before the shared normalisation drops it.
    // `split: false` wrote delimiters but never cut, so there is no `Split` to hand back.
    if !metaspace
        .get("split")
        .and_then(Value::as_bool)
        .unwrap_or(true)
    {
        return Err(ConvertError::MetaspaceNoSplit);
    }
    // Reuse the field normalisation written for the decoder case, so the `add_prefix_space` quirk
    // is applied in exactly one place.
    let mut metaspace = metaspace;
    canonicalize_metaspace(&mut metaspace)?;

    let replacement = metaspace
        .get("replacement")
        .and_then(Value::as_str)
        .ok_or(ConvertError::MetaspaceNoReplacement)?
        .to_string();
    let prepend = match metaspace.get("prepend_scheme").and_then(Value::as_str) {
        Some("always") => true,
        Some("never") => false,
        // No canonical spelling: the pipeline applies a normalizer per segment and cannot know it
        // is the first. See the `TODO(v1)` in tk-serialize's `from_json::pre_tokenizers`.
        Some("first") => return Err(ConvertError::MetaspacePrependSchemeFirst),
        other => {
            return Err(ConvertError::UnknownPrependScheme {
                scheme: other.unwrap_or_default().to_string(),
            });
        }
    };

    append_normalizer(
        root,
        tagged(
            "MetaspaceNormalizer",
            &[
                ("replacement", Value::String(replacement.clone())),
                ("prepend", Value::Bool(prepend)),
                ("drop_whitespace", Value::Bool(drop_whitespace)),
            ],
        ),
    );
    root.insert(
        "pre_tokenizer".to_string(),
        tagged(
            "Split",
            &[
                ("pattern", serde_json::json!({ "String": replacement })),
                ("behavior", Value::String("MergedWithNext".to_string())),
                ("invert", Value::Bool(false)),
            ],
        ),
    );
    Ok(())
}

/// A `ByteLevel` pre-tokenizer says two things, and only one of them is about splitting.
///
/// The byte map is a property of the vocabulary, so it becomes `"byte_level": true` on the model.
/// What is left is the split it asked for: the GPT-2 regex when `use_regex` (the default), and
/// nothing at all when not.
fn lower_byte_level_pre_tokenizer(root: &mut Map<String, Value>) -> Result<(), ConvertError> {
    let Some(pretok) = root.get_mut("pre_tokenizer") else {
        return Ok(());
    };
    let is_byte_level = |v: &Value| v.get("type").and_then(Value::as_str) == Some("ByteLevel");

    // `use_regex` defaults to true, which is what gpt2 and roberta rely on.
    let split_for = |bl: &Value| -> Result<Option<Value>, ConvertError> {
        if bl.get("add_prefix_space").and_then(Value::as_bool) == Some(true) {
            return Err(ConvertError::ByteLevelAddPrefixSpace);
        }
        Ok(
            match bl.get("use_regex").and_then(Value::as_bool).unwrap_or(true) {
                true => Some(tagged(
                    "Split",
                    &[(
                        "pattern",
                        serde_json::json!({ "Regex": bitsplit::regexes::GPT2 }),
                    )],
                )),
                false => None,
            },
        )
    };

    let replacement = if is_byte_level(pretok) {
        Some(split_for(pretok)?)
    } else if pretok.get("type").and_then(Value::as_str) == Some("Sequence") {
        let members = pretok
            .get_mut("pretokenizers")
            .and_then(Value::as_array_mut);
        match members {
            Some(members) if members.last().is_some_and(is_byte_level) => {
                let bl = members.pop().expect("checked by `last`");
                if let Some(split) = split_for(&bl)? {
                    members.push(split);
                }
                // A `Sequence` whose only member was the dropped `ByteLevel` is not an empty
                // sequence, it is no pre-tokenizer at all -- which is what `use_regex: false`
                // lowered to before.
                if members.is_empty() {
                    Some(None)
                } else {
                    None // the `Sequence` was edited in place
                }
            }
            // A `ByteLevel` anywhere but last never loaded: the byte map has to apply after every
            // split, and guessing an order would produce a different tokenizer rather than an error.
            Some(members) if members.iter().any(is_byte_level) => {
                return Err(ConvertError::ByteLevelNotLast);
            }
            _ => return Ok(()),
        }
    } else {
        return Ok(());
    };

    let kind = root
        .get("model")
        .and_then(|m| m.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("an untyped model")
        .to_string();
    if kind != "BPE" {
        return Err(ConvertError::ByteLevelOnNonBpeModel { model: kind });
    }

    if let Some(split) = replacement {
        match split {
            Some(split) => root.insert("pre_tokenizer".to_string(), split),
            None => root.remove("pre_tokenizer"),
        };
    }
    set_model_flag(root, "byte_level", true)?;
    Ok(())
}

/// Set a boolean on the `model` object.
fn set_model_flag(
    root: &mut Map<String, Value>,
    key: &str,
    value: bool,
) -> Result<(), ConvertError> {
    let model = root.get_mut("model").ok_or(ConvertError::MissingModel)?;
    let found = kind_of(model);
    model
        .as_object_mut()
        .ok_or(ConvertError::ModelNotObject { found })?
        .insert(key.to_string(), Value::Bool(value));
    Ok(())
}

/// Everything the reader requires of a model that a legacy writer left out.
fn fill_model_defaults(root: &mut Map<String, Value>) -> Result<(), ConvertError> {
    let model = root.get_mut("model").ok_or(ConvertError::MissingModel)?;
    let found = kind_of(model);
    let model = model
        .as_object_mut()
        .ok_or(ConvertError::ModelNotObject { found })?;
    if model.get("type").and_then(Value::as_str) == Some("BPE") {
        // Only set when the `ByteLevel` lowering did not already say otherwise.
        model.entry("byte_level").or_insert(Value::Bool(false));
    }
    Ok(())
}

/// A `TemplateProcessing` used to name its special tokens and carry a table to look them up in.
/// The pipeline keeps only the ids, so the canonical piece states them directly and the table goes:
///
/// ```text
///   {"SpecialToken": {"id": "[CLS]", "type_id": 0}}   ->  {"ids": [2], "type_id": 0}
///   {"Sequence":     {"id": "A",     "type_id": 0}}   ->  {"seq": "A", "type_id": 0}
/// ```
fn lower_template_processing(root: &mut Map<String, Value>) -> Result<(), ConvertError> {
    let Some(pp) = root.get_mut("post_processor") else {
        return Ok(());
    };
    lower_template_node(pp)
}

/// A `TemplateProcessing` can sit inside a `Sequence` post-processor -- llama-3 puts one behind a
/// `ByteLevel` -- so the whole subtree is walked rather than just the slot.
fn lower_template_node(node: &mut Value) -> Result<(), ConvertError> {
    if let Value::Array(items) = node {
        for item in items {
            lower_template_node(item)?;
        }
        return Ok(());
    }
    let Some(pp) = node.as_object_mut() else {
        return Ok(());
    };
    if pp.get("type").and_then(Value::as_str) == Some("Sequence")
        && let Some(children) = pp.get_mut("processors")
    {
        return lower_template_node(children);
    }
    if pp.get("type").and_then(Value::as_str) != Some("TemplateProcessing") {
        return Ok(());
    }
    let specials = pp.get("special_tokens").cloned().unwrap_or(Value::Null);
    // Already canonical: the table is what a legacy file has, and the pieces are flat without it.
    if specials.is_null() {
        return Ok(());
    }

    for key in ["single", "pair"] {
        let Some(pieces) = pp.get(key).and_then(Value::as_array).cloned() else {
            continue;
        };
        let mut out = Vec::with_capacity(pieces.len());
        for piece in &pieces {
            // Already flat, so leave it: this pass has to be safe to run twice.
            if piece.get("seq").is_some() || piece.get("ids").is_some() {
                out.push(piece.clone());
                continue;
            }
            let mut flat = Map::new();
            if let Some(seq) = piece.get("Sequence") {
                flat.insert(
                    "seq".to_string(),
                    seq.get("id")
                        .cloned()
                        .ok_or(ConvertError::BadTemplatePiece)?,
                );
                if let Some(t) = seq.get("type_id") {
                    flat.insert("type_id".to_string(), t.clone());
                }
            } else if let Some(tok) = piece.get("SpecialToken") {
                let name = tok
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or(ConvertError::BadTemplatePiece)?;
                let ids = specials
                    .get(name)
                    .and_then(|e| e.get("ids"))
                    .cloned()
                    .ok_or_else(|| ConvertError::UnknownTemplateSpecial {
                        name: name.to_string(),
                    })?;
                flat.insert("ids".to_string(), ids);
                if let Some(t) = tok.get("type_id") {
                    flat.insert("type_id".to_string(), t.clone());
                }
            } else {
                return Err(ConvertError::BadTemplatePiece);
            }
            out.push(Value::Object(flat));
        }
        pp.insert(key.to_string(), Value::Array(out));
    }
    pp.remove("special_tokens");
    Ok(())
}
