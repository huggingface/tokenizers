//! The JSON→JSON pass that turns an old `tokenizer.json` into a canonical one.
//!
//! This is the point of the crate, stated as a function. `tk-serialize` is a hand-rolled,
//! serde-free reader for the format *as it is written today*; every shape older than that is
//! refused there with a message naming what to convert, and this module is what does the
//! converting. It fills in the fields the canonical reader expects to find and rewrites the ones
//! that changed spelling, so that the reader needs no backwards-compatibility branch of its own.
//!
//! Nothing here builds a component. It reads a `serde_json::Value`, edits it, and hands it back:
//! no `ModelWrapper`, no `Metaspace`, no regex compiled. That matters because a converter that
//! went through the config types would have to link every one of them, and the whole reason the
//! three crates are split the way they are is that *naming* a wrapper enum makes every variant of
//! it reachable. A converter is allowed to be a text transformation, so it is one.
//!
//! ## What it fills, and how often that shape actually occurs
//!
//! Measured over the 20 real `tokenizer.json` files in `data/` (`gpt2-vocab.json` is a bare vocab
//! map and `unigram.json` a bare Unigram *model*, so neither is a tokenizer config):
//!
//! | shape | files | occurrences |
//! |---|---|---|
//! | a `model` with no `"type"` | 7 | 7 |
//! | `merges` as `"a b"` strings rather than `["a", "b"]` pairs | 7 | 7 |
//! | a `[token, score]` array-shaped Unigram `vocab` | 4 | 4 |
//! | a `Metaspace` spelled with `add_prefix_space` | 2 | 4 |
//! | a `str_rep` field | 2 | 4 |
//! | a `Metaspace` with no `prepend_scheme` | 2 | 4 |
//! | a vocabulary given as a file path (`files`) | 0 | 0 |
//!
//! The Metaspace rows are 2 files but 4 objects because `Metaspace` is written down twice in both
//! t5 and albert — once as a pre-tokenizer and once as a decoder — and each copy carries the legacy
//! spelling. That is why the walk below visits every position rather than just the pre-tokenizer.
//!
//! The no-`"type"` row is worth pinning down, because it is easy to count as 8: `data/unigram.json`
//! is a bare Unigram *model* with no surrounding config, and it too has no `"type"` and an
//! array-shaped vocab. It is not a `tokenizer.json`, so it is not in the 20 and not in the 7 — but
//! anything that counts model objects rather than configs will see it.
//!
//! The array-shaped Unigram vocab is the odd one out: it is *not* rewritten. An array of
//! `[token, score]` pairs is what the canonical reader wants for a `Unigram`, and the scores would
//! have nowhere to go in an object map. It is listed because it is the shape that *identifies* a
//! `Unigram` when the model carries no `"type"` — rule 3 of the inference below — so the converter
//! has to recognise it even though it leaves it alone.
//!
//! `files` never occurs in the fixtures; it is supported because the canonical reader names it as
//! a shape it refuses, which makes it this module's problem by construction.

use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

/// Anything that stops a config from being canonicalised.
///
/// Deliberately not `tk_encode::Error` (a boxed `dyn Error`): a converter's failures are a short,
/// closed list, and a caller that wants to say "this file needs a human" wants to match on it.
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

    /// The config path splits a legacy merge on `' '` and demands exactly two parts
    /// (`convert_merges_to_hashmap`, which raises `BadMerges(line)`). A token containing a space
    /// therefore has never loaded, and the converter must not invent a reading for it — that is
    /// exactly the ambiguity that made `["a", "b"]` pairs canonical.
    #[error("legacy merge {merge:?} does not split into exactly two tokens on a space")]
    BadMerge { merge: String },

    #[error("a merge is neither a `[\"a\", \"b\"]` pair nor an `\"a b\"` string")]
    BadMergeShape,

    #[error("`merges` must be an array, found {found}")]
    MergesNotArray { found: &'static str },

    /// Reproduced verbatim from both existing read paths, because a caller may well be matching on
    /// the text: `tk-convert/src/decoders/mirror.rs`'s `metaspace::deserialize` and
    /// `tk-serialize/src/from_json.rs`'s `read_prepend_scheme` both raise this exact string.
    #[error("add_prefix_space does not match declared prepend_scheme")]
    PrefixSpaceMismatch,

    #[error("unknown metaspace prepend_scheme {scheme:?}")]
    UnknownPrependScheme { scheme: String },

    #[error("a `Metaspace` has no `replacement`")]
    MetaspaceNoReplacement,

    #[error("a `Metaspace` `replacement` must be exactly one character, got {got:?}")]
    MetaspaceBadReplacement { got: String },

    #[error(
        "`files` must be a path, a `[vocab, merges]` array, or a `{{\"vocab\": .., \"merges\": ..}}` object"
    )]
    BadFilesShape,

    #[error(
        "{path} is not a vocabulary: expected a JSON object of token -> id, or one token per line"
    )]
    BadVocabFile { path: PathBuf },
}

/// Canonicalise a `tokenizer.json` held as a string.
///
/// The output is pretty-printed, because the thing that reads it next is usually a human deciding
/// whether the conversion did what they wanted.
///
/// A `vocab`/`merges` given as a *relative* file path resolves against the process working
/// directory here — a string has no directory of its own. Use [`canonicalize_file`] when the paths
/// are meant to be read relative to the config, which is what a model directory on the Hub looks
/// like.
pub fn canonicalize_str(json: &str) -> Result<String, ConvertError> {
    let mut value: Value = serde_json::from_str(json)?;
    canonicalize_in(&mut value, None)?;
    Ok(serde_json::to_string_pretty(&value)?)
}

/// Canonicalise a `tokenizer.json` read from a file.
///
/// The file's own directory is the base for any relative path inside it, which is the only reason
/// this is not a one-line wrapper over [`canonicalize_str`].
pub fn canonicalize_file(path: impl AsRef<Path>) -> Result<String, ConvertError> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| ConvertError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut value: Value = serde_json::from_str(&text)?;
    canonicalize_in(&mut value, path.parent())?;
    Ok(serde_json::to_string_pretty(&value)?)
}

/// Canonicalise a parsed config in place. The core; the other two entry points wrap it.
///
/// Idempotent: every step below either recognises the canonical shape and returns, or rewrites the
/// legacy one *into* the canonical shape. Running this on its own output changes nothing, which is
/// what makes it safe to put in front of a reader unconditionally.
pub fn canonicalize_value(value: &mut Value) -> Result<(), ConvertError> {
    canonicalize_in(value, None)
}

/// The real signature. `base` is the directory a relative `files` path resolves against; the public
/// surface is the three named functions, so this stays private rather than growing a fourth.
fn canonicalize_in(value: &mut Value, base: Option<&Path>) -> Result<(), ConvertError> {
    let found = kind_of(value);
    let root = value
        .as_object_mut()
        .ok_or(ConvertError::NotAnObject { found })?;

    let model = root.get_mut("model").ok_or(ConvertError::MissingModel)?;
    let found = kind_of(model);
    let model = model
        .as_object_mut()
        .ok_or(ConvertError::ModelNotObject { found })?;
    // Order matters: resolving `files` is what *produces* the `merges` key that the type inference
    // then reads, so a `files: [vocab.json, merges.txt]` model has to be inlined before it can be
    // recognised as a BPE.
    inline_vocab_files(model, base)?;
    fill_model_type(model);
    canonicalize_merges(model)?;

    // Every position a component can occupy. `Metaspace` genuinely appears in three of them: as a
    // pre-tokenizer (where it is the SentencePiece rewrite-and-cut), as a decoder (where it undoes
    // that), and -- for a config that spells its normalizer chain out by hand -- in the normalizer
    // slot. `post_processor` cannot hold one today, and is walked anyway so that the set of places
    // this function looks is "all of them" rather than a list to keep in sync.
    for slot in ["normalizer", "pre_tokenizer", "post_processor", "decoder"] {
        if let Some(node) = root.get_mut(slot) {
            canonicalize_component(node)?;
        }
    }
    Ok(())
}

// -------------------------------------------------------------------------------------------------
// Model
// -------------------------------------------------------------------------------------------------

/// Give the model a `"type"` if it has none.
///
/// The order of the tests is the load-bearing part, and it is the order both existing read paths
/// use (`ModelUntagged`'s variant order in `models.rs`, and `model_kind` in
/// `tk-serialize/src/from_json.rs`):
///
/// 1. has `merges` ⇒ BPE
/// 2. else has `continuing_subword_prefix` ⇒ WordPiece
/// 3. else the vocab is a `[token, score]` array ⇒ Unigram
/// 4. else ⇒ WordLevel
///
/// **The BPE test must come before the WordPiece test.** A serialized BPE writes out all of its
/// optional fields including `"continuing_subword_prefix": null`, so a BPE and a WordPiece are not
/// distinguishable by that key at all — every modern BPE has it. Testing `merges` first is what
/// keeps `gpt2.json` (no `"type"`, string merges, `"continuing_subword_prefix": null`) from being
/// read as a WordPiece with no merges. `gpt2` is the single most common config on the Hub, so this
/// is not a hypothetical ordering.
///
/// Presence is *key* presence, not "present and not null", for the same reason: the discriminating
/// keys are all spelled with an explicit `null` by the serializer, and `get`-not-`get_some` is what
/// the reader does. Steps 2 and 3 are in the reader's opposite order, which cannot matter — a
/// Unigram has no `continuing_subword_prefix` key and nothing with that key has an array vocab —
/// but this is the order the task fixes, so this is the order it is written in.
fn fill_model_type(model: &mut Map<String, Value>) {
    // Already canonical. Not an error even if the tag is unknown: which model types exist is the
    // reader's business, and refusing here would make this pass fail on a config a newer reader
    // could load.
    if model.get("type").and_then(Value::as_str).is_some() {
        return;
    }

    let kind = if model.contains_key("merges") {
        "BPE"
    } else if model.contains_key("continuing_subword_prefix") {
        "WordPiece"
    } else if model.get("vocab").is_some_and(Value::is_array) {
        // Unigram's vocab is an array of [token, score] pairs; every other model's is an object.
        "Unigram"
    } else {
        "WordLevel"
    };
    model.insert("type".to_string(), Value::String(kind.to_string()));
}

/// Rewrite `merges` from the `merges.txt` spelling into pairs.
///
/// Two rules copied from `convert_merges_to_hashmap`, which is what the config path runs on this
/// exact data:
///
/// - a `#version` line is dropped. It is the header `merges.txt` carries, and a config built by
///   pasting that file's lines into a JSON array keeps it. The config path filters it *before*
///   enumerating, so the merge ranks are numbered as if it were never there — dropping the element
///   reproduces that, where rewriting it into a pair would shift every rank by one.
/// - the split is on `' '` and must yield exactly two parts. `split_once` would be more forgiving,
///   but a merge whose token contains a space has never loaded through the config path, and
///   silently picking the first space would produce a *different tokenizer* rather than an error.
fn canonicalize_merges(model: &mut Map<String, Value>) -> Result<(), ConvertError> {
    let Some(merges) = model.get_mut("merges") else {
        return Ok(());
    };
    // An explicit null is how "not a BPE" is spelled by some writers; leave it exactly as it is.
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
            // Already canonical.
            Value::Array(ref pair) if pair.len() == 2 && pair.iter().all(|p| p.is_string()) => {
                out.push(entry);
            }
            Value::Array(_) => return Err(ConvertError::BadMergeShape),
            Value::String(s) => {
                if s.starts_with("#version") {
                    continue;
                }
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

/// Inline a vocabulary (and merges) that the config names by path rather than spelling out.
///
/// Three spellings are accepted, because the field was a builder argument long before it was a
/// serialized one and each layer wrote it down differently:
///
/// - `"files": "vocab.txt"` — the one-file models (`WordPiece`, `WordLevel`).
/// - `"files": ["vocab.json", "merges.txt"]` — `BpeBuilder::files(vocab, merges)`, which held a
///   `(String, String)`. A one-element array is the one-file case.
/// - `"files": {"vocab": .., "merges": ..}` — the same thing spelled by key.
///
/// A `"vocab"` (or `"merges"`) whose value is a *string* is treated as the same thing, because that
/// is what the Python bindings' `BPE("vocab.json", "merges.txt")` produces when it is written out.
///
/// Which of the two vocabulary formats a file is in is decided by its content, not its extension:
/// a JSON object is a `{token: id}` map (`BPE::read_file`, `WordLevel::read_file`), anything else is
/// one token per line with the line number as the id (`WordPiece::read_file`). Content-sniffing
/// rather than switching on the model kind, because the model kind is not known yet — inferring it
/// needs the vocab this function is producing.
fn inline_vocab_files(
    model: &mut Map<String, Value>,
    base: Option<&Path>,
) -> Result<(), ConvertError> {
    // `vocab`/`merges` spelled as a path, with no `files` key at all.
    let inline_str_field = |model: &mut Map<String, Value>, key: &str| -> Option<String> {
        match model.get(key) {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        }
    };
    let mut vocab_path = inline_str_field(model, "vocab");
    let mut merges_path = inline_str_field(model, "merges");

    match model.get("files") {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) => vocab_path = Some(s.clone()),
        Some(Value::Array(items)) => match items.as_slice() {
            [v] => vocab_path = as_path(v)?,
            [v, m] => {
                vocab_path = as_path(v)?;
                merges_path = as_path(m)?;
            }
            _ => return Err(ConvertError::BadFilesShape),
        },
        Some(Value::Object(map)) => {
            if let Some(v) = map.get("vocab") {
                vocab_path = as_path(v)?;
            }
            if let Some(m) = map.get("merges") {
                merges_path = as_path(m)?;
            }
        }
        Some(_) => return Err(ConvertError::BadFilesShape),
    }

    if vocab_path.is_none() && merges_path.is_none() {
        // Nothing to inline. Still drop a `files` that only spelled nulls, so the reader does not
        // see a key it refuses on sight.
        model.remove("files");
        return Ok(());
    }

    if let Some(p) = vocab_path {
        let path = resolve(base, &p);
        let text = read(&path)?;
        model.insert("vocab".to_string(), read_vocab(&text, &path)?);
    }
    if let Some(p) = merges_path {
        let path = resolve(base, &p);
        let text = read(&path)?;
        let mut merges = Vec::new();
        for line in text.lines() {
            if line.starts_with("#version") {
                continue;
            }
            // Skip the blank final line a text file ends with; anything else short of two parts is
            // the config path's `BadMerges`.
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split(' ').collect();
            if parts.len() != 2 {
                return Err(ConvertError::BadMerge {
                    merge: line.to_string(),
                });
            }
            merges.push(Value::Array(vec![
                Value::String(parts[0].to_string()),
                Value::String(parts[1].to_string()),
            ]));
        }
        model.insert("merges".to_string(), Value::Array(merges));
    }
    model.remove("files");
    Ok(())
}

fn as_path(value: &Value) -> Result<Option<String>, ConvertError> {
    match value {
        Value::String(s) => Ok(Some(s.clone())),
        Value::Null => Ok(None),
        _ => Err(ConvertError::BadFilesShape),
    }
}

fn resolve(base: Option<&Path>, path: &str) -> PathBuf {
    match base {
        Some(dir) => dir.join(path),
        None => PathBuf::from(path),
    }
}

fn read(path: &Path) -> Result<String, ConvertError> {
    std::fs::read_to_string(path).map_err(|source| ConvertError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// A vocabulary file, in whichever of the two formats it turns out to be in.
///
/// The JSON branch keeps `BPE::read_file`'s quirk of *skipping* an entry whose id is not a number
/// rather than failing on it, because a converter that refused those would reject files the config
/// path loads.
fn read_vocab(text: &str, path: &Path) -> Result<Value, ConvertError> {
    if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(text) {
        let kept = map
            .into_iter()
            .filter(|(_, id)| id.is_number())
            .collect::<Map<_, _>>();
        return Ok(Value::Object(kept));
    }
    // One token per line, id = line number. `trim_end` matches `WordPiece::read_file`, which is
    // reading a `vocab.txt` whose lines may carry a `\r`.
    let mut out = Map::new();
    for (index, line) in text.lines().enumerate() {
        out.insert(line.trim_end().to_string(), Value::from(index as u32));
    }
    if out.is_empty() {
        return Err(ConvertError::BadVocabFile {
            path: path.to_path_buf(),
        });
    }
    Ok(Value::Object(out))
}

// -------------------------------------------------------------------------------------------------
// Components
// -------------------------------------------------------------------------------------------------

/// The `Sequence` child key, one per component position. A `Sequence` is the only component that
/// holds others, and it names them after itself.
const SEQUENCE_CHILDREN: [&str; 4] = ["normalizers", "pretokenizers", "decoders", "processors"];

/// Walk one component position, recursing through `Sequence` children.
///
/// Only the four `Sequence` child keys are descended into, rather than every nested object. A
/// post-processor's `special_tokens` map and a normalizer's `precompiled_charsmap` are data, not
/// components, and a blind walk would eventually meet a data object with a `"type"` key and rewrite
/// it.
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

/// `Metaspace`, the one component whose spelling changed in a way that is not a rename.
///
/// The rule is reproduced from `tk-convert/src/decoders/mirror.rs`'s `metaspace` module and
/// `tk-serialize/src/from_json.rs`'s `read_prepend_scheme`, which already agree with each other,
/// quirks included:
///
/// - **an absent `prepend_scheme` is `"always"`, not `"never"`.** This is the one that surprises
///   people. `add_prefix_space` was the old key and it defaulted to `true`, so a config that spells
///   neither key is asking for a prefix space, and the scheme that means that is `Always`.
/// - `add_prefix_space: true` is **ignored outright**. It agrees with the `"always"` default when
///   the config spells only the old key (t5, albert), and it *loses* to an explicit
///   `prepend_scheme` when both are spelled — so `{add_prefix_space: true, prepend_scheme:
///   "never"}` is `"never"`, not a contradiction.
/// - `add_prefix_space: false` is checked against the *already defaulted* scheme, which is
///   `"always"`. So the old key alone can never spell `false`: it is a hard error unless
///   `prepend_scheme: "never"` is spelled out beside it, at which point it changes nothing.
/// - `str_rep` is read and thrown away. It was a cache of `replacement` as a `String`, kept in sync
///   by hand; nothing reads it.
///
/// Surprising, and reproduced rather than fixed, because token ids depend on it. A converter that
/// "corrected" the `add_prefix_space: false` case into `"never"` would silently change what t5
/// tokenizes to.
fn canonicalize_metaspace(obj: &mut Map<String, Value>) -> Result<(), ConvertError> {
    // `get`-then-filter-null, not `get`: `"prepend_scheme": null` means "not spelled", which is
    // what `get_some` does on the reader side.
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

    // `replacement` is required and is a single character. Checked here rather than left to the
    // reader because a two-character `replacement` must not be silently truncated to its first
    // char, and this pass is the last place that can say so with the file name still in hand.
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
    // `split` defaults to `true` on both read paths. Spelled out here so the canonical file says
    // what it means and the reader has one fewer default to carry.
    if obj.get("split").filter(|v| !v.is_null()).is_none() {
        obj.insert("split".to_string(), Value::Bool(true));
    }
    Ok(())
}

/// For error messages. `serde_json` has no public name for a `Value`'s variant.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-data root, spelled the same way `tests/common/mod.rs` spells it.
    const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data");

    /// Wrap a `model` (and optionally other top-level fields) in the smallest config that is a
    /// valid `tokenizer.json`.
    fn config(model: &str, extra: &str) -> Value {
        let text = format!(r#"{{"version": "1.0", "model": {model}{extra}}}"#);
        serde_json::from_str(&text).expect("the test literal is not JSON")
    }

    fn done(model: &str, extra: &str) -> Value {
        let mut v = config(model, extra);
        canonicalize_value(&mut v).expect("canonicalisation failed");
        assert_no_legacy_residue(&v, "the hand-written literal");
        v
    }

    fn model_type(v: &Value) -> &str {
        v["model"]["type"].as_str().expect("model has no `type`")
    }

    /// Assert that not one legacy shape survived the pass.
    ///
    /// This is the check that says what the module is *for*, and it is the one that does not depend
    /// on `tk-serialize` still carrying its own backwards-compatibility branches. "The canonical
    /// reader accepted the output" is satisfied today by a reader that would also have accepted the
    /// input; "the output contains no legacy shape" is the property that lets those branches be
    /// deleted, and it is checkable on the JSON alone.
    fn assert_no_legacy_residue(v: &Value, what: &str) {
        let model = v["model"].as_object().expect("no model object");
        assert!(
            model.get("type").and_then(Value::as_str).is_some(),
            "{what}: the model still has no `type`"
        );
        assert!(
            model.get("files").is_none(),
            "{what}: the model still names its vocabulary by path"
        );
        assert!(
            !model.get("vocab").is_some_and(Value::is_string),
            "{what}: `vocab` is still a path"
        );
        if let Some(merges) = model.get("merges").and_then(Value::as_array) {
            for m in merges {
                assert!(
                    !m.is_string(),
                    "{what}: a merge is still spelled {m} rather than as a pair"
                );
            }
        }
        // Every `Metaspace` anywhere in the tree, not just the ones this pass is known to visit:
        // a residue check that only looked where the walk looks could not catch a missed position.
        fn every_metaspace(node: &Value, what: &str) {
            match node {
                Value::Object(obj) => {
                    if obj.get("type").and_then(Value::as_str) == Some("Metaspace") {
                        assert!(
                            obj.get("add_prefix_space").is_none(),
                            "{what}: a Metaspace still spells `add_prefix_space`"
                        );
                        assert!(
                            obj.get("str_rep").is_none(),
                            "{what}: a Metaspace still carries `str_rep`"
                        );
                        assert!(
                            obj.get("prepend_scheme").and_then(Value::as_str).is_some(),
                            "{what}: a Metaspace still has no `prepend_scheme`"
                        );
                        assert!(
                            obj.get("split").and_then(Value::as_bool).is_some(),
                            "{what}: a Metaspace still has no `split`"
                        );
                    }
                    for v in obj.values() {
                        every_metaspace(v, what);
                    }
                }
                Value::Array(items) => {
                    for v in items {
                        every_metaspace(v, what);
                    }
                }
                _ => {}
            }
        }
        every_metaspace(v, what);
    }

    // ---------------------------------------------------------------------------------------------
    // Rule 1: model type inference
    // ---------------------------------------------------------------------------------------------

    #[test]
    fn fills_a_missing_model_type() {
        // 1. merges ⇒ BPE.
        assert_eq!(
            model_type(&done(
                r#"{"vocab": {"a": 0, "b": 1}, "merges": [["a", "b"]]}"#,
                ""
            )),
            "BPE"
        );
        // 2. continuing_subword_prefix ⇒ WordPiece.
        assert_eq!(
            model_type(&done(
                r#"{"vocab": {"a": 0}, "continuing_subword_prefix": "@@",
                    "unk_token": "[UNK]", "max_input_chars_per_word": 100}"#,
                ""
            )),
            "WordPiece"
        );
        // 3. array-shaped vocab ⇒ Unigram.
        assert_eq!(
            model_type(&done(
                r#"{"vocab": [["a", 0.0], ["b", -1.0]], "unk_id": 0}"#,
                ""
            )),
            "Unigram"
        );
        // 4. nothing else ⇒ WordLevel.
        assert_eq!(
            model_type(&done(r#"{"vocab": {"a": 0}, "unk_token": "<unk>"}"#, "")),
            "WordLevel"
        );
    }

    /// The ordering trap, and the reason rule 1 is written the way it is.
    ///
    /// A serialized BPE writes every optional field, so it carries `"continuing_subword_prefix":
    /// null`. If the WordPiece test ran first, `gpt2.json` -- no `"type"`, string merges, that null
    /// prefix -- would come out a WordPiece with no merges, and would either fail to load or
    /// tokenize as something else entirely.
    #[test]
    fn merges_beats_continuing_subword_prefix() {
        let v = done(
            r#"{"dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false,
                "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]}"#,
            "",
        );
        assert_eq!(model_type(&v), "BPE");
        // And the null prefix survives untouched: it is a BPE field too.
        assert!(v["model"]["continuing_subword_prefix"].is_null());
    }

    /// A non-null `continuing_subword_prefix` with no merges is still a WordPiece, so the ordering
    /// above is not just "always BPE".
    #[test]
    fn a_prefix_without_merges_is_a_wordpiece() {
        let v = done(
            r#"{"vocab": {"[UNK]": 0, "ab": 1, "@@c": 2}, "unk_token": "[UNK]",
                "continuing_subword_prefix": "@@", "max_input_chars_per_word": 100}"#,
            "",
        );
        assert_eq!(model_type(&v), "WordPiece");
    }

    #[test]
    fn an_existing_model_type_is_never_second_guessed() {
        // This *looks* like a BPE by every shape rule, and the tag still wins.
        let v = done(
            r#"{"type": "WordLevel", "vocab": {"a": 0}, "merges": []}"#,
            "",
        );
        assert_eq!(model_type(&v), "WordLevel");
    }

    // ---------------------------------------------------------------------------------------------
    // Legacy merges
    // ---------------------------------------------------------------------------------------------

    #[test]
    fn rewrites_space_joined_merges_into_pairs() {
        let v = done(
            r#"{"vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b", "ab ab"]}"#,
            "",
        );
        assert_eq!(
            v["model"]["merges"],
            serde_json::json!([["a", "b"], ["ab", "ab"]])
        );
    }

    #[test]
    fn a_merges_array_may_already_be_pairs() {
        let v = done(r#"{"vocab": {"a": 0}, "merges": [["a", "b"], "c d"]}"#, "");
        assert_eq!(
            v["model"]["merges"],
            serde_json::json!([["a", "b"], ["c", "d"]])
        );
    }

    /// The `merges.txt` header, which a config built by pasting that file's lines still carries. It
    /// is dropped rather than converted, because the config path filters it before numbering the
    /// ranks -- converting it would push every merge one rank later.
    #[test]
    fn drops_the_merges_txt_version_header() {
        let v = done(
            r##"{"vocab": {"a": 0}, "merges": ["#version: 0.2", "a b"]}"##,
            "",
        );
        assert_eq!(v["model"]["merges"], serde_json::json!([["a", "b"]]));
    }

    /// A token containing a space is exactly the ambiguity pairs were introduced to remove. The
    /// config path errors (`BadMerges`); guessing here would produce a different tokenizer.
    #[test]
    fn an_ambiguous_legacy_merge_is_an_error() {
        let mut v = config(r#"{"vocab": {"a": 0}, "merges": ["a b c"]}"#, "");
        let err = canonicalize_value(&mut v).unwrap_err();
        assert!(
            matches!(err, ConvertError::BadMerge { .. }),
            "expected BadMerge, got {err}"
        );
    }

    // ---------------------------------------------------------------------------------------------
    // Unigram's array vocab
    // ---------------------------------------------------------------------------------------------

    /// The array is left exactly as it is: it is the canonical Unigram shape, and it is also the
    /// signal that made rule 3 fire.
    #[test]
    fn an_array_shaped_unigram_vocab_is_left_alone() {
        let v = done(
            r#"{"vocab": [["<unk>", 0.0], ["a", -1.0], ["ab", -0.5]], "unk_id": 0}"#,
            "",
        );
        assert_eq!(model_type(&v), "Unigram");
        assert_eq!(
            v["model"]["vocab"],
            serde_json::json!([["<unk>", 0.0], ["a", -1.0], ["ab", -0.5]])
        );
    }

    // ---------------------------------------------------------------------------------------------
    // Rule 2: Metaspace
    // ---------------------------------------------------------------------------------------------

    fn metaspace(spelling: &str) -> Result<Value, ConvertError> {
        let mut v = config(
            r#"{"type": "Unigram", "vocab": [["a", 0.0]], "unk_id": 0}"#,
            &format!(r#", "pre_tokenizer": {spelling}"#),
        );
        canonicalize_value(&mut v)?;
        Ok(v["pre_tokenizer"].clone())
    }

    #[test]
    fn an_absent_prepend_scheme_is_always_not_never() {
        let ms = metaspace(r#"{"type": "Metaspace", "replacement": "▁"}"#).unwrap();
        assert_eq!(ms["prepend_scheme"], "always");
    }

    #[test]
    fn add_prefix_space_true_is_dropped_and_never_overrides_a_scheme() {
        // Only the old key: agrees with the default.
        let ms =
            metaspace(r#"{"type": "Metaspace", "replacement": "▁", "add_prefix_space": true}"#)
                .unwrap();
        assert_eq!(ms["prepend_scheme"], "always");
        assert!(ms.get("add_prefix_space").is_none());

        // Both keys: the explicit scheme wins, and `true` is not treated as a contradiction.
        let ms = metaspace(
            r#"{"type": "Metaspace", "replacement": "▁",
                "add_prefix_space": true, "prepend_scheme": "never"}"#,
        )
        .unwrap();
        assert_eq!(ms["prepend_scheme"], "never");

        let ms = metaspace(
            r#"{"type": "Metaspace", "replacement": "▁",
                "add_prefix_space": true, "prepend_scheme": "first"}"#,
        )
        .unwrap();
        assert_eq!(ms["prepend_scheme"], "first");
    }

    #[test]
    fn add_prefix_space_false_needs_an_agreeing_never() {
        // Agrees: fine, and changes nothing.
        let ms = metaspace(
            r#"{"type": "Metaspace", "replacement": "▁",
                "add_prefix_space": false, "prepend_scheme": "never"}"#,
        )
        .unwrap();
        assert_eq!(ms["prepend_scheme"], "never");

        // Alone: checked against the *defaulted* `always`, so it is a hard error.
        let err =
            metaspace(r#"{"type": "Metaspace", "replacement": "▁", "add_prefix_space": false}"#)
                .unwrap_err();
        assert!(matches!(err, ConvertError::PrefixSpaceMismatch), "{err}");

        // Explicitly disagreeing: also a hard error.
        let err = metaspace(
            r#"{"type": "Metaspace", "replacement": "▁",
                "add_prefix_space": false, "prepend_scheme": "always"}"#,
        )
        .unwrap_err();
        assert!(matches!(err, ConvertError::PrefixSpaceMismatch), "{err}");

        let err = metaspace(
            r#"{"type": "Metaspace", "replacement": "▁",
                "add_prefix_space": false, "prepend_scheme": "first"}"#,
        )
        .unwrap_err();
        assert!(matches!(err, ConvertError::PrefixSpaceMismatch), "{err}");
    }

    #[test]
    fn str_rep_is_thrown_away_and_split_is_spelled_out() {
        let ms = metaspace(
            r#"{"type": "Metaspace", "replacement": "▁",
                "str_rep": "▁", "add_prefix_space": true}"#,
        )
        .unwrap();
        assert!(ms.get("str_rep").is_none());
        assert_eq!(ms["split"], true);
        // An explicit `split` is kept.
        let ms = metaspace(r#"{"type": "Metaspace", "replacement": "▁", "split": false}"#).unwrap();
        assert_eq!(ms["split"], false);
    }

    #[test]
    fn an_unknown_prepend_scheme_is_refused() {
        let err = metaspace(
            r#"{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "sometimes"}"#,
        )
        .unwrap_err();
        assert!(
            matches!(err, ConvertError::UnknownPrependScheme { .. }),
            "{err}"
        );
    }

    #[test]
    fn a_metaspace_replacement_must_be_one_character() {
        let err = metaspace(r#"{"type": "Metaspace"}"#).unwrap_err();
        assert!(matches!(err, ConvertError::MetaspaceNoReplacement), "{err}");
        let err = metaspace(r#"{"type": "Metaspace", "replacement": "__"}"#).unwrap_err();
        assert!(
            matches!(err, ConvertError::MetaspaceBadReplacement { .. }),
            "{err}"
        );
    }

    /// t5 and albert both spell `Metaspace` twice -- inside a pre-tokenizer `Sequence` and again as
    /// the decoder -- and both copies carry the legacy spelling. All three positions plus the
    /// nesting are exercised here in one config.
    #[test]
    fn every_metaspace_position_is_walked() {
        let mut v = config(
            r#"{"type": "Unigram", "vocab": [["a", 0.0]], "unk_id": 0}"#,
            r#", "normalizer": {"type": "Sequence", "normalizers": [
                   {"type": "Metaspace", "replacement": "▁", "str_rep": "▁"}]}
               , "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                   {"type": "WhitespaceSplit"},
                   {"type": "Sequence", "pretokenizers": [
                     {"type": "Metaspace", "replacement": "▁", "add_prefix_space": true}]}]}
               , "decoder": {"type": "Sequence", "decoders": [
                   {"type": "Metaspace", "replacement": "▁", "str_rep": "▁",
                    "add_prefix_space": true}]}"#,
        );
        canonicalize_value(&mut v).unwrap();

        for found in [
            &v["normalizer"]["normalizers"][0],
            &v["pre_tokenizer"]["pretokenizers"][1]["pretokenizers"][0],
            &v["decoder"]["decoders"][0],
        ] {
            assert_eq!(found["prepend_scheme"], "always");
            assert_eq!(found["split"], true);
            assert!(found.get("add_prefix_space").is_none());
            assert!(found.get("str_rep").is_none());
        }
    }

    /// A `Metaspace` inside a post-processor's `special_tokens` is *data*, not a component, and must
    /// not be rewritten. This pins the "only descend through the four `Sequence` child keys"
    /// decision.
    #[test]
    fn data_objects_are_not_mistaken_for_components() {
        let mut v = config(
            r#"{"type": "Unigram", "vocab": [["a", 0.0]], "unk_id": 0}"#,
            r#", "post_processor": {"type": "TemplateProcessing", "single": [],
                  "pair": [], "special_tokens": {"x": {"type": "Metaspace"}}}"#,
        );
        // No `replacement`, so if the walk reached it this would be `MetaspaceNoReplacement`.
        canonicalize_value(&mut v).unwrap();
        assert!(v["post_processor"]["special_tokens"]["x"]["type"] == "Metaspace");
    }

    // ---------------------------------------------------------------------------------------------
    // Vocabulary given as a file path
    // ---------------------------------------------------------------------------------------------

    #[test]
    fn inlines_a_bpe_given_as_two_file_paths() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("vocab.json"),
            r#"{"a": 0, "b": 1, "ab": 2}"#,
        )
        .unwrap();
        std::fs::write(dir.path().join("merges.txt"), "#version: 0.2\na b\n").unwrap();
        let cfg = dir.path().join("tokenizer.json");
        std::fs::write(
            &cfg,
            r#"{"version": "1.0", "model":
                 {"files": ["vocab.json", "merges.txt"], "unk_token": null}}"#,
        )
        .unwrap();

        let out: Value = serde_json::from_str(&canonicalize_file(&cfg).unwrap()).unwrap();
        assert_eq!(model_type(&out), "BPE");
        assert!(out["model"].get("files").is_none());
        assert_eq!(
            out["model"]["vocab"],
            serde_json::json!({"a": 0, "b": 1, "ab": 2})
        );
        assert_eq!(out["model"]["merges"], serde_json::json!([["a", "b"]]));
    }

    /// The one-file spellings, and the `vocab.txt` format: one token per line, id = line number.
    #[test]
    fn inlines_a_wordpiece_vocab_txt() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("vocab.txt"), "[UNK]\nab\n@@c\n").unwrap();
        let cfg = dir.path().join("tokenizer.json");
        std::fs::write(
            &cfg,
            r#"{"version": "1.0", "model": {"files": "vocab.txt", "unk_token": "[UNK]",
                 "continuing_subword_prefix": "@@", "max_input_chars_per_word": 100}}"#,
        )
        .unwrap();

        let out: Value = serde_json::from_str(&canonicalize_file(&cfg).unwrap()).unwrap();
        assert_eq!(model_type(&out), "WordPiece");
        assert_eq!(
            out["model"]["vocab"],
            serde_json::json!({"[UNK]": 0, "ab": 1, "@@c": 2})
        );
    }

    /// The by-key spelling, and a `vocab` that is itself a path rather than a map.
    #[test]
    fn inlines_the_object_and_bare_string_spellings() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("v.json"), r#"{"a": 0, "b": 1}"#).unwrap();
        std::fs::write(dir.path().join("m.txt"), "a b\n").unwrap();

        let cfg = dir.path().join("by_key.json");
        std::fs::write(
            &cfg,
            r#"{"version": "1.0", "model": {"files": {"vocab": "v.json", "merges": "m.txt"}}}"#,
        )
        .unwrap();
        let out: Value = serde_json::from_str(&canonicalize_file(&cfg).unwrap()).unwrap();
        assert_eq!(model_type(&out), "BPE");
        assert_eq!(out["model"]["vocab"], serde_json::json!({"a": 0, "b": 1}));

        let cfg = dir.path().join("bare.json");
        std::fs::write(
            &cfg,
            r#"{"version": "1.0", "model": {"vocab": "v.json", "unk_token": "<unk>"}}"#,
        )
        .unwrap();
        let out: Value = serde_json::from_str(&canonicalize_file(&cfg).unwrap()).unwrap();
        assert_eq!(model_type(&out), "WordLevel");
        assert_eq!(out["model"]["vocab"], serde_json::json!({"a": 0, "b": 1}));
    }

    #[test]
    fn a_missing_vocab_file_names_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = dir.path().join("tokenizer.json");
        std::fs::write(
            &cfg,
            r#"{"version": "1.0", "model": {"files": "nope.txt"}}"#,
        )
        .unwrap();
        let err = canonicalize_file(&cfg).unwrap_err();
        assert!(err.to_string().contains("nope.txt"), "{err}");
    }

    // ---------------------------------------------------------------------------------------------
    // Shape of the pass itself
    // ---------------------------------------------------------------------------------------------

    #[test]
    fn refuses_something_that_is_not_a_tokenizer_config() {
        assert!(matches!(
            canonicalize_str("[]").unwrap_err(),
            ConvertError::NotAnObject { .. }
        ));
        assert!(matches!(
            canonicalize_str("{}").unwrap_err(),
            ConvertError::MissingModel
        ));
        assert!(matches!(
            canonicalize_str(r#"{"model": 3}"#).unwrap_err(),
            ConvertError::ModelNotObject { .. }
        ));
        assert!(matches!(
            canonicalize_str("not json").unwrap_err(),
            ConvertError::Json(_)
        ));
    }

    /// Canonicalising an already-canonical file must be a no-op that still succeeds -- that is what
    /// makes it safe to run unconditionally in front of a reader.
    #[test]
    fn is_idempotent() {
        let legacy = r##"{"version": "1.0",
            "normalizer": null,
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "WhitespaceSplit"},
                {"type": "Metaspace", "replacement": "▁", "str_rep": "▁",
                 "add_prefix_space": true}]},
            "decoder": {"type": "Metaspace", "replacement": "▁", "str_rep": "▁",
                        "add_prefix_space": true},
            "model": {"continuing_subword_prefix": null, "vocab": {"a": 0, "b": 1, "ab": 2},
                      "merges": ["#version: 0.2", "a b"]}}"##;

        let once = canonicalize_str(legacy).unwrap();
        let twice = canonicalize_str(&once).unwrap();
        assert_eq!(once, twice, "the pass is not idempotent");
        assert_no_legacy_residue(&serde_json::from_str(&once).unwrap(), "the legacy literal");

        // And the third pass over the *value* API agrees with the string one.
        let mut v: Value = serde_json::from_str(&twice).unwrap();
        canonicalize_value(&mut v).unwrap();
        assert_eq!(serde_json::to_string_pretty(&v).unwrap(), twice);
    }

    // ---------------------------------------------------------------------------------------------
    // End to end: every fixture, through the canonical reader
    // ---------------------------------------------------------------------------------------------

    /// The only test that proves the point of the module: after this pass, the *canonical* reader
    /// can read the file. `tk_serialize::from_json` is a dev-dependency with every component
    /// feature on, precisely so this cannot silently drop to "read nothing and pass".
    ///
    /// ## Why the reader's verdict is compared before *and* after
    ///
    /// "The canonical reader accepts the output" is not quite the property to assert, because that
    /// reader also refuses things for reasons that have nothing to do with the file's age: the
    /// pipeline cannot express a `ByteLevel` with `add_prefix_space: true`, or a `Metaspace` with
    /// `prepend_scheme: first`, and no amount of converting will change that. `data/tokenizer.json`
    /// is exactly that case, and asserting "everything reads" would leave this test with a
    /// hand-maintained exception list that quietly grows.
    ///
    /// So the assertion is the one a converter actually owes its caller, in two halves:
    ///
    /// - **no regression** — a file the reader accepted before must still be accepted after;
    /// - **no residue** — a file the reader still refuses must fail with the *same error it already
    ///   failed with*. An unchanged message is proof the refusal is about a component the pipeline
    ///   cannot build, not about a field this pass was supposed to fill.
    ///
    /// The err → ok column is the conversion doing visible work. It is small today only because
    /// `tk-serialize` has not yet dropped its own three legacy branches; when it does, this column
    /// is what will carry the fixtures that only load through this pass.
    ///
    /// Two fixtures in `data/` are not tokenizer configs and are reported as skipped rather than
    /// failed: `gpt2-vocab.json` is a bare `{token: id}` map (which happens to contain a token
    /// spelled `model`, hence the "must be an object" check rather than a "has a model" one) and
    /// `unigram.json` is a bare Unigram model with no surrounding config.
    ///
    /// A missing `data/` skips the whole test: the directory is gitignored and populated by
    /// `make fixtures`, so a fresh checkout has none of it.
    #[test]
    fn every_fixture_canonicalises_into_something_the_canonical_reader_accepts() {
        let dir = std::path::Path::new(DATA);
        let Ok(entries) = std::fs::read_dir(dir) else {
            eprintln!("skipping: no fixture directory at {DATA}");
            return;
        };
        let mut files: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .collect();
        files.sort();
        if files.is_empty() {
            eprintln!("skipping: no *.json fixtures in {DATA}");
            return;
        }

        let read = |text: &str| {
            tk_serialize::from_json(text)
                .map(|_| ())
                .map_err(|e| e.to_string())
        };

        let (mut ok, mut fixed, mut unreadable, mut skipped) = (0usize, 0usize, 0usize, 0usize);
        for (i, path) in files.iter().enumerate() {
            let at = format!("[{}/{}]", i + 1, files.len());
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            let text = std::fs::read_to_string(path).unwrap();
            let parsed: Value = match serde_json::from_str(&text) {
                Ok(v) => v,
                Err(e) => panic!("{at} {name}: not JSON at all: {e}"),
            };
            if !parsed.get("model").is_some_and(Value::is_object) {
                eprintln!("{at} {name}  skipped (not a tokenizer.json)");
                skipped += 1;
                continue;
            }

            let before = read(&text);
            let canonical = canonicalize_file(path).unwrap_or_else(|e| panic!("{at} {name}: {e}"));
            // The check that does not depend on what the reader currently tolerates.
            assert_no_legacy_residue(&serde_json::from_str(&canonical).unwrap(), &name);
            let after = read(&canonical);

            match (before, after) {
                (Ok(()), Ok(())) => {
                    eprintln!("{at} {name}  ok");
                    ok += 1;
                }
                (Err(was), Ok(())) => {
                    eprintln!("{at} {name}  ok (the conversion fixed it: {was})");
                    fixed += 1;
                }
                (Ok(()), Err(now)) => panic!(
                    "{at} {name}: the canonical reader read the ORIGINAL and refuses the \
                     canonicalised one -- this pass broke it: {now}"
                ),
                (Err(was), Err(now)) => {
                    assert_eq!(
                        was, now,
                        "{at} {name}: still refused, and for a different reason than before, \
                         so this pass left a field unfilled"
                    );
                    eprintln!("{at} {name}  refused before and after, unchanged: {now}");
                    unreadable += 1;
                }
            }
        }
        eprintln!(
            "{ok} already read + {fixed} fixed by this pass = {} read through the canonical \
             reader; {unreadable} refused for a reason this pass cannot affect; {skipped} not \
             tokenizer configs",
            ok + fixed
        );
        assert!(ok + fixed > 0, "no fixture was actually read");
    }
}
