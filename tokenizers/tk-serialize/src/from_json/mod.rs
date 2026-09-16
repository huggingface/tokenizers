//! Build a [`PipelineTokenizer`] from a `tokenizer.json`.
//!
//! Reads the canonical `2.0` format and nothing else. A legacy file is tk-convert's input, not
//! this reader's: it produces a `2.0` file, and only then does this run.
//!
//! ## TODO(pr3): port to tk-convert
//!
//! Each of these was accepted here and is now refused. tk-convert has to rewrite them:
//!
//! - `"version": "1.0"` — everything below is what makes a file one
//! - a `model` with no `"type"` — the kind was inferred from which keys are present: `merges` meant
//!   BPE, an array `vocab` meant Unigram, `continuing_subword_prefix` meant WordPiece, else
//!   WordLevel
//! - `merges` spelled as `"a b"` strings rather than `["a", "b"]` pairs — split on the first space
//! - a `Metaspace` *pre-tokenizer*, which is two components: it becomes a `MetaspaceNormalizer` in
//!   the `normalizer` slot plus a `Split` on the delimiter with `MergedWithNext`
//! - `Sequence[WhitespaceSplit, Metaspace]`, the t5/albert shape — the same, with
//!   `drop_whitespace: true`
//! - `add_prefix_space` on a `Metaspace`, with its quirk: `false` is an error unless
//!   `prepend_scheme: "never"` is spelled beside it, and `true` is ignored outright
//! - a `Metaspace` decoder with no `prepend_scheme` — it defaulted to `always`
//! - a `ByteLevel` *pre-tokenizer* — it becomes `"byte_level": true` on the model plus, for
//!   `use_regex: true`, a `Split` on the GPT-2 regex with `Isolated`; `use_regex: false` leaves no
//!   pre-tokenizer at all. `add_prefix_space: true` on one is still unsupported downstream
//!
//! Refused by both, because nothing converts it: a vocabulary named by path (`files`).
mod added_tokens;
mod decoders;
mod model;
mod normalizers;
mod padding;
mod post_processors;
mod pre_tokenizers;

#[cfg(test)]
mod tests;

use self::added_tokens::read_added_tokens;
use self::decoders::read_decoder;
use self::model::read_bpe;
#[cfg(feature = "unigram")]
use self::model::read_unigram;
#[cfg(feature = "wordlevel")]
use self::model::read_wordlevel;
#[cfg(feature = "wordpiece")]
use self::model::read_wordpiece;
use self::normalizers::read_normalizers;
use self::padding::read_padding;
use self::post_processors::read_post_processor;
use self::pre_tokenizers::read_pre_tokenizer;
use crate::json::Json;
use std::collections::BTreeMap;
use std::sync::Arc;
use tk_encode::models::bpe::{BpeConfig, PipelineBPE};
use tk_encode::pipeline::{PipelineModel, PipelineTokenizer, TokenizerBuilder};
use tk_encode::tokenizer::Result;

fn unsupported(what: &str) -> tk_encode::Error {
    format!(
        "the slim JSON reader does not support {what}. Convert the config offline \
         (tk-convert), or build with the `config` feature for the full reader. You are probably attempting to read a legacy tokenizer.json file saved before v1."
    )
    .into()
}

/// A component this build *could* read if a feature were on. Distinct from [`unsupported`], because
/// the fix is a `--features` flag rather than an offline conversion.
// One arm per gated component calls this; it exists as soon as any of them is compiled out.
#[cfg(any(not(feature = "normalizers"), not(feature = "unicode-scripts")))]
fn needs_feature(what: &str, feature: &str) -> tk_encode::Error {
    format!("reading {what} needs the `{feature}` feature, which is off in this build").into()
}

// Free functions rather than inherent methods: `PipelineTokenizer` is defined in `tk-encode`, and
// reading a `tokenizer.json` is this crate's whole job.
/// Read a `tokenizer.json` from a string.
pub fn from_json(text: &str) -> Result<PipelineTokenizer> {
    let doc = Json::parse(text).map_err(|e| -> tk_encode::Error { e.to_string().into() })?;
    from_json_value(&doc)
}

/// Read a `tokenizer.json` from a file.
pub fn from_json_file(path: impl AsRef<std::path::Path>) -> Result<PipelineTokenizer> {
    let text = std::fs::read_to_string(path)?;
    from_json(&text)
}

/// Read a lone `post_processor` -- the value of that one key -- rather than a whole
/// `tokenizer.json`. `null` and the empty string are the default frame, the same as a config that
/// leaves the key out.
///
/// The bindings hold a post-processor on its own, so they need this half of the reader; going
/// through it is what keeps one parser for the canonical spelling.
pub fn post_processor_from_json(text: &str) -> Result<tk_encode::pipeline::PipelinePostProcessor> {
    let text = text.trim();
    if text.is_empty() || text == "null" {
        return Ok(tk_encode::pipeline::PipelinePostProcessor::default());
    }
    let doc = Json::parse(text).map_err(|e| -> tk_encode::Error { e.to_string().into() })?;
    read_post_processor(Some(&doc))
}

fn from_json_value(doc: &Json<'_>) -> Result<PipelineTokenizer> {
    // `2.0` is the canonical format this crate reads and writes. A `1.0` file is a legacy file:
    // tk-convert turns one into a `2.0` file, and this reader never sees it.
    match doc.field("version").and_then(Json::as_str) {
        Some("2.0") => {}
        Some(v) => {
            return Err(format!("tokenizer version '{v}' is not `2.0`; convert it first").into());
        }
        None => return Err(unsupported("a config with no `version`")),
    }

    let normalizers = read_normalizers(doc.field("normalizer"))?;
    let pre_tokenizer = read_pre_tokenizer(doc.field("pre_tokenizer"))?;

    let model_cfg = doc
        .field("model")
        .ok_or_else(|| -> tk_encode::Error { "config has no `model`".into() })?;
    let kind = model_kind(model_cfg)?;
    if model_cfg.get("files").is_some() {
        return Err(unsupported("a model whose vocab is a file path (`files`)"));
    }

    let model = match kind {
        "BPE" => {
            let (vocab, merges, options) = read_bpe(model_cfg)?;
            PipelineModel::BPE(PipelineBPE::from_config(BpeConfig {
                vocab,
                merges,
                ..options
            })?)
        }
        #[cfg(feature = "wordpiece")]
        "WordPiece" => PipelineModel::WordPiece(read_wordpiece(model_cfg)?.try_into()?),
        #[cfg(feature = "unigram")]
        "Unigram" => PipelineModel::Unigram(read_unigram(model_cfg)?),
        #[cfg(feature = "wordlevel")]
        "WordLevel" => PipelineModel::WordLevel(read_wordlevel(model_cfg)?),
        // Covers both an unrecognised `"type"` and a known one whose per-model feature is off,
        // because from here the two are indistinguishable: the arm simply is not compiled.
        other => {
            return Err(format!(
                "the slim JSON reader cannot read the `{other}` model: either it is not a model \
                 type this crate knows, or its feature (`unigram`, `wordpiece`, `wordlevel`) is \
                 off in this build"
            )
            .into());
        }
    };

    TokenizerBuilder {
        added_tokens: read_added_tokens(doc.field("added_tokens"))?,
        normalizers,
        pre_tokenizer,
        model: Arc::new(model),
        post_processor: read_post_processor(doc.field("post_processor"))?,
        decoder: read_decoder(doc.field("decoder"))?,
        role_to_token: read_role_to_token(doc.field("role_to_token"))?,
        padding: read_padding(doc.field("padding"))?,
    }
    .build()
}

/// `{"eos_token": "</s>", ...}`. Absent or `null` means the config declares no roles.
fn read_role_to_token(cfg: Option<&Json<'_>>) -> Result<BTreeMap<String, String>> {
    let Some(cfg) = cfg else {
        return Ok(BTreeMap::new());
    };
    let entries = cfg
        .entries()
        .ok_or_else(|| unsupported("a `role_to_token` that is not an object"))?;
    entries
        .map(|(role, token)| {
            let token = token
                .as_str()
                .ok_or_else(|| unsupported("a `role_to_token` value that is not a string"))?;
            Ok((role.to_string(), token.to_string()))
        })
        .collect()
}

/// Which model a config declares. Canonical files tag it; inferring the kind from which keys are
/// present is what tk-convert does to a legacy file before this reader sees it.
fn model_kind(cfg: &Json<'_>) -> Result<&'static str> {
    Ok(match cfg.type_tag() {
        Some("BPE") => "BPE",
        Some("Unigram") => "Unigram",
        Some("WordPiece") => "WordPiece",
        Some("WordLevel") => "WordLevel",
        Some(_) => "unknown",
        None => return Err(unsupported("a `model` with no `type`")),
    })
}
