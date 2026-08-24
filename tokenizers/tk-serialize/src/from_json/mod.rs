//! Build a [`PipelineTokenizer`] from a `tokenizer.json`.
//!
//! ## Legacy shapes accepted
//!
//! Three, because between them they cover most of the Hub (`gpt2` hits the first two):
//!
//! - a `model` with no `"type"` — the kind is inferred from which keys are present
//! - `merges` spelled as `"a b"` strings rather than `["a", "b"]` pairs
//! - a `Metaspace` spelled with `add_prefix_space` rather than `prepend_scheme`, which is what t5
//!   and albert still ship (see [`read_prepend_scheme`] for the exact rule, quirk included)
//!
//! Everything else legacy — vocab-as-file-path, an untagged component object — is refused with a
//! message naming what to convert.
mod added_tokens;
mod decoders;
mod model;
mod normalizers;
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
use self::post_processors::read_post_processor;
use self::pre_tokenizers::read_pre_tokenizer;
use crate::json::Json;
use tk_encode::models::bpe::{BpeConfig, PipelineBPE, Vocab};
use tk_encode::pipeline::{
    NormalizerChain, PipelineModel, PipelineNormalizer, PipelinePreTokenizer, PipelineTokenizer,
};
use tk_encode::tokenizer::Result;
use tk_encode::vocab::bucket_added_vocabulary::AddedVocabulary as BucketAddedVocabulary;

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

fn from_json_value(doc: &Json<'_>) -> Result<PipelineTokenizer> {
    // The config path gates on this exact string, so a future format bump is a loud failure
    // rather than a silently mis-read file.
    if let Some(v) = doc.field("version").and_then(Json::as_str)
        && v != "1.0"
    {
        return Err(format!("unknown tokenizer version '{v}'").into());
    }

    let mut normalizers = read_normalizers(doc.field("normalizer"))?;
    // Takes `normalizers` because a `Metaspace` pre-tokenizer contributes one, and it has to
    // land *after* the declared normalizer — the config asks for the whole normalizer first,
    // then the pre-tokenizer.
    let (pre_tokenizer, byte_level) =
        read_pre_tokenizer(doc.field("pre_tokenizer"), &mut normalizers)?;

    let model_cfg = doc
        .field("model")
        .ok_or_else(|| -> tk_encode::Error { "config has no `model`".into() })?;
    let kind = model_kind(model_cfg);
    if byte_level && kind != "BPE" {
        return Err(format!("ByteLevel pre tokenizer is not supported with model {kind}").into());
    }
    if model_cfg.get("files").is_some() {
        return Err(unsupported("a model whose vocab is a file path (`files`)"));
    }

    match kind {
        "BPE" => {
            let (vocab, merges, options) = read_bpe(model_cfg, byte_level)?;
            build(
                doc,
                normalizers,
                pre_tokenizer,
                vocab,
                |v| v.len(),
                |v, t| v.get(t).copied(),
                |vocab| {
                    Ok(PipelineModel::BPE(PipelineBPE::from_config(BpeConfig {
                        vocab,
                        merges,
                        ..options
                    })?))
                },
            )
        }
        #[cfg(feature = "wordpiece")]
        "WordPiece" => build(
            doc,
            normalizers,
            pre_tokenizer,
            read_wordpiece(model_cfg)?,
            tk_encode::Model::get_vocab_size,
            |m, t| tk_encode::Model::token_to_id(m, t),
            |wp| Ok(PipelineModel::WordPiece(wp.try_into()?)),
        ),
        #[cfg(feature = "unigram")]
        "Unigram" => build(
            doc,
            normalizers,
            pre_tokenizer,
            read_unigram(model_cfg)?,
            tk_encode::Model::get_vocab_size,
            |m, t| tk_encode::Model::token_to_id(m, t),
            |u| Ok(PipelineModel::Unigram(u)),
        ),
        #[cfg(feature = "wordlevel")]
        "WordLevel" => build(
            doc,
            normalizers,
            pre_tokenizer,
            read_wordlevel(model_cfg)?,
            tk_encode::Model::get_vocab_size,
            |m, t| tk_encode::Model::token_to_id(m, t),
            |wl| Ok(PipelineModel::WordLevel(wl)),
        ),
        // Covers both an unrecognised `"type"` and a known one whose per-model feature is off,
        // because from here the two are indistinguishable: the arm simply is not compiled.
        other => Err(format!(
            "the slim JSON reader cannot read the `{other}` model: either it is not a model \
             type this crate knows, or its feature (`unigram`, `wordpiece`, `wordlevel`) is \
             off in this build"
        )
        .into()),
    }
}

fn build<M>(
    doc: &Json<'_>,
    normalizers: Vec<PipelineNormalizer>,
    pre_tokenizer: PipelinePreTokenizer,
    concrete: M,
    vocab_size: impl FnOnce(&M) -> usize,
    token_to_id: impl Fn(&M, &str) -> Option<u32>,
    lower: impl FnOnce(M) -> Result<PipelineModel>,
) -> Result<PipelineTokenizer> {
    let added = read_added_tokens(doc.field("added_tokens"))?;

    let mut added_vocabulary = BucketAddedVocabulary::new();
    // TODO: this has nothing to do here.
    let declared = match normalizers.last() {
        Some(PipelineNormalizer::Metaspace(_)) => &normalizers[..normalizers.len() - 1],
        _ => &normalizers[..],
    };
    let chain = NormalizerChain(declared);
    added_vocabulary.add_tokens(
        added,
        vocab_size(&concrete),
        |t| token_to_id(&concrete, t),
        Some(&chain),
    )?;
    // TODO: we need to read encode_special_tokens from the config as well.
    let model = lower(concrete)?;

    Ok(PipelineTokenizer::from_parts(
        added_vocabulary,
        normalizers,
        pre_tokenizer,
        model,
        read_post_processor(doc.field("post_processor"))?,
        read_decoder(doc.field("decoder"))?,
    ))
}

/// Which model a config declares. `"type"` when it has one, otherwise inferred from the keys —
/// `gpt2` and a few other don't have a type, we try to still infer it from the saved keys.
fn model_kind(cfg: &Json<'_>) -> &'static str {
    if let Some(t) = cfg.type_tag() {
        return match t {
            "BPE" => "BPE",
            "Unigram" => "Unigram",
            "WordPiece" => "WordPiece",
            "WordLevel" => "WordLevel",
            _ => "unknown",
        };
    }
    if cfg.get("merges").is_some() {
        "BPE"
    } else if cfg.get("vocab").and_then(Json::as_array).is_some() {
        // Unigram's vocab is an array of [token, score] pairs; everyone else's is an object.
        "Unigram"
    } else if cfg.get("continuing_subword_prefix").is_some() {
        "WordPiece"
    } else {
        "WordLevel"
    }
}
