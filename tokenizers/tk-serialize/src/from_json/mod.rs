//! Build a [`PipelineTokenizer`] straight from a `tokenizer.json`, with no serde anywhere.
//!
//! This is the slim read path. It parses the config with [`tk_encode::tokenizer::json`] and constructs
//! the pipeline-native types directly, because naming a wrapper enum makes every variant of it
//! reachable, and that reachability is most of what this exercise exists to remove. In particular
//! neither `ModelWrapper` nor `NormalizerWrapper` is ever named: the concrete model goes straight
//! into a [`PipelineModel`], and each normalizer into its own [`PipelineNormalizer`] variant.
//!
//! The decoder is the single exception, and a deliberate one: it is off the encode path and all of
//! its variants are small, so the reader builds a [`DecoderRuntime`] directly. See [`read_decoder`].
//!
//! A child module of `pipeline` on purpose: `Slice` and `Template` are reused rather than copied, so
//! the post-processor is lowered by exactly the same code the config path uses rather than a second
//! copy of the rules. The pipeline itself is assembled through
//! [`PipelineTokenizer::from_parts`], which both readers share.
//!
//! ## Construction order is load-bearing
//!
//! Added tokens must be replayed into the vocabulary **before** the model is lowered to a
//! [`PipelineModel`], against the *concrete* model, and in id order. `add_tokens` reuses a model id
//! when the token is already in the vocabulary, so replaying in order is what reproduces the id
//! assignment the config path produces. Getting this wrong drifts ids silently — `json_oracle` is
//! the gate that catches it.
//!
//! The concrete model differs per model kind, which is exactly why [`build`] is generic over it:
//! the shared tail is written once and each kind hands it its own model plus the closure that
//! lowers it. That avoids a `ModelWrapper` — the one type whose mere mention drags in every model's
//! deserializer.
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
//! message naming what to convert. That is the whole point of the split: the BC lives in the
//! offline crate.

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
// The writer's base64 encoder is checked against this decoder rather than against a table of
// expected strings, so the pair is tested as a pair.
#[cfg(test)]
pub(crate) use self::normalizers::base64_decode;
use self::post_processors::read_post_processor;
use self::pre_tokenizers::read_pre_tokenizer;
use crate::json::{Json, JsonExt};
use tk_encode::models::bpe::{PipelineBPE, Vocab};
use tk_encode::pipeline::{
    NormalizerChain, PipelineModel, PipelineNormalizer, PipelinePreTokenizer, PipelineTokenizer,
};
use tk_encode::tokenizer::Result;
use tk_encode::vocab::bucket_added_vocabulary::AddedVocabulary as BucketAddedVocabulary;

/// What the reader will not do, phrased as an instruction rather than a complaint.
fn unsupported(what: &str) -> tk_encode::Error {
    format!(
        "the slim JSON reader does not support {what}. Convert the config offline \
         (tk-convert), or build with the `config` feature for the full reader."
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
    if let Some(v) = doc.get_some("version").and_then(Json::as_str)
        && v != "1.0"
    {
        return Err(format!("unknown tokenizer version '{v}'").into());
    }

    let mut normalizers = read_normalizers(doc.get_some("normalizer"))?;
    // Takes `normalizers` because a `Metaspace` pre-tokenizer contributes one, and it has to
    // land *after* the declared normalizer — the config asks for the whole normalizer first,
    // then the pre-tokenizer.
    let (pre_tokenizer, with_byte_level) =
        read_pre_tokenizer(doc.get_some("pre_tokenizer"), &mut normalizers)?;

    let model_cfg = doc
        .get_some("model")
        .ok_or_else(|| -> tk_encode::Error { "config has no `model`".into() })?;
    let kind = model_kind(model_cfg);
    if with_byte_level && kind != "BPE" {
        return Err(format!("ByteLevel pre tokenizer is not supported with model {kind}").into());
    }
    if model_cfg.get("files").is_some() {
        return Err(unsupported("a model whose vocab is a file path (`files`)"));
    }

    match kind {
        // The only model with no `tk_encode::Model` of its own to hand to `build`: the
        // config-shaped `BPE` that used to play that part lives in `tk-convert`, which this crate
        // must not depend on. `build` gets the vocabulary instead -- see [`VocabOnly`] -- and
        // lowers it into the model afterwards, which keeps the replay-then-lower order intact.
        "BPE" => {
            let (vocab, merges, options) = read_bpe(model_cfg, with_byte_level)?;
            build(doc, normalizers, pre_tokenizer, VocabOnly(vocab), |vocab| {
                Ok(PipelineModel::BPE(PipelineBPE::from_vocab_and_merges(
                    vocab.0, merges, options,
                )?))
            })
        }
        #[cfg(feature = "wordpiece")]
        "WordPiece" => build(
            doc,
            normalizers,
            pre_tokenizer,
            read_wordpiece(model_cfg)?,
            |wp| Ok(PipelineModel::WordPiece(wp.try_into()?)),
        ),
        #[cfg(feature = "unigram")]
        "Unigram" => build(
            doc,
            normalizers,
            pre_tokenizer,
            read_unigram(model_cfg)?,
            |u| Ok(PipelineModel::Unigram(u)),
        ),
        #[cfg(feature = "wordlevel")]
        "WordLevel" => build(
            doc,
            normalizers,
            pre_tokenizer,
            read_wordlevel(model_cfg)?,
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

/// The half of construction that does not depend on which model was read.
///
/// `concrete` is the model as its own type, which is what `add_tokens` has to see; `lower` turns it
/// into a [`PipelineModel`] afterwards. Splitting it that way is what keeps the order in the module
/// docs enforceable — the model cannot be lowered before the replay, because `lower` consumes it.
fn build<M: tk_encode::Model>(
    doc: &Json<'_>,
    normalizers: Vec<PipelineNormalizer>,
    pre_tokenizer: PipelinePreTokenizer,
    concrete: M,
    lower: impl FnOnce(M) -> Result<PipelineModel>,
) -> Result<PipelineTokenizer> {
    let added = read_added_tokens(doc.get_some("added_tokens"))?;

    let mut added_vocabulary = BucketAddedVocabulary::new();
    // The whole chain, not just the first member: a config `Sequence` was flattened into
    // `normalizers`, and a `normalized: true` added token has to see all of it or its id moves.
    let chain = NormalizerChain(&normalizers);
    added_vocabulary.add_tokens(added, &concrete, Some(&chain))?;
    // `encode_special_tokens` is deliberately not read: it is runtime state, not one of the nine
    // fields a `tokenizer.json` carries, so the config path never sets it from a file either. Both
    // vocabularies default it to `false`, and honouring the key here would be the one way to end up
    // with a *different* setting than the config path for the same file.

    let model = lower(concrete)?;

    Ok(PipelineTokenizer::from_parts(
        added_vocabulary,
        normalizers,
        pre_tokenizer,
        model,
        read_post_processor(doc.get_some("post_processor"))?,
        read_decoder(doc.get_some("decoder"))?,
    ))
}

/// Which model a config declares. `"type"` when it has one, otherwise inferred from the keys —
/// `gpt2.json` and five other fixtures in `data/` carry no `"type"` at all.
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
    // Key presence, not shape-matching over every variant: ~4 lines instead of the untagged
    // machinery's ~480. The order mirrors `ModelUntagged`'s, which is what decides ties on the
    // config path: BPE first, then WordPiece before WordLevel (WordLevel's shape is a subset).
    if cfg.get("merges").is_some() {
        "BPE"
    } else if cfg.get("vocab").and_then(Json::as_arr).is_some() {
        // Unigram's vocab is an array of [token, score] pairs; everyone else's is an object.
        "Unigram"
    } else if cfg.get("continuing_subword_prefix").is_some() {
        "WordPiece"
    } else {
        "WordLevel"
    }
}

/// A vocabulary standing in for a model, so that `build` can replay the added tokens.
///
/// [`BucketAddedVocabulary::add_tokens`] wants a [`tk_encode::Model`], and it asks exactly two
/// things of it: how many entries the vocabulary has, and whether it already contains a given
/// token. Every model but BPE hands over its own value, which implements that trait. `PipelineBPE`
/// cannot: it implements the *pipeline* `Model` trait instead, and -- decisively -- a byte-level one
/// has already decoded its vocabulary at load, so asking it for `"\u{120}the"` would miss where the
/// config's own vocabulary hits, and every added token after it would get a different id.
///
/// So the reader keeps the vocabulary object it parsed and answers from that. The other four methods
/// are never reached; they are here because the trait requires them.
struct VocabOnly(Vocab);

impl tk_encode::Model for VocabOnly {
    fn get_vocab_size(&self) -> usize {
        self.0.len()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.0.get(token).copied()
    }

    fn get_vocab(&self) -> std::collections::HashMap<String, u32> {
        self.0.iter().map(|(k, &v)| (k.clone(), v)).collect()
    }

    /// A scan, because this direction is not what the type is for -- see the note above.
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.0
            .iter()
            .find(|&(_, &v)| v == id)
            .map(|(token, _)| token.clone())
    }

    fn tokenize(&self, _sequence: &str) -> Result<Vec<tk_encode::Token>> {
        Err("this is a vocabulary lookup for added tokens, not an encoder".into())
    }

    fn save(
        &self,
        _folder: &std::path::Path,
        _prefix: Option<&str>,
    ) -> Result<Vec<std::path::PathBuf>> {
        Err("this is a vocabulary lookup for added tokens, not a model".into())
    }
}
