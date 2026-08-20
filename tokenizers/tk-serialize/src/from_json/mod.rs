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

use crate::json::Json;
use tk_encode::decoders::DecoderRuntime;
use tk_encode::decoders::bpe::BPEDecoder;
use tk_encode::decoders::byte_fallback::ByteFallback;
use tk_encode::decoders::byte_level::ByteLevelDecoder;
use tk_encode::decoders::ctc::CTC;
use tk_encode::decoders::fuse::Fuse;
use tk_encode::decoders::metaspace::MetaspaceDecoder;
use tk_encode::decoders::replace::ReplaceDecoder;
use tk_encode::decoders::strip::Strip as StripDecoder;
use tk_encode::decoders::wordpiece::WordPiece as WordPieceDecoder;
use tk_encode::models::bpe::{Merges, PipelineBPE, PipelineBpeOptions, Vocab};
use tk_encode::normalizers::byte_level::ByteLevel as ByteLevelNormalizer;
use tk_encode::normalizers::metaspace::MetaspaceNormalizer;
use tk_encode::normalizers::prepend::Prepend;
use tk_encode::normalizers::replace::{Replace, ReplacePattern};
use tk_encode::normalizers::strip::Strip;
use tk_encode::normalizers::utils::Lowercase;
use tk_encode::pipeline::{
    NormalizerChain, PipelineModel, PipelineNormalizer, PipelinePostProcessor,
    PipelinePreTokenizer, PipelineToken, PipelineTokenizer, Seq, Slice, Template, compose,
};
use tk_encode::pre_tokenizers::bert::BertPreTokenizer;
use tk_encode::pre_tokenizers::delimiter::CharDelimiterSplit;
use tk_encode::pre_tokenizers::digits::Digits;
use tk_encode::pre_tokenizers::fixed_length::FixedLength;
use tk_encode::pre_tokenizers::metaspace::PrependScheme;
use tk_encode::pre_tokenizers::punctuation::Punctuation;
use tk_encode::pre_tokenizers::sequence::PipelineSequence;
use tk_encode::pre_tokenizers::split::{Split as SplitPretok, SplitPattern};
#[cfg(feature = "unicode-scripts")]
use tk_encode::pre_tokenizers::unicode_scripts::UnicodeScripts;
use tk_encode::pre_tokenizers::whitespace::{Whitespace, WhitespaceSplit};
use tk_encode::tokenizer::{Result, SplitDelimiterBehavior};
use tk_encode::vocab::bucket_added_vocabulary::{
    AddedToken as BucketAddedToken, AddedVocabulary as BucketAddedVocabulary,
};

#[cfg(feature = "normalizers")]
use tk_encode::normalizers::{
    bert::BertNormalizer,
    precompiled::Precompiled,
    strip::StripAccents,
    unicode::{NFC, NFD, NFKC, NFKD, Nmt},
};

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

/// The three parts [`PipelineBPE::from_vocab_and_merges`] takes, read out of the config. They are
/// returned rather than consumed here so that the vocabulary can serve as [`VocabOnly`] first.
fn read_bpe(cfg: &Json<'_>, with_byte_level: bool) -> Result<(Vocab, Merges, PipelineBpeOptions)> {
    let vocab = read_vocab_object(cfg)?;

    let merges_arr = cfg
        .get_some("merges")
        .and_then(Json::as_arr)
        .ok_or_else(|| -> tk_encode::Error { "BPE model has no `merges` array".into() })?;
    let mut merges: Merges = Vec::with_capacity(merges_arr.len());
    for entry in merges_arr {
        match entry {
            // Canonical: ["a", "b"].
            Json::Arr(pair) if pair.len() == 2 => {
                let (a, b) = (pair[0].as_str(), pair[1].as_str());
                match (a, b) {
                    (Some(a), Some(b)) => merges.push((a.to_string(), b.to_string())),
                    _ => return Err("a merge pair is not a pair of strings".into()),
                }
            }
            // Legacy: "a b". Split on the first space, the way the config path does. Ambiguous when
            // a token contains a space, which is exactly why pairs became canonical.
            Json::Str(s) => {
                let (a, b) = s.split_once(' ').ok_or_else(|| -> tk_encode::Error {
                    format!("legacy merge {s:?} has no space to split on").into()
                })?;
                merges.push((a.to_string(), b.to_string()));
            }
            _ => return Err("a merge is neither a pair nor a string".into()),
        }
    }

    // Each option is left at its default unless the config names it, which is what the builder's
    // `if let Some(..)` chain used to say. A key that is present but null reads as absent, exactly
    // as it did.
    let options = PipelineBpeOptions {
        dropout: cfg
            .get_some("dropout")
            .and_then(Json::as_f64)
            .map(|v| v as f32),
        unk_token: cfg
            .get_some("unk_token")
            .and_then(Json::as_str)
            .map(str::to_string),
        continuing_subword_prefix: cfg
            .get_some("continuing_subword_prefix")
            .and_then(Json::as_str)
            .map(str::to_string),
        end_of_word_suffix: cfg
            .get_some("end_of_word_suffix")
            .and_then(Json::as_str)
            .map(str::to_string),
        fuse_unk: cfg
            .get_some("fuse_unk")
            .and_then(Json::as_bool)
            .unwrap_or_default(),
        byte_fallback: cfg
            .get_some("byte_fallback")
            .and_then(Json::as_bool)
            .unwrap_or_default(),
        ignore_merges: cfg
            .get_some("ignore_merges")
            .and_then(Json::as_bool)
            .unwrap_or_default(),
        with_byte_level,
        ..PipelineBpeOptions::default()
    };
    Ok((vocab, merges, options))
}

/// `{"token": id, ...}`, which is how every model but Unigram spells its vocabulary.
fn read_vocab_object(cfg: &Json<'_>) -> Result<Vocab> {
    let vocab_obj = cfg
        .get_some("vocab")
        .and_then(Json::as_obj)
        .ok_or_else(|| -> tk_encode::Error { "model has no `vocab` object".into() })?;
    let mut vocab = Vocab::with_capacity(vocab_obj.len());
    for (token, id) in vocab_obj {
        let id = id.as_u32().ok_or_else(|| -> tk_encode::Error {
            format!("vocab entry {token:?} has a bad id").into()
        })?;
        vocab.insert(token.to_string(), id);
    }
    Ok(vocab)
}

/// All four fields are required, exactly as on the config path — `WordPiece`'s deserializer collects
/// missing ones and errors, so a config that omits `unk_token` has never loaded.
#[cfg(feature = "wordpiece")]
fn read_wordpiece(cfg: &Json<'_>) -> Result<tk_encode::models::wordpiece::WordPiece> {
    let vocab = read_vocab_object(cfg)?;
    let field = |name: &str| -> Result<&str> {
        cfg.get_some(name)
            .and_then(Json::as_str)
            .ok_or_else(|| format!("WordPiece model has no `{name}`").into())
    };
    let max_input_chars_per_word = cfg
        .get_some("max_input_chars_per_word")
        .and_then(Json::as_usize)
        .ok_or_else(|| -> tk_encode::Error {
            "WordPiece model has no `max_input_chars_per_word`".into()
        })?;
    tk_encode::models::wordpiece::WordPiece::builder()
        .vocab(vocab)
        .unk_token(field("unk_token")?.to_string())
        .continuing_subword_prefix(field("continuing_subword_prefix")?.to_string())
        .max_input_chars_per_word(max_input_chars_per_word)
        .build()
}

/// Unigram's vocab is an array of `[token, score]` pairs, and the scores decide the lattice, so a
/// score that is not a number is an error rather than a `0.0`.
///
/// ## The scores do not bit-match the config path, and this reader is the correct one
///
/// [`tk_encode::tokenizer::json`] parses numbers with `f64::from_str`, which is correctly rounded.
/// `serde_json` without its `float_roundtrip` feature does not: it accumulates the digits into a
/// `u64` and divides by a power of ten, which is off by one ULP for 8334 of t5's 32100 scores. Every
/// score in that file is exactly an `f32` widened to `f64`, and `from_str` lands on it; `serde_json`
/// lands next to it.
///
/// The lattice is a sum of scores, so one ULP only matters where two segmentations very nearly tie —
/// 2 ids out of 1.25 M on the english fixture, always a run of one repeated character. `json_oracle`
/// reports those cells as `SLIM MISMATCH`, and it is the *config* side that is imprecise. Closing it
/// means turning on `serde_json/float_roundtrip` and re-recording the digests, which is a decision
/// about the config path, not something to paper over here by reproducing the error.
#[cfg(feature = "unigram")]
fn read_unigram(cfg: &Json<'_>) -> Result<tk_encode::models::unigram::Unigram> {
    let entries = cfg
        .get_some("vocab")
        .and_then(Json::as_arr)
        .ok_or_else(|| -> tk_encode::Error { "Unigram model has no `vocab` array".into() })?;
    let mut vocab = Vec::with_capacity(entries.len());
    for entry in entries {
        let pair = entry
            .as_arr()
            .filter(|p| p.len() == 2)
            .ok_or_else(|| -> tk_encode::Error {
                "a Unigram vocab entry is not a [token, score] pair".into()
            })?;
        let token = pair[0].as_str().ok_or_else(|| -> tk_encode::Error {
            "a Unigram vocab token is not a string".into()
        })?;
        let score = pair[1].as_f64().ok_or_else(|| -> tk_encode::Error {
            "a Unigram vocab score is not a number".into()
        })?;
        vocab.push((token.to_string(), score));
    }
    // `get_some`, so an explicit `"unk_id": null` reads as "no unk", which is what it means.
    let unk_id = match cfg.get_some("unk_id") {
        Some(v) => Some(v.as_usize().ok_or_else(|| -> tk_encode::Error {
            "Unigram `unk_id` is not a usable index".into()
        })?),
        None => None,
    };
    let byte_fallback = cfg
        .get_some("byte_fallback")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    tk_encode::models::unigram::Unigram::from(vocab, unk_id, byte_fallback)
}

#[cfg(feature = "wordlevel")]
fn read_wordlevel(cfg: &Json<'_>) -> Result<tk_encode::models::wordlevel::WordLevel> {
    let vocab = read_vocab_object(cfg)?;
    let unk_token = cfg
        .get_some("unk_token")
        .and_then(Json::as_str)
        .ok_or_else(|| -> tk_encode::Error { "WordLevel model has no `unk_token`".into() })?;
    tk_encode::models::wordlevel::WordLevel::builder()
        .vocab(vocab)
        .unk_token(unk_token.to_string())
        .build()
}

/// Added tokens, in ascending id order — `add_tokens` depends on that order to reproduce ids.
fn read_added_tokens(cfg: Option<&Json<'_>>) -> Result<Vec<BucketAddedToken>> {
    let Some(arr) = cfg.and_then(Json::as_arr) else {
        return Ok(Vec::new());
    };
    // The ids alone decide the order, so only the ids get sorted: each key is its token's id in
    // the high half and the token's position in the array in the low half, which makes ascending
    // u64 order identical to a *stable* sort by id -- ties, if a config ever repeats an id, still
    // come out in file order. Sorting `[u64]` rather than `[(u32, AddedToken)]` also keeps this
    // path on the one sort instantiation the crate already pays for, instead of a second
    // ~3 KB copy of driftsort specialised for the pair.
    let mut order: Vec<u64> = Vec::with_capacity(arr.len());
    for (i, entry) in arr.iter().enumerate() {
        let id = entry
            .get("id")
            .and_then(Json::as_u32)
            .ok_or_else(|| -> tk_encode::Error { "an added token has no usable `id`".into() })?;
        order.push(((id as u64) << 32) | i as u64);
    }
    order.sort_unstable();

    let mut out: Vec<BucketAddedToken> = Vec::with_capacity(arr.len());
    for key in &order {
        let entry = &arr[(*key & 0xFFFF_FFFF) as usize];
        let content = entry
            .get("content")
            .and_then(Json::as_str)
            .ok_or_else(|| -> tk_encode::Error { "an added token has no `content`".into() })?;
        // `AddedToken` has no serde defaults on the config path either: all six flags are required.
        let flag = |name: &str| -> Result<bool> {
            entry
                .get(name)
                .and_then(Json::as_bool)
                .ok_or_else(|| format!("added token {content:?} has no `{name}`").into())
        };
        out.push(BucketAddedToken {
            content: content.to_string(),
            single_word: flag("single_word")?,
            lstrip: flag("lstrip")?,
            rstrip: flag("rstrip")?,
            normalized: flag("normalized")?,
            special: flag("special")?,
        });
    }
    Ok(out)
}

/// Normalizers, flattened. A `Sequence` becomes its members in order, which is what the pipeline's
/// `Vec<PipelineNormalizer>` already means — and an empty one (deepseek ships one) disappears
/// instead of costing a no-op call per segment.
fn read_normalizers(cfg: Option<&Json<'_>>) -> Result<Vec<PipelineNormalizer>> {
    let mut out = Vec::new();
    if let Some(cfg) = cfg {
        push_normalizer(cfg, &mut out)?;
    }
    Ok(out)
}

fn push_normalizer(cfg: &Json<'_>, out: &mut Vec<PipelineNormalizer>) -> Result<()> {
    let kind = cfg
        .type_tag()
        .ok_or_else(|| unsupported("a normalizer with no `type`"))?;
    let flag = |name: &str| -> Result<bool> {
        cfg.get_some(name)
            .and_then(Json::as_bool)
            .ok_or_else(|| format!("the `{kind}` normalizer has no `{name}`").into())
    };
    match kind {
        "Sequence" => {
            for member in cfg
                .get_some("normalizers")
                .and_then(Json::as_arr)
                .unwrap_or(&[])
            {
                push_normalizer(member, out)?;
            }
        }
        "Replace" => out.push(PipelineNormalizer::Replace(read_replace(cfg)?)),
        "Prepend" => {
            let prepend = cfg
                .get_some("prepend")
                .and_then(Json::as_str)
                .ok_or_else(|| -> tk_encode::Error { "Prepend has no `prepend`".into() })?;
            out.push(PipelineNormalizer::Prepend(Prepend::new(
                prepend.to_string(),
            )));
        }
        "Strip" => out.push(PipelineNormalizer::Strip(Strip::new(
            flag("strip_left")?,
            flag("strip_right")?,
        ))),
        "Lowercase" => out.push(PipelineNormalizer::Lowercase(Lowercase)),
        "ByteLevel" => out.push(PipelineNormalizer::ByteLevel(ByteLevelNormalizer)),
        #[cfg(feature = "normalizers")]
        "BertNormalizer" => {
            // `strip_accents` is `Option<bool>`, and serde still demands the key be present — a
            // `null` is what means "decide from `lowercase`". Requiring it keeps the two paths
            // agreeing on which configs load at all.
            let strip_accents = cfg
                .get("strip_accents")
                .ok_or_else(|| -> tk_encode::Error {
                    "BertNormalizer has no `strip_accents` (spell it `null` for the default)".into()
                })?
                .as_bool();
            out.push(PipelineNormalizer::Bert(BertNormalizer::new(
                flag("clean_text")?,
                flag("handle_chinese_chars")?,
                strip_accents,
                flag("lowercase")?,
            )));
        }
        #[cfg(feature = "normalizers")]
        "StripAccents" => out.push(PipelineNormalizer::StripAccents(StripAccents)),
        #[cfg(feature = "normalizers")]
        "NFC" => out.push(PipelineNormalizer::NFC(NFC)),
        #[cfg(feature = "normalizers")]
        "NFD" => out.push(PipelineNormalizer::NFD(NFD)),
        #[cfg(feature = "normalizers")]
        "NFKC" => out.push(PipelineNormalizer::NFKC(NFKC)),
        #[cfg(feature = "normalizers")]
        "NFKD" => out.push(PipelineNormalizer::NFKD(NFKD)),
        #[cfg(feature = "normalizers")]
        "Nmt" => out.push(PipelineNormalizer::Nmt(Nmt)),
        #[cfg(feature = "normalizers")]
        "Precompiled" => {
            let charsmap = cfg
                .get_some("precompiled_charsmap")
                .and_then(Json::as_str)
                .ok_or_else(|| -> tk_encode::Error {
                    "Precompiled has no `precompiled_charsmap`".into()
                })?;
            let bytes = base64_decode(charsmap)?;
            out.push(PipelineNormalizer::Precompiled(
                Precompiled::from(&bytes)
                    .map_err(|e| -> tk_encode::Error { e.to_string().into() })?,
            ));
        }
        #[cfg(not(feature = "normalizers"))]
        "BertNormalizer" | "StripAccents" | "NFC" | "NFD" | "NFKC" | "NFKD" | "Nmt"
        | "Precompiled" => {
            return Err(needs_feature(
                &format!("the `{kind}` normalizer"),
                "normalizers",
            ));
        }
        other => return Err(unsupported(&format!("the `{other}` normalizer"))),
    }
    Ok(())
}

/// Shared by the `Replace` normalizer and the `Replace` decoder — the same type in both roles.
///
/// A regex pattern is forwarded rather than refused: `Replace::new` compiles it with the system
/// backend, and the build without one already errors with a message naming `fancy-regex`. Refusing
/// here would only replace that message with a worse one.
/// The two fields a `Replace` is spelled with, shared by the normalizer and the decoder -- which
/// are separate types with the same JSON shape, so the *parsing* is shared rather than the type.
fn read_replace_fields<'a>(cfg: &'a Json<'a>) -> Result<(ReplacePattern, &'a str)> {
    let pattern = cfg
        .get_some("pattern")
        .ok_or_else(|| -> tk_encode::Error { "Replace has no `pattern`".into() })?;
    let content = cfg
        .get_some("content")
        .and_then(Json::as_str)
        .ok_or_else(|| -> tk_encode::Error { "Replace has no `content`".into() })?;
    let pattern = if let Some(s) = pattern.get_some("String").and_then(Json::as_str) {
        ReplacePattern::String(s.to_string())
    } else if let Some(r) = pattern.get_some("Regex").and_then(Json::as_str) {
        ReplacePattern::Regex(r.to_string())
    } else {
        return Err("Replace `pattern` is neither `String` nor `Regex`".into());
    };
    Ok((pattern, content))
}

fn read_replace(cfg: &Json<'_>) -> Result<Replace> {
    let (pattern, content) = read_replace_fields(cfg)?;
    Replace::new(pattern, content)
}

fn read_replace_decoder(cfg: &Json<'_>) -> Result<ReplaceDecoder> {
    let (pattern, content) = read_replace_fields(cfg)?;
    ReplaceDecoder::new(pattern, content)
}

/// Standard-alphabet base64, for `Precompiled`'s charsmap — the one place a `tokenizer.json` holds
/// binary. Written out rather than pulled in as a dependency: it is 20 lines, and the slim path
/// exists to shed dependencies.
// Only `Precompiled` needs it, so a build without `normalizers` has no caller — but its tests are
// worth running in every build, hence the `test` arm.
#[cfg(any(feature = "normalizers", test))]
fn base64_decode(s: &str) -> Result<Vec<u8>> {
    fn sextet(b: u8) -> Option<u32> {
        Some(match b {
            b'A'..=b'Z' => u32::from(b - b'A'),
            b'a'..=b'z' => u32::from(b - b'a') + 26,
            b'0'..=b'9' => u32::from(b - b'0') + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        })
    }
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for &byte in s.as_bytes() {
        // Padding only ever trails, so the first `=` ends the data.
        if byte == b'=' {
            break;
        }
        let Some(six) = sextet(byte) else {
            return Err("`precompiled_charsmap` is not valid base64".into());
        };
        acc = (acc << 6) | six;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    // A whole group leaves 0 bits over, a 3- or 2-character tail 2 or 4. Six means a lone trailing
    // character, which encodes nothing and can only be a truncated file.
    if bits >= 6 {
        return Err("`precompiled_charsmap` ends mid-group".into());
    }
    Ok(out)
}

/// The pre-tokenizer, plus whether a `ByteLevel` is in play — the model needs to know, and the
/// config path derives the same flag the same way.
///
/// `normalizers` is appended to for the one pre-tokenizer that is also a rewrite: see
/// [`read_metaspace`].
fn read_pre_tokenizer(
    cfg: Option<&Json<'_>>,
    normalizers: &mut Vec<PipelineNormalizer>,
) -> Result<(PipelinePreTokenizer, bool)> {
    let Some(cfg) = cfg else {
        return Ok((PipelinePreTokenizer::None, false));
    };
    let kind = cfg
        .type_tag()
        .ok_or_else(|| unsupported("a pre-tokenizer with no `type`"))?;

    if kind == "Metaspace" {
        return Ok((read_metaspace(cfg, false, normalizers)?, false));
    }

    if kind == "Sequence" {
        let members = cfg
            .get_some("pretokenizers")
            .and_then(Json::as_arr)
            .unwrap_or(&[]);
        // t5 and albert: throw the whitespace away first, then mark where words start. The config
        // path recognises this exact pair and collapses it to a single `Split`, so a `Sequence`
        // here would not be the same pipeline.
        if let [first, second] = members
            && first.type_tag() == Some("WhitespaceSplit")
            && second.type_tag() == Some("Metaspace")
        {
            return Ok((read_metaspace(second, true, normalizers)?, false));
        }
        if members.iter().any(|m| m.type_tag() == Some("Sequence")) {
            return Err("Nesting Sequence pre tokenizers is not supported".into());
        }
        let byte_level_at = members
            .iter()
            .position(|m| m.type_tag() == Some("ByteLevel"));
        if let Some(pos) = byte_level_at
            && pos != members.len() - 1
        {
            return Err(
                "ByteLevel pre tokenizer must be the last pre tokenizer in the Sequence".into(),
            );
        }
        let mut built = Vec::with_capacity(members.len());
        for member in members {
            built.push(read_one_pre_tokenizer(member)?);
        }
        return Ok((
            PipelinePreTokenizer::Sequence(PipelineSequence::new(built)),
            byte_level_at.is_some(),
        ));
    }

    let one = read_one_pre_tokenizer(cfg)?;
    Ok((one, kind == "ByteLevel"))
}

fn read_one_pre_tokenizer(cfg: &Json<'_>) -> Result<PipelinePreTokenizer> {
    let kind = cfg
        .type_tag()
        .ok_or_else(|| unsupported("a pre-tokenizer with no `type`"))?;
    let b = |name: &str, default: bool| {
        cfg.get_some(name)
            .and_then(Json::as_bool)
            .unwrap_or(default)
    };

    Ok(match kind {
        "ByteLevel" => byte_level_pre_tokenizer(cfg)?,
        "Split" => PipelinePreTokenizer::Split(read_split(cfg)?),
        "Whitespace" => PipelinePreTokenizer::Whitespace(Whitespace),
        "WhitespaceSplit" => PipelinePreTokenizer::WhitespaceSplit(WhitespaceSplit),
        "BertPreTokenizer" => PipelinePreTokenizer::Bert(BertPreTokenizer),
        #[cfg(feature = "unicode-scripts")]
        "UnicodeScripts" => PipelinePreTokenizer::UnicodeScripts(UnicodeScripts::new()),
        #[cfg(not(feature = "unicode-scripts"))]
        "UnicodeScripts" => {
            return Err(needs_feature(
                "the `UnicodeScripts` pre-tokenizer",
                "unicode-scripts",
            ));
        }
        "Digits" => PipelinePreTokenizer::Digits(Digits::new(b("individual_digits", false))),
        "Punctuation" => PipelinePreTokenizer::Punctuation(Punctuation::new(read_behavior(
            cfg,
            SplitDelimiterBehavior::Isolated,
        )?)),
        "CharDelimiterSplit" => {
            let d = cfg
                .get_some("delimiter")
                .and_then(Json::as_str)
                .and_then(|s| s.chars().next())
                .ok_or_else(|| -> tk_encode::Error {
                    "CharDelimiterSplit has no `delimiter`".into()
                })?;
            PipelinePreTokenizer::Delimiter(CharDelimiterSplit::new(d))
        }
        "FixedLength" => {
            let n = cfg
                .get_some("length")
                .and_then(Json::as_usize)
                .ok_or_else(|| -> tk_encode::Error { "FixedLength has no `length`".into() })?;
            PipelinePreTokenizer::FixedLength(FixedLength::new(n))
        }
        // Only the two shapes `read_pre_tokenizer` intercepts can be rebuilt as normalizer + split;
        // a `Metaspace` anywhere else in a `Sequence` cannot, and the config path rejects it too.
        "Metaspace" => {
            return Err(unsupported(
                "a `Metaspace` pre-tokenizer other than on its own or after a `WhitespaceSplit`",
            ));
        }
        other => return Err(unsupported(&format!("the `{other}` pre-tokenizer"))),
    })
}

/// A `ByteLevel` pre-tokenizer is two unrelated switches. With `use_regex` it splits on the GPT-2
/// regex, which `gpt_fsm` recognises, so it drives `atomsplit` natively and needs no regex backend.
/// Without it, it only asks for the byte map — which the model half already applies — so the
/// splitting step is the identity. That is the `Sequence[Split, ByteLevel]` idiom, and
/// `PipelineSequence` relies on seeing a `None` there to fuse the pair.
fn byte_level_pre_tokenizer(cfg: &Json<'_>) -> Result<PipelinePreTokenizer> {
    if cfg
        .get_some("add_prefix_space")
        .and_then(Json::as_bool)
        .unwrap_or(false)
    {
        return Err("ByteLevel add_prefix_space=true is not supported by the pipeline yet".into());
    }
    let use_regex = cfg
        .get_some("use_regex")
        .and_then(Json::as_bool)
        .unwrap_or(true);
    if !use_regex {
        return Ok(PipelinePreTokenizer::None);
    }
    // `native` rather than `new`: this pattern is FSM-recognised, so never ask for an engine.
    Ok(PipelinePreTokenizer::Split(SplitPretok::native(
        SplitPattern::Regex(atomsplit::regexes::GPT2.to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )?))
}

/// A `Metaspace` pre-tokenizer does two jobs at once: it writes `▁` delimiters into the text, then
/// cuts on them. The pipeline keeps rewriting and cutting apart, so it is rebuilt as a normalizer
/// plus a `Split` — the same decomposition `metaspace::to_normalizer_and_split` performs for the
/// config path, and the same settings are refused.
///
/// `drop_whitespace` is the `WhitespaceSplit` that t5 and albert run in front of theirs.
fn read_metaspace(
    cfg: &Json<'_>,
    drop_whitespace: bool,
    normalizers: &mut Vec<PipelineNormalizer>,
) -> Result<PipelinePreTokenizer> {
    let replacement = read_char(cfg, "replacement")?;
    // `split: false` writes the delimiters but never cuts the text, so there is no `Split` to hand
    // back, and no way to express "rewrite only" as a pre-tokenizer.
    if !cfg
        .get_some("split")
        .and_then(Json::as_bool)
        .unwrap_or(true)
    {
        return Err(unsupported(
            "a `Metaspace` pre-tokenizer with `split: false`",
        ));
    }
    let prepend = match read_prepend_scheme(cfg)? {
        PrependScheme::Always => true,
        PrependScheme::Never => false,
        // `First` writes the delimiter only on the piece at the very start of the text it came
        // from. A normalizer is handed one chunk at a time, without that context.
        PrependScheme::First => {
            return Err(unsupported(
                "a `Metaspace` pre-tokenizer with `prepend_scheme: first`",
            ));
        }
    };
    if drop_whitespace && !prepend {
        return Err(unsupported(
            "a `WhitespaceSplit` + `Metaspace` that neither keeps whitespace nor prepends \
             (nothing would show where words begin)",
        ));
    }
    normalizers.push(PipelineNormalizer::Metaspace(MetaspaceNormalizer::new(
        replacement,
        prepend,
        drop_whitespace,
    )));
    // `MergedWithNext` keeps each delimiter attached to the word it opens (`▁hello`), which is how
    // SentencePiece vocabularies spell their tokens. A literal needs no regex backend.
    Ok(PipelinePreTokenizer::Split(SplitPretok::native(
        SplitPattern::String(replacement.to_string()),
        SplitDelimiterBehavior::MergedWithNext,
        false,
    )?))
}

/// `prepend_scheme`, including the pre-`prepend_scheme` `add_prefix_space` spelling.
///
/// The rule is the config path's, both quirks included:
///
/// - `add_prefix_space: true` is **ignored**. It agrees with the `Always` default when the config
///   spells only the old key (t5, albert), and loses to `prepend_scheme` when it spells both — so
///   `{add_prefix_space: true, prepend_scheme: "never"}` is `Never`, not a contradiction.
/// - `add_prefix_space: false` is checked against the *already defaulted* scheme, which is `Always`.
///   So the old key alone can never spell `false`: it is an error unless `prepend_scheme: "never"`
///   is spelled out beside it, at which point it changes nothing.
///
/// Surprising, and reproduced rather than fixed, because ids depend on it.
fn read_prepend_scheme(cfg: &Json<'_>) -> Result<PrependScheme> {
    let mut scheme = match cfg.get_some("prepend_scheme").and_then(Json::as_str) {
        None => PrependScheme::Always,
        Some("always") => PrependScheme::Always,
        Some("first") => PrependScheme::First,
        Some("never") => PrependScheme::Never,
        Some(other) => return Err(format!("unknown metaspace prepend_scheme {other:?}").into()),
    };
    if cfg.get_some("add_prefix_space").and_then(Json::as_bool) == Some(false) {
        if scheme != PrependScheme::Never {
            return Err("add_prefix_space does not match declared prepend_scheme".into());
        }
        scheme = PrependScheme::Never;
    }
    Ok(scheme)
}

/// A one-character field. JSON has no char, so serde reads these as a string of length one and
/// rejects anything else; a two-character `replacement` must not silently become its first char.
fn read_char(cfg: &Json<'_>, key: &str) -> Result<char> {
    let s = cfg
        .get_some(key)
        .and_then(Json::as_str)
        .ok_or_else(|| -> tk_encode::Error { format!("missing `{key}`").into() })?;
    let mut chars = s.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) => Ok(c),
        _ => Err(format!("`{key}` must be exactly one character, got {s:?}").into()),
    }
}

fn read_split(cfg: &Json<'_>) -> Result<SplitPretok> {
    let pattern = cfg
        .get_some("pattern")
        .ok_or_else(|| -> tk_encode::Error { "Split has no `pattern`".into() })?;
    let behavior = read_behavior(cfg, SplitDelimiterBehavior::Isolated)?;
    let invert = cfg
        .get_some("invert")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    let pattern = if let Some(s) = pattern.get_some("String").and_then(Json::as_str) {
        SplitPattern::String(s.to_string())
    } else if let Some(r) = pattern.get_some("Regex").and_then(Json::as_str) {
        SplitPattern::Regex(r.to_string())
    } else {
        return Err("Split `pattern` is neither `String` nor `Regex`".into());
    };
    // `native`: a recognised GPT pattern, a member of a natively-run composition (deepseek's three),
    // or a literal. `Split::new` would demand a regex engine for the middle case.
    SplitPretok::native(pattern, behavior, invert)?.canonicalized_for_pipeline()
}

fn read_behavior(
    cfg: &Json<'_>,
    default: SplitDelimiterBehavior,
) -> Result<SplitDelimiterBehavior> {
    let Some(name) = cfg.get_some("behavior").and_then(Json::as_str) else {
        return Ok(default);
    };
    // Spelled out to match the serialized names exactly; `Display for SplitDelimiterBehavior` is
    // pinned against serde by `display_matches_serde`, and this is the inverse of it.
    Ok(match name {
        "Removed" => SplitDelimiterBehavior::Removed,
        "Isolated" => SplitDelimiterBehavior::Isolated,
        "MergedWithPrevious" => SplitDelimiterBehavior::MergedWithPrevious,
        "MergedWithNext" => SplitDelimiterBehavior::MergedWithNext,
        "Contiguous" => SplitDelimiterBehavior::Contiguous,
        other => return Err(format!("unknown split behavior {other:?}").into()),
    })
}

/// The post-processor, lowered into the same `Template` IR the config path produces.
fn read_post_processor(cfg: Option<&Json<'_>>) -> Result<PipelinePostProcessor> {
    let Some(cfg) = cfg else {
        return Ok(PipelinePostProcessor::default());
    };
    let kind = cfg
        .type_tag()
        .ok_or_else(|| unsupported("a post-processor with no `type`"))?;
    match kind {
        // ByteLevel only trims offsets, which the pipeline does not track: a pass-through.
        "ByteLevel" => Ok(PipelinePostProcessor::default()),
        "Sequence" => {
            let members = cfg
                .get_some("processors")
                .and_then(Json::as_arr)
                .unwrap_or(&[]);
            let built = members
                .iter()
                .map(|m| read_post_processor(Some(m)))
                .collect::<Result<Vec<_>>>()?;
            // Reuse the config path's composition rules rather than restating them.
            Ok(PipelinePostProcessor::new(
                compose(built.iter().map(|m| m.templates().0))?,
                compose(built.iter().map(|m| m.templates().1))?,
            ))
        }
        "TemplateProcessing" => read_template(cfg),
        // The two frames that predate `TemplateProcessing`. Slice shapes copied from
        // `TryFrom<&PostProcessorWrapper>`; note roberta pairs with a doubled sep and keeps every
        // type id at 0, where bert retags the second sequence.
        "BertProcessing" | "RobertaProcessing" => {
            let cls = read_special_id(cfg, "cls")?;
            let sep = read_special_id(cfg, "sep")?;
            let one = |id: u32, type_id: u8| Slice::Specials {
                tokens: Box::new([PipelineToken::from(id)]),
                type_id,
            };
            let sq = |seq, type_id| Slice::Sequence { seq, type_id };
            let single = Template::new(vec![one(cls, 0), sq(Seq::A, 0), one(sep, 0)]);
            let pair = if kind == "BertProcessing" {
                Template::new(vec![
                    one(cls, 0),
                    sq(Seq::A, 0),
                    one(sep, 0),
                    sq(Seq::B, 1),
                    one(sep, 1),
                ])
            } else {
                Template::new(vec![
                    one(cls, 0),
                    sq(Seq::A, 0),
                    Slice::Specials {
                        tokens: Box::new([PipelineToken::from(sep), PipelineToken::from(sep)]),
                        type_id: 0,
                    },
                    sq(Seq::B, 0),
                    one(sep, 0),
                ])
            };
            Ok(PipelinePostProcessor::new(single, pair))
        }
        other => Err(unsupported(&format!("the `{other}` post-processor"))),
    }
}

/// A `["<token>", id]` pair, which is how `BertProcessing` and `RobertaProcessing` spell `cls` and
/// `sep`. Only the id reaches the pipeline; the string is the human-readable half.
fn read_special_id(cfg: &Json<'_>, key: &str) -> Result<u32> {
    cfg.get_some(key)
        .and_then(Json::as_arr)
        .filter(|pair| pair.len() == 2)
        .and_then(|pair| pair[1].as_u32())
        .ok_or_else(|| format!("`{key}` is not a [token, id] pair").into())
}

fn read_template(cfg: &Json<'_>) -> Result<PipelinePostProcessor> {
    // `special_tokens` maps a placeholder to the ids it expands to.
    let specials = cfg
        .get_some("special_tokens")
        .ok_or_else(|| -> tk_encode::Error {
            "TemplateProcessing has no `special_tokens`".into()
        })?;

    let ids_for =
        |name: &str| -> Result<Vec<u32>> {
            let entry = specials.get(name).ok_or_else(|| -> tk_encode::Error {
                format!("template references unknown special token {name:?}").into()
            })?;
            let ids = entry.get_some("ids").and_then(Json::as_arr).ok_or_else(
                || -> tk_encode::Error { format!("special token {name:?} has no `ids`").into() },
            )?;
            ids.iter()
                .map(|i| {
                    i.as_u32().ok_or_else(|| -> tk_encode::Error {
                        format!("special token {name:?} has a bad id").into()
                    })
                })
                .collect()
        };

    let slices_for = |key: &str| -> Result<Vec<Slice>> {
        let pieces =
            cfg.get_some(key)
                .and_then(Json::as_arr)
                .ok_or_else(|| -> tk_encode::Error {
                    format!("TemplateProcessing has no `{key}`").into()
                })?;
        let mut out = Vec::with_capacity(pieces.len());
        for piece in pieces {
            if let Some(seq) = piece.get_some("Sequence") {
                let id = seq.get_some("id").and_then(Json::as_str).ok_or_else(
                    || -> tk_encode::Error { "template Sequence has no `id`".into() },
                )?;
                let type_id = seq.get_some("type_id").and_then(Json::as_u32).unwrap_or(0) as u8;
                let seq = match id {
                    "A" => Seq::A,
                    "B" => Seq::B,
                    other => return Err(format!("unknown template sequence {other:?}").into()),
                };
                out.push(Slice::Sequence { seq, type_id });
            } else if let Some(tok) = piece.get_some("SpecialToken") {
                let id = tok.get_some("id").and_then(Json::as_str).ok_or_else(
                    || -> tk_encode::Error { "template SpecialToken has no `id`".into() },
                )?;
                let type_id = tok.get_some("type_id").and_then(Json::as_u32).unwrap_or(0) as u8;
                out.push(Slice::Specials {
                    tokens: ids_for(id)?.into_iter().map(PipelineToken::from).collect(),
                    type_id,
                });
            } else {
                return Err("a template piece is neither Sequence nor SpecialToken".into());
            }
        }
        Ok(out)
    };

    Ok(PipelinePostProcessor::new(
        Template::new(slices_for("single")?),
        Template::new(slices_for("pair")?),
    ))
}

/// The decoder, as a [`DecoderRuntime`].
///
/// The wrapper is fine here where a `ModelWrapper` would not be: only its `Deserialize` impl pulls
/// serde in, and building the enum by hand does not name it. Decode is not on the encode hot path
/// and every variant is small, so there is nothing to gain from a second, pipeline-native enum.
fn read_decoder(cfg: Option<&Json<'_>>) -> Result<Option<DecoderRuntime>> {
    match cfg {
        Some(cfg) => Ok(Some(read_one_decoder(cfg)?)),
        None => Ok(None),
    }
}

fn read_one_decoder(cfg: &Json<'_>) -> Result<DecoderRuntime> {
    let kind = cfg
        .type_tag()
        .ok_or_else(|| unsupported("a decoder with no `type`"))?;
    let flag = |name: &str| -> Result<bool> {
        cfg.get_some(name)
            .and_then(Json::as_bool)
            .ok_or_else(|| format!("the `{kind}` decoder has no `{name}`").into())
    };
    let text = |name: &str| -> Result<String> {
        cfg.get_some(name)
            .and_then(Json::as_str)
            .map(str::to_string)
            .ok_or_else(|| format!("the `{kind}` decoder has no `{name}`").into())
    };
    let count = |name: &str| -> Result<usize> {
        cfg.get_some(name)
            .and_then(Json::as_usize)
            .ok_or_else(|| format!("the `{kind}` decoder has no `{name}`").into())
    };

    Ok(match kind {
        "ByteLevel" => DecoderRuntime::ByteLevel(ByteLevelDecoder::new(
            flag("add_prefix_space")?,
            flag("trim_offsets")?,
            // The one decoder field with a serde default, and it is `true`.
            cfg.get_some("use_regex")
                .and_then(Json::as_bool)
                .unwrap_or(true),
        )),
        "Replace" => DecoderRuntime::Replace(read_replace_decoder(cfg)?),
        "ByteFallback" => DecoderRuntime::ByteFallback(ByteFallback::new()),
        "Fuse" => DecoderRuntime::Fuse(Fuse::new()),
        "Strip" => DecoderRuntime::Strip(StripDecoder::new(
            read_char(cfg, "content")?,
            count("start")?,
            count("stop")?,
        )),
        // Spelled `BPEDecoder` in the file, unlike every other tag, which matches its type name.
        "BPEDecoder" => DecoderRuntime::BPE(BPEDecoder::new(text("suffix")?)),
        "WordPiece" => {
            DecoderRuntime::WordPiece(WordPieceDecoder::new(text("prefix")?, flag("cleanup")?))
        }
        "Metaspace" => DecoderRuntime::Metaspace(MetaspaceDecoder::new(
            read_char(cfg, "replacement")?,
            read_prepend_scheme(cfg)?,
            cfg.get_some("split")
                .and_then(Json::as_bool)
                .unwrap_or(true),
        )),
        "CTC" => DecoderRuntime::CTC(CTC::new(
            text("pad_token")?,
            text("word_delimiter_token")?,
            flag("cleanup")?,
        )),
        "Sequence" => {
            let members = cfg
                .get_some("decoders")
                .and_then(Json::as_arr)
                .unwrap_or(&[]);
            DecoderRuntime::Sequence(
                members
                    .iter()
                    .map(read_one_decoder)
                    .collect::<Result<Vec<_>>>()?,
            )
        }
        other => return Err(unsupported(&format!("the `{other}` decoder"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal BPE that needs no data files: two merges over a four-token vocab.
    const TINY_BPE: &str = r#"{
        "version": "1.0",
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "ab": 2, "abab": 3},
            "merges": [["a", "b"], ["ab", "ab"]]
        }
    }"#;

    /// Swap `TINY_BPE`'s model out for another kind. Only the per-model tests need it.
    #[cfg(any(feature = "unigram", feature = "wordpiece", feature = "wordlevel"))]
    fn with_model(model: &str) -> String {
        TINY_BPE.replace(
            r#""model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1, "ab": 2, "abab": 3},
            "merges": [["a", "b"], ["ab", "ab"]]
        }"#,
            model,
        )
    }

    /// Swap one top-level component into `TINY_BPE`. The field is always spelled `null` there, so a
    /// plain textual replace is unambiguous.
    fn with_component(field: &str, json: &str) -> String {
        TINY_BPE.replace(
            &format!(r#""{field}": null"#),
            &format!(r#""{field}": {json}"#),
        )
    }

    fn read(text: &str) -> Result<PipelineTokenizer> {
        from_json(text)
    }

    /// The message from a config the reader must refuse. Not `unwrap_err`: `PipelineTokenizer` has
    /// no `Debug`, which is what that would need on the `Ok` side.
    fn read_err(text: &str) -> String {
        match read(text) {
            Ok(_) => panic!("expected the reader to refuse this config"),
            Err(e) => e.to_string(),
        }
    }

    fn ids(tok: &PipelineTokenizer, text: &str) -> Vec<u32> {
        tok.encode(text, true)
            .wait()
            .unwrap()
            .iter()
            .flat_map(|e| e.ids())
            .map(|t| t.id())
            .collect()
    }

    #[test]
    fn reads_a_tiny_bpe() {
        let tok = read(TINY_BPE).unwrap();
        assert_eq!(ids(&tok, "abab"), vec![3]);
    }

    // ---- base64, the one piece of parsing that is not a field read ------------------------------

    #[test]
    fn base64_round_trips_every_tail_length() {
        // Reference vectors from RFC 4648 §10, which cover all three padding cases.
        for (encoded, decoded) in [
            ("", ""),
            ("Zg==", "f"),
            ("Zm8=", "fo"),
            ("Zm9v", "foo"),
            ("Zm9vYg==", "foob"),
            ("Zm9vYmE=", "fooba"),
            ("Zm9vYmFy", "foobar"),
        ] {
            assert_eq!(
                base64_decode(encoded).unwrap(),
                decoded.as_bytes(),
                "{encoded}"
            );
        }
    }

    #[test]
    fn base64_decodes_without_padding_and_covers_the_alphabet() {
        // The same three bytes, padded and not: `spm_precompiled`'s own decoder is lenient here.
        assert_eq!(base64_decode("Zg").unwrap(), b"f");
        // `+` and `/` are the two non-alphanumeric symbols, and the ones a URL-safe alphabet moves.
        assert_eq!(base64_decode("++//").unwrap(), [0xfb, 0xef, 0xff]);
    }

    #[test]
    fn base64_rejects_junk_and_truncation() {
        assert!(base64_decode("Zm9v!").is_err(), "an illegal character");
        assert!(base64_decode("Z").is_err(), "a lone trailing character");
        assert!(base64_decode("Zm9vZ").is_err(), "a truncated final group");
    }

    // ---- models ---------------------------------------------------------------------------------

    #[test]
    fn infers_the_model_kind_without_a_type_tag() {
        let cases = [
            (r#"{"merges": [], "vocab": {}}"#, "BPE"),
            (r#"{"vocab": [["a", 0.0]]}"#, "Unigram"),
            (
                r#"{"vocab": {}, "continuing_subword_prefix": "@@"}"#,
                "WordPiece",
            ),
            (r#"{"vocab": {}, "unk_token": "<unk>"}"#, "WordLevel"),
            (r#"{"type": "Unigram", "vocab": []}"#, "Unigram"),
            (r#"{"type": "Nonsense"}"#, "unknown"),
        ];
        for (json, want) in cases {
            let doc = Json::parse(json).unwrap();
            assert_eq!(model_kind(&doc), want, "{json}");
        }
    }

    #[test]
    fn accepts_legacy_string_merges() {
        let legacy = TINY_BPE.replace(r#"[["a", "b"], ["ab", "ab"]]"#, r#"["a b", "ab ab"]"#);
        assert_eq!(ids(&read(&legacy).unwrap(), "abab"), vec![3]);
    }

    #[test]
    fn refuses_a_merge_it_cannot_split() {
        let bad = TINY_BPE.replace(r#""a b""#, r#""ab""#);
        let bad = bad.replace(r#"[["a", "b"], ["ab", "ab"]]"#, r#"["ab"]"#);
        assert!(read_err(&bad).contains("no space"));
    }

    #[test]
    #[cfg(feature = "unigram")]
    fn reads_a_unigram_vocab_of_pairs() {
        let json = with_model(
            r#""model": {"type": "Unigram", "unk_id": 0, "byte_fallback": false,
                "vocab": [["<unk>", 0.0], ["a", -1.0], ["ab", -0.5]]}"#,
        );
        // `ab` scores better than `a` + `a`, so the lattice must pick the pair.
        assert_eq!(ids(&read(&json).unwrap(), "ab"), vec![2]);
    }

    #[test]
    #[cfg(feature = "unigram")]
    fn refuses_a_unigram_entry_that_is_not_a_pair() {
        let json = with_model(r#""model": {"type": "Unigram", "vocab": [["a"]]}"#);
        assert!(read_err(&json).contains("[token, score] pair"));
    }

    #[test]
    #[cfg(feature = "wordpiece")]
    fn reads_a_wordpiece_and_requires_every_field() {
        let full = with_model(
            r#""model": {"type": "WordPiece", "unk_token": "[UNK]",
                "continuing_subword_prefix": "@@", "max_input_chars_per_word": 100,
                "vocab": {"[UNK]": 0, "ab": 1, "@@c": 2}}"#,
        );
        assert_eq!(ids(&read(&full).unwrap(), "abc"), vec![1, 2]);

        // The config path's deserializer has no defaults either, so dropping a field must fail.
        let missing = full.replace(r#""max_input_chars_per_word": 100,"#, "");
        assert!(read_err(&missing).contains("max_input_chars_per_word"));
    }

    #[test]
    #[cfg(feature = "wordlevel")]
    fn reads_a_wordlevel() {
        let json = with_model(
            r#""model": {"type": "WordLevel", "unk_token": "<unk>",
                "vocab": {"<unk>": 0, "hello": 1}}"#,
        );
        assert_eq!(ids(&read(&json).unwrap(), "hello"), vec![1]);
    }

    // ---- normalizers ----------------------------------------------------------------------------

    #[test]
    fn flattens_a_normalizer_sequence_and_drops_an_empty_one() {
        let seq = with_component(
            "normalizer",
            r#"{"type": "Sequence", "normalizers": [
                {"type": "Lowercase"},
                {"type": "Strip", "strip_left": true, "strip_right": true}
            ]}"#,
        );
        let doc = Json::parse(&seq).unwrap();
        assert_eq!(
            read_normalizers(doc.get_some("normalizer")).unwrap().len(),
            2
        );

        let empty = with_component("normalizer", r#"{"type": "Sequence", "normalizers": []}"#);
        let doc = Json::parse(&empty).unwrap();
        assert!(
            read_normalizers(doc.get_some("normalizer"))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn strip_needs_both_sides_spelled_out() {
        let json = with_component("normalizer", r#"{"type": "Strip", "strip_left": true}"#);
        assert!(read_err(&json).contains("strip_right"));
    }

    #[test]
    #[cfg(feature = "normalizers")]
    fn bert_normalizer_wants_strip_accents_present_even_as_null() {
        let ok = with_component(
            "normalizer",
            r#"{"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true,
                "strip_accents": null, "lowercase": true}"#,
        );
        assert!(read(&ok).is_ok());

        let missing = with_component(
            "normalizer",
            r#"{"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true,
                "lowercase": true}"#,
        );
        assert!(read_err(&missing).contains("strip_accents"));
    }

    #[test]
    fn refuses_a_normalizer_it_does_not_know() {
        let json = with_component("normalizer", r#"{"type": "Invented"}"#);
        assert!(read_err(&json).contains("`Invented` normalizer"));
    }

    // ---- pre-tokenizers -------------------------------------------------------------------------

    #[test]
    fn byte_level_without_use_regex_is_the_identity_split() {
        // The `Sequence[Split, ByteLevel]` idiom: the trailing ByteLevel only asks for the byte map,
        // which the model applies, so as a *splitter* it must be a no-op.
        let json = with_component(
            "pre_tokenizer",
            r#"{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true,
                "use_regex": false}"#,
        );
        let doc = Json::parse(&json).unwrap();
        let mut normalizers = Vec::new();
        let (pretok, with_byte_level) =
            read_pre_tokenizer(doc.get_some("pre_tokenizer"), &mut normalizers).unwrap();
        assert!(matches!(pretok, PipelinePreTokenizer::None));
        // Still byte-level for the *model*, which is a separate switch.
        assert!(with_byte_level);
    }

    #[test]
    fn byte_level_add_prefix_space_is_refused() {
        let json = with_component(
            "pre_tokenizer",
            r#"{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true}"#,
        );
        assert!(read_err(&json).contains("add_prefix_space"));
    }

    #[test]
    fn metaspace_becomes_a_normalizer_plus_a_split() {
        let json = with_component(
            "pre_tokenizer",
            r#"{"type": "Metaspace", "replacement": "▁", "add_prefix_space": true}"#,
        );
        let doc = Json::parse(&json).unwrap();
        let mut normalizers = Vec::new();
        let (pretok, with_byte_level) =
            read_pre_tokenizer(doc.get_some("pre_tokenizer"), &mut normalizers).unwrap();
        assert!(!with_byte_level);
        assert!(matches!(pretok, PipelinePreTokenizer::Split(_)));
        assert!(matches!(
            normalizers.as_slice(),
            [PipelineNormalizer::Metaspace(_)]
        ));
    }

    #[test]
    fn t5_shape_collapses_to_one_split_not_a_sequence() {
        let json = with_component(
            "pre_tokenizer",
            r#"{"type": "Sequence", "pretokenizers": [
                {"type": "WhitespaceSplit"},
                {"type": "Metaspace", "replacement": "▁", "add_prefix_space": true}
            ]}"#,
        );
        let doc = Json::parse(&json).unwrap();
        let mut normalizers = Vec::new();
        let (pretok, _) =
            read_pre_tokenizer(doc.get_some("pre_tokenizer"), &mut normalizers).unwrap();
        // A `Sequence` here would run the whitespace split again over already-marked text.
        assert!(matches!(pretok, PipelinePreTokenizer::Split(_)));
        assert_eq!(normalizers.len(), 1);
    }

    #[test]
    fn the_metaspace_normalizer_lands_after_the_declared_one() {
        let json = with_component("normalizer", r#"{"type": "Lowercase"}"#);
        let json = json.replace(
            r#""pre_tokenizer": null"#,
            r#""pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "add_prefix_space": true}"#,
        );
        let doc = Json::parse(&json).unwrap();
        let mut normalizers = read_normalizers(doc.get_some("normalizer")).unwrap();
        read_pre_tokenizer(doc.get_some("pre_tokenizer"), &mut normalizers).unwrap();
        // The config asks for the whole normalizer first, then the pre-tokenizer.
        assert!(matches!(
            normalizers.as_slice(),
            [
                PipelineNormalizer::Lowercase(_),
                PipelineNormalizer::Metaspace(_)
            ]
        ));
    }

    #[test]
    fn prepend_scheme_reproduces_the_config_paths_rule() {
        let parse = |json: &str| {
            let doc = Json::parse(json).unwrap();
            read_prepend_scheme(&doc)
        };
        // Neither key: the default.
        assert_eq!(parse("{}").unwrap(), PrependScheme::Always);
        // The old key alone, which is what t5 and albert ship.
        assert_eq!(
            parse(r#"{"add_prefix_space": true}"#).unwrap(),
            PrependScheme::Always
        );
        // `add_prefix_space: true` is ignored outright, so an explicit scheme wins over it — even a
        // contradicting one.
        assert_eq!(
            parse(r#"{"add_prefix_space": true, "prepend_scheme": "never"}"#).unwrap(),
            PrependScheme::Never
        );
        // And `false` is checked against the *defaulted* scheme, which is `Always`. So the old key
        // alone can never spell `false`, and `false` is only accepted next to the `never` it would
        // have set. Both quirks are the config path's, reproduced because ids depend on them.
        assert!(parse(r#"{"add_prefix_space": false}"#).is_err());
        assert!(parse(r#"{"add_prefix_space": false, "prepend_scheme": "always"}"#).is_err());
        assert_eq!(
            parse(r#"{"add_prefix_space": false, "prepend_scheme": "never"}"#).unwrap(),
            PrependScheme::Never
        );
        assert!(parse(r#"{"prepend_scheme": "sometimes"}"#).is_err());
    }

    #[test]
    fn refuses_the_metaspace_settings_it_cannot_rebuild() {
        for (why, pretok) in [
            (
                "split: false",
                r#"{"type": "Metaspace", "replacement": "▁", "split": false}"#,
            ),
            (
                "prepend_scheme: first",
                r#"{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first"}"#,
            ),
            (
                "a metaspace buried in a sequence",
                r#"{"type": "Sequence", "pretokenizers": [
                    {"type": "Whitespace"},
                    {"type": "Metaspace", "replacement": "▁"}
                ]}"#,
            ),
        ] {
            let json = with_component("pre_tokenizer", pretok);
            assert!(read(&json).is_err(), "{why}");
        }
    }

    #[test]
    fn a_multi_character_replacement_is_not_truncated() {
        let json = with_component(
            "pre_tokenizer",
            r#"{"type": "Metaspace", "replacement": "ab"}"#,
        );
        assert!(read_err(&json).contains("exactly one character"));
    }

    // ---- post-processors ------------------------------------------------------------------------

    #[test]
    fn bert_and_roberta_processors_build_their_frames() {
        let bert = with_component(
            "post_processor",
            r#"{"type": "BertProcessing", "sep": ["b", 1], "cls": ["a", 0]}"#,
        );
        // [CLS] $A [SEP] around the single sequence.
        assert_eq!(ids(&read(&bert).unwrap(), "abab"), vec![0, 3, 1]);

        let roberta = with_component(
            "post_processor",
            r#"{"type": "RobertaProcessing", "sep": ["b", 1], "cls": ["a", 0],
                "trim_offsets": true, "add_prefix_space": true}"#,
        );
        assert_eq!(ids(&read(&roberta).unwrap(), "abab"), vec![0, 3, 1]);
    }

    #[test]
    fn a_special_pair_must_carry_an_id() {
        let json = with_component(
            "post_processor",
            r#"{"type": "BertProcessing", "sep": ["b"], "cls": ["a", 0]}"#,
        );
        assert!(read_err(&json).contains("[token, id] pair"));
    }

    // ---- decoders -------------------------------------------------------------------------------

    #[test]
    fn builds_every_decoder_variant() {
        // One config per variant, so a field rename in any of them fails here rather than silently
        // producing a decoder that decodes nothing.
        let cases = [
            r#"{"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true}"#,
            r#"{"type": "Replace", "pattern": {"String": "▁"}, "content": " "}"#,
            r#"{"type": "ByteFallback"}"#,
            r#"{"type": "Fuse"}"#,
            r#"{"type": "Strip", "content": " ", "start": 1, "stop": 0}"#,
            r#"{"type": "BPEDecoder", "suffix": "</w>"}"#,
            r#"{"type": "WordPiece", "prefix": "@@", "cleanup": true}"#,
            r#"{"type": "Metaspace", "replacement": "▁", "add_prefix_space": true}"#,
            r#"{"type": "CTC", "pad_token": "<pad>", "word_delimiter_token": "|", "cleanup": true}"#,
            r#"{"type": "Sequence", "decoders": [{"type": "Fuse"}, {"type": "ByteFallback"}]}"#,
        ];
        for json in cases {
            let doc = Json::parse(json).unwrap();
            read_one_decoder(&doc).unwrap_or_else(|e| panic!("{json}: {e}"));
        }
    }

    #[test]
    fn refuses_a_decoder_it_does_not_know() {
        let doc = Json::parse(r#"{"type": "Invented"}"#).unwrap();
        assert!(
            read_one_decoder(&doc)
                .unwrap_err()
                .to_string()
                .contains("`Invented` decoder")
        );
    }

    #[test]
    fn a_decoder_reads_back_what_the_slim_path_wired_up() {
        let json = with_component(
            "decoder",
            r#"{"type": "Sequence", "decoders": [
                {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                {"type": "Fuse"}
            ]}"#,
        );
        let tok = read(&json).unwrap();
        // `abab` is id 3; without a decoder the ids would join with a space instead.
        assert_eq!(tok.decode(&[3, 0], false).unwrap(), "ababa");
    }

    // ---- the real configs, when they are present ------------------------------------------------

    /// t5's Unigram scores, which are the reason this reader emulates `serde_json`'s float
    /// arithmetic instead of using `f64::from_str`.
    ///
    /// A SentencePiece vocabulary is trained in `f32`, so every score in the file is exactly an
    /// `f32` widened to `f64`, and `f64::from_str` lands on it. `serde_json` (without
    /// `float_roundtrip`, which is off by default) misses 8,334 of t5's 32,100 by one ULP. Scores
    /// feed a Viterbi lattice, so that flips a near-tie roughly twice per 1.25M tokens.
    ///
    /// We reproduce serde rather than improve on it, because the ids that ship today are the
    /// contract. So this pins two things: the scores are bit-identical to the config path (which is
    /// what makes t5 byte-exact in `json_oracle`), and they are deliberately *not* all the
    /// correctly-rounded `f32` values — with any deviation bounded to one ULP, so a real parsing
    /// bug could not hide behind this allowance.
    /// The hand-rolled parser reads Unigram scores as `f64` from decimal text; `serde_json` reads
    /// the same text and rounds identically. That equality is pinned directly against `serde_json`
    /// in `json.rs` (`matches_serde_not_from_str_on_a_real_unigram_score` and
    /// `numbers_are_bit_identical_to_serde_json`), so what is left to check here is the *bound* on
    /// the error over a whole real vocabulary: every score is within one ULP of the `f32` the file
    /// actually encodes, and at least one is not exactly it -- which is the cost we knowingly accept
    /// for not pulling in `serde_json/float_roundtrip`.
    #[test]
    #[cfg(feature = "unigram")]
    fn unigram_scores_stay_within_one_ulp_of_the_f32_the_file_encodes() {
        let path = "../data/t5-base.json";
        if !std::path::Path::new(path).exists() {
            return;
        }
        let text = std::fs::read_to_string(path).unwrap();
        let doc = Json::parse(&text).unwrap();
        let slim = read_unigram(doc.get_some("model").unwrap()).unwrap();

        let mut off_by_one_ulp = 0usize;
        for (tok, score) in slim.iter() {
            let correctly_rounded = f64::from(*score as f32);
            if score.to_bits() != correctly_rounded.to_bits() {
                let delta = (score.to_bits() as i64).abs_diff(correctly_rounded.to_bits() as i64);
                assert!(
                    delta <= 1,
                    "score for {tok:?} is {delta} ULP from the f32 the file encodes, which is more \
                     than serde's rounding can explain"
                );
                off_by_one_ulp += 1;
            }
        }
        assert!(
            off_by_one_ulp > 0,
            "every score is now correctly rounded, so the parser's float path must have changed: \
             re-check it against `serde_json` before relaxing this test"
        );
    }
}
