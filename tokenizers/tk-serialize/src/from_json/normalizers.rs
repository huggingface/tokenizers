//! The `normalizer` object, flattened into the pipeline's `Vec<PipelineNormalizer>`.

#[cfg(not(feature = "normalizers"))]
use super::needs_feature;
use super::unsupported;
use crate::json::Json;
#[cfg(feature = "normalizers")]
use base64::Engine as _;
use tk_encode::normalizers::byte_level::ByteLevel as ByteLevelNormalizer;
use tk_encode::normalizers::metaspace::{MetaspaceNormalizer, PrependBehavior};
use tk_encode::normalizers::prepend::Prepend;
use tk_encode::normalizers::replace::{Replace, ReplacePattern};
use tk_encode::normalizers::strip::Strip;
use tk_encode::normalizers::utils::Lowercase;
use tk_encode::pipeline::PipelineNormalizer;
use tk_encode::tokenizer::Result;

#[cfg(feature = "normalizers")]
use tk_encode::normalizers::{
    bert::BertNormalizer,
    precompiled::PrecompiledNormalizer,
    strip::StripAccents,
    unicode::{NFC, NFD, NFKC, NFKD, Nmt},
};

pub(super) fn read_normalizers(cfg: Option<&Json<'_>>) -> Result<Vec<PipelineNormalizer>> {
    let mut out = Vec::new();
    if let Some(cfg) = cfg {
        push_normalizer(cfg, &mut out)?;
    }
    Ok(out)
}

fn read_metaspace_prepend_behavior(cfg: &Json<'_>, owner: &str) -> Result<PrependBehavior> {
    match cfg.need(owner, "prepend", Json::as_str)? {
        "always" => Ok(PrependBehavior::Always),
        "first" => Ok(PrependBehavior::First),
        "never" => Ok(PrependBehavior::Never),
        other => Err(format!("unknown metaspace prepend {other:?}").into()),
    }
}

fn push_normalizer(cfg: &Json<'_>, out: &mut Vec<PipelineNormalizer>) -> Result<()> {
    let kind = cfg
        .type_tag()
        .ok_or_else(|| unsupported("a normalizer with no `type`"))?;
    let owner = format!("the `{kind}` normalizer");
    match kind {
        "Sequence" => {
            for member in cfg
                .field("normalizers")
                .and_then(Json::as_array)
                .unwrap_or(&[])
            {
                push_normalizer(member, out)?;
            }
        }
        "MetaspaceNormalizer" => out.push(PipelineNormalizer::Metaspace(MetaspaceNormalizer::new(
            super::pre_tokenizers::read_char(cfg, "replacement")?,
            read_metaspace_prepend_behavior(cfg, &owner)?,
            cfg.need(&owner, "drop_whitespace", Json::as_bool)?,
        ))),
        "Replace" => out.push(PipelineNormalizer::Replace(read_replace(cfg)?)),
        "Prepend" => {
            let prepend = cfg
                .field("prepend")
                .and_then(Json::as_str)
                .ok_or_else(|| -> tk_encode::Error { "Prepend has no `prepend` field".into() })?;
            out.push(PipelineNormalizer::Prepend(Prepend::new(
                prepend.to_string(),
            )));
        }
        "Strip" => out.push(PipelineNormalizer::Strip(Strip::new(
            cfg.need(&owner, "strip_left", Json::as_bool)?,
            cfg.need(&owner, "strip_right", Json::as_bool)?,
        ))),
        "Lowercase" => out.push(PipelineNormalizer::Lowercase(Lowercase)),
        "ByteLevel" => out.push(PipelineNormalizer::ByteLevel(ByteLevelNormalizer)),
        #[cfg(feature = "normalizers")]
        "BertNormalizer" => {
            let strip_accents = cfg
                .get("strip_accents")
                .ok_or_else(|| -> tk_encode::Error {
                    "BertNormalizer has no `strip_accents` (spell it `null` for the default)".into()
                })?
                .as_bool();
            out.push(PipelineNormalizer::Bert(BertNormalizer::new(
                cfg.need(&owner, "clean_text", Json::as_bool)?,
                cfg.need(&owner, "handle_chinese_chars", Json::as_bool)?,
                strip_accents,
                cfg.need(&owner, "lowercase", Json::as_bool)?,
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
                .field("precompiled_charsmap")
                .and_then(Json::as_str)
                .ok_or_else(|| -> tk_encode::Error {
                    "Precompiled has no `precompiled_charsmap`".into()
                })?;
            let bytes = crate::BASE64
                .decode(charsmap)
                .map_err(|e| -> tk_encode::Error {
                    format!("`precompiled_charsmap` is not valid base64: {e}").into()
                })?;
            // The bytes go in beside the parsed value: they are the whole of this normalizer's
            // configuration, and `spm_precompiled` publishes them only through serde.
            out.push(PipelineNormalizer::Precompiled(
                PrecompiledNormalizer::from_charsmap(&bytes)?,
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
pub(super) fn read_replace_fields<'a>(cfg: &'a Json<'a>) -> Result<(ReplacePattern, &'a str)> {
    let pattern = cfg
        .field("pattern")
        .ok_or_else(|| -> tk_encode::Error { "Replace has no `pattern`".into() })?;
    let content = cfg
        .field("content")
        .and_then(Json::as_str)
        .ok_or_else(|| -> tk_encode::Error { "Replace has no `content`".into() })?;
    let pattern = if let Some(s) = pattern.field("String").and_then(Json::as_str) {
        ReplacePattern::String(s.to_string())
    } else if let Some(r) = pattern.field("Regex").and_then(Json::as_str) {
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
