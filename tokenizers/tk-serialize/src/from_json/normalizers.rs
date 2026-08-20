//! The `normalizer` object, flattened into the pipeline's `Vec<PipelineNormalizer>`.

#[cfg(not(feature = "normalizers"))]
use super::needs_feature;
use super::unsupported;
use crate::json::{Json, JsonExt};
use tk_encode::normalizers::byte_level::ByteLevel as ByteLevelNormalizer;
use tk_encode::normalizers::prepend::Prepend;
use tk_encode::normalizers::replace::{Replace, ReplacePattern};
use tk_encode::normalizers::strip::Strip;
use tk_encode::normalizers::utils::Lowercase;
use tk_encode::pipeline::PipelineNormalizer;
use tk_encode::tokenizer::Result;

#[cfg(feature = "normalizers")]
use tk_encode::normalizers::{
    bert::BertNormalizer,
    precompiled::Precompiled,
    strip::StripAccents,
    unicode::{NFC, NFD, NFKC, NFKD, Nmt},
};

/// Normalizers, flattened. A `Sequence` becomes its members in order, which is what the pipeline's
/// `Vec<PipelineNormalizer>` already means — and an empty one (deepseek ships one) disappears
/// instead of costing a no-op call per segment.
pub(super) fn read_normalizers(cfg: Option<&Json<'_>>) -> Result<Vec<PipelineNormalizer>> {
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
pub(super) fn read_replace_fields<'a>(cfg: &'a Json<'a>) -> Result<(ReplacePattern, &'a str)> {
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

/// Standard-alphabet base64, for `Precompiled`'s charsmap — the one place a `tokenizer.json` holds
/// binary. Written out rather than pulled in as a dependency: it is 20 lines, and the slim path
/// exists to shed dependencies.
// Only `Precompiled` needs it, so a build without `normalizers` has no caller — but its tests are
// worth running in every build, hence the `test` arm.
#[cfg(any(feature = "normalizers", test))]
pub(super) fn base64_decode(s: &str) -> Result<Vec<u8>> {
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
