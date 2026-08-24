//! The `pre_tokenizer` object, plus the field readers a `Metaspace` and a `Split` share.

#[cfg(not(feature = "unicode-scripts"))]
use super::needs_feature;
use super::unsupported;
use crate::json::Json;
use tk_encode::pipeline::PipelinePreTokenizer;
use tk_encode::pre_tokenizers::bert::BertPreTokenizer;
use tk_encode::pre_tokenizers::delimiter::CharDelimiterSplit;
use tk_encode::pre_tokenizers::digits::Digits;
use tk_encode::pre_tokenizers::fixed_length::FixedLength;
use tk_encode::pre_tokenizers::punctuation::Punctuation;
use tk_encode::pre_tokenizers::sequence::PipelineSequence;
use tk_encode::pre_tokenizers::split::{Split as SplitPretok, SplitPattern};
#[cfg(feature = "unicode-scripts")]
use tk_encode::pre_tokenizers::unicode_scripts::UnicodeScripts;
use tk_encode::pre_tokenizers::whitespace::{Whitespace, WhitespaceSplit};
use tk_encode::tokenizer::{Result, SplitDelimiterBehavior};

pub(super) fn read_pre_tokenizer(cfg: Option<&Json<'_>>) -> Result<PipelinePreTokenizer> {
    let Some(cfg) = cfg else {
        return Ok(PipelinePreTokenizer::None);
    };
    let kind = cfg
        .type_tag()
        .ok_or_else(|| unsupported("a pre-tokenizer with no `type`"))?;

    if kind == "Sequence" {
        let members = cfg
            .field("pretokenizers")
            .and_then(Json::as_array)
            .unwrap_or(&[]);
        if members.iter().any(|m| m.type_tag() == Some("Sequence")) {
            return Err("Nesting Sequence pre tokenizers is not supported".into());
        }
        let mut built = Vec::with_capacity(members.len());
        for member in members {
            built.push(read_one_pre_tokenizer(member)?);
        }
        return Ok(PipelinePreTokenizer::Sequence(PipelineSequence::new(built)));
    }

    read_one_pre_tokenizer(cfg)
}

fn read_one_pre_tokenizer(cfg: &Json<'_>) -> Result<PipelinePreTokenizer> {
    let kind = cfg
        .type_tag()
        .ok_or_else(|| unsupported("a pre-tokenizer with no `type`"))?;
    let b = |name: &str, default: bool| cfg.field(name).and_then(Json::as_bool).unwrap_or(default);

    Ok(match kind {
        // Not canonical: a `Metaspace` is two components, and the canonical file spells them as
        // what they are -- a `MetaspaceNormalizer` in the `normalizer` slot and a `Split` here.
        "Metaspace" => return Err(unsupported("a `Metaspace` pre-tokenizer")),
        // Not canonical: byte-level is a model property now, and the split it asked for is a
        // plain `Split` on the GPT-2 regex (or nothing, for `use_regex: false`).
        "ByteLevel" => return Err(unsupported("a `ByteLevel` pre-tokenizer")),
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
                .field("delimiter")
                .and_then(Json::as_str)
                .and_then(|s| s.chars().next())
                .ok_or_else(|| -> tk_encode::Error {
                    "CharDelimiterSplit has no `delimiter`".into()
                })?;
            PipelinePreTokenizer::Delimiter(CharDelimiterSplit::new(d))
        }
        "FixedLength" => {
            let n = cfg.need("FixedLength", "length", Json::as_usize)?;
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

pub(super) fn read_char(cfg: &Json<'_>, key: &str) -> Result<char> {
    let s = cfg
        .field(key)
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
        .field("pattern")
        .ok_or_else(|| -> tk_encode::Error { "Split has no `pattern`".into() })?;
    let behavior = read_behavior(cfg, SplitDelimiterBehavior::Isolated)?;
    let invert = cfg.field("invert").and_then(Json::as_bool).unwrap_or(false);
    let pattern = if let Some(s) = pattern.field("String").and_then(Json::as_str) {
        SplitPattern::String(s.to_string())
    } else if let Some(r) = pattern.field("Regex").and_then(Json::as_str) {
        SplitPattern::Regex(r.to_string())
    } else {
        return Err("Split `pattern` is neither `String` nor `Regex`".into());
    };
    SplitPretok::native(pattern, behavior, invert)?.canonicalized_for_pipeline()
}

fn read_behavior(
    cfg: &Json<'_>,
    default: SplitDelimiterBehavior,
) -> Result<SplitDelimiterBehavior> {
    let Some(name) = cfg.field("behavior").and_then(Json::as_str) else {
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
