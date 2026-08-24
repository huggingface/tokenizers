//! The `pre_tokenizer` object, plus the field readers a `Metaspace` and a `Split` share.

#[cfg(not(feature = "unicode-scripts"))]
use super::needs_feature;
use super::unsupported;
use crate::json::Json;
use tk_encode::decoders::metaspace::PrependScheme;
use tk_encode::normalizers::metaspace::MetaspaceNormalizer;
use tk_encode::pipeline::{PipelineNormalizer, PipelinePreTokenizer};
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

pub(super) fn read_pre_tokenizer(
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
            .field("pretokenizers")
            .and_then(Json::as_array)
            .unwrap_or(&[]);
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
    let b = |name: &str, default: bool| cfg.field(name).and_then(Json::as_bool).unwrap_or(default);

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

fn byte_level_pre_tokenizer(cfg: &Json<'_>) -> Result<PipelinePreTokenizer> {
    if cfg
        .field("add_prefix_space")
        .and_then(Json::as_bool)
        .unwrap_or(false)
    {
        // TODO: this is something we prob want for v1 depending on the usage.
        return Err("ByteLevel add_prefix_space=true is not supported by the pipeline yet".into());
    }
    let use_regex = cfg
        .field("use_regex")
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

/// We removed the Metaspace pre tokenizer in favor of 2 normalizers as it makes more sense
/// regarding the flow of code.
/// `drop_whitespace` is the `WhitespaceSplit` that t5 and albert run in front of theirs.
fn read_metaspace(
    cfg: &Json<'_>,
    drop_whitespace: bool,
    normalizers: &mut Vec<PipelineNormalizer>,
) -> Result<PipelinePreTokenizer> {
    let replacement = read_char(cfg, "replacement")?;
    if !cfg.field("split").and_then(Json::as_bool).unwrap_or(true) {
        return Err(unsupported(
            "a `Metaspace` pre-tokenizer with `split: false`",
        ));
    }
    let prepend = match read_prepend_scheme(cfg)? {
        PrependScheme::Always => true,
        PrependScheme::Never => false,
        // TODO: v1 probably needs to supports that?
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
pub(super) fn read_prepend_scheme(cfg: &Json<'_>) -> Result<PrependScheme> {
    let mut scheme = match cfg.field("prepend_scheme").and_then(Json::as_str) {
        None => PrependScheme::Always,
        Some("always") => PrependScheme::Always,
        Some("first") => PrependScheme::First,
        Some("never") => PrependScheme::Never,
        Some(other) => return Err(format!("unknown metaspace prepend_scheme {other:?}").into()),
    };
    if cfg.field("add_prefix_space").and_then(Json::as_bool) == Some(false) {
        if scheme != PrependScheme::Never {
            return Err("add_prefix_space does not match declared prepend_scheme".into());
        }
        scheme = PrependScheme::Never;
    }
    Ok(scheme)
}

/// A one-character field. JSON has no char, so serde reads these as a string of length one and
/// rejects anything else; a two-character `replacement` must not silently become its first char.
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
    // `native`: a recognised GPT pattern, a member of a natively-run composition (deepseek's three),
    // or a literal. `Split::new` would demand a regex engine for the middle case.
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
