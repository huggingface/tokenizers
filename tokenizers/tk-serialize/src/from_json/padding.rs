//! The `padding` object, read into the `PaddingParams` the tokenizer pads with.

use crate::json::Json;
use tk_encode::tokenizer::{PaddingDirection, PaddingParams, PaddingStrategy, Result};

/// `{"strategy": {"Fixed": 128}, "direction": "Right", ...}`, which every `encode` then applies.
/// Absent or `null` means the config declares no padding.
pub(super) fn read_padding(cfg: Option<&Json<'_>>) -> Result<Option<PaddingParams>> {
    let Some(cfg) = cfg else {
        return Ok(None);
    };
    Ok(Some(PaddingParams {
        strategy: read_strategy(cfg)?,
        direction: read_direction(cfg)?,
        // The one key a config may leave out: real files ship it as `null` when there is no
        // multiple to round the padded length up to.
        pad_to_multiple_of: cfg.field("pad_to_multiple_of").and_then(Json::as_usize),
        pad_id: cfg.need("the `padding` config", "pad_id", Json::as_u32)?,
        pad_type_id: cfg.need("the `padding` config", "pad_type_id", Json::as_u32)?,
        pad_token: cfg
            .need("the `padding` config", "pad_token", Json::as_str)?
            .to_string(),
    }))
}

/// `"BatchLongest"`, the longest sequence in the batch, or `{"Fixed": <size>}`, one length for
/// every sequence.
fn read_strategy(cfg: &Json<'_>) -> Result<PaddingStrategy> {
    let strategy = cfg
        .field("strategy")
        .ok_or_else(|| -> tk_encode::Error { "the `padding` config has no `strategy`".into() })?;
    match strategy.as_str() {
        Some("BatchLongest") => Ok(PaddingStrategy::BatchLongest),
        Some(other) => Err(format!("unknown padding strategy {other:?}").into()),
        None => strategy
            .field("Fixed")
            .and_then(Json::as_usize)
            .map(PaddingStrategy::Fixed)
            .ok_or_else(|| {
                "a padding strategy is `\"BatchLongest\"` or `{\"Fixed\": <size>}`".into()
            }),
    }
}

fn read_direction(cfg: &Json<'_>) -> Result<PaddingDirection> {
    match cfg.need("the `padding` config", "direction", Json::as_str)? {
        "Left" => Ok(PaddingDirection::Left),
        "Right" => Ok(PaddingDirection::Right),
        other => Err(format!("unknown padding direction {other:?}").into()),
    }
}
