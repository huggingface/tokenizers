use super::unsupported;
use crate::json::Json;
use tk_encode::tokenizer::Result;
use tk_encode::{PaddingDirection, PaddingParams, PaddingStrategy};

pub(super) fn read_padding(cfg: Option<&Json<'_>>) -> Result<Option<PaddingParams>> {
    let Some(cfg) = cfg else {
        return Ok(None);
    };
    let strategy = cfg
        .field("strategy")
        .ok_or_else(|| unsupported("padding configuration with no `strategy`"))?;
    Ok(Some(PaddingParams {
        strategy: read_strategy(strategy)?,
        direction: read_direction(cfg.need("padding", "direction", Json::as_str)?)?,
        pad_to_multiple_of: cfg.field("pad_to_multiple_of").and_then(Json::as_usize),
        pad_id: cfg.need("padding", "pad_id", Json::as_u32)?,
        pad_type_id: cfg.need("padding", "pad_type_id", Json::as_u32)?,
        pad_token: cfg.need("padding", "pad_token", Json::as_str)?.to_string(),
    }))
}

fn read_strategy(cfg: &Json<'_>) -> Result<PaddingStrategy> {
    match cfg.as_str() {
        Some("BatchLongest") => Ok(PaddingStrategy::BatchLongest),
        Some(other) => Err(format!("unknown padding strategy {other:?}").into()),
        None => cfg
            .field("Fixed")
            .and_then(Json::as_usize)
            .map(PaddingStrategy::Fixed)
            .ok_or_else(|| {
                "padding strategy must be `\"BatchLongest\"` or `{\"Fixed\": <size>}`".into()
            }),
    }
}

fn read_direction(s: &str) -> Result<PaddingDirection> {
    match s {
        "Left" => Ok(PaddingDirection::Left),
        "Right" => Ok(PaddingDirection::Right),
        other => Err(format!("unknown padding direction {other:?}").into()),
    }
}
