//! The `truncation` object, read into the `TruncationParams` the tokenizer cuts with.

use crate::json::Json;
use tk_encode::tokenizer::{Result, TruncationDirection, TruncationParams, TruncationStrategy};

/// `{"direction": "Right", "max_length": 128, "strategy": "LongestFirst", "stride": 0}`, which
/// every `encode` then applies. Absent or `null` means the config declares no truncation.
pub(super) fn read_truncation(cfg: Option<&Json<'_>>) -> Result<Option<TruncationParams>> {
    let Some(cfg) = cfg else {
        return Ok(None);
    };
    // A `stride` asks for what the released crate returns as `overflowing` encodings: windows of
    // `max_length` tokens that consecutive windows share `stride` of. The pipeline keeps the first
    // window and drops the rest, so the value is read and carried rather than refused. 395 of the
    // 5,053 popular Hub configs surveyed carry one, all question-answering fine-tunes, and
    // transformers rebuilds truncation from the per-call `stride` anyway, so the file's value never
    // reaches encode there either.
    Ok(Some(TruncationParams {
        max_length: cfg.need("the `truncation` config", "max_length", Json::as_usize)?,
        strategy: read_strategy(cfg)?,
        direction: read_direction(cfg)?,
        stride: cfg.need("the `truncation` config", "stride", Json::as_usize)?,
    }))
}

fn read_strategy(cfg: &Json<'_>) -> Result<TruncationStrategy> {
    match cfg.need("the `truncation` config", "strategy", Json::as_str)? {
        "LongestFirst" => Ok(TruncationStrategy::LongestFirst),
        "OnlyFirst" => Ok(TruncationStrategy::OnlyFirst),
        "OnlySecond" => Ok(TruncationStrategy::OnlySecond),
        other => Err(format!("unknown truncation strategy {other:?}").into()),
    }
}

/// The one key a config may leave out. The released crate added `direction` after `truncation`
/// first shipped, with a serde default, so a file saved before that has no such key:
/// `sentence-transformers/all-MiniLM-L6-v2` is one. Absent means `Right`.
fn read_direction(cfg: &Json<'_>) -> Result<TruncationDirection> {
    let Some(direction) = cfg.field("direction") else {
        return Ok(TruncationDirection::Right);
    };
    match direction.as_str() {
        Some("Left") => Ok(TruncationDirection::Left),
        Some("Right") => Ok(TruncationDirection::Right),
        Some(other) => Err(format!("unknown truncation direction {other:?}").into()),
        None => Err("the `truncation` config's `direction` is not a string".into()),
    }
}
