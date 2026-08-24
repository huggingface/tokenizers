use crate::json::{Json, JsonExt};
use tk_encode::tokenizer::Result;
use tk_encode::vocab::bucket_added_vocabulary::AddedToken as BucketAddedToken;

/// Added tokens, in ascending id order — `add_tokens` depends on that order
pub(super) fn read_added_tokens(cfg: Option<&Json<'_>>) -> Result<Vec<BucketAddedToken>> {
    let Some(arr) = cfg.and_then(Json::as_arr) else {
        return Ok(Vec::new());
    };
    // The ids alone decide the order, so only the ids get sorted: each key is its token's id in
    // the high half and the token's position in the array in the low half, which makes ascending
    // u64 order identical to a *stable* sort by id
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
