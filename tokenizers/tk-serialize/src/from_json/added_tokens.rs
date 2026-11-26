use crate::json::Json;
use tk_encode::tokenizer::Result;
use tk_encode::vocab::bucket_added_vocabulary::AddedToken as BucketAddedToken;

/// The added tokens of a config, in the order of the config.
///
/// That order gives the ids, not the `id` each token declares: `add_tokens` numbers the tokens the
/// model lacks as they come.
pub(super) fn read_added_tokens(cfg: Option<&Json<'_>>) -> Result<Vec<BucketAddedToken>> {
    let Some(arr) = cfg.and_then(Json::as_array) else {
        return Ok(Vec::new());
    };
    let mut out: Vec<BucketAddedToken> = Vec::with_capacity(arr.len());
    for entry in arr {
        entry
            .get("id")
            .and_then(Json::as_u32)
            .ok_or_else(|| -> tk_encode::Error { "an added token has no usable `id`".into() })?;
        let content = entry
            .get("content")
            .and_then(Json::as_str)
            .ok_or_else(|| -> tk_encode::Error { "an added token has no `content`".into() })?;
        // `AddedToken` has no serde defaults on the config path either: all six flags are required.
        let owner = format!("added token {content:?}");
        let mut token = BucketAddedToken::default();
        token.content = content.to_string();
        token.single_word = entry.need(&owner, "single_word", Json::as_bool)?;
        token.lstrip = entry.need(&owner, "lstrip", Json::as_bool)?;
        token.rstrip = entry.need(&owner, "rstrip", Json::as_bool)?;
        token.normalized = entry.need(&owner, "normalized", Json::as_bool)?;
        token.special = entry.need(&owner, "special", Json::as_bool)?;
        token.control = entry.need(&owner, "control", Json::as_bool)?;
        out.push(token);
    }
    Ok(out)
}
