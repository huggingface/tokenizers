//! The serialized shape of an [`AddedToken`], owned here.
//!
//! `tk_encode`'s `AddedToken` carries no serde: what a token looks like on disk is the config
//! layer's business, and the config layer is not the inference crate. The trainers need it to read
//! back a trainer's JSON, so the shape lives with them.
//!
//! Use it on a field: `#[serde(with = "crate::added_token_serde")]`.

use serde::{Deserialize, Deserializer};
use tk_encode::vocab::bucket_added_vocabulary::AddedToken;

/// A stand-in with the same fields, so serde can derive against a foreign type.
///
/// The six fields, in this order, are what an `added_tokens` entry in a `tokenizer.json` holds
/// (minus its `id`, which a trainer does not assign).
#[derive(Deserialize)]
#[serde(remote = "AddedToken")]
struct AddedTokenDef {
    content: String,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
    special: bool,
}

pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<AddedToken>, D::Error> {
    #[derive(Deserialize)]
    struct Wrapper(#[serde(with = "AddedTokenDef")] AddedToken);

    Ok(Vec::<Wrapper>::deserialize(d)?
        .into_iter()
        .map(|Wrapper(token)| token)
        .collect())
}
