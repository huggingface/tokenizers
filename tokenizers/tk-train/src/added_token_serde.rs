//! The serialized shape of an [`AddedToken`], owned here.
//!
//! `tk_encode`'s `AddedToken` carries no serde: what a token looks like on disk is the config
//! layer's business, and the config layer is not the inference crate. `tk-convert` used to hold this
//! shape; with its strip, the only thing left that needs one is the trainers -- the Python bindings
//! pickle a `TrainerWrapper` through `serde_json` -- so the shape lives with them.
//!
//! Use it on a field: `#[serde(with = "crate::added_token_serde")]`.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tk_encode::vocab::bucket_added_vocabulary::AddedToken;

/// The serialized form of an [`AddedToken`].
///
/// The six fields, in this order, are what an `added_tokens` entry in a `tokenizer.json` holds
/// (minus its `id`, which a trainer does not assign).
#[derive(Serialize, Deserialize)]
struct AddedTokenDef {
    content: String,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
    special: bool,
}

impl From<&AddedToken> for AddedTokenDef {
    fn from(token: &AddedToken) -> Self {
        Self {
            content: token.content.clone(),
            single_word: token.single_word,
            lstrip: token.lstrip,
            rstrip: token.rstrip,
            normalized: token.normalized,
            special: token.special,
        }
    }
}

impl From<AddedTokenDef> for AddedToken {
    fn from(def: AddedTokenDef) -> Self {
        let mut token = AddedToken::default();
        token.content = def.content;
        token.single_word = def.single_word;
        token.lstrip = def.lstrip;
        token.rstrip = def.rstrip;
        token.normalized = def.normalized;
        token.special = def.special;
        token
    }
}

pub fn serialize<S: Serializer>(tokens: &[AddedToken], s: S) -> Result<S::Ok, S::Error> {
    s.collect_seq(tokens.iter().map(AddedTokenDef::from))
}

pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<AddedToken>, D::Error> {
    let defs: Vec<AddedTokenDef> = Vec::deserialize(d)?;
    let tokens: Vec<AddedToken> = defs.into_iter().map(Into::into).collect();
    Ok(tokens)
}
