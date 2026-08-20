//! serde for the added vocabulary: the `added_tokens` array of a `tokenizer.json`.
//!
//! [`AddedToken`] itself is a plain derive on the type. What needs writing out by hand is
//! [`AddedVocabulary`], because what goes on the wire is its *logical* token list — id, content and
//! flags, ordered by id — and never the derived `Buckets`/`VocabStore`, which `add_tokens` rebuilds
//! from that list on the way back in. There is deliberately no `Deserialize`: an added vocabulary is
//! only ever rebuilt by replaying `add_tokens`, which is what maintains those derived structures.

use serde::ser::SerializeSeq;
use serde::{Serialize, Serializer};

use super::bucket_added_vocabulary::{AddedToken, AddedVocabulary};

/// One entry of the `added_tokens` array: an [`AddedToken`] with the id it was assigned, flattened
/// into a single object.
#[derive(Debug, Serialize, serde::Deserialize)]
pub struct AddedTokenWithId {
    /// The id assigned to this token
    pub id: u32,
    /// The target AddedToken
    #[serde(flatten)]
    pub token: AddedToken,
}

impl Serialize for AddedVocabulary {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // `get_added_tokens_decoder` is keyed by id, so the ids are unique and sorting by them is a
        // total order — an unstable sort is enough to make two saves of the same tokenizer identical.
        let mut added_tokens: Vec<AddedTokenWithId> = self
            .get_added_tokens_decoder()
            .into_iter()
            .map(|(id, token)| AddedTokenWithId { id, token })
            .collect();
        added_tokens.sort_unstable_by_key(|o| o.id);

        let mut vocabulary = serializer.serialize_seq(Some(added_tokens.len()))?;
        for token in &added_tokens {
            vocabulary.serialize_element(token)?;
        }
        vocabulary.end()
    }
}
