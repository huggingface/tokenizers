//! Added-vocabulary subsystem: the `AddedVocabulary`/`AddedToken` logic plus the two structures it
//! relies on — the MPHF `VocabStore` and the special-token `Buckets` matcher.
#[allow(clippy::module_inception)] // keep the AddedVocabulary logic in its own file within this folder
mod added_vocabulary;
pub mod buckets;
pub mod vocab_store;

pub use added_vocabulary::*;
