//! The tokens a long word may still have to be looked up as.

use crate::utils::signature::Signature;
use crate::vocab::bucket_vocab_store::BucketVocabStore;

/// The tokens longer than a [`Signature`] holds whole that merging their own text does not give.
///
/// A word that long skips the vocabulary: merging it gives the ids its token would.
/// These tokens are the exception. Most models have none.
#[derive(Default)]
pub(super) struct LongFolds {
    /// Their [`Signature::hash`]es, sorted.
    hashes: Box<[u64]>,
}

impl LongFolds {
    /// `merged` says, per vocabulary id, whether merging the token's own text gives that id.
    pub(super) fn new(vocab: &BucketVocabStore, merged: &[bool]) -> Self {
        let mut hashes: Vec<u64> = Vec::new();
        for id in 0..vocab.id_space() as u32 {
            let Some(token) = vocab.id_to_token_bytes(id) else {
                continue;
            };
            // A word is a `&str`, so it never equals a token that is not UTF-8.
            if token.len() > Signature::BYTES
                && std::str::from_utf8(token).is_ok()
                && !merged[id as usize]
            {
                hashes.push(Signature::new(token).hash);
            }
        }
        hashes.sort_unstable();
        Self {
            hashes: hashes.into_boxed_slice(),
        }
    }

    /// Whether a word with this signature may be one of the tokens. Another word can have the
    /// same hash, so the vocabulary confirms it.
    pub(super) fn may_hold(&self, signature: Signature) -> bool {
        self.hashes.binary_search(&signature.hash).is_ok()
    }
}
