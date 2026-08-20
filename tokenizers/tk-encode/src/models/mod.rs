//! Popular tokenizer models.

pub mod bpe;
#[cfg(feature = "unigram")]
pub mod unigram;
#[cfg(feature = "wordlevel")]
pub mod wordlevel;
#[cfg(feature = "wordpiece")]
pub mod wordpiece;

// `ModelWrapper` and its hand-written `Deserialize` are gone, and so is the config-shaped `BPE`:
// both were deleted with the config layer. What is left in this tree is the engines that actually
// encode -- `PipelineBPE` is the only BPE among them -- plus, below, the one serde helper they
// share.

/// Wraps a vocab mapping (ID -> token) to a struct that will be serialized in order
/// of token ID, smallest to largest.
///
/// It exists to give a `{token: id}` object a *deterministic* order: a hash map's iteration order is
/// unspecified, and a `vocab.json` that reorders itself between two saves of the same model is a
/// diff nobody wants to read. `pub` because the `Serialize` impls that use it live in the
/// `serialization.rs` next to each model rather than here, and because it has been public API
/// since before the split.
#[cfg(feature = "serde")]
pub struct OrderedVocabIter<'a> {
    vocab_r: &'a ahash::AHashMap<u32, String>,
}

#[cfg(feature = "serde")]
impl<'a> OrderedVocabIter<'a> {
    pub fn new(vocab_r: &'a ahash::AHashMap<u32, String>) -> Self {
        Self { vocab_r }
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for OrderedVocabIter<'_> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // There could be holes so max + 1 is more correct than vocab_r.len()
        let mut holes = vec![];
        let result = if let Some(max) = self.vocab_r.keys().max() {
            let iter = (0..*max + 1).filter_map(|i| {
                if let Some(token) = self.vocab_r.get(&i) {
                    Some((token, i))
                } else {
                    holes.push(i);
                    None
                }
            });
            serializer.collect_map(iter)
        } else {
            serializer.collect_map(std::iter::empty::<(&str, u32)>())
        };

        if !holes.is_empty() {
            warn!(
                "The OrderedVocab you are attempting to serialize contains holes for indices {holes:?}, your vocabulary could be corrupted!"
            );
        }
        result
    }
}
