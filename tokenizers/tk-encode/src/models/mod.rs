//! Popular tokenizer models.

pub mod bpe;
#[cfg(feature = "unigram")]
pub mod unigram;
#[cfg(feature = "wordlevel")]
pub mod wordlevel;
#[cfg(feature = "wordpiece")]
pub mod wordpiece;

// Nothing else lives here. `ModelWrapper`, its hand-written `Deserialize`, the per-model serde and
// `OrderedVocabIter` (which only ever existed to give those impls a deterministic vocabulary order)
// are all `tk-convert`'s now. The config-shaped `BPE` went with them: what is left in this tree is
// the four engines that actually encode, and `PipelineBPE` is the only BPE among them.
