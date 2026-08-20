pub mod bert;
pub mod byte_level;
pub mod delimiter;
pub mod digits;
pub mod fixed_length;
pub mod metaspace;
pub mod punctuation;
pub mod sequence;
#[cfg(feature = "serde")]
mod serialization;
pub mod split;
#[cfg(feature = "unicode-scripts")]
pub mod unicode_scripts;
pub mod whitespace;

// `PreTokenizerWrapper` and its hand-written `Deserialize` (tagged, with an untagged legacy
// fallback) are gone, deleted with the config layer, and so is the `Sequence` pre-tokenizer that
// held a `Vec` of it. What the encode path runs is `pipeline::PipelinePreTokenizer`, whose
// `Sequence` variant holds a `sequence::PipelineSequence` -- a `Vec` of the runtime type, needing
// no wrapper -- built straight from the config by `tk-serialize`.
