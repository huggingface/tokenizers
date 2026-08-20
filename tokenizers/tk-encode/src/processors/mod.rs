pub mod bert;
pub mod roberta;
#[cfg(feature = "serde")]
pub(crate) mod serialization;
pub mod template;

// Re-export these as processors
pub use super::pre_tokenizers::byte_level;

// `PostProcessorWrapper` (an untagged enum whose variant order is load-bearing — Roberta must come
// before Bert, since serde does not validate tags) and the `Sequence` processor that holds a
// `Vec<PostProcessorWrapper>` live in `tk-convert`. What the encode path runs is
// `pipeline::PipelinePostProcessor`, which the config layer lowers a wrapper into.
