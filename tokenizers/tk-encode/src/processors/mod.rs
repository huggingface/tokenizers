pub mod bert;
pub mod roberta;
#[cfg(feature = "serde")]
pub(crate) mod serialization;
pub mod template;

// Re-export these as processors
pub use super::pre_tokenizers::byte_level;

// `PostProcessorWrapper` (an untagged enum whose variant order was load-bearing — Roberta had to
// come before Bert, since serde does not validate tags) and the `Sequence` processor that held a
// `Vec<PostProcessorWrapper>` were deleted with the config layer. What the encode path runs is
// `pipeline::PipelinePostProcessor`: two templates of resolved ids, which is what the reader
// lowers every processor kind into -- a config `Sequence` included, by composing its members.
