//! The original `BPE` model, as the released `tokenizers` crate implements it. The encode path
//! replaces it with [`PipelineBPE`](super::PipelineBPE), which is built from a [`BPE`](model::BPE)
//! by consuming it; it is still what serde reads and writes, and what `tk-train` trains into.
pub(super) mod model;
mod serialization;
