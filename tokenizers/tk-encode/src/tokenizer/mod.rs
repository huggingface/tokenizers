//! Represents a tokenization pipeline.
//!
//! The components a [`pipeline::PipelineTokenizer`] is built out of. Normalization,
//! pre-tokenization and post-processing are defined by the traits in [`pipeline`]; what stays here
//! is the input/output vocabulary they are written against.

pub mod pattern;
pub mod pipeline;

pub use crate::decoders::{Decoder, DecoderRuntime};
pub use crate::utils::padding::{PaddingDirection, PaddingParams, PaddingStrategy, pad_encodings};
pub use crate::utils::truncation::{
    TruncationDirection, TruncationParams, TruncationStrategy, truncate_pair,
};

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;

/// Defines the expected behavior for the delimiter of a Split Pattern
/// When splitting on `'-'` for example, with input `the-final--countdown`:
///  - Removed => `[ "the", "final", "countdown" ]`
///  - Isolated => `[ "the", "-", "final", "-", "-", "countdown" ]`
///  - MergedWithPrevious => `[ "the-", "final-", "-", "countdown" ]`
///  - MergedWithNext => `[ "the", "-final", "-", "-countdown" ]`
///  - Contiguous => `[ "the", "-", "final", "--", "countdown" ]`
///
/// On disk it is the bare variant name, which the hand-written `Display` below spells verbatim.
/// This one needs no `serialization.rs` of its own: those five names are the whole on-disk shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SplitDelimiterBehavior {
    Removed,
    Isolated,
    MergedWithPrevious,
    MergedWithNext,
    Contiguous,
}

impl std::fmt::Display for SplitDelimiterBehavior {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Spelled out rather than handed to the serializer, so the name survives a build with no
        // serde -- and it is a `match` instead of abusing a `Formatter` as a `Serializer`. The
        // name `tk-serialize` reads and writes is the variant name verbatim.
        f.write_str(match self {
            Self::Removed => "Removed",
            Self::Isolated => "Isolated",
            Self::MergedWithPrevious => "MergedWithPrevious",
            Self::MergedWithNext => "MergedWithNext",
            Self::Contiguous => "Contiguous",
        })
    }
}
