//! Represents a tokenization pipeline.
//!
//! The components a [`pipeline::PipelineTokenizer`] is built out of. Normalization,
//! pre-tokenization and post-processing are defined by the traits in [`pipeline`]; what stays here
//! is the input/output vocabulary they are written against.

// The `Path` types are reachable only from `Model::save`.

mod encoding;
pub mod pattern;
pub mod pipeline;

// `Tokenizer`, `TokenizerImpl`, `TokenizerBuilder`, `DecodeStream`, the legacy `AddedVocabulary`
// (and its `AddedToken`) and the whole serde load/save surface live in `tk-convert`. What
// stays here is the vocabulary of traits every component implements, plus the input/output types
// they are written against.
pub use crate::decoders::{Decoder, DecoderRuntime};
pub use crate::utils::iter::LinesWithEnding;
pub use crate::utils::padding::{PaddingDirection, PaddingParams, PaddingStrategy, pad_encodings};
pub use crate::utils::truncation::{
    TruncationDirection, TruncationParams, TruncationStrategy, truncate_encodings,
};
pub use encoding::*;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;
pub type Offsets = (usize, usize);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub id: u32,
    pub value: String,
    pub offsets: (usize, usize),
}
impl Token {
    pub fn new(id: u32, value: String, offsets: (usize, usize)) -> Self {
        Self { id, value, offsets }
    }
}

use std::borrow::Cow;

#[derive(Debug, Clone)]
pub enum InputSequence<'s> {
    Raw(Cow<'s, str>),
    PreTokenized(Cow<'s, [&'s str]>),
    PreTokenizedOwned(Cow<'s, [String]>),
    PreTokenizedCow(Cow<'s, [Cow<'s, str>]>),
}

impl<'s> From<Cow<'s, str>> for InputSequence<'s> {
    fn from(input: Cow<'s, str>) -> Self {
        Self::Raw(input)
    }
}

impl<'s> From<&'s str> for InputSequence<'s> {
    fn from(input: &'s str) -> Self {
        Self::Raw(Cow::Borrowed(input))
    }
}

impl From<String> for InputSequence<'_> {
    fn from(input: String) -> Self {
        Self::Raw(Cow::Owned(input))
    }
}

impl<'s> From<&'s [&'s str]> for InputSequence<'s> {
    fn from(input: &'s [&'s str]) -> Self {
        Self::PreTokenized(Cow::Borrowed(input))
    }
}

impl<'s> From<Vec<&'s str>> for InputSequence<'s> {
    fn from(input: Vec<&'s str>) -> Self {
        Self::PreTokenized(Cow::Owned(input))
    }
}

impl<'s> From<&'s [String]> for InputSequence<'s> {
    fn from(input: &'s [String]) -> Self {
        Self::PreTokenizedOwned(Cow::Borrowed(input))
    }
}

impl From<Vec<String>> for InputSequence<'_> {
    fn from(input: Vec<String>) -> Self {
        Self::PreTokenizedOwned(Cow::Owned(input))
    }
}

impl<'s> From<Vec<Cow<'s, str>>> for InputSequence<'s> {
    fn from(input: Vec<Cow<'s, str>>) -> Self {
        Self::PreTokenizedCow(Cow::Owned(input))
    }
}

impl<'s> From<&'s [Cow<'s, str>]> for InputSequence<'s> {
    fn from(input: &'s [Cow<'s, str>]) -> Self {
        Self::PreTokenizedCow(Cow::Borrowed(input))
    }
}

#[derive(Debug, Clone)]
pub enum EncodeInput<'s> {
    Single(InputSequence<'s>),
    Dual(InputSequence<'s>, InputSequence<'s>),
}

impl<'s, I: Into<InputSequence<'s>>> From<I> for EncodeInput<'s> {
    fn from(input: I) -> Self {
        Self::Single(input.into())
    }
}

impl<'s, I1, I2> From<(I1, I2)> for EncodeInput<'s>
where
    I1: Into<InputSequence<'s>>,
    I2: Into<InputSequence<'s>>,
{
    fn from(input: (I1, I2)) -> Self {
        Self::Dual(input.0.into(), input.1.into())
    }
}

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
