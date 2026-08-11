use serde::{Deserialize, Serialize};

/// Defines the expected behavior for the delimiter of a Split Pattern
/// When splitting on `'-'` for example, with input `the-final--countdown`:
///  - Removed => `[ "the", "final", "countdown" ]`
///  - Isolated => `[ "the", "-", "final", "-", "-", "countdown" ]`
///  - MergedWithPrevious => `[ "the-", "final-", "-", "countdown" ]`
///  - MergedWithNext => `[ "the", "-final", "-", "-countdown" ]`
///  - Contiguous => `[ "the", "-", "final", "--", "countdown" ]`
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Eq)]
pub enum SplitDelimiterBehavior {
    Removed,
    Isolated,
    MergedWithPrevious,
    MergedWithNext,
    Contiguous,
}

impl std::fmt::Display for SplitDelimiterBehavior {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.serialize(f)
    }
}
