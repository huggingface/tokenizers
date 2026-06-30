use std::ops::Range;

use super::Result;

/// A pre-token split, a range into the input text.
pub struct Split {
    pub range: Range<usize>,
}

pub trait PreTokenizer {
    /// Split `text` into pre-tokens, appending to `out`. Ranges are into `text`.
    fn pre_tokenize(&self, text: &str, out: &mut Vec<Split>) -> Result<()>;
}
