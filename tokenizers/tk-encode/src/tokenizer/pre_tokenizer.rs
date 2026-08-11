use crate::{NormalizedString, Offsets, Token};
use std::collections::HashMap;

/// Various possible types of offsets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffsetType {
    Byte,
    Char,
    None,
}

/// Wrapper for a subpart of a `NormalizedString`.
///
/// This Split contains the underlying `NormalizedString` as well as its offsets
/// in the original string. These offsets are in the `original` referential.
/// It also contains any `Token` associated to the current split
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Split {
    /// The underlying `NormalizedString`. Each SubString is represented by a `NormalizedString`
    /// and in the end we might be carrying a lot of SubString representing various parts of the
    /// original input string.
    normalized: NormalizedString,
    /// Optional Tokens associated to this Split
    tokens: Option<Vec<Token>>,
}

impl From<NormalizedString> for Split {
    fn from(n: NormalizedString) -> Self {
        Self {
            normalized: n,
            tokens: None,
        }
    }
}

impl From<(NormalizedString, Option<Vec<Token>>)> for Split {
    fn from(f: (NormalizedString, Option<Vec<Token>>)) -> Self {
        Self {
            normalized: f.0,
            tokens: f.1,
        }
    }
}

struct BytesToCharOffsetConverter {
    map: HashMap<usize, usize>,
}

impl BytesToCharOffsetConverter {
    pub fn new(sequence: &str) -> Self {
        Self {
            map: sequence
                .char_indices()
                .enumerate()
                .flat_map(|(i, (b, c))| {
                    let mut n = 0;
                    std::iter::repeat_with(move || {
                        let o = (b + n, i);
                        n += 1;
                        o
                    })
                    .take(c.len_utf8())
                })
                .collect(),
        }
    }

    pub fn convert(&self, offsets: Offsets) -> Option<Offsets> {
        match (self.map.get(&offsets.0), self.map.get(&offsets.1)) {
            (Some(start), Some(end)) => Some((*start, *end)),
            // If we reached the end, `end` is not in the map
            (Some(start), None) => {
                // But the one just before should be
                let last = self.map.get(&(offsets.1 - 1)).copied().unwrap_or(start + 1);
                Some((*start, last + 1))
            }
            _ => None,
        }
    }
}
