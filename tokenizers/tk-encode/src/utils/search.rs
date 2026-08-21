//! Looking for a pattern in a string, for the components spelled as `{pattern, content}`.
//!
//! This is a *utility*, not a component: both the `Replace` normalizer and the `Replace` decoder
//! need to find the same matches, and neither should have to name the other to do it. Keeping the
//! matcher here is what lets those two be genuinely separate types rather than one type wearing
//! two hats.

use atomsplit::literal::Literal;

use crate::tokenizer::Result;
use crate::tokenizer::pattern::Pattern;
use crate::utils::SysRegex;

/// Represents the different patterns that a `Replace` can use.
///
/// Externally tagged on disk — `{"String":"…"}` / `{"Regex":"…"}` — which is what the bare derive
/// gives and what is in every `tokenizer.json` that has one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplacePattern {
    String(String),
    Regex(String),
}

impl From<String> for ReplacePattern {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}

impl From<&str> for ReplacePattern {
    fn from(v: &str) -> Self {
        Self::String(v.to_owned())
    }
}

/// How a `Replace` looks for its pattern.
#[derive(Debug)]
pub(crate) enum Search {
    /// A plain string, scanned for directly — no regex engine involved, so this works in every
    /// build.
    Literal(Literal),
    /// A real regex, which needs the system backend (the `fancy-regex` feature).
    Regex(SysRegex),
    /// The empty string, which matches nothing.
    Nothing,
}

impl Search {
    /// Compile a pattern into something that can be searched for.
    pub(crate) fn new(pattern: &ReplacePattern) -> Result<Self> {
        Ok(match pattern {
            ReplacePattern::String(s) if s.is_empty() => Self::Nothing,
            ReplacePattern::String(s) => Self::Literal(Literal::new(s.as_bytes())?),
            ReplacePattern::Regex(r) => Self::Regex(SysRegex::new(r)?),
        })
    }

    /// Every span of `inside`, flagged with whether it matched. Used by the decoder, which rebuilds
    /// the string span by span.
    pub(crate) fn find_matches(&self, inside: &str) -> Result<Vec<((usize, usize), bool)>> {
        match self {
            Self::Literal(literal) => literal.find_matches(inside),
            Self::Regex(regex) => regex.find_matches(inside),
            Self::Nothing => Ok(vec![((0, inside.len()), false)]),
        }
    }
}
