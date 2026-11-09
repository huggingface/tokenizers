//! The [`Search`] that `Split` and the `Replace` normalizer and decoder hold.
//!
//! The `Replace` normalizer and decoder find the same matches, and neither should have to name the
//! other to do it.

use bitcannon::literal::Literal;

use super::SysRegex;
use super::matcher::{self, Pattern};
use crate::Offsets;
use crate::tokenizer::Result;
use std::borrow::Cow;

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

/// How a `Split` or a `Replace` looks for its pattern.
#[derive(Debug)]
pub(crate) enum Search {
    /// A plain string, scanned for directly. It needs no regex engine, so it works in every build.
    Literal(Literal),
    /// A real regex, which needs the system backend (the `fancy-regex` feature).
    Regex(SysRegex),
    /// The empty string, which matches nothing.
    Nothing,
}

impl Search {
    /// Compile a `Replace` pattern into something that can be searched for.
    pub(crate) fn new(pattern: &ReplacePattern) -> Result<Self> {
        Ok(match pattern {
            ReplacePattern::String(s) if s.is_empty() => Self::Nothing,
            ReplacePattern::String(s) => Self::Literal(Literal::new(s.as_bytes())?),
            ReplacePattern::Regex(r) => Self::Regex(SysRegex::new(r)?),
        })
    }

    /// Every piece of `inside`, in order and covering all of it, flagged with whether it matched.
    pub(crate) fn find_matches(&self, inside: &str) -> Result<Vec<(Offsets, bool)>> {
        match self {
            Self::Literal(literal) => matcher::find_matches(literal, inside),
            Self::Regex(regex) => matcher::find_matches(regex, inside),
            Self::Nothing => Ok(vec![((0, inside.len()), false)]),
        }
    }

    /// The text with every match swapped for `content`. Borrows `input` when nothing matched.
    pub(crate) fn replace<'a>(&self, input: &'a str, content: &str) -> Result<Cow<'a, str>> {
        match self {
            Self::Literal(literal) => replace_matches(input, content, literal),
            Self::Regex(regex) => replace_matches(input, content, regex),
            Self::Nothing => Ok(Cow::Borrowed(input)),
        }
    }
}

fn replace_matches<'a>(
    input: &'a str,
    content: &str,
    pattern: &impl Pattern,
) -> Result<Cow<'a, str>> {
    let mut replaced: Option<String> = None;
    let mut last_end = 0;
    for found in matcher::matches(pattern, input) {
        let (start, end) = found?;
        let replaced: &mut String =
            replaced.get_or_insert_with(|| String::with_capacity(input.len()));
        replaced.push_str(&input[last_end..start]);
        replaced.push_str(content);
        last_end = end;
    }
    Ok(match replaced {
        Some(mut replaced) => {
            replaced.push_str(&input[last_end..]);
            Cow::Owned(replaced)
        }
        None => Cow::Borrowed(input),
    })
}
