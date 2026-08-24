//! We decided to use `hifijson` as it has 0 dependencies and is very lightweight. What is left
//! here is the tree and its accessors; the code that reproduces the previous behavior when parsing
//! a tokenizer.json and fp64 values is a small part of the serde json library, vendored with
//! attritbution in [`crate::vendored`].
//!

use std::borrow::Cow;
use std::fmt;

use hifijson::token::Lex as _;

use crate::vendored::{f64_from_parts, number};

/// Deepest nesting accepted. A real `tokenizer.json` nests about six levels max
/// (`post_processor.special_tokens.<tok>.tokens[0]`); this leaves plenty of room while keeping a
/// hostile file from recursing the parser off the stack.
pub const MAX_DEPTH: usize = 64;

/// `hifijson`'s tree instantiated for slice input, which is why the two type parameters are what
/// they are: a string borrows from the input unless it carries an escape (`Cow`), and a number is
/// the `&str` it was written as, paired with the parts (`zero`/`dot`/`exp`) the lexer saw.
type Raw<'a> = hifijson::value::Value<&'a str, Cow<'a, str>>;

/// A parsed `tokenizer.json`.
///
/// Nothing becomes an `f64` until [`Json::as_f64`] asks, which is the whole reason this parser and
/// not another -- see [`crate::vendored`].
///
/// A newtype rather than an alias so the accessors below are inherent methods: no trait to import
/// at every reader, and `hifijson` stays an implementation detail of this module.
#[repr(transparent)]
#[derive(Debug, PartialEq)]
pub struct Json<'a>(Raw<'a>);

/// A parse failure, and how far into the document it happened.
#[derive(Debug, PartialEq)]
pub struct Error {
    /// Byte offset of the first byte the lexer had not consumed when it gave up. Worth keeping: a
    /// `tokenizer.json` runs to tens of megabytes, so "invalid JSON" on its own says very little.
    pub at: usize,
    /// What `hifijson` refused.
    pub kind: hifijson::Error,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid JSON at byte {}: {}", self.at, self.kind)
    }
}

impl std::error::Error for Error {}

/// The lookups the reader needs.
///
/// Every accessor here answers `None` for "not there or not that kind", which is what an optional
/// field wants: an absent field and a wrongly-typed one both mean "not available". A *required*
/// field goes through [`Json::need`] instead, which turns the same `None` into an error naming
/// the field and what was reading it.
impl<'a> Json<'a> {
    /// Parse a whole document. Trailing content other than whitespace is an error.
    pub fn parse(input: &'a str) -> Result<Self, Error> {
        let mut lexer = hifijson::SliceLexer::new(input.as_bytes());
        // `exactly_one` is what rejects trailing content; `parse_bounded` is what caps recursion.
        let parsed = lexer.exactly_one(hifijson::token::Lex::ws_peek, |next, lexer| {
            hifijson::value::parse_bounded(MAX_DEPTH, next, lexer)
        });
        parsed.map(Json).map_err(|kind| Error {
            // What the lexer has not eaten yet, measured against the whole input.
            at: input.len() - lexer.as_slice().len(),
            kind,
        })
    }

    /// A borrowed [`Raw`] is a borrowed [`Json`]: the newtype is `#[repr(transparent)]`, so the
    /// two have identical layout and this is the standard newtype-reference cast.
    #[inline]
    fn wrap<'r>(raw: &'r Raw<'a>) -> &'r Self {
        // SAFETY: `Json` is `#[repr(transparent)]` over exactly `Raw`.
        unsafe { &*(raw as *const Raw<'a> as *const Self) }
    }

    /// Look a key up in an object. `None` for a missing key *or* a non-object. A `null` value
    /// reads as present; [`Json::field`] is the one that treats it as absent.
    pub fn get(&self, key: &str) -> Option<&Json<'a>> {
        match &self.0 {
            // Linear, because an object here has a handful of keys and hashing them costs more.
            Raw::Object(entries) => entries
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| Json::wrap(v)),
            _ => None,
        }
    }

    /// [`get`](Json::get), but `null` reads as absent. `tokenizer.json` spells "no normalizer"
    /// as `"normalizer": null`, and every caller wants that to behave like a missing key.
    pub fn field(&self, key: &str) -> Option<&Json<'a>> {
        self.get(key).filter(|v| !matches!(v.0, Raw::Null))
    }

    /// The `"type"` tag. `None` when the object has none, which every canonical component has.
    pub fn type_tag(&self) -> Option<&str> {
        self.get("type")?.as_str()
    }

    pub fn as_str(&self) -> Option<&str> {
        match &self.0 {
            Raw::String(s) => Some(&**s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match &self.0 {
            Raw::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// The number as `serde_json`'s default build would have parsed it. See [`crate::vendored`].
    pub fn as_f64(&self) -> Option<f64> {
        match &self.0 {
            Raw::Number((digits, _)) => {
                let (positive, significand, exponent) = number(digits);
                Some(f64_from_parts(positive, significand, exponent))
            }
            _ => None,
        }
    }

    /// A token id or a count. Rejects fractions, negatives and anything past `u32::MAX`, so a
    /// malformed file cannot silently truncate an id.
    pub fn as_u32(&self) -> Option<u32> {
        let n = self.as_f64()?;
        if n.fract() != 0.0 || !(0.0..=f64::from(u32::MAX)).contains(&n) {
            return None;
        }
        Some(n as u32)
    }

    pub fn as_usize(&self) -> Option<usize> {
        let n = self.as_f64()?;
        if n.fract() != 0.0 || !(0.0..=9_007_199_254_740_992.0).contains(&n) {
            return None;
        }
        Some(n as usize)
    }

    pub fn as_array(&self) -> Option<&[Json<'a>]> {
        match &self.0 {
            // SAFETY: `Json` is `#[repr(transparent)]` over `Raw`, so `[Raw]` and `[Json]` have
            // the same layout and element count.
            Raw::Array(items) => {
                Some(unsafe { &*(items.as_slice() as *const [Raw<'a>] as *const [Json<'a>]) })
            }
            _ => None,
        }
    }

    /// The `"key": value` pairs, in the order the file wrote them. `ExactSizeIterator` so a
    /// caller can still size its map up front.
    pub fn entries(&self) -> Option<impl ExactSizeIterator<Item = (&str, &Json<'a>)>> {
        match &self.0 {
            Raw::Object(entries) => Some(entries.iter().map(|(k, v)| (&**k, Json::wrap(v)))),
            _ => None,
        }
    }

    /// A required field: the optional accessor above, plus the one error every reader used to
    /// spell as its own local closure -- "`<owner>` has no `<name>`".
    pub fn need<'s, T>(
        &'s self,
        owner: &str,
        name: &str,
        as_kind: impl FnOnce(&'s Json<'a>) -> Option<T>,
    ) -> tk_encode::Result<T> {
        self.field(name)
            .and_then(as_kind)
            .ok_or_else(|| format!("{owner} has no `{name}`").into())
    }
}

#[cfg(test)]
mod tests;
