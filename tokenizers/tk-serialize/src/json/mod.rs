//! The parsed tree and its accessors. `hifijson` does the parsing; the float arithmetic that
//! reproduces `serde_json`'s rounding lives in [`crate::vendored`].

use std::borrow::Cow;
use std::fmt;

use hifijson::token::Lex as _;

use crate::vendored::{f64_from_parts, number};

/// A real `tokenizer.json` nests about six levels; this keeps a hostile one off the stack.
pub const MAX_DEPTH: usize = 64;

type Raw<'a> = hifijson::value::Value<&'a str, Cow<'a, str>>;

#[repr(transparent)]
#[derive(Debug, PartialEq)]
/// A parsed `tokenizer.json`. A newtype, not an alias, so the accessors are inherent methods
/// and `hifijson` stays inside this module. Nothing becomes an `f64` until [`Json::as_f64`].
pub struct Json<'a>(Raw<'a>);

#[derive(Debug, PartialEq)]
pub struct Error {
    pub at: usize,
    pub kind: hifijson::Error,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid JSON at byte {}: {}", self.at, self.kind)
    }
}

impl std::error::Error for Error {}

impl<'a> Json<'a> {
    pub fn parse(input: &'a str) -> Result<Self, Error> {
        let mut lexer = hifijson::SliceLexer::new(input.as_bytes());
        let parsed = lexer.exactly_one(hifijson::token::Lex::ws_peek, |next, lexer| {
            hifijson::value::parse_bounded(MAX_DEPTH, next, lexer)
        });
        parsed.map(Json).map_err(|kind| Error {
            at: input.len() - lexer.as_slice().len(),
            kind,
        })
    }

    #[inline]
    fn wrap<'r>(raw: &'r Raw<'a>) -> &'r Self {
        // SAFETY: `Json` is `#[repr(transparent)]` over exactly `Raw`.
        unsafe { &*(raw as *const Raw<'a> as *const Self) }
    }

    /// A `null` value reads as present; [`Json::field`] treats it as absent.
    pub fn get(&self, key: &str) -> Option<&Json<'a>> {
        match &self.0 {
            Raw::Object(entries) => entries
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| Json::wrap(v)),
            _ => None,
        }
    }

    /// [`get`](Json::get), but `null` reads as absent -- `"normalizer": null` means no normalizer.
    pub fn field(&self, key: &str) -> Option<&Json<'a>> {
        self.get(key).filter(|v| !matches!(v.0, Raw::Null))
    }

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

    pub fn as_f64(&self) -> Option<f64> {
        match &self.0 {
            Raw::Number((digits, _)) => {
                let (positive, significand, exponent) = number(digits);
                Some(f64_from_parts(positive, significand, exponent))
            }
            _ => None,
        }
    }

    /// Rejects fractions, negatives and anything past `u32::MAX`, so a malformed file cannot
    /// silently truncate an id.
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
            Raw::Array(items) => {
                Some(unsafe { &*(items.as_slice() as *const [Raw<'a>] as *const [Json<'a>]) })
            }
            _ => None,
        }
    }

    pub fn entries(&self) -> Option<impl ExactSizeIterator<Item = (&str, &Json<'a>)>> {
        match &self.0 {
            Raw::Object(entries) => Some(entries.iter().map(|(k, v)| (&**k, Json::wrap(v)))),
            _ => None,
        }
    }

    /// A required field: the accessor plus the one error -- "`<owner>` has no `<name>`".
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
