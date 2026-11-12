//! Finding a pattern in a text: a plain string or a regex.
//!
//! [`Search`] is what `Split`, the `Replace` normalizer and the `Replace` decoder hold. The `Pattern`
//! trait behind it is visible only inside this module, so every match goes through its order check.

mod matcher;
mod search;

// Optional system-regex backend, needed only for a *regex* pattern that bitcannon does not cover.
// With `fancy-regex` off a stub compiles and those patterns error at load.
#[cfg(feature = "fancy-regex")]
mod fancy;
#[cfg(not(feature = "fancy-regex"))]
mod no_regex;

#[cfg(feature = "fancy-regex")]
pub use fancy::SysRegex;
#[cfg(not(feature = "fancy-regex"))]
pub use no_regex::SysRegex;
pub use search::ReplacePattern;
pub(crate) use search::Search;
