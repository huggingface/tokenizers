// `pub` because `Tokenizer::from_pretrained` — the only caller — lives in `tk-convert`.
#[cfg(feature = "http")]
pub mod from_pretrained;
pub(crate) mod word_cache;

// Optional system-regex backend, needed only for a *regex* pattern that bitcannon does not cover.
// With `fancy-regex` off a stub compiles and those patterns error at load. Everything else works
// regardless: the bitcannon-native pre-tokenizers, and any `Split` or `Replace` whose pattern is a
// plain string (searched for directly, see `bitcannon::literal`).
#[cfg(feature = "fancy-regex")]
mod fancy;
#[cfg(feature = "fancy-regex")]
pub use fancy::SysRegex;
#[cfg(not(feature = "fancy-regex"))]
mod no_regex;
#[cfg(not(feature = "fancy-regex"))]
pub use no_regex::SysRegex;

// Recognize known GPT pre-tokenization regexes and route them to bitcannon's native (unrolled) FSM.
mod unrolled_regex;
pub use unrolled_regex::{DEEPSEEK_PATTERNS, Grammar, GrammarPattern, is_deepseek, recognize};

pub mod byte_level;
pub mod iter;
pub mod padding;
#[cfg(feature = "parallelism")]
pub mod parallelism;
pub mod progress;
pub mod search;
pub mod truncation;

// Re-export ProgressFormat for public API
pub use progress::ProgressFormat;


pub const DEFAULT_CACHE_CAPACITY: usize = 65536;