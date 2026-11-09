pub(crate) mod cache;
// `pub` because `Tokenizer::from_pretrained` — the only caller — lives in `tk-convert`.
#[cfg(feature = "http")]
pub mod from_pretrained;
pub(crate) mod word_cache;

pub use crate::tokenizer::pattern::SysRegex;

// Recognize known GPT pre-tokenization regexes and route them to bitcannon's native (unrolled) FSM.
mod unrolled_regex;
pub use unrolled_regex::{DEEPSEEK_PATTERNS, Grammar, is_deepseek, recognize};

pub mod byte_level;
pub mod iter;
pub mod padding;
#[cfg(feature = "parallelism")]
pub mod parallelism;
pub(crate) mod perfect_hash;
pub mod truncation;

#[cfg(feature = "normalizers")]
pub(crate) mod unicode;
