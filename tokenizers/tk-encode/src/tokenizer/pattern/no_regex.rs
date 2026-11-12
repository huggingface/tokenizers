//! Stub `SysRegex` for builds with **no** system-regex backend (`fancy-regex` off — the default).
//!
//! The type stays present so `Split` / `Replace` still compile, but construction always fails. Only a
//! *regex* pattern ever asks for it: the bitcannon-native pre-tokenizers (GPT-2, cl100k, deepseek, the
//! class family, char-delimiter) need no backend, and a plain string pattern is searched for directly
//! (`bitcannon::literal`). A regex bitcannon does not cover errors at load time with a clear message.
//! Enable `fancy-regex` to get a real backend.
use super::matcher::Pattern;
use crate::{Offsets, Result};

#[derive(Debug)]
pub struct SysRegex {
    // Never populated — `new` always errors — but keeps the type inhabited so structs that hold a
    // `SysRegex` field (e.g. `Replace`) remain constructible in principle.
    _priv: (),
}

impl SysRegex {
    pub fn new(_regex_str: &str) -> Result<Self> {
        Err(
            "no system-regex backend compiled: enable the `fancy-regex` feature to use a regex \
             pattern in a `Split` pre-tokenizer or a `Replace` normalizer (a plain string pattern \
             needs no backend)"
                .into(),
        )
    }
}

impl Pattern for SysRegex {
    fn scan<'t>(&'t self, _inside: &'t str) -> impl Iterator<Item = Result<Offsets>> {
        // Unreachable: no `SysRegex` is ever constructed without a backend.
        std::iter::empty()
    }
}
