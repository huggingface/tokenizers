//! Stub `SysRegex` for builds with **no** system-regex backend (`fancy-regex` off — the default).
//!
//! The type stays present so `Split` / `Replace` still compile, but construction always fails. Only a
//! *regex* pattern ever asks for it: the atomsplit-native pre-tokenizers (GPT-2, cl100k, deepseek, the
//! class family, char-delimiter) need no backend, and a plain string pattern is searched for directly
//! (`atomsplit::literal`). A regex atomsplit does not cover errors at load time with a clear message.
//! Enable `fancy-regex` to get a real backend.
use std::error::Error;

#[derive(Debug)]
pub struct SysRegex {
    // Never populated — `new` always errors — but keeps the type inhabited so structs that hold a
    // `SysRegex` field (e.g. `Replace`) remain constructible in principle.
    _priv: (),
}

impl SysRegex {
    pub fn new(_regex_str: &str) -> Result<Self, Box<dyn Error + Send + Sync + 'static>> {
        Err(
            "no system-regex backend compiled: enable the `fancy-regex` feature to use a regex \
             pattern in a `Split` pre-tokenizer or a `Replace` normalizer (a plain string pattern \
             needs no backend)"
                .into(),
        )
    }

    pub fn find_iter(&self, _inside: &str) -> std::iter::Empty<(usize, usize)> {
        // Unreachable: no `SysRegex` is ever constructed without a backend.
        std::iter::empty()
    }
}
