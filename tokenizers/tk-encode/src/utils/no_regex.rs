//! Stub `SysRegex` for builds with **no** system-regex backend (`fancy-regex` and `onig` both off).
//!
//! The type stays present so `Split` / `Replace` still compile, but construction always fails: the
//! atomsplit-native pre-tokenizers (GPT-2, cl100k, deepseek, the class family, char-delimiter) need
//! no backend, while a `Split` with an *arbitrary* regex or the `Replace` normalizer error at load
//! time with a clear message. Enable `fancy-regex` (default) or `onig` to get a real backend.
use std::error::Error;

#[derive(Debug)]
pub struct SysRegex {
    // Never populated — `new` always errors — but keeps the type inhabited so structs that hold a
    // `SysRegex` field (e.g. `Replace`) remain constructible in principle.
    _priv: (),
}

impl SysRegex {
    pub fn new(_regex_str: &str) -> Result<Self, Box<dyn Error + Send + Sync + 'static>> {
        Err("no system-regex backend compiled: enable the `fancy-regex` (default) or `onig` feature \
             to use a `Split` pre-tokenizer with a custom regex, or the `Replace` normalizer"
            .into())
    }

    pub fn find_iter(&self, _inside: &str) -> std::iter::Empty<(usize, usize)> {
        // Unreachable: no `SysRegex` is ever constructed without a backend.
        std::iter::empty()
    }
}
