//! Data-driven Unicode normalization — NFC / NFD / NFKC / NFKD as one lean design.
//!
//! Real tokenizer input rarely needs normalization work, so the architecture optimizes the skip, not
//! the transform: **layer 0** skips whole windows whose *lead bytes* provably start no char the form
//! cares about (per-form 64-byte lead masks — ASCII, continuations, and clean scripts like Han never
//! leave this loop); **layer 1** rides word-structured suspect scripts with fused byte-class kernels
//! (2-byte + ASCII in one loop, the union bit probed in-register); **layer 2** touches only
//! confirmed-suspect chars through ONE flat per-codepoint tag byte (`TAG[cp]`, direct index):
//!
//! ```text
//! 0x00        inert — stable under every form
//! 0x01..=0x3B identity combining mark; the value is its ccc RANK (order-preserving, so canonical
//!             order is a byte compare — no ccc anywhere at runtime)
//! 0x3C        compat-changing starter (NFKD/NFKC break; NFD/NFC stable), e.g. `ﬁ`
//! 0x3D        compat-changing mark (rare; conservative under NFD)
//! 0x40 | r    NFC quick-check Maybe (composes as a second) — rank rides in the low bits
//! 0x7D        canonically decomposes AND composition-relevant: compose must recompose
//! 0x7E        canonically decomposes, composition-stable (é, ά, Hangul): compose skips it
//! ```
//!
//! Decompose *writes* are a table index → one blob copy (`[first_rank, last_rank, mark_run_off]`
//! headers keep cross-char canonical order a byte compare); Hangul is arithmetic. Compose is a pair
//! lookup (`COMPOSE`) inside a recompose window opened only where a relevant char was actually found.
//! Already-normalized input returns `Cow::Borrowed` untouched.
//!
//! Two paths, like `atomsplit`: the scalar core in `norm.rs` is the complete, portable
//! implementation; `simd_norm.rs` (aarch64) accelerates the three skip kernels as pure prefix
//! processors — they stop where SIMD chunking ends and the scalar loops finish, so the paths are
//! byte-exact by construction. Zero runtime dependencies; tables are committed, generated from
//! `unicode-normalization` (also the test oracle):
//! `cargo test -p atomnorm --release generate -- --ignored`. Inputs must be valid UTF-8.
//!
//! The same architecture also powers the **scan normalizers** ([`lowercase`], [`strip_accents`],
//! [`nmt`], [`bert`] — see `scan.rs`): one skip-scan over per-rule property sets baked from the
//! exact predicates the legacy tk-encode normalizers use, with an in-register ASCII transform lane.

use std::borrow::Cow;

mod norm;
mod scan;
#[cfg(target_arch = "aarch64")]
mod simd_norm;
mod tables;

/// NFD-normalize. Byte-exact with `str::nfd()`; borrows when already normalized.
pub fn nfd(input: &str) -> Cow<'_, str> {
    norm::decompose::<false, true>(input)
}
/// NFKD-normalize. Byte-exact with `str::nfkd()`; borrows when already normalized.
pub fn nfkd(input: &str) -> Cow<'_, str> {
    norm::decompose::<true, true>(input)
}
/// NFC-normalize. Byte-exact with `str::nfc()`; borrows when already normalized.
pub fn nfc(input: &str) -> Cow<'_, str> {
    norm::compose::<false, true>(input)
}
/// NFKC-normalize. Byte-exact with `str::nfkc()`; borrows when already normalized.
pub fn nfkc(input: &str) -> Cow<'_, str> {
    norm::compose::<true, true>(input)
}
/// NFD-decompose a single char, invoking `f` with each output char in canonical order (a stable char
/// calls `f(c)` once). For char-at-a-time consumers like BERT `strip_accents`.
pub fn nfd_char(c: char, f: impl FnMut(char)) {
    norm::nfd_char(c, f)
}

// ── scan normalizers (see `scan.rs`): same skip architecture over per-rule property sets ──────────

/// Unicode-lowercase. Byte-exact with `chars().flat_map(char::to_lowercase)`; borrows when already
/// lowercase. ASCII rides an in-register `|0x20` lane.
pub fn lowercase(input: &str) -> Cow<'_, str> {
    scan::lowercase::<true>(input)
}
/// Remove combining marks (general category M — the tk `StripAccents` predicate; no decomposition).
/// Borrows when mark-free.
pub fn strip_accents(input: &str) -> Cow<'_, str> {
    scan::strip_accents::<true>(input)
}
/// The NMT normalizer: drop its control set, fold its whitespace set to `' '`. Borrows when clean.
pub fn nmt(input: &str) -> Cow<'_, str> {
    scan::nmt::<true>(input)
}
/// The fused BERT normalizer — clean_text + handle_chinese_chars + strip_accents (NFD, drop Mn) +
/// lowercase in ONE pass. Callers resolve `strip_accents = strip_accents.unwrap_or(lowercase)`.
/// Borrows when no enabled rule touches the input.
pub fn bert(
    input: &str,
    clean_text: bool,
    handle_chinese_chars: bool,
    strip_accents: bool,
    lowercase: bool,
) -> Cow<'_, str> {
    scan::bert::<true>(
        input,
        clean_text,
        handle_chinese_chars,
        strip_accents,
        lowercase,
    )
}

/// Scalar-only entry points (the SIMD prefixes disabled) — for benchmarking and differential testing
/// of the two paths; not part of the supported API surface.
#[doc(hidden)]
pub mod scalar {
    use std::borrow::Cow;
    pub fn nfd(input: &str) -> Cow<'_, str> {
        crate::norm::decompose::<false, false>(input)
    }
    pub fn nfkd(input: &str) -> Cow<'_, str> {
        crate::norm::decompose::<true, false>(input)
    }
    pub fn nfc(input: &str) -> Cow<'_, str> {
        crate::norm::compose::<false, false>(input)
    }
    pub fn nfkc(input: &str) -> Cow<'_, str> {
        crate::norm::compose::<true, false>(input)
    }
    pub fn lowercase(input: &str) -> Cow<'_, str> {
        crate::scan::lowercase::<false>(input)
    }
    pub fn strip_accents(input: &str) -> Cow<'_, str> {
        crate::scan::strip_accents::<false>(input)
    }
    pub fn nmt(input: &str) -> Cow<'_, str> {
        crate::scan::nmt::<false>(input)
    }
    pub fn bert(input: &str, ct: bool, cc: bool, sa: bool, lc: bool) -> Cow<'_, str> {
        crate::scan::bert::<false>(input, ct, cc, sa, lc)
    }
}
