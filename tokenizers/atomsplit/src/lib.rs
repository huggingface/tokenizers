//! `atomsplit` — SIMD Unicode classification + finite-state pre-tokenization.
//!
//! One SIMD pass ([`classify`]) maps every codepoint to a tiny "atom" alphabet; a family of no-push
//! FSMs ([`fsm`]) turn that atom stream into token spans (byte ranges) — the pre-tokenizer stage that
//! runs before a BPE/WordPiece model. Pre-tokenizers implemented: `WhitespaceSplit`, `Punctuation`,
//! `Digits`, `Whitespace`, `Bert`, `Cl100k`, `DeepSeek`, `ByteLevel`, `CharDelimiterSplit` (o200k and
//! Mistral's tekken are exposed as the [`fsm::fsm_o200k`] / [`fsm::fsm_tekken`] functions rather than
//! recipe structs).
//!
//! For pre-tokenizers that split on single characters (such as the Metaspace `▁` delimiter),
//! we skip the atom classification pass entirely and search for the raw  bytes instead.
//! See [`literal`].
//!
//! Design: every fsm is *no-push* — it writes spans into a caller-preallocated `&mut [fsm::Span]`
//! (length ≥ `text.len()`) and returns the token count; there is no `Vec`/allocation on the hot path.
//!
//! # Preconditions
//! Inputs are `&[u8]` (not `&str`) for zero-copy, but **must be well-formed UTF-8**: a buffer that ends
//! mid-codepoint is a precondition violation and may panic. `tags`/`out` scratch buffers must be
//! `≥ text.len()` (asserted in [`classify`]; documented per-fsm).
mod atom_tables;
pub mod classify;
pub mod fsm;
pub mod literal;
pub mod regexes;
#[cfg(target_arch = "x86_64")]
mod simd_avx_classify;
#[cfg(target_arch = "aarch64")]
mod simd_classify;
mod simd_fsm;
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
mod simd_literal;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod simd_wasm_classify;
pub mod tables;
