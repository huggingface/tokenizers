//! FSM layer: turn the `Atom` tag stream (from [`crate::classify`]) into token spans. Every fsm is
//! NO-PUSH — it writes spans into a caller-preallocated `&mut [Span]` (len ≥ `text.len()`) and returns
//! the token count; no `Vec`, no realloc. Inputs must be well-formed UTF-8 (see the crate-level docs).
//!
//! The class family (WhitespaceSplit / Punctuation / Digits / Whitespace / Bert) goes through
//! [`class_runs_into`]: on aarch64/wasm the SIMD movemask boundary-extractor + homogeneous-chunk
//! early-out (in `simd_fsm`), elsewhere the scalar run-end core ([`emit_class_spans`]). The
//! regex-shaped ones ([`fsm_cl100k`] / [`fsm_o200k`] / [`fsm_tekken`] / [`fsm_deepseek`] /
//! [`fsm_byte_level`]) are scalar jump-tables (only the class family's [`class_runs_into`] has a SIMD
//! path).

pub(crate) use crate::classify::{Atom, char_len, classify, in_mask, mask};
// Atom-tag aliases, shared with the per-tokenizer FSM submodules (`fsm/*.rs`) via `use super::*`.
pub(crate) const LET: u8 = Atom::Letter as u8;
pub(crate) const NW: u8 = Atom::NumWord as u8;
pub(crate) const NO: u8 = Atom::NumOther as u8;
pub(crate) const NLN: u8 = Atom::Newline as u8;
pub(crate) const SPC: u8 = Atom::Space as u8;
pub(crate) const WSO: u8 = Atom::WsOther as u8;
pub(crate) const MRK: u8 = Atom::Mark as u8;
pub(crate) const CON: u8 = Atom::Connector as u8;
pub(crate) const PUN: u8 = Atom::Punct as u8;
pub(crate) const APO: u8 = Atom::Apostrophe as u8;
pub(crate) const SYM: u8 = Atom::SymOther as u8;
pub(crate) const NMO: u8 = Atom::NumericOther as u8;
pub(crate) const CTL: u8 = Atom::Control as u8;
pub(crate) const CONT: u8 = Atom::Cont as u8;
pub(crate) const ASM: u8 = Atom::AlphaSymMark as u8;
pub(crate) const ZWJ: u8 = Atom::Zwj as u8; // 0x26 — ZWJ/ZWNJ, tagged in classify so FSMs skip the text peek

/// Advance over a maximal `m`-membership run (m is a mask); returns the byte index past it.
/// `inline(always)`: it's called once per token (~200K/MB on English) — a real call here doubles fsm cost.
///
/// logos-style "fast loop": process 16 tags per iteration with ONE bounds check per chunk and unchecked
/// reads (the loop condition proves `i + 16 <= end`), so short runs pay ~1 bounds check instead of one
/// per byte and long runs stay a tight unrolled scan. Byte-identical to the plain `while in_mask` scan.
#[inline(always)]
pub(crate) fn run_end(tags: &[u8], mut i: usize, end: usize, mut m: u16) -> usize {
    m |= Atom::Cont.bit();
    debug_assert!(end <= tags.len());
    // SAFETY: `i + 16 <= end <= tags.len()` in the unrolled body, so every `get_unchecked(i + k)`
    // (k < 16) is in bounds. The tail is the plain checked scan.
    while i + 16 <= end {
        for k in 0..16 {
            if !in_mask(unsafe { *tags.get_unchecked(i + k) }, m) {
                return i + k;
            }
        }
        i += 16;
    }
    while i < end && in_mask(tags[i], m) {
        i += 1;
    }
    i
}

/// End of a whitespace token starting at `i`, for the tail shared byte-for-byte by cl100k and o200k
/// (`\s*[\r\n]+ | \s+(?!\S) | \s+`): through the last `\r\n` if any (rule 5), else the whole run at
/// EOF (rule 7), else give the final ws char back to the next token (rule 6). `#[inline]` — hot,
/// called once per whitespace token. (deepseek's tail differs — it also stops before a digit/CJK — and
/// byte_level has no `[\r\n]` rule; both keep their own.)
#[inline]
pub(crate) fn ws_tail(text: &[u8], tags: &[u8], i: usize, end: usize) -> usize {
    let re = run_end(tags, i, end, mask::WS);
    if let Some(r) = text[i..re].iter().rposition(|&x| x == 0x0A || x == 0x0D) {
        i + r + 1
    } else if re == end {
        re
    } else {
        let mut last = re - 1;
        while last > i && text[last] & 0xC0 == 0x80 {
            last -= 1;
        }
        if last > i { last } else { re }
    }
}

/// Case-insensitive contraction match at `i` (`'s 't 're 've 'm 'll 'd`), shared by cl100k and o200k:
/// byte length (2 or 3) or 0 if none. Self-guarding — `text[i]` need not be an apostrophe. `#[inline]`.
/// (byte_level's contraction is case-SENSITIVE, so it keeps its own.)
#[inline]
pub(crate) fn contraction(text: &[u8], i: usize) -> usize {
    let end = text.len();
    if i >= end || text[i] != 0x27 || i + 1 >= end || text[i + 1] >= 0x80 {
        return 0;
    }
    let lc = text[i + 1] | 0x20;
    match lc {
        b's' | b't' | b'm' | b'd' => 2,
        b'r' | b'v' | b'l' if i + 2 < end && text[i + 2] < 0x80 => {
            let l2 = text[i + 2] | 0x20;
            usize::from((matches!(lc, b'r' | b'v') && l2 == b'e') || (lc == b'l' && l2 == b'l')) * 3
        }
        _ => 0,
    }
}

/// A token span: byte offsets `[start, end)` into the input. `#[repr(C)]` so the FSM output buffer has a
/// stable `[start, end]` layout — the pipeline reuses it with zero conversion, and it can be reinterpreted
/// as bytes / handed across the crate boundary.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash, PartialOrd, Ord)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Span {
    #[inline]
    pub const fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }

    /// `[start, end)` as a `usize` range — for slicing the input text.
    #[inline]
    pub fn range(self) -> core::ops::Range<usize> {
        self.start as usize..self.end as usize
    }
}

/// Compare against a bare `(start, end)` tuple — convenience for tests/interop.
impl PartialEq<(u32, u32)> for Span {
    #[inline]
    fn eq(&self, o: &(u32, u32)) -> bool {
        self.start == o.0 && self.end == o.1
    }
}

/// No-`push` class-family pre-tokenizer core: writes spans into the preallocated `out` slice and returns
/// the count. ONE shape covers the whole class family via `<DROP, ISOLATE, KEEP_A>`:
///   WhitespaceSplit `<{WS},0,0>` · Punctuation `<0,{PUNCT},0>` · Digits `<0,0,{NUMERIC}>` ·
///   Whitespace `<{WS},0,{WORD}>` · Bert `<{WS},{PUNCT},0>`.
/// Class of a char: `DROP`→dropped, `ISOLATE`→own token, `KEEP_A`→run "A", else→run "B" (A/B cut apart).
/// TODO: find a better explanation
#[inline]
#[must_use]
pub fn class_runs_into<const DROP: u16, const ISOLATE: u16, const KEEP_A: u16>(
    text: &[u8],
    tags: &[u8],
    out: &mut [Span],
) -> usize {
    debug_assert!(out.len() >= text.len() && tags.len() >= text.len());
    #[cfg(target_arch = "aarch64")]
    {
        crate::simd_fsm::class_runs_neon::<DROP, ISOLATE, KEEP_A>(text, tags, out)
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        crate::simd_fsm::class_runs_wasm::<DROP, ISOLATE, KEEP_A>(text, tags, out)
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    )))]
    {
        emit_class_spans::<DROP, ISOLATE, KEEP_A>(text, tags, out, 0, 0, 0, None)
    }
}

/// This is the most important function as it's the core of the scalar finite state machine.
/// It allows to emit class spans with different behaviours for tags we want to drop, tags we want
/// to isolate and tags we want to keep. Any other tags are assumed to be keept.
///
/// This function is used as a fallback to the SIMD fast fsm. It is used for most pre tokenizers
/// but the unrolled regex, which have more complex variations that cannot be expressed with drop,
/// isolate, keep. These 3 generic parameters are u16 bitmap masks over the 16 classes we have and
/// define the behaviour. They are usally one of the [`crate::classify::mask`]. They allow dropping
/// words, isolating whitespace and keeping new line for example.
#[must_use]
#[inline]
pub fn emit_class_spans<const DROP: u16, const ISOLATE: u16, const KEEP_A: u16>(
    text: &[u8],
    tags: &[u8],
    out: &mut [Span],
    mut write_index: usize,     // in the out slice
    mut text_pointer: usize,    // in the text slice
    segment_start: usize,       // previous segment_start
    segment_class: Option<u16>, // previous segment's class
) -> usize {
    debug_assert!(out.len() >= text.len() && tags.len() >= text.len());
    let n = text.len();
    // Tie `tags.len() == text.len() == n` so the optimizer drops the interior `tags[i]` / `text[i]`
    // bounds checks (callers guarantee len ≥ n; same trick as `cl100k`). Per-byte scanning already
    // avoids checks via `run_end`'s unrolled `get_unchecked`; this covers the per-token accesses.
    let tags = &tags[..n];
    let text = &text[..n];
    let other = !(DROP | ISOLATE | KEEP_A); // None of the above correspond to a continuation
    if let Some(segment_class) = segment_class {
        // this will usually be at the tail of a SIMD call.
        text_pointer = run_end(tags, text_pointer, n, segment_class); // skip the whole drop run at once
        if segment_class != DROP {
            out[write_index] = Span {
                start: segment_start as u32,
                end: text_pointer as u32,
            };
            if text_pointer == n {
                return write_index + 1;
            }
            write_index += 1;
        }
    }
    while text_pointer < n {
        let t = tags[text_pointer];
        if t == Atom::Cont as u8 {
            text_pointer += 1;
            continue;
        }
        // classify the first char.
        if in_mask(t, DROP) {
            text_pointer = run_end(tags, text_pointer, n, DROP); // skip the whole drop run at once
        } else if in_mask(t, ISOLATE) {
            let s = text_pointer;
            text_pointer += char_len(text[text_pointer]);
            out[write_index] = Span {
                start: s as u32,
                end: text_pointer as u32,
            }; // isolate: one char = one token
            write_index += 1;
        } else {
            let s = text_pointer;
            text_pointer = if in_mask(t, KEEP_A) {
                run_end(tags, text_pointer, n, KEEP_A)
            } else {
                run_end(tags, text_pointer, n, other)
            };
            out[write_index] = Span {
                start: s as u32,
                end: text_pointer as u32,
            };
            write_index += 1;
        }
    }
    write_index
}

// ── per-tokenizer unrolled FSMs (one file each; shared helpers above via `use super::*`) ──
mod byte_level;
mod cl100k;
mod deepseek;
mod o200k;
pub use byte_level::fsm_byte_level;
pub use cl100k::{fsm_cl100k, fsm_cl100k_cap};
pub use deepseek::fsm_deepseek;
pub use o200k::{fsm_o200k, fsm_tekken};

// ── Composition recipes ────────────────────────────────────────────────────────────────────────
// Each pre-tokenizer = (classify → fsm shape + params). `tags` and `out` are caller-owned
// scratch, reused across calls — NO per-call alloc, NO push. The class family writes spans into the
// preallocated `out: &mut [Span]` (len ≥ text.len()) via `class_runs_into` and returns the token count.
// In `tk-encode` these delegate from the `pipeline::PreTokenizer` impls (offset conversion happens there).

/// `WhitespaceSplit` — split on Unicode whitespace and drop it; keeps maximal non-whitespace runs.
pub struct WhitespaceSplit;
impl WhitespaceSplit {
    #[inline]
    #[must_use]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut [Span]) -> usize {
        classify(text, tags);
        class_runs_into::<{ mask::WS }, 0, 0>(text, tags, out)
    }
}

/// `Punctuation` — isolate each punctuation char as its own token; non-punct grouped into runs.
pub struct Punctuation;
impl Punctuation {
    #[inline]
    #[must_use]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut [Span]) -> usize {
        classify(text, tags);
        class_runs_into::<0, { mask::PUNCT }, 0>(text, tags, out)
    }
}

/// `Digits` — cut numeric runs apart from non-numeric runs (contiguous), keeping both.
pub struct Digits;
impl Digits {
    #[inline]
    #[must_use]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut [Span]) -> usize {
        classify(text, tags);
        class_runs_into::<0, 0, { mask::NUMERIC }>(text, tags, out)
    }
}

/// `Whitespace` — the `\w+|[^\w\s]+` pre-tokenizer: drop whitespace, cut word runs from symbol runs.
pub struct Whitespace;
impl Whitespace {
    #[inline]
    #[must_use]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut [Span]) -> usize {
        classify(text, tags);
        class_runs_into::<{ mask::WS }, 0, { mask::WORD }>(text, tags, out)
    }
}

/// `Bert` — the BERT basic pre-tokenizer: drop whitespace, isolate punctuation, keep the rest as runs.
pub struct Bert;
impl Bert {
    #[inline]
    #[must_use]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut [Span]) -> usize {
        classify(text, tags);
        class_runs_into::<{ mask::WS }, { mask::PUNCT }, 0>(text, tags, out)
    }
}

/// `Cl100k` — the tiktoken cl100k_base / Llama-3 pre-tokenizer (7-rule regex). (o200k is a distinct,
/// case-aware FSM — [`fsm_o200k`].)
pub struct Cl100k;
impl Cl100k {
    /// Uses the scalar run-end core: cl100k's letter/ws runs are short on Latin/code (the common case),
    /// where a SIMD run-end's setup would lose; only long CJK runs would benefit.
    #[inline]
    #[must_use]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut [Span]) -> usize {
        classify(text, tags);
        fsm_cl100k(text, tags, out)
    }
}

/// `DeepSeek` — the DeepSeek-V3/R1 pre-tokenizer (digits{1,3} → CJK-range → big regex, composed).
pub struct DeepSeek;
impl DeepSeek {
    #[inline]
    #[must_use]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut [Span]) -> usize {
        classify(text, tags);
        fsm_deepseek(text, tags, out)
    }
}

/// `ByteLevel` — the GPT-2 / Llama / Qwen byte-level pre-tokenizer regex (before byte-mapping).
pub struct ByteLevel;
impl ByteLevel {
    #[inline]
    #[must_use]
    pub fn pre_tokenize(&self, text: &[u8], tags: &mut [u8], out: &mut [Span]) -> usize {
        classify(text, tags);
        fsm_byte_level(text, tags, out)
    }
}
