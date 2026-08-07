//! The class-run family: WhitespaceSplit / Punctuation / Digits / Whitespace / Bert, plus
//! CharDelimiterSplit. Not regex-shaped -- these cut where the atom class changes, so one shape
//! (`class_runs_into<DROP, ISOLATE, KEEP_A>`) covers all of them.
//!
//! ponytail: still the scalar/NEON run extractor moved over from atomsplit, not a bitstream
//! program. A class-run boundary is `c & !(c << 1)`, so porting it onto the `Blk` streams would
//! delete `simd_classes.rs` outright -- worth doing, but it is not what makes these correct today.

use crate::Span;
use crate::classify::{Atom, char_len, classify, in_mask, mask};
use crate::run_end;

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
        crate::simd_classes::class_runs_neon::<DROP, ISOLATE, KEEP_A>(text, tags, out)
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        crate::simd_classes::class_runs_wasm::<DROP, ISOLATE, KEEP_A>(text, tags, out)
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




/// `Split(char, Removed)` — the only pre-tokenizer that keys on a *literal char* rather than an atom
/// class, so it scans bytes directly (no classify pass). UTF-8 is self-synchronizing, so the
/// delimiter's byte pattern only matches on char boundaries.
pub struct CharDelimiterSplit(pub char);
impl CharDelimiterSplit {
    /// Split on the literal char (Removed); writes spans into `out` (len ≥ `text.len()`), returns count.
    #[inline]
    #[must_use]
    pub fn pre_tokenize(&self, text: &[u8], _tags: &mut [u8], out: &mut [Span]) -> usize {
        debug_assert!(out.len() >= text.len());
        let mut buf = [0u8; 4];
        let delim = self.0.encode_utf8(&mut buf).as_bytes();
        let (n, dl) = (text.len(), delim.len());
        let (mut start, mut i, mut w) = (0usize, 0usize, 0usize);
        while i + dl <= n {
            // memchr the first delimiter byte, then confirm the full pattern. memchr (already a
            // workspace dep) beats a scalar scan 1.4–23× here — the gap widening as the delimiter
            // gets rarer over large inputs, since its SIMD skips whole 16/32/64-byte strides.
            match memchr::memchr(delim[0], &text[i..n - dl + 1]) {
                Some(off) if text[i + off..i + off + dl] == *delim => {
                    let m = i + off;
                    if m > start {
                        out[w] = Span {
                            start: start as u32,
                            end: m as u32,
                        }; // gap before the delimiter (Removed)
                        w += 1;
                    }
                    i = m + dl;
                    start = i;
                }
                Some(off) => i += off + 1, // first byte matched mid-pattern; keep scanning
                None => break,
            }
        }
        if start < n {
            out[w] = Span {
                start: start as u32,
                end: n as u32,
            };
            w += 1;
        }
        w
    }
}
