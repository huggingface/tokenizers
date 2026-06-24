//! Steps [4]-[6] of the encode pipeline — the pre-tokenizer half.
//!
//! Steps [1]-[3] (pre-norm special extract -> normalizer chain -> post-norm
//! special extract) are owned by the other half and produce a *sequence* of
//! [`PostNormChunk`]s — borrowed views: no ownership, no copy. This module
//! consumes them:
//!
//!   [4] PRE-TOKENIZER       split each text chunk into `Split`s (pre-tokens)
//!   [5] MODEL               tokenize each split -> ids (+ offsets if tracked)
//!   [6] WRITE ADDED TOKENS  interleave the pre-resolved special/added tokens
//!
//! ## Naming
//!
//! Keeps the existing `pre_tokenizer` / `Split` vocabulary (not IREE's
//! "segmenter" / "segment"). The [`PreTokenizer`] trait and [`Split`] here are
//! the lean, range-based v2 forms intended to *replace* the heavyweight
//! `tokenizer::PreTokenizer` (which takes `&mut PreTokenizedString`) and
//! `pre_tokenizer::Split` (which owns a `NormalizedString`) once the v2 encode
//! path lands — same concepts, zero-copy shapes.
//!
//! ## Interface (matches IREE's pre-tokenizer contract)
//!
//! IREE's pre-tokenizer takes a raw `iree_string_view_t` and emits pre-token
//! byte ranges *relative to the input* — no metadata threaded in. We mirror it:
//!   - A `Text` chunk is a **raw `&str`**; the pre-tokenizer needs nothing else.
//!     (`&str` carries the UTF-8 invariant for free and yields `as_bytes()` at
//!     zero cost; `&[u8]` is equally valid — IREE's UTF-8 DFA splits bytes
//!     char-boundary-safely.)
//!   - normalized->original **offset mapping is a separate, opt-in side table**
//!     ([`OffsetRuns`], IREE's `offset_run_list`), populated by [1]-[3] and
//!     applied at the end — NOT bundled per chunk. The ids-only serving/TTFT
//!     path omits it entirely.
//!   - The input is a typed chunk *sequence* (not one slice) because [6] must
//!     interleave the special/added tokens resolved in [1]/[3] in original order.

use std::ops::Range;

use super::{Encoding, Model, Result};

/// One unit handed from [1]-[3] to [4]-[6]. Borrowed; its lifetime is the
/// normalizer's output buffer, which must outlive this half.
pub enum PostNormChunk<'a> {
    /// A special/added token already resolved to an id (and original range) in
    /// [1] or [3]. Emitted verbatim in [6]; never pre-tokenized or modeled.
    Added { id: u32, orig: Range<usize> },
    /// A run of normalized text to pre-tokenize [4] + model [5]. Just the bytes
    /// — no wrapper. (`&str`; use `.as_bytes()` for the byte-level path.)
    Text(&'a str),
}

/// Opt-in normalized->original offset mapping (RLE), the analogue of IREE's
/// `iree_tokenizer_offset_run_list_t`. Owned/populated by the [1]-[3] half;
/// passed to [`encode_post_norm`] ONLY when offset tracking is requested. The
/// pre-tokenizer never sees it.
pub struct OffsetRuns {
    // TODO: run-length-encoded (normalized_offset, original_offset) pairs.
}

impl OffsetRuns {
    /// Map a transform-buffer (normalized) byte range to the ORIGINAL input.
    pub fn to_original(&self, norm: Range<usize>) -> Range<usize> {
        let _ = norm;
        todo!("walk the RLE runs; identity-shift within a length-preserving run")
    }
}

/// One pre-token produced by [4]: a byte range RELATIVE to the chunk text it was
/// produced from (caller rebases to absolute, as in IREE).
///
/// Lean v2 form — just a range — vs the heavyweight `pre_tokenizer::Split` that
/// owns a `NormalizedString`.
pub struct Split {
    pub range: Range<usize>,
}

/// [4] PRE-TOKENIZER. Takes a raw text view, appends pre-token ranges.
///
/// Single-threaded. The authority-zone intra-sequence parallel layer is a
/// deferred phase-2 seam (it would partition `text` into overlapping zones here
/// and merge boundary matches); intentionally absent until the single-threaded
/// core lands.
///
/// This is the lean, range-based replacement for the existing
/// `tokenizer::PreTokenizer` (whose `pre_tokenize` takes `&mut
/// PreTokenizedString`); kept in its own module while the v1 path still exists.
pub trait PreTokenizer {
    /// Split `text` into pre-tokens, appending to `out`. Ranges are into `text`.
    fn pre_tokenize(&self, text: &str, out: &mut Vec<Split>) -> Result<()>;
}

/// Drives [4]-[6] over the chunk sequence from [1]-[3].
///
/// `offsets`: `Some` on the span/NER path (map token ranges back to original via
/// the side table); `None` on the ids-only serving/TTFT path (skip offset work).
///
/// Per chunk, in order:
///   - `Added { id, orig }` -> [6] emit the id (offset = `orig`) directly.
///   - `Text(t)`            -> [4] `pre_tokenizer.pre_tokenize(t, …)`, then per
///     split [5] `model.tokenize(&t[split.range])`; if `offsets`, map each token
///     range (split-relative -> chunk -> original via `OffsetRuns::to_original`).
pub fn encode_post_norm<P, M>(
    chunks: &[PostNormChunk<'_>],
    pre_tokenizer: &P,
    model: &M,
    offsets: Option<&OffsetRuns>,
) -> Result<Encoding>
where
    P: PreTokenizer,
    M: Model,
{
    let _ = (chunks, pre_tokenizer, model, offsets);
    todo!()
}
