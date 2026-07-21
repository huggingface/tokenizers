//! Parallel encode runtime for [`PipelineTokenizer`].
//!
//! [`PipelineTokenizer::encode`] returns an owned [`EncodeHandle`] immediately;
//! workers encode in the background. Results surface in input order through the
//! handle's three faces: the blocking [`Iterator`] (which also *assists* —
//! the caller drains pending work instead of idling), [`EncodeHandle::wait_for_completion`],
//! and, behind the `async` feature, `impl futures_core::Stream`.
//!
//! **Job model.** Work is split into `Item`s behind one atomic claim cursor in
//! `JobCore`; rayon pool tasks and the consuming thread both drain it, so
//! there is no scheduler — just a cursor and caller-assist. Per-worker
//! `EncodeState` scratch persists (thread-local, reset-not-realloc) for cache
//! warmth. Panics are caught per unit and delivered as that input's `Err`.
//!
//! **Rung ladder.** Every encode is decomposed the same way, outermost stage
//! first, cheapest split first:
//!
//!  * **Rung 0 (always):** special tokens are the outermost pipeline stage, so
//!    cutting at each match is unconditionally safe. The builder peels them
//!    (matched specials resolve immediately, no worker) and hands each
//!    special-free text segment down to the sub-rung. This costs one matcher
//!    scan (~0.1% of encode) and is what makes the byte-level boundary able to
//!    cut at number transitions — a strided segment can never contain a special.
//!  * **Sub-rung (`ParallelPlan`, chosen per config):** applied to each segment.
//!    `SplitRaw` strides it into ~`PARALLEL_MIN_TOTAL_BYTES` windows workers
//!    boundary-snap locally via a `StrideBoundary` (a pure predicate over the
//!    `atomsplit` tag stream the config's grammar provably cannot cross);
//!    `SplitNormalized` serial-normalizes the segment then strides the buffer;
//!    `SplitAtModel` serial-normalizes + pre-tokenizes and parallelizes the model
//!    over span-groups; `WholeInput` leaves the segment to one worker (batch-level
//!    parallelism only). The two serial-prefix rungs run only when the batch alone
//!    can't feed the pool. Never wrong, only less parallel.
//!
//! **Fork safety** lives in `pool`: a `pthread_atfork` child abandons the
//! stale pool without touching it and lazily rebuilds.

use std::cell::RefCell;
use std::convert::TryInto;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex, PoisonError};
use std::{borrow::Cow, convert::TryFrom};

use atomsplit::classify::{classify, in_mask, mask, Atom};

use crate::models::bpe::{BpeScratch, PipelineBPE};
use crate::models::unigram::{Unigram, UnigramScratch};
use crate::models::wordlevel::WordLevel;
use crate::models::wordpiece::{PipelineWordPiece, WordPieceScratch};
use crate::utils::byte_level::GPT2_REGEX_STR;
use crate::vocab::bucket_added_vocabulary::{
    AddedToken as BucketAddedToken, AddedVocabulary as BucketAddedVocabulary,
};
use crate::{
    normalizers::NormalizerWrapper,
    pre_tokenizers::{
        bert::BertPreTokenizer,
        delimiter::CharDelimiterSplit,
        digits::Digits,
        fixed_length::FixedLength,
        punctuation::Punctuation,
        sequence::PipelineSequence,
        split::{Split as SplitPretok, SplitPattern},
        unicode_scripts::UnicodeScripts,
        whitespace::{Whitespace, WhitespaceSplit},
    },
    ModelWrapper, PostProcessorWrapper, PreTokenizerWrapper, Token, Tokenizer,
};

use super::{pool, Result, SplitDelimiterBehavior};

pub use atomsplit::fsm::Span;

/// We use a thread local scratch for the tags (per byte class) and for the split spans.
pub(crate) fn classify_into_spans(
    bytes: &[u8],
    fsm: impl FnOnce(&[u8], &[u8], &mut [Span]) -> usize,
    out: &mut Vec<Span>,
) {
    thread_local! {
        static SCRATCH: RefCell<(Vec<u8>, Vec<Span>)> = const { RefCell::new((Vec::new(), Vec::new())) };
    }
    let n = bytes.len();
    SCRATCH.with(|cell| {
        let (tags, spans) = &mut *cell.borrow_mut();
        if tags.len() < n {
            tags.resize(n, 0); // grow-only: after the largest segment, no realloc / no re-zeroing
        }
        if spans.len() < n + 1 {
            spans.resize(n + 1, Span::default());
        }
        classify(bytes, &mut tags[..n]);
        let k = fsm(bytes, &tags[..n], &mut spans[..n + 1]);
        out.extend_from_slice(&spans[..k]); // same type now — plain memcpy, no per-token conversion
    });
}

pub trait Normalizer {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>>;
}

/// Range-based pre-tokenization: yields spans into the input rather than owned
/// substrings, so the pipeline can pre-tokenize without allocating.
pub trait PreTokenizer {
    /// Span `text` into pre-tokens, appending to `out`. Ranges are into `text`.
    fn pre_tokenize(&self, text: &str, out: &mut Vec<Span>) -> Result<()>;
}

/// The pre-tokenizers a [`PipelineTokenizer`] can run.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum PipelinePreTokenizer {
    Bert(BertPreTokenizer),
    Delimiter(CharDelimiterSplit),
    Digits(Digits),
    FixedLength(FixedLength),
    Punctuation(Punctuation),
    Sequence(PipelineSequence),
    Split(SplitPretok),
    UnicodeScripts(UnicodeScripts),
    Whitespace(Whitespace),
    WhitespaceSplit(WhitespaceSplit),
    None,
}

impl PreTokenizer for PipelinePreTokenizer {
    fn pre_tokenize(&self, text: &str, out: &mut Vec<Span>) -> Result<()> {
        match self {
            Self::None => {
                out.push(Span {
                    start: 0,
                    end: text.len() as u32,
                });
                Ok(())
            }
            Self::Bert(pretok) => pretok.pre_tokenize(text, out),
            Self::Delimiter(pretok) => pretok.pre_tokenize(text, out),
            Self::Digits(pretok) => pretok.pre_tokenize(text, out),
            Self::FixedLength(pretok) => pretok.pre_tokenize(text, out),
            Self::Punctuation(pretok) => pretok.pre_tokenize(text, out),
            Self::Sequence(pretok) => pretok.pre_tokenize(text, out),
            Self::Split(pretok) => pretok.pre_tokenize(text, out),
            Self::UnicodeScripts(pretok) => pretok.pre_tokenize(text, out),
            Self::Whitespace(pretok) => pretok.pre_tokenize(text, out),
            Self::WhitespaceSplit(pretok) => pretok.pre_tokenize(text, out),
        }
    }
}

impl PipelinePreTokenizer {
    /// The `StrideBoundary` this pre-tokenizer guarantees, if any: a cut
    /// finder its splitting rules provably cannot tokenize across. This is the
    /// extension point for raw-text parallel chunking — a pre-tokenizer that
    /// can prove a boundary for its grammar adds an arm here and large inputs
    /// split under it automatically; anything without one falls back to the
    /// model-stage split (still correct, just a serial pre-tokenize).
    pub(crate) fn stride_boundary(&self) -> Option<StrideBoundary> {
        match self {
            // Whitespace-delimiting: whitespace is a dropped delimiter, so any
            // whitespace character is a safe cut (see `boundary_at_whitespace`).
            Self::Whitespace(_) | Self::WhitespaceSplit(_) | Self::Bert(_) => {
                Some(boundary_at_whitespace)
            }
            // atomsplit-FSM byte-level families: cut at a space after non-ws, or
            // a newline after a word character (see `boundary_fsm`).
            Self::Split(s) if s.has_pipeline_fsm() => Some(boundary_fsm),
            Self::Sequence(seq) if seq.is_deepseek() => Some(boundary_fsm),
            Self::Sequence(seq) => match seq.members() {
                [only] => only.stride_boundary(),
                // The llama/GPT-4 shape: [Split(FSM), ByteLevel(no regex) -> None].
                [split, Self::None] => split.stride_boundary(),
                _ => None,
            },
            _ => None,
        }
    }
}

impl TryFrom<PreTokenizerWrapper> for PipelinePreTokenizer {
    type Error = crate::Error;

    fn try_from(value: PreTokenizerWrapper) -> Result<Self> {
        match value {
            PreTokenizerWrapper::BertPreTokenizer(p) => Ok(PipelinePreTokenizer::Bert(p)),
            PreTokenizerWrapper::Delimiter(p) => Ok(PipelinePreTokenizer::Delimiter(p)),
            PreTokenizerWrapper::Digits(p) => Ok(PipelinePreTokenizer::Digits(p)),
            PreTokenizerWrapper::FixedLength(p) => Ok(PipelinePreTokenizer::FixedLength(p)),
            PreTokenizerWrapper::Punctuation(p) => Ok(PipelinePreTokenizer::Punctuation(p)),
            PreTokenizerWrapper::Split(p) => {
                Ok(PipelinePreTokenizer::Split(p.canonicalized_for_pipeline()?))
            }
            PreTokenizerWrapper::UnicodeScripts(p) => Ok(PipelinePreTokenizer::UnicodeScripts(p)),
            PreTokenizerWrapper::Whitespace(p) => Ok(PipelinePreTokenizer::Whitespace(p)),
            PreTokenizerWrapper::WhitespaceSplit(p) => Ok(PipelinePreTokenizer::WhitespaceSplit(p)),
            PreTokenizerWrapper::ByteLevel(byte_level) => {
                if byte_level.add_prefix_space {
                    return Err(
                        "ByteLevel add_prefix_space=true is not supported by the pipeline yet"
                            .into(),
                    );
                }
                if byte_level.use_regex {
                    Ok(PipelinePreTokenizer::Split(SplitPretok::new(
                        SplitPattern::Regex(GPT2_REGEX_STR.to_owned()),
                        SplitDelimiterBehavior::Isolated,
                        false,
                    )?))
                } else {
                    Ok(PipelinePreTokenizer::None)
                }
            }
            PreTokenizerWrapper::Sequence(p) => Ok(PipelinePreTokenizer::Sequence(p.try_into()?)),
            other => {
                Err(format!("PipelineTokenizer does not support PreTokenizer: {other:?}").into())
            }
        }
    }
}

/// An output token. Carries only the vocabulary `id` — offsets and the token
/// string are dropped, which is all an encode-only caller needs.
#[derive(Debug, Clone, Copy)]
pub struct PipelineToken {
    pub id: u32,
}

impl From<Token> for PipelineToken {
    fn from(value: Token) -> Self {
        Self { id: value.id }
    }
}

/// Finds special/added tokens in a text segment so the pipeline can carve them
/// out before running the model.
pub trait PipelinePatternMatcher {
    /// Return the first special token in `input` as `Some(((start, end), id))`, where
    /// `start..end` is its byte range. `normalized` selects whether to match the
    /// tokens declared on normalized or on raw text.
    /// Returns `None` if there is no special tokens in input.
    fn extract_next(
        &self,
        full_input: &[u8],
        search_offset: usize,
        normalized: bool,
    ) -> Option<((usize, usize), u32)>;
}

/// A piece of the input produced by [`SpecialSegmentIterator`].
pub enum Segment<'a> {
    /// Ordinary text still to be (optonally normalized), pre-tokenized and run through the model.
    Text(&'a str),
    /// A matched special token, identified by its vocabulary id.
    SpecialToken(u32),
}

/// Splits `input` into [`Segment`]s, in order: runs of ordinary text
/// ([`Segment::Text`]) interleaved with the special tokens
/// ([`Segment::SpecialToken`]) matched by the [`PipelinePatternMatcher`].
///
/// ```ignore
/// for segment in SpecialSegmentIterator::new(input, pattern_matcher, false) {
///     match segment {
///         Segment::SpecialToken(id) => { /* emit the special token */ }
///         Segment::Text(chunk) => { /* tokenize this chunk */ }
///     }
/// }
/// ```
pub struct SpecialSegmentIterator<'a, 'b, PatternMatcher: PipelinePatternMatcher> {
    /// The chunk of text from which we want to extract special tokens
    input: &'a str,
    /// Implementor of [`PipelinePatternMatcher`] - the engine to match special tokens
    pattern_matcher: &'b PatternMatcher,
    /// Whether the input is normalized
    normalized: bool,
    offset: usize,
    pending: Option<u32>,
}

impl<'a, 'b, PatternMatcher: PipelinePatternMatcher>
    SpecialSegmentIterator<'a, 'b, PatternMatcher>
{
    /// Create a new iterator over the [`Segment`]s of `input`.
    /// This iterator will yield [`Segment`] in order.
    pub fn new(input: &'a str, pattern_matcher: &'b PatternMatcher, normalized: bool) -> Self {
        Self {
            input,
            pattern_matcher,
            normalized,
            pending: None,
            offset: 0,
        }
    }
}

impl<'a, 'b, PatternMatcher: PipelinePatternMatcher> Iterator
    for SpecialSegmentIterator<'a, 'b, PatternMatcher>
{
    type Item = Segment<'a>;

    /// Get the next segment of the input.
    fn next(&mut self) -> Option<Self::Item> {
        // take resets the pending option to None
        if let Some(special_token) = self.pending.take() {
            return Some(Segment::SpecialToken(special_token));
        }

        let remaining_input = &self.input[self.offset..];
        if remaining_input.is_empty() {
            // We've processed all the input string, return
            return None;
        }
        if let Some(((start, end), token)) =
            self.pattern_matcher
                .extract_next(self.input.as_bytes(), self.offset, self.normalized)
        {
            // `extract_next` positions are absolute in `input`, not relative to `offset`.
            let before_token = &self.input[self.offset..start];
            self.offset = end;
            if !before_token.is_empty() {
                // The iterator returns segments in order: we need to return the chunk of text and then the special token.
                // Store the special token to return in the next call and return a [`Segment::Text`]
                self.pending = Some(token);
                return Some(Segment::Text(before_token));
            } else {
                return Some(Segment::SpecialToken(token));
            }
        }
        self.offset = self.input.len();
        Some(Segment::Text(remaining_input))
    }
}

/// Experimental encode-only pipeline built from a [`Tokenizer`]. Runs the same
/// stages (special-token split → normalize → pre-tokenize → model) over borrowed
/// ranges to avoid the reference path's allocations.
///
/// This is a cheap-clone **handle** over an `Arc`'d frozen core (the v2
/// `Arc<Compiled>` shape): cloning shares the vocabulary/model, and an owned
/// [`EncodeHandle`] keeps the tokenizer alive past the `encode` call by holding a
/// clone.
#[derive(Clone)]
pub struct PipelineTokenizer {
    inner: Arc<PipelineInner>,
}

/// The frozen pipeline components, shared by every clone of the handle.
struct PipelineInner {
    added_vocabulary: BucketAddedVocabulary,
    normalizer: Option<NormalizerWrapper>,
    pre_tokenizer: PipelinePreTokenizer,
    model: PipelineModel,
    _post_processor: Option<PostProcessorWrapper>,
    /// Any `normalized` added token whose content contains an internal cut of
    /// the pre-tokenizer's [`StrideBoundary`] could be split by a stride (each
    /// half then mis-framing it), so its presence disables `SplitRaw`/
    /// `SplitNormalized`. Only `normalized` tokens count: raw-matched tokens are
    /// peeled by rung 0 before any striding. Computed once at build via
    /// [`has_internal_boundary`].
    added_token_splits_boundary: bool,
}

// comptime verification that PipelinePreTokenizer is Send + Sync
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PipelineTokenizer>();
};

/// Reusable per-encode scratch, kept alive across sequences processed by the
/// same worker thread (`reset`, never realloc'd). Today it holds only the
/// pre-token span buffer; as the fast (SIMD) pre-tokenizer lands, the byte
/// `tags` buffer and chunk spans will live here too, so the parallel runtime
/// keeps a single reset-not-realloc scratch per worker.
#[derive(Default)]
pub(crate) struct EncodeState {
    pre_tokens: Vec<Span>,
    /// Classify tag scratch for the stride boundary scans (grow-and-reuse).
    tags: Vec<u8>,
    /// Model-specific heap buffers (merge heap, word symbols, candidate string,
    /// …), lazily built for the model kind this thread last encoded with and
    /// re-initialized on a kind switch — thread-local state is shared across
    /// tokenizers on a thread. Deliberately *not* cleared by [`reset`](Self::reset):
    /// persistence across sequences and encode calls is the point.
    model_scratch: Option<PipelineModelScratch>,
}

impl EncodeState {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Clear the scratch for reuse on the next sequence, keeping capacity.
    pub(crate) fn reset(&mut self) {
        self.pre_tokens.clear();
    }
}

/// Total input bytes below which `encode` runs inline on the calling thread (no pool):
/// under this, the pool's dispatch/wakeup cost outweighs the parallelism gain. The same
/// threshold sets the stride size for raw-cut chunking and the span-group target for
/// `SplitAtModel`, so it is the single "unit of parallel work" knob. Bench-tunable —
/// splitting the three roles into separate knobs is a tracked tuning item.
const PARALLEL_MIN_TOTAL_BYTES: usize = 8 * 1024;

/// The byte range that stride `index` of `text` owns after boundary snapping,
/// or `None` when the stride's window contains no safe boundary (its bytes
/// belong to the previous stride's chunk, which extends across it — that
/// stride emits an empty result to keep per-input chunk counts static).
///
/// This is the **fused executor**'s scheduling primitive: there is no central
/// cut list and no serial pre-scan — each worker claims a stride index and
/// derives its chunk locally. Strides are fixed
/// `PARALLEL_MIN_TOTAL_BYTES`-sized windows; a chunk starts at the first safe
/// boundary inside its own window (stride 0 starts at 0) and ends at the first
/// safe boundary found in the following windows — the exact scan those
/// windows' owners also perform, so adjacent workers agree on every cut with
/// no communication. Total scanning is O(input): each window is scanned at
/// most twice (once by its owner, once by an extending predecessor). Text with
/// no boundaries at all degrades to stride 0 owning the whole input.
fn stride_range(
    text: &str,
    index: usize,
    boundary: StrideBoundary,
    tags: &mut Vec<u8>,
) -> Option<Range<usize>> {
    const STRIDE: usize = PARALLEL_MIN_TOTAL_BYTES;
    let len = text.len();
    let start = if index == 0 {
        0
    } else {
        boundary_in_window(
            text,
            index * STRIDE,
            ((index + 1) * STRIDE).min(len),
            boundary,
            tags,
        )?
    };
    let mut window = index + 1;
    let end = loop {
        let lo = window * STRIDE;
        if lo >= len {
            break len;
        }
        if let Some(b) =
            boundary_in_window(text, lo, ((window + 1) * STRIDE).min(len), boundary, tags)
        {
            break b;
        }
        window += 1;
    };
    (start < end).then_some(start..end)
}

/// A safe-cut predicate for one pre-tokenizer family, expressed over the
/// `atomsplit` **tag stream** (the same SIMD `classify` substrate the FSM
/// pre-tokenizers run on — boundary logic is not reimplemented byte-matching).
/// `window_tags[0]` is one byte of left context; the fn returns the first
/// window-relative position `i >= 1` that is a safe cut: no pre-token — under
/// that family's splitting rules — can span it, and tokenization of each side
/// is independent of the other.
///
/// The boundary definition **belongs to the pre-tokenizer**
/// ([`PipelinePreTokenizer::stride_boundary`]): each configuration exposes a
/// predicate its own token grammar provably cannot cross, and the scheduler
/// ([`stride_range`]) is agnostic to what the boundary is. Must be a pure
/// function of the tags — workers rely on recomputing identical results.
type StrideBoundary = fn(window_tags: &[u8]) -> Option<usize>;

/// Tag classes that are a complete non-whitespace character — the previous-char
/// requirement for [`boundary_fsm`]'s space cut. Excludes whitespace and the
/// bytes that don't name a class on their own (`Cont`/`MultiByte`/`Sentinel`)
/// plus control∪unassigned. Read via [`prev_char_tag`], so a multibyte
/// character is seen at its resolved lead, never a continuation byte.
const NON_WS_PREV: u16 = Atom::Letter.bit()
    | Atom::NumWord.bit()
    | Atom::NumOther.bit()
    | Atom::Mark.bit()
    | Atom::Connector.bit()
    | Atom::Punct.bit()
    | Atom::Apostrophe.bit()
    | Atom::SymOther.bit()
    | Atom::NumericOther.bit();

/// Tag classes whose token can never carry a *trailing* newline — the
/// previous-char requirement for [`boundary_fsm`]'s newline cut. Letters
/// (`\p{L}+`, which is also every CJK character), numbers (`\p{N}{1,3}`), and
/// marks all live in rules that exclude `\r\n`, so the word token ends before
/// the newline. Punctuation/symbol classes are deliberately absent: the
/// cl100k/o200k/deepseek punct rule ` ?[…]+[\r\n]*` absorbs following newlines
/// into the punct token, so a cut after punct-then-newline is not
/// side-independent.
const NEWLINE_PREV: u16 =
    Atom::Letter.bit() | Atom::NumWord.bit() | Atom::NumOther.bit() | Atom::Mark.bit();

/// Whitespace classes other than newline (space + other unicode whitespace) —
/// the `cur` side of [`boundary_fsm`]'s whitespace cut. Newline is handled on
/// its own ([`NEWLINE_PREV`]) because it alone can be absorbed by a trailing
/// `[\r\n]*` in the punct rule.
const WS_NON_NEWLINE: u16 = Atom::Space.bit() | Atom::WsOther.bit();

/// Tag classes a number run can start *after* — the previous-char requirement
/// for [`safe_fsm_cut`]'s letter/connector/punct→number cut. The byte-level
/// families tokenize digit runs as `\p{N}{1,3}`, which never joins a preceding
/// non-number, and every preceding token rule (letters, punct) excludes `\p{N}`,
/// so the number run starts a fresh token regardless of side. (Number→number is
/// excluded — a long digit run splits into `{1,3}` groups the cut would break.)
const NUM_START_PREV: u16 = Atom::Letter.bit()
    | Atom::Mark.bit()
    | Atom::Connector.bit()
    | Atom::Punct.bit()
    | Atom::Apostrophe.bit()
    | Atom::SymOther.bit()
    | Atom::NumericOther.bit();

/// The class tag of the character ending just before window position `i`: walk
/// back over `Cont` continuation bytes to the character's lead. `i >= 1`, and
/// `window_tags[0]` is always a lead (the classify window starts on a char
/// boundary), so the walk stays in bounds. Lets a boundary predicate judge the
/// preceding character by its real class even when it is multibyte (e.g. CJK,
/// whose lead is `Letter` and whose trailing bytes are `Cont`).
fn prev_char_tag(window_tags: &[u8], i: usize) -> u8 {
    let mut j = i - 1;
    while j > 0 && in_mask(window_tags[j], Atom::Cont.bit()) {
        j -= 1;
    }
    window_tags[j]
}

/// `StrideBoundary` for whitespace-delimiting pre-tokenizers (`Whitespace`,
/// `WhitespaceSplit`, `Bert`): *any* whitespace character is a safe cut. These
/// splitters drop whitespace as a delimiter and never emit a token that spans
/// it, so a cut placed at a whitespace character leaves the tokens on either
/// side unchanged — even one inside a whitespace run, since each side then
/// drops its own run. Cutting *at* the character (it joins the right chunk)
/// keeps the cut on a char boundary: a whitespace lead is classed `WS`, its
/// continuation bytes `Cont`. Denser than a newline-only rule, which matters
/// for single-line inputs (minified JSON, long CSV rows) that would otherwise
/// see no intra-input split.
fn boundary_at_whitespace(window_tags: &[u8]) -> Option<usize> {
    (1..window_tags.len()).find(|&i| in_mask(window_tags[i], mask::WS))
}

/// Whether a cut at a `prev`→`cur` character transition is side-independent for
/// the atomsplit-FSM byte-level families (gpt2 / cl100k / o200k / deepseek) —
/// the cut character `cur` joins the right chunk. Two safe transitions, each
/// checked branch by branch against `atomsplit::regexes` for all four families:
///
/// * **→ space / other whitespace** after a non-whitespace character — the ws
///   joins the right chunk, where a leading space binds to the following token
///   (` ?\p{L}+`); no rule matches a non-whitespace character directly followed
///   by a (non-newline) space *inside* one token.
/// * **→ newline** after a word character ([`NEWLINE_PREV`]) — a letter/number/
///   mark token never carries a trailing newline, so the newline opens a fresh
///   whitespace match (`\s*[\r\n]+`). Gives space-sparse scripts (CJK)
///   intra-document parallelism.
///
/// * **→ number** after a letter/connector/punct/symbol ([`NUM_START_PREV`]) —
///   a `\p{N}{1,3}` run opens a fresh token; no preceding rule absorbs digits.
/// * **number → letter** — the digit run ended (`\p{N}{1,3}`), so a following
///   letter opens a ` ?\p{L}+` token. Together these two give whitespace-free
///   byte-level content (base64, minified JSON, long IDs) intra-input cuts.
///
/// The number transitions *can* fall inside a special (`<…token_0|>`), so they
/// are only sound because **rung 0 always peels specials before any striding** —
/// a strided segment is special-free, so no cut can bisect one. The gate then
/// only needs to guard normalized-form added tokens (see `added_token_splits_boundary`).
fn safe_fsm_cut(prev: u8, cur: u8) -> bool {
    (in_mask(cur, WS_NON_NEWLINE) && in_mask(prev, NON_WS_PREV))
        || (in_mask(cur, mask::NEWLINE) && in_mask(prev, NEWLINE_PREV))
        || (in_mask(cur, mask::NUMBER) && in_mask(prev, NUM_START_PREV))
        || (in_mask(cur, mask::LETTER) && in_mask(prev, mask::NUMBER))
}

/// `StrideBoundary` for the atomsplit-FSM byte-level families: the first
/// [`safe_fsm_cut`] transition in the window. The previous character is read
/// through [`prev_char_tag`], so every branch also fires after a multibyte
/// character (e.g. CJK).
fn boundary_fsm(window_tags: &[u8]) -> Option<usize> {
    (1..window_tags.len())
        .find(|&i| safe_fsm_cut(prev_char_tag(window_tags, i), window_tags[i]))
}

/// Whether `token`'s own bytes contain a cut of `boundary` strictly inside it —
/// i.e. a stride could split this token and each half would mis-frame it. Used
/// by `added_token_splits_boundary` to disqualify striding for a `normalized`
/// added token whose content carries an internal cut (e.g. a `_→0` number
/// transition). Raw-matched tokens don't need this — rung 0 peels them before
/// any striding — so only normalized-form tokens reach this check.
fn has_internal_boundary(token: &str, boundary: StrideBoundary) -> bool {
    if token.len() < 2 {
        return false;
    }
    let mut tags = vec![0u8; token.len()];
    classify(token.as_bytes(), &mut tags);
    boundary(&tags).is_some()
}

/// Find the first safe cut in `text[lo..hi)` (absolute byte index; `lo >= 1`):
/// classify in blocks — SIMD [`classify`] into the caller's `tags` scratch,
/// with one byte of left context so the predicate can see the previous tag —
/// and apply `boundary` per block, early-exiting on the first hit. Blocked so
/// the common case (a boundary within the first block) classifies ~1 KB, not
/// the whole window.
///
/// Block edges are snapped to char boundaries (`classify` must see complete
/// characters). The snap is a pure function of the text, so adjacent workers
/// still agree; the extra bytes a snap adds are continuation bytes, which tag
/// as `Cont` and can never satisfy a boundary predicate.
fn boundary_in_window(
    text: &str,
    lo: usize,
    hi: usize,
    boundary: StrideBoundary,
    tags: &mut Vec<u8>,
) -> Option<usize> {
    const BLOCK: usize = 1024;
    debug_assert!(1 <= lo && lo <= hi && hi <= text.len());
    let mut block_lo = lo;
    while block_lo < hi {
        let mut ctx = block_lo - 1;
        while !text.is_char_boundary(ctx) {
            ctx -= 1;
        }
        let mut block_hi = (block_lo + BLOCK).min(hi);
        while block_hi < text.len() && !text.is_char_boundary(block_hi) {
            block_hi += 1;
        }
        let window = &text.as_bytes()[ctx..block_hi];
        tags.clear();
        tags.resize(window.len(), 0);
        classify(window, tags);
        if let Some(rel) = boundary(tags) {
            return Some(ctx + rel);
        }
        block_lo = block_hi;
    }
    None
}

/// The **sub-rung** applied to each special-free text segment (rung 0 always
/// peels the specials first), chosen per tokenizer configuration: cut the raw
/// text when the components allow it; otherwise normalize serially and cut the
/// normalized text; otherwise pre-tokenize serially and parallelize the model;
/// otherwise rely on batch-level parallelism only. Never wrong, only less
/// parallel.
#[derive(Clone, Copy, Debug)]
enum ParallelPlan {
    /// Rung 1: cut raw text at safe boundaries; each chunk runs the full
    /// pipeline (`encode_one_with`). Requires a boundary-preserving normalizer.
    SplitRaw(StrideBoundary),
    /// Rung 2: frame + normalize serially on the caller, then cut the normalized
    /// text at the pre-tokenizer's boundary and parallelize pre-tokenize + model
    /// (`encode_normalized`) per chunk. The serial normalize takes the normalizer
    /// out of the correctness equation, so this needs only a splittable
    /// pre-tokenizer — the fallback when a non-boundary-preserving normalizer
    /// rules out `SplitRaw` but the pre-tokenizer still cuts.
    SplitNormalized(StrideBoundary),
    /// Rung 3: frame + normalize + pre-tokenize serially on the caller;
    /// parallelize the model over ~`PARALLEL_MIN_TOTAL_BYTES`-sized groups of
    /// pre-token spans. Valid for every config (each pre-token is modeled
    /// independently).
    SplitAtModel,
    /// No intra-input parallelism.
    WholeInput,
}

/// Serial-prefix output of the `SplitAtModel` sub-rung for **one special-free
/// text segment** (rung 0 already peeled raw specials): normalization,
/// normalized-special framing and pre-tokenization are done; what remains is
/// model work, packaged as span-groups. Everything is index/range-based (no
/// borrows), so both the scoped and the owned paths can consume it.
struct ModelPrefix {
    /// Ordered pipeline units for this segment.
    units: Vec<PrefixUnit>,
    /// Owned normalized segments ([`PrefixText::Norm`] points here) — only
    /// populated when the normalizer actually rewrote text.
    norm_bufs: Vec<String>,
    /// Flat pool of pre-token spans; groups reference sub-ranges of it.
    spans: Vec<Span>,
}

enum PrefixUnit {
    /// A matched special/added token — already resolved, no worker needed.
    Special(u32),
    /// A group of consecutive pre-token spans over one text segment.
    Group {
        text: PrefixText,
        spans: Range<usize>,
    },
}

/// Where a span-group's text lives. Spans are relative to the resolved text.
enum PrefixText {
    /// A sub-range of the raw input (normalizer absent or returned `Borrowed`).
    Raw(Range<usize>),
    /// A sub-range of `norm_bufs[buf]` (the normalizer rewrote this segment).
    Norm { buf: usize, range: Range<usize> },
}

/// Byte offset of the subslice `sub` within its base string `base`.
/// Both must be views into the same allocation (`sub` derived from `base`).
fn offset_within(base: &str, sub: &str) -> usize {
    let off = sub.as_ptr() as usize - base.as_ptr() as usize;
    debug_assert!(off + sub.len() <= base.len());
    off
}

thread_local! {
    /// Per-thread reusable encode scratch: pool drainers, inline callers and the
    /// caller-assist path all keep their buffers warm across encode calls
    /// (reset, never realloc'd) — pool threads persist for the process, so this
    /// is the cache-warmth contract of `EncodeState`.
    static SCRATCH: RefCell<EncodeState> = RefCell::new(EncodeState::new());
}

/// One unit's result travelling from a worker back to the job: the tokens (or
/// error) of the `idx`-th chunk of input `seq`.
struct UnitResult {
    seq: usize,
    idx: usize,
    tokens: Result<Vec<PipelineToken>>,
}

enum HandleInner {
    /// Fully computed, drained in input order.
    Ready(std::vec::IntoIter<Result<Vec<PipelineToken>>>),
    /// Being filled by pool workers over a channel; reassembled in input order
    /// on the fly.
    Streaming {
        core: Arc<JobCore>,
        state: StreamState,
    },
}

/// Reassembly state for a streaming job: buffers out-of-order chunk arrivals and yields
/// each input once all of its chunks are in, in input order.
struct StreamState {
    rx: mpsc::Receiver<UnitResult>,
    /// `slots[seq][chunk_idx]` — a chunk's tokens once received; the row length
    /// is the number of chunks expected for that input.
    slots: Vec<Vec<Option<Vec<PipelineToken>>>>,
    /// chunks received so far per input.
    filled: Vec<usize>,
    /// first error seen per input (if any).
    err: Vec<Option<super::Error>>,
    /// next input index to emit (input-order cursor).
    next_yield: usize,
    /// Caller-assist bookkeeping: set once the job's cursor is exhausted so
    /// the blocking faces stop trying to claim units.
    assist_done: bool,
}

impl StreamState {
    fn new(rx: mpsc::Receiver<UnitResult>, counts: Vec<usize>) -> Self {
        let slots: Vec<Vec<Option<Vec<PipelineToken>>>> = counts
            .iter()
            .map(|&n| (0..n).map(|_| None).collect())
            .collect();
        let filled = vec![0; counts.len()];
        let err = counts.iter().map(|_| None).collect();
        Self {
            rx,
            slots,
            filled,
            err,
            next_yield: 0,
            assist_done: false,
        }
    }

    fn record(&mut self, UnitResult { seq, idx, tokens }: UnitResult) {
        match tokens {
            Ok(t) => self.slots[seq][idx] = Some(t),
            Err(e) => {
                if self.err[seq].is_none() {
                    self.err[seq] = Some(e);
                }
                self.slots[seq][idx] = Some(Vec::new());
            }
        }
        self.filled[seq] += 1;
    }

    /// Channel closed with the current input still short of its chunks: a worker died
    /// without sending (a bug or a killed thread). Surface that as an error for it.
    fn disconnected(&mut self) -> Result<Vec<PipelineToken>> {
        self.next_yield += 1;
        Err("encode: worker exited before all chunks were produced".into())
    }

    /// If the next input in order is complete, emit it and advance the cursor.
    fn try_emit_next(&mut self) -> Option<Result<Vec<PipelineToken>>> {
        let k = self.next_yield;
        if self.filled[k] < self.slots[k].len() {
            return None;
        }
        self.next_yield += 1;
        if let Some(e) = self.err[k].take() {
            return Some(Err(e));
        }
        let toks: Vec<PipelineToken> = self.slots[k]
            .iter_mut()
            .flat_map(|slot| slot.take().unwrap_or_default())
            .collect();
        Some(Ok(toks))
    }

    /// Yield the next input's result in input order. While waiting it absorbs whatever the
    /// channel already holds, then encodes pending items itself (caller-assist, when the
    /// backend supports it) rather than blocking. Returns `None` when every input has been
    /// emitted.
    fn next_in_order(&mut self, core: &JobCore) -> Option<Result<Vec<PipelineToken>>> {
        loop {
            if self.next_yield >= self.slots.len() {
                return None;
            }
            if let Some(out) = self.try_emit_next() {
                return Some(out);
            }
            // Record anything already delivered without blocking.
            match self.rx.try_recv() {
                Ok(msg) => {
                    self.record(msg);
                    continue;
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => return Some(self.disconnected()),
            }
            // Nothing ready: claim a unit of work ourselves rather than idling.
            if !self.assist_done {
                if SCRATCH.with(|st| core.run_one(&mut st.borrow_mut())) {
                    continue;
                }
                // Cursor exhausted: nothing left to claim.
                self.assist_done = true;
                continue;
            }
            match self.rx.recv() {
                Ok(msg) => self.record(msg),
                Err(_) => return Some(self.disconnected()),
            }
        }
    }

    /// Non-blocking, waker-registering flavor of [`next_in_order`](Self::next_in_order) for
    /// the `Stream` impl. Deliberately does **not** assist: encoding a chunk inside `poll`
    /// would block the async executor's thread; the pool workers make the progress.
    #[cfg(feature = "async")]
    fn poll_next_in_order(
        &mut self,
        core: &JobCore,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<Vec<PipelineToken>>>> {
        use std::task::Poll;
        loop {
            if self.next_yield >= self.slots.len() {
                return Poll::Ready(None);
            }
            if let Some(out) = self.try_emit_next() {
                return Poll::Ready(Some(out));
            }
            match self.rx.try_recv() {
                Ok(msg) => {
                    self.record(msg);
                    continue;
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Poll::Ready(Some(self.disconnected()))
                }
            }
            // Register-then-recheck: a send that raced the registration is caught
            // by the second `try_recv`; a send after it wakes the stored waker.
            core.set_waker(cx.waker());
            match self.rx.try_recv() {
                Ok(msg) => {
                    self.record(msg);
                    continue;
                }
                Err(mpsc::TryRecvError::Empty) => return Poll::Pending,
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Poll::Ready(Some(self.disconnected()))
                }
            }
        }
    }
}

/// Input storage an [`EncodeHandle`] keeps alive for its whole lifetime — the
/// contract that makes the returnable handle sound with no scoped lifetimes:
/// implementors promise every `&str` returned by [`get`](Self::get) stays valid
/// and unchanged for as long as the value lives. Owned storage (`String`,
/// `Vec<String>`) qualifies trivially; bindings can wrap refcounted foreign
/// strings (e.g. a `Py<PyString>` keep-alive over Python's stable UTF-8 buffer)
/// for zero-copy owned encodes.
#[allow(clippy::len_without_is_empty)] // an empty batch is legal; nothing branches on it
pub trait Inputs: Send + Sync + 'static {
    fn len(&self) -> usize;
    fn get(&self, i: usize) -> &str;
}

impl Inputs for String {
    fn len(&self) -> usize {
        1
    }
    fn get(&self, _i: usize) -> &str {
        self
    }
}
impl Inputs for Vec<String> {
    fn len(&self) -> usize {
        self.as_slice().len()
    }
    fn get(&self, i: usize) -> &str {
        &self[i]
    }
}

/// Conversion into [`Inputs`] for [`PipelineTokenizer::encode`]. Owned
/// inputs convert for free (a move); `&str`-family inputs pay **one copy** at
/// the boundary (measured at low single-digit % of encode cost).
pub trait IntoInputs {
    type Inputs: Inputs;
    fn into_inputs(self) -> Self::Inputs;
}

impl IntoInputs for String {
    type Inputs = String;
    fn into_inputs(self) -> String {
        self
    }
}
impl IntoInputs for Vec<String> {
    type Inputs = Vec<String>;
    fn into_inputs(self) -> Vec<String> {
        self
    }
}
impl IntoInputs for &str {
    type Inputs = String;
    fn into_inputs(self) -> String {
        self.to_owned()
    }
}
impl IntoInputs for &String {
    type Inputs = String;
    fn into_inputs(self) -> String {
        self.clone()
    }
}
impl IntoInputs for &[&str] {
    type Inputs = Vec<String>;
    fn into_inputs(self) -> Vec<String> {
        self.iter().map(|s| (*s).to_owned()).collect()
    }
}
impl IntoInputs for &Vec<&str> {
    type Inputs = Vec<String>;
    fn into_inputs(self) -> Vec<String> {
        self.iter().map(|s| (*s).to_owned()).collect()
    }
}

/// One owned encode call's worth of work, shared with the pool workers via
/// `Arc`. Fully safe: the job owns its storage and a tokenizer handle, and the
/// chunks are byte ranges into the storage — nothing borrows the caller, so the
/// job outlives the `encode` call freely. Dropping the [`EncodeHandle`] sets
/// `cancelled`; workers stop claiming, in-flight chunks finish into the closed
/// channel, and the last `Arc` holder releases the storage.
struct JobCore {
    storage: Box<dyn Inputs>,
    /// Ordered worker units, (seq, idx)-tagged for reassembly.
    units: Vec<Item>,
    /// Owned normalized segments for `SplitAtModel` units
    /// ([`PrefixText::Norm`] points here).
    norm_bufs: Vec<String>,
    /// Flat pool of pre-token spans for `SplitAtModel` units.
    spans: Vec<Span>,
    tokenizer: PipelineTokenizer,
    cursor: AtomicUsize,
    cancelled: AtomicBool,
    tx: mpsc::Sender<UnitResult>,
    /// Wakes an async consumer after each delivered chunk; blocking consumers
    /// are woken by the channel itself.
    waker: Mutex<Option<std::task::Waker>>,
}

/// One owned work unit: the `idx`-th piece of input `seq`.
struct Item {
    seq: usize,
    idx: usize,
    work: Work,
}

/// A unit's byte region within its source text: either a fixed range, or a
/// fused stride the worker boundary-snaps itself (no serial pre-scan).
#[derive(Clone)]
enum Region {
    Full(Range<usize>),
    Stride {
        index: usize,
        boundary: StrideBoundary,
        /// The special-free segment (byte range within the source) this stride
        /// tiles. Strides index from the segment start, so a rung-0-peeled text
        /// segment is strided independently of the specials around it.
        seg: Range<usize>,
    },
}

impl Region {
    /// Tile a special-free `seg` (byte range within its source) into work
    /// regions: fixed ~`PARALLEL_MIN_TOTAL_BYTES` fused strides (workers
    /// boundary-snap them), or a single `Full` region when the segment is too
    /// small to be worth striding. The shared "stride-or-whole" decision for the
    /// raw (`SplitRaw`) and normalized (`SplitNormalized`) paths.
    fn tile(seg: Range<usize>, boundary: StrideBoundary) -> Vec<Region> {
        let len = seg.len();
        if len >= 2 * PARALLEL_MIN_TOTAL_BYTES {
            (0..len.div_ceil(PARALLEL_MIN_TOTAL_BYTES))
                .map(|index| Region::Stride {
                    index,
                    boundary,
                    seg: seg.clone(),
                })
                .collect()
        } else {
            vec![Region::Full(seg)]
        }
    }

    /// Resolve to an absolute byte range within `text`. `None` when a stride's
    /// window has no boundary — that stride yields an empty result, keeping the
    /// per-input chunk count static for reassembly. A stride resolves against its
    /// own `seg` sub-slice, then shifts the result back to `text` coordinates.
    fn resolve(&self, text: &str, tags: &mut Vec<u8>) -> Option<Range<usize>> {
        match self {
            Region::Full(r) => Some(r.clone()),
            Region::Stride {
                index,
                boundary,
                seg,
            } => stride_range(&text[seg.clone()], *index, *boundary, tags)
                .map(|r| seg.start + r.start..seg.start + r.end),
        }
    }

    /// Rough byte size, for LPT ordering.
    fn approx_len(&self) -> usize {
        match self {
            Region::Full(r) => r.len(),
            Region::Stride { seg, .. } => PARALLEL_MIN_TOTAL_BYTES.min(seg.len()),
        }
    }
}

/// The pipeline stage a unit enters at (and the text its region indexes into).
enum Work {
    /// Full pipeline (`encode_one_with`) over a raw region of input `seq`.
    Raw(Region),
    /// Pre-tokenize + model (`encode_normalized`) over a region of the
    /// already-normalized buffer `JobCore::norm_bufs[buf]` (`SplitNormalized`).
    Normalized {
        buf: usize,
        region: Region,
    },
    /// Model-only over a span-group (`SplitAtModel`): `text` resolves against
    /// input `seq` (`Raw`) or `JobCore::norm_bufs` (`Norm`); `spans` indexes
    /// `JobCore::spans`.
    Model {
        text: PrefixText,
        spans: Range<usize>,
    },
}

impl JobCore {
    /// Claim and run one unit; `false` when the cursor is exhausted (or the job
    /// cancelled). A panicking unit is delivered as that input's `Err` so the
    /// consumer never hangs on a chunk that will not arrive.
    fn run_one(&self, scratch: &mut EncodeState) -> bool {
        if self.cancelled.load(Ordering::Relaxed) {
            return false;
        }
        let i = self.cursor.fetch_add(1, Ordering::Relaxed);
        if i >= self.units.len() {
            return false;
        }
        let unit = &self.units[i];
        let seq = unit.seq;
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match &unit.work {
            Work::Raw(region) => {
                let text = self.storage.get(seq);
                match region.resolve(text, &mut scratch.tags) {
                    Some(range) => self.tokenizer.encode_one_with(&text[range], scratch),
                    None => Ok(Vec::new()),
                }
            }
            Work::Normalized { buf, region } => {
                let text = &self.norm_bufs[*buf];
                match region.resolve(text, &mut scratch.tags) {
                    Some(range) => {
                        let mut output = Vec::with_capacity(range.len() / 4);
                        self.tokenizer
                            .encode_normalized(&text[range], scratch, &mut output)
                            .map(|()| output)
                    }
                    None => Ok(Vec::new()),
                }
            }
            Work::Model { text, spans } => {
                let text = match text {
                    PrefixText::Raw(r) => &self.storage.get(seq)[r.clone()],
                    PrefixText::Norm { buf, range } => &self.norm_bufs[*buf][range.clone()],
                };
                self.tokenizer
                    .encode_spans(text, &self.spans[spans.clone()], scratch)
            }
        }))
        .unwrap_or_else(|_| Err("encode worker panicked".into()));
        // A dropped receiver (job dropped early) just discards the result.
        let _ = self.tx.send(UnitResult {
            seq,
            idx: unit.idx,
            tokens: res,
        });
        if let Some(w) = self
            .waker
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
        {
            w.wake();
        }
        true
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    #[cfg(feature = "async")]
    fn set_waker(&self, w: &std::task::Waker) {
        *self.waker.lock().unwrap_or_else(PoisonError::into_inner) = Some(w.clone());
    }
}

/// Caller-assist for the owned path: the blocking consumer claims chunks
/// straight off the job core with its thread-local scratch.
/// An owned, escapable encode — what [`PipelineTokenizer::encode`] returns.
/// Workers keep encoding in the background while the caller holds this; results
/// surface in **input order** through:
/// - the blocking `Iterator` (which also *assists*: the calling thread claims
///   and encodes pending chunks instead of idling),
/// - [`EncodeHandle::wait_for_completion`](Self::wait_for_completion) /
///   [`into_single`](Self::into_single),
/// - `impl futures_core::Stream` with the `async` feature (poll-based, does not
///   assist — workers make the progress).
///
/// Dropping the job cancels all unclaimed work; chunks already being encoded
/// finish into the void. Worker panics surface as the affected input's `Err`.
pub struct EncodeHandle {
    inner: HandleInner,
}

impl EncodeHandle {
    fn ready(results: Vec<Result<Vec<PipelineToken>>>) -> Self {
        Self {
            inner: HandleInner::Ready(results.into_iter()),
        }
    }

    fn streaming(core: Arc<JobCore>, rx: mpsc::Receiver<UnitResult>, counts: Vec<usize>) -> Self {
        Self {
            inner: HandleInner::Streaming {
                core,
                state: StreamState::new(rx, counts),
            },
        }
    }

    /// Block until all inputs are encoded (assisting the pool from this thread);
    /// return per-input token lists in input order. Fails on the first input
    /// that errored.
    pub fn wait_for_completion(self) -> Result<Vec<Vec<PipelineToken>>> {
        self.collect()
    }

    /// Convenience for a single-input encode: block for and return the sole
    /// result (or the first, if called on a multi-input job).
    ///
    /// Not an efficient path — it blocks until every chunk of the input is done
    /// and concatenates them on the calling thread (a serial tail with nothing
    /// to overlap it). For throughput, prefer the streaming [`Iterator`] over a
    /// batch, where each input's reassembly overlaps the others' encoding.
    pub fn into_single(self) -> Result<Vec<PipelineToken>> {
        self.wait_for_completion().map(|mut all| {
            if all.is_empty() {
                Vec::new()
            } else {
                all.swap_remove(0)
            }
        })
    }
}

impl Iterator for EncodeHandle {
    type Item = Result<Vec<PipelineToken>>;
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            HandleInner::Ready(it) => it.next(),
            HandleInner::Streaming { core, state } => state.next_in_order(core),
        }
    }
}

impl Drop for EncodeHandle {
    fn drop(&mut self) {
        // Cancel whatever hasn't been claimed. Harmless after a full drain (the
        // cursor is already exhausted); on an early drop it stops the workers
        // from encoding into the void.
        if let HandleInner::Streaming { core, .. } = &self.inner {
            core.cancel();
        }
    }
}

#[cfg(feature = "async")]
impl futures_core::Stream for EncodeHandle {
    type Item = Result<Vec<PipelineToken>>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        match &mut self.get_mut().inner {
            HandleInner::Ready(it) => std::task::Poll::Ready(it.next()),
            HandleInner::Streaming { core, state } => state.poll_next_in_order(core, cx),
        }
    }
}

impl TryFrom<&Tokenizer> for PipelineTokenizer {
    type Error = super::Error;

    /// Build a pipeline from an existing [`Tokenizer`], cloning its components.
    ///
    /// The base [`Tokenizer`] carries the legacy [`crate::AddedVocabulary`]; the pipeline uses the
    /// fast bucket [`BucketAddedVocabulary`], so we rebuild it from the tokenizer's added tokens.
    /// Adding them in id order preserves ids (tokens present in the model reuse their model id, the
    /// rest keep their dense order), so the pipeline emits the same ids as the reference tokenizer.
    fn try_from(tok: &Tokenizer) -> Result<Self> {
        let pre_tokenizer: PipelinePreTokenizer = tok
            .get_pre_tokenizer()
            .cloned()
            .map(TryInto::try_into)
            .transpose()?
            .unwrap_or(PipelinePreTokenizer::None);

        let legacy_av = tok.get_added_vocabulary();
        let mut added_tokens: Vec<_> = legacy_av.get_added_tokens_decoder().iter().collect();
        added_tokens.sort_by_key(|(id, _)| **id);
        // A stride cut must never land inside an added/special token (each half
        // would mis-frame it). Rung 0 always peels the *raw*-matched (non-
        // normalized) tokens before any striding, so those can never be bisected
        // regardless of the boundary. Only `normalized`-form tokens — matched
        // after normalization, so invisible to rung 0's raw peel — can still be
        // split, and only if the boundary-preserving normalizer keeps their
        // internal cut. Disqualify striding just for those (e.g. a normalized
        // added token whose content carries a `boundary_fsm` transition).
        let added_token_splits_boundary = pre_tokenizer.stride_boundary().is_some_and(|boundary| {
            added_tokens
                .iter()
                .filter(|(_, t)| t.normalized)
                .any(|(_, t)| has_internal_boundary(&t.content, boundary))
        });
        let mut added_vocabulary = BucketAddedVocabulary::new();
        added_vocabulary.add_tokens(
            added_tokens.into_iter().map(|(_, t)| BucketAddedToken {
                content: t.content.clone(),
                single_word: t.single_word,
                lstrip: t.lstrip,
                rstrip: t.rstrip,
                normalized: t.normalized,
                special: t.special,
            }),
            tok.get_model(),
            tok.get_normalizer(),
        )?;
        added_vocabulary.set_encode_special_tokens(legacy_av.get_encode_special_tokens());

        let with_byte_level = {
            if let Some(pt) = tok.get_pre_tokenizer() {
                if let PreTokenizerWrapper::ByteLevel(_) = pt {
                    true
                } else if let PreTokenizerWrapper::Sequence(seq) = pt {
                    if seq
                        .as_ref()
                        .iter()
                        .any(|pt| matches!(pt, PreTokenizerWrapper::Sequence(_)))
                    {
                        return Err("Nesting Sequence pre tokenizers is not supported".into());
                    }
                    if let Some(pos) = seq
                        .as_ref()
                        .iter()
                        .position(|p| matches!(p, PreTokenizerWrapper::ByteLevel(_)))
                    {
                        if pos != seq.as_ref().len() - 1 {
                            return Err("ByteLevel pre tokenizer must be the last pre tokenizer in the Sequence".into());
                        }
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            }
        };

        let model = tok.get_model();
        if with_byte_level && !matches!(&model, ModelWrapper::BPE(_)) {
            let model_name = match model {
                ModelWrapper::BPE(_) => "BPE",
                ModelWrapper::Unigram(_) => "Unigram",
                ModelWrapper::WordLevel(_) => "WordLevel",
                ModelWrapper::WordPiece(_) => "WordPiece",
            };
            return Err(format!(
                "ByteLevel pre tokenizer is not supported with model {model_name}"
            )
            .into());
        }

        let model = match model.clone() {
            ModelWrapper::BPE(model) => {
                PipelineModel::BPE(PipelineBPE::from_bpe(model, with_byte_level)?)
            }
            ModelWrapper::Unigram(model) => PipelineModel::Unigram(model),
            ModelWrapper::WordLevel(model) => PipelineModel::WordLevel(model),
            ModelWrapper::WordPiece(model) => PipelineModel::WordPiece(model.try_into()?),
        };

        Ok(Self {
            inner: Arc::new(PipelineInner {
                added_vocabulary,
                normalizer: tok.get_normalizer().cloned(),
                pre_tokenizer,
                model,
                _post_processor: tok.get_post_processor().cloned(),
                added_token_splits_boundary,
            }),
        })
    }
}

impl PipelineTokenizer {
    // Component accessors: introspection for callers (and for external benches
    // that recompose partial pipelines — the library itself carries no staging
    // or timing scaffolding).

    pub fn get_model(&self) -> &PipelineModel {
        &self.inner.model
    }

    /// The added/special-token matcher the frame passes run with.
    pub fn get_added_vocabulary(&self) -> &impl PipelinePatternMatcher {
        &self.inner.added_vocabulary
    }

    pub fn get_normalizer(&self) -> Option<&NormalizerWrapper> {
        self.inner.normalizer.as_ref()
    }

    pub fn get_pre_tokenizer(&self) -> &PipelinePreTokenizer {
        &self.inner.pre_tokenizer
    }

    /// Encode one or many sequences, returning an [`EncodeHandle`] — an owned,
    /// escapable handle: workers keep encoding in the background while the caller
    /// holds it, drains it (blocking `Iterator` / [`EncodeHandle::wait_for_completion`]), or polls
    /// it as a `Stream` (with the `async` feature). **This is the default API.**
    ///
    /// The job is `'static`: it holds the input storage (see [`IntoInputs`] —
    /// moving a `String`/`Vec<String>` is zero-copy;
    /// passing `&str` pays one copy at the boundary) and a cheap clone of this
    /// tokenizer handle. Chunks are byte *ranges* into that storage, so there is no
    /// unsafe and no join barrier on this path — dropping the job cancels unclaimed
    /// work, and the last worker releases the storage.
    ///
    /// Scheduling: cost gate below `PARALLEL_MIN_TOTAL_BYTES` (the job comes
    /// back already complete, no pool touched); otherwise each input is decomposed
    /// rung 0 first (peel specials) then per the sub-[`ParallelPlan`], into work
    /// units drained from one shared claim cursor by pool tasks and by the job's
    /// blocking faces (caller-assist). Results surface in input order; worker
    /// panics surface as that input's `Err`.
    ///
    /// [`EncodeHandle::wait_for_completion`]: EncodeHandle::wait_for_completion
    pub fn encode<I: IntoInputs>(&self, inputs: I) -> EncodeHandle {
        let storage = inputs.into_inputs();
        let n_inputs = storage.len();
        let refs: Vec<&str> = (0..n_inputs).map(|i| storage.get(i)).collect();

        // Cost gate: small work is encoded right here; the job comes back Ready.
        let total_bytes: usize = refs.iter().map(|s| s.len()).sum();
        if total_bytes < PARALLEL_MIN_TOTAL_BYTES {
            return EncodeHandle::ready(self.encode_serial(&refs));
        }
        let Some(pool) = pool::rayon() else {
            return EncodeHandle::ready(self.encode_serial(&refs));
        };

        // Pick the parallel plan and build the (seq, idx)-tagged unit list.
        // Everything is range-based, so nothing borrows `refs`/`storage`.
        let mut counts = vec![0usize; n_inputs];
        let mut units: Vec<Item> = Vec::new();
        let mut norm_bufs: Vec<String> = Vec::new();
        let mut spans: Vec<Span> = Vec::new();
        let mut preresolved: Vec<UnitResult> = Vec::new();

        // Rung 0 is the outermost pipeline stage and unconditionally safe: special
        // tokens are framed before anything else, so cutting at each match and
        // encoding the inter-special text independently reproduces the whole
        // encode exactly. So EVERY plan begins here — peel specials (a matcher
        // scan the worker runs anyway, ~0.1% of encode) and hand each special-free
        // text segment to the config's deepest sub-rung. `plan()` names that
        // *sub*-rung; rung 0 wraps them all, so a strided segment can never
        // bisect a special (which is what lets the byte-level boundary include
        // number transitions — see `safe_fsm_cut`).
        let plan = self.plan();
        // The expensive sub-rungs (rung 2/3) run a serial normalize / pre-tokenize
        // prefix on the calling thread — only worth it when the batch alone can't
        // feed the pool. Rung 0/1 add no such prefix, so they always apply.
        let prefix_worth_it = n_inputs < pool.current_num_threads();

        for (seq, &input) in refs.iter().enumerate() {
            // An input too small to split into even two strides gains nothing from
            // intra-input decomposition — and running rung 0's matcher scan per
            // input on the calling thread would just serialize overhead across a
            // large batch of small inputs. Emit one whole unit; the worker frames
            // its specials itself (in parallel). No striding here, so this needs
            // no rung-0 peel to be safe.
            if input.len() < 2 * PARALLEL_MIN_TOTAL_BYTES {
                self.emit_full(seq, 0..input.len(), &mut counts, &mut units);
                continue;
            }
            for segment in SpecialSegmentIterator::new(input, &self.inner.added_vocabulary, false) {
                let chunk = match segment {
                    // Rung 0: a matched special resolves immediately, no worker.
                    Segment::SpecialToken(id) => {
                        let idx = counts[seq];
                        counts[seq] += 1;
                        preresolved.push(UnitResult {
                            seq,
                            idx,
                            tokens: Ok(vec![PipelineToken { id }]),
                        });
                        continue;
                    }
                    Segment::Text(chunk) => chunk,
                };
                let base = offset_within(input, chunk);
                let seg = base..base + chunk.len();
                let big_enough = chunk.len() >= 2 * PARALLEL_MIN_TOTAL_BYTES;

                match &plan {
                    // Rung 1: stride the raw segment; workers boundary-snap
                    // locally with no serial pre-scan.
                    ParallelPlan::SplitRaw(boundary) => {
                        for region in Region::tile(seg.clone(), *boundary) {
                            let idx = counts[seq];
                            counts[seq] += 1;
                            units.push(Item {
                                seq,
                                idx,
                                work: Work::Raw(region),
                            });
                        }
                    }
                    // Rung 2: serial-normalize the segment into an owned buffer,
                    // then stride the buffer (pre-tokenize + model in parallel).
                    ParallelPlan::SplitNormalized(boundary) if prefix_worth_it && big_enough => {
                        match self.inner.normalizer.as_ref().map_or_else(
                            || Ok(chunk.to_owned()),
                            |n| n.normalize(chunk).map(|c| c.into_owned()),
                        ) {
                            Ok(buf_str) => {
                                let buf = norm_bufs.len();
                                norm_bufs.push(buf_str);
                                for region in Region::tile(0..norm_bufs[buf].len(), *boundary) {
                                    let idx = counts[seq];
                                    counts[seq] += 1;
                                    units.push(Item {
                                        seq,
                                        idx,
                                        work: Work::Normalized { buf, region },
                                    });
                                }
                            }
                            // Normalize error → a whole-segment unit surfaces it.
                            Err(_) => self.emit_full(seq, seg.clone(), &mut counts, &mut units),
                        }
                    }
                    // Rung 3: serial normalize + pre-tokenize the segment, package
                    // the model work as span-groups.
                    ParallelPlan::SplitAtModel if prefix_worth_it && big_enough => {
                        match self.model_prefix_chunk(chunk, input) {
                            Ok(prefix) => {
                                // Merge this segment's prefix into the job pools.
                                let buf_base = norm_bufs.len();
                                let span_base = spans.len();
                                norm_bufs.extend(prefix.norm_bufs);
                                spans.extend_from_slice(&prefix.spans);
                                for unit in prefix.units {
                                    let idx = counts[seq];
                                    counts[seq] += 1;
                                    match unit {
                                        PrefixUnit::Special(t) => preresolved.push(UnitResult {
                                            seq,
                                            idx,
                                            tokens: Ok(vec![PipelineToken { id: t }]),
                                        }),
                                        PrefixUnit::Group { text, spans: sr } => {
                                            // Norm buffers were merged at `buf_base`;
                                            // Raw offsets are already input-absolute.
                                            let text = match text {
                                                PrefixText::Norm { buf, range } => {
                                                    PrefixText::Norm {
                                                        buf: buf + buf_base,
                                                        range,
                                                    }
                                                }
                                                raw => raw,
                                            };
                                            units.push(Item {
                                                seq,
                                                idx,
                                                work: Work::Model {
                                                    text,
                                                    spans: sr.start + span_base
                                                        ..sr.end + span_base,
                                                },
                                            });
                                        }
                                    }
                                }
                            }
                            Err(_) => self.emit_full(seq, seg.clone(), &mut counts, &mut units),
                        }
                    }
                    // `WholeInput`, or an expensive sub-rung whose prefix isn't
                    // worth it here (batch saturates the pool, or a small segment):
                    // one whole-segment unit, full pipeline in the worker.
                    _ => self.emit_full(seq, seg.clone(), &mut counts, &mut units),
                }
            }
        }

        // Too little schedulable work to be worth the pool. (Preresolved specials
        // need no worker, so only worker `units` count; falling back re-encodes
        // serially, which is byte-identical and cheap at this size.)
        if units.len() < 2 {
            return EncodeHandle::ready(self.encode_serial(&refs));
        }
        drop(refs);

        // LPT: claim the largest units first so no giant chunk lands last
        // (straggler tail); the (seq, idx) tags keep reassembly order-free.
        units.sort_by_key(|unit| {
            std::cmp::Reverse(match &unit.work {
                Work::Raw(region) => region.approx_len(),
                Work::Normalized { region, .. } => region.approx_len(),
                Work::Model { spans: sr, .. } => {
                    let group = &spans[sr.clone()];
                    group
                        .last()
                        .map(|last| (last.end - group[0].start) as usize)
                        .unwrap_or(0)
                }
            })
        });

        let (tx, rx) = mpsc::channel::<UnitResult>();
        // Units the serial prefix already resolved (special tokens) go straight
        // into the channel; reassembly absorbs them like any other chunk.
        for msg in preresolved {
            let _ = tx.send(msg);
        }
        let n_units = units.len();
        let core = Arc::new(JobCore {
            storage: Box::new(storage),
            units,
            norm_bufs,
            spans,
            tokenizer: self.clone(),
            cursor: AtomicUsize::new(0),
            cancelled: AtomicBool::new(false),
            tx,
            waker: Mutex::new(None),
        });

        // Spawn one cursor-drainer per potential worker ('static: each holds an
        // Arc of the core). Drainers that find the cursor exhausted are cheap
        // no-ops.
        let drainers = n_units.min(pool.current_num_threads());
        for _ in 0..drainers {
            let core = Arc::clone(&core);
            pool.spawn(move || {
                SCRATCH.with(|st| {
                    let scratch = &mut *st.borrow_mut();
                    while core.run_one(scratch) {}
                })
            });
        }

        EncodeHandle::streaming(core, rx, counts)
    }

    /// Model-only kernel for the `SplitAtModel` plan: tokenize each pre-token
    /// span of (already normalized) `text` in order.
    fn encode_spans(
        &self,
        text: &str,
        spans: &[Span],
        state: &mut EncodeState,
    ) -> Result<Vec<PipelineToken>> {
        // ~4.3 input bytes per token measured on English corpora; /4 is a
        // conservative reserve that avoids most growth reallocations.
        let covered = spans
            .last()
            .map(|last| (last.end - spans[0].start) as usize)
            .unwrap_or(0);
        let mut output = Vec::with_capacity(covered / 4);
        let scratch = self.inner.model.scratch_in(&mut state.model_scratch);
        for span in spans {
            self.inner
                .model
                .tokenize_pipeline(&text[span.range()], scratch, &mut output)?;
        }
        Ok(output)
    }

    /// Encode every input serially on the calling thread, reusing the thread's
    /// warm scratch across the whole batch (one result per input).
    fn encode_serial(&self, inputs: &[&str]) -> Vec<Result<Vec<PipelineToken>>> {
        SCRATCH.with(|st| {
            let state = &mut *st.borrow_mut();
            inputs
                .iter()
                .map(|&input| self.encode_one_with(input, state))
                .collect()
        })
    }

    /// The post-normalize suffix of the pipeline: frame (special tokens on
    /// already-normalized text) → pre-tokenize → model, appending to `output`.
    /// This is the entry point workers use when normalization has already run
    /// (rung 2 / the `SplitNormalized` plan), and the shared tail of
    /// [`encode_one_with`](Self::encode_one_with). Reuses the caller's
    /// `EncodeState` scratch (pre-token buffer + model scratch).
    fn encode_normalized(
        &self,
        normalized: &str,
        state: &mut EncodeState,
        output: &mut Vec<PipelineToken>,
    ) -> Result<()> {
        let scratch = self.inner.model.scratch_in(&mut state.model_scratch);
        let pre_tokens = &mut state.pre_tokens;
        for segment in SpecialSegmentIterator::new(normalized, &self.inner.added_vocabulary, true) {
            match segment {
                Segment::SpecialToken(token) => output.push(PipelineToken { id: token }),
                Segment::Text(chunk) => {
                    pre_tokens.clear();
                    self.inner.pre_tokenizer.pre_tokenize(chunk, pre_tokens)?;
                    for pre_token in pre_tokens.iter() {
                        self.inner.model.tokenize_pipeline(
                            &chunk[pre_token.range()],
                            scratch,
                            output,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    /// The full single-sequence kernel: frame (special tokens on raw text) →
    /// normalize → [`encode_normalized`](Self::encode_normalized), reusing the
    /// caller's `EncodeState` scratch. The entry point when a worker holds a raw
    /// region (rung 0/1: inline, pool drainers, caller-assist); span-group model
    /// work goes through [`encode_spans`](Self::encode_spans) instead.
    ///
    /// Note: the post-processor (special-token framing like BOS/EOS) is not
    /// wired yet; when it lands it applies per input after chunk concatenation.
    fn encode_one_with(
        &self,
        input: &str,
        state: &mut EncodeState,
    ) -> Result<Vec<PipelineToken>> {
        state.reset();
        // ~4.3 input bytes per token measured on English corpora; /4 is a
        // conservative reserve that avoids most growth reallocations.
        let mut output = Vec::with_capacity(input.len() / 4);

        // Extract the special tokens declared on raw text, normalize each text
        // segment, then run the post-normalize suffix.
        for segment in SpecialSegmentIterator::new(input, &self.inner.added_vocabulary, false) {
            match segment {
                Segment::SpecialToken(token) => output.push(PipelineToken { id: token }),
                Segment::Text(chunk) => {
                    let normalized: Cow<str> = match &self.inner.normalizer {
                        Some(normalizer) => normalizer.normalize(chunk)?,
                        None => Cow::Borrowed(chunk),
                    };
                    self.encode_normalized(&normalized, state, &mut output)?;
                }
            }
        }
        Ok(output)
    }

    /// The sub-rung this config applies to each special-free text segment (rung 0
    /// always peels specials first — see [`encode`](Self::encode)). The cheapest
    /// safe rung wins (see `PARALLEL_RUNTIME_DESIGN.md` §8).
    fn plan(&self) -> ParallelPlan {
        if let Some(boundary) = self.stride_boundary() {
            // Rung 1: no serial prefix, so always worth it when available.
            ParallelPlan::SplitRaw(boundary)
        } else if matches!(self.inner.model, PipelineModel::WordLevel(_)) {
            // A WordLevel "model step" is one hash lookup per pre-token, and its
            // pre-tokenize is cheap too — the serial prefix that rungs 2/3 need
            // would dwarf the parallelized part, so lean on batch parallelism.
            ParallelPlan::WholeInput
        } else if let Some(boundary) = self.normalized_stride_boundary() {
            // Rung 2: the normalizer rules out a raw cut, but the pre-tokenizer
            // can cut the (serially) normalized text — parallelize pretok+model.
            ParallelPlan::SplitNormalized(boundary)
        } else {
            ParallelPlan::SplitAtModel
        }
    }

    /// The pre-tokenizer's cut boundary for **already-normalized** text, gated
    /// only on the added vocabulary (no normalizer requirement — rung 2 runs the
    /// normalizer serially first, so it's out of the correctness equation). A
    /// cut still must not strip-absorb or split an added token, hence the
    /// `has_affix_strip` / `added_token_splits_boundary` gates.
    fn normalized_stride_boundary(&self) -> Option<StrideBoundary> {
        if self.inner.added_vocabulary.has_affix_strip() || self.inner.added_token_splits_boundary {
            return None;
        }
        self.inner.pre_tokenizer.stride_boundary()
    }

    /// Whether — and where — the config lets **raw** text be cut and each piece
    /// encoded independently with the same result as the whole (rung 1). Composes
    /// the per-stage split-safety declarations: the normalizer must preserve
    /// boundaries ([`NormalizerWrapper::preserves_stride_boundaries`]) *and* the
    /// added vocabulary + pre-tokenizer must permit a cut
    /// ([`normalized_stride_boundary`](Self::normalized_stride_boundary)). When
    /// the normalizer doesn't preserve boundaries, rung 2 (`SplitNormalized`)
    /// takes over via `normalized_stride_boundary` after a serial normalize.
    fn stride_boundary(&self) -> Option<StrideBoundary> {
        let norm_ok = self
            .inner
            .normalizer
            .as_ref()
            .is_none_or(NormalizerWrapper::preserves_stride_boundaries);
        if !norm_ok {
            return None;
        }
        self.normalized_stride_boundary()
    }

    /// Emit one whole-segment `Work::Raw` unit (full pipeline in the worker) for
    /// the byte `range` of input `seq`. The shared fallback for `WholeInput` and
    /// for the expensive sub-rungs when their serial prefix isn't worth running.
    fn emit_full(&self, seq: usize, range: Range<usize>, counts: &mut [usize], units: &mut Vec<Item>) {
        let idx = counts[seq];
        counts[seq] += 1;
        units.push(Item {
            seq,
            idx,
            work: Work::Raw(Region::Full(range)),
        });
    }

    /// Serial prefix of the [`ParallelPlan::SplitAtModel`] plan (rung 3) over one
    /// **special-free** text segment (rung 0 already peeled raw specials): normalize
    /// → frame(normalized specials) → pre-tokenize, packaging the remaining model
    /// work as ~`PARALLEL_MIN_TOTAL_BYTES`-sized span-groups. `input` is the whole
    /// source the borrowed (`PrefixText::Raw`) offsets resolve against; `chunk` is a
    /// subslice of it. Mirrors the walker in [`encode_one_with`](Self::encode_one_with)
    /// stage by stage — the two must agree on unit order for reassembly to be
    /// byte-identical to the serial encode.
    fn model_prefix_chunk(&self, chunk: &str, input: &str) -> Result<ModelPrefix> {
        let mut prefix = ModelPrefix {
            units: Vec::new(),
            norm_bufs: Vec::new(),
            spans: Vec::new(),
        };
        let mut pre_tokens: Vec<Span> = Vec::new();
        let normalized: Cow<str> = match &self.inner.normalizer {
            Some(normalizer) => normalizer.normalize(chunk)?,
            None => Cow::Borrowed(chunk),
        };
        let owned = matches!(normalized, Cow::Owned(_));
        for seg in SpecialSegmentIterator::new(&normalized, &self.inner.added_vocabulary, true) {
            match seg {
                Segment::SpecialToken(token) => prefix.units.push(PrefixUnit::Special(token)),
                Segment::Text(nchunk) => {
                    pre_tokens.clear();
                    self.inner
                        .pre_tokenizer
                        .pre_tokenize(nchunk, &mut pre_tokens)?;
                    if pre_tokens.is_empty() {
                        continue;
                    }
                    // Where this segment's text will live once the prefix is
                    // finalized (`normalized` is pushed to `norm_bufs` below, so
                    // its future index is the current length).
                    let base = if owned {
                        offset_within(&normalized, nchunk)
                    } else {
                        // Borrowed all the way down: nchunk is a subslice of the
                        // raw input.
                        offset_within(input, nchunk)
                    };
                    let text_range = base..base + nchunk.len();
                    // Group consecutive spans into ~target-byte units.
                    let mut g = 0;
                    while g < pre_tokens.len() {
                        let group_start = pre_tokens[g].start;
                        let mut e = g + 1;
                        while e < pre_tokens.len()
                            && ((pre_tokens[e - 1].end - group_start) as usize)
                                < PARALLEL_MIN_TOTAL_BYTES
                        {
                            e += 1;
                        }
                        let span_range = prefix.spans.len()..prefix.spans.len() + (e - g);
                        prefix.spans.extend_from_slice(&pre_tokens[g..e]);
                        let text = if owned {
                            PrefixText::Norm {
                                buf: prefix.norm_bufs.len(),
                                range: text_range.clone(),
                            }
                        } else {
                            PrefixText::Raw(text_range.clone())
                        };
                        prefix.units.push(PrefixUnit::Group {
                            text,
                            spans: span_range,
                        });
                        g = e;
                    }
                }
            }
        }
        if let Cow::Owned(s) = normalized {
            prefix.norm_bufs.push(s);
        }
        Ok(prefix)
    }
}

/// What [`split`] does with each split it forms
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SplitPolicy {
    /// Drop it, emit no split
    Remove,
    /// Emit it whole as one split
    Keep,
    /// Emit each character as its own split
    Isolate,
}

/// Splits `text` into same-class groups, emitting each as a [`Span`]
/// according to its [`SplitPolicy`].
///
/// `classify` maps each char to a small `Copy + Eq` class, the current
/// split ends whenever the class changes (or on every char of an `Isolate`
/// class), and `policy` decides what becomes of it. Ranges are byte offsets
/// into `text`.
#[inline(always)]
pub(crate) fn split<C: Copy + PartialEq>(
    text: &str,
    out: &mut Vec<Span>,
    classify: impl Fn(char) -> C,
    policy: impl Fn(C) -> SplitPolicy,
) {
    let mut start: u32 = 0;
    let mut prev: Option<C> = None;

    for (i, ch) in text.char_indices() {
        let c = classify(ch);
        if let Some(p) = prev {
            if p != c || policy(c) == SplitPolicy::Isolate {
                if policy(p) != SplitPolicy::Remove {
                    out.push(Span {
                        start,
                        end: i as u32,
                    });
                }
                start = i as u32;
            }
        }
        prev = Some(c);
    }

    if let Some(p) = prev {
        if policy(p) != SplitPolicy::Remove {
            out.push(Span {
                start,
                end: text.len() as u32,
            });
        }
    }
}

/// Splits `text` around a single delimiter predicate, honoring the full
/// [`SplitDelimiterBehavior`] contract. The pipeline-side equivalent of
/// `NormalizedString::split(pattern, behavior)` for a char predicate.
///
/// The three non-merging behaviors reduce to a [`SplitPolicy`] on the delimiter
/// class and reuse [`split`]. The two merge variants are their own single pass:
/// - `MergedWithPrevious` cuts the split *after* each delimiter, so a delimiter
///   joins the run before it (`"the-final"` -> `["the-", "final"]`).
/// - `MergedWithNext` cuts *before* each delimiter, so it joins the run after it
///   (`"the-final"` -> `["the", "-final"]`).
pub(crate) fn split_delimiter(
    text: &str,
    out: &mut Vec<Span>,
    is_delim: impl Fn(char) -> bool,
    behavior: SplitDelimiterBehavior,
) {
    use SplitDelimiterBehavior::*;

    let delim_policy = match behavior {
        Removed => SplitPolicy::Remove,
        Isolated => SplitPolicy::Isolate,
        Contiguous => SplitPolicy::Keep,
        MergedWithPrevious => {
            let mut start: u32 = 0;
            for (i, ch) in text.char_indices() {
                if is_delim(ch) {
                    let end = (i + ch.len_utf8()) as u32;
                    out.push(Span { start, end });
                    start = end;
                }
            }
            if (start as usize) < text.len() {
                out.push(Span {
                    start,
                    end: text.len() as u32,
                });
            }
            return;
        }
        MergedWithNext => {
            let mut start: u32 = 0;
            for (i, ch) in text.char_indices() {
                if is_delim(ch) {
                    let i = i as u32;
                    // skip the empty span before a leading run of delimiters
                    if i > start {
                        out.push(Span { start, end: i });
                    }
                    start = i;
                }
            }
            if (start as usize) < text.len() {
                out.push(Span {
                    start,
                    end: text.len() as u32,
                });
            }
            return;
        }
    };

    split(text, out, is_delim, |d| {
        if d {
            delim_policy
        } else {
            SplitPolicy::Keep
        }
    });
}

/// Applies a [`SplitDelimiterBehavior`] to a match segmentation and appends the
/// resulting pieces to `out`.
///
/// `matches` is the `(offsets, is_match)` sequence covering the whole input,
/// so regex matches interleaved with the gaps between them (exactly what
/// `Pattern::find_matches` produces). This is the pipeline-side equivalent of
/// the fold in `NormalizedString::split`; the arms mirror it exactly. Empty and
/// removed pieces are dropped.
pub(crate) fn split_matches(
    out: &mut Vec<Span>,
    matches: Vec<((usize, usize), bool)>,
    behavior: SplitDelimiterBehavior,
) {
    use SplitDelimiterBehavior::*;

    // (offsets, should_remove) — mirrors `NormalizedString::split`.
    let splits: Vec<((usize, usize), bool)> = match behavior {
        Isolated => matches.into_iter().map(|(o, _)| (o, false)).collect(),
        Removed => matches, // should_remove == is_match
        Contiguous => {
            let mut previous_match = false;
            matches
                .into_iter()
                .fold(vec![], |mut acc, (offsets, is_match)| {
                    if is_match == previous_match {
                        if let Some(((_, end), _)) = acc.last_mut() {
                            *end = offsets.1;
                        } else {
                            acc.push((offsets, false));
                        }
                    } else {
                        acc.push((offsets, false));
                    }
                    previous_match = is_match;
                    acc
                })
        }
        MergedWithPrevious => {
            let mut previous_match = false;
            matches
                .into_iter()
                .fold(vec![], |mut acc, (offsets, is_match)| {
                    if is_match && !previous_match {
                        if let Some(((_, end), _)) = acc.last_mut() {
                            *end = offsets.1;
                        } else {
                            acc.push((offsets, false));
                        }
                    } else {
                        acc.push((offsets, false));
                    }
                    previous_match = is_match;
                    acc
                })
        }
        MergedWithNext => {
            let mut previous_match = false;
            let mut splits =
                matches
                    .into_iter()
                    .rev()
                    .fold(vec![], |mut acc, (offsets, is_match)| {
                        if is_match && !previous_match {
                            if let Some(((start, _), _)) = acc.last_mut() {
                                *start = offsets.0;
                            } else {
                                acc.push((offsets, false));
                            }
                        } else {
                            acc.push((offsets, false));
                        }
                        previous_match = is_match;
                        acc
                    });
            splits.reverse();
            splits
        }
    };

    for ((start, end), should_remove) in splits {
        if !should_remove && start != end {
            out.push(Span {
                start: start as u32,
                end: end as u32,
            });
        }
    }
}

pub trait ModelScratch {}

pub trait Model {
    type Scratch: ModelScratch;

    fn tokenize_pipeline(
        &self,
        sequence: &str,
        scratch: &mut Self::Scratch,
        output: &mut Vec<PipelineToken>,
    ) -> Result<()>;

    fn init_scratch(&self) -> Self::Scratch;
}

#[allow(
    clippy::large_enum_variant,
    reason = "PipelineBPE holds a 1kB byte -> id lookup table"
)]
pub enum PipelineModel {
    BPE(PipelineBPE),
    Unigram(Unigram),
    WordLevel(WordLevel),
    WordPiece(PipelineWordPiece),
}

impl PipelineModel {
    /// Get this model's per-thread scratch out of `slot`, (re)building it when
    /// the slot is empty or was built for a different model kind — the
    /// thread-local `EncodeState` is shared across tokenizers on a thread.
    fn scratch_in<'a>(
        &self,
        slot: &'a mut Option<PipelineModelScratch>,
    ) -> &'a mut PipelineModelScratch {
        if !slot.as_ref().is_some_and(|s| s.matches(self)) {
            *slot = Some(self.init_scratch());
        }
        slot.as_mut().unwrap()
    }
}

impl Model for PipelineModel {
    type Scratch = PipelineModelScratch;

    fn tokenize_pipeline(
        &self,
        sequence: &str,
        scratch: &mut Self::Scratch,
        output: &mut Vec<PipelineToken>,
    ) -> Result<()> {
        match (self, scratch) {
            (Self::BPE(model), PipelineModelScratch::BPE(scratch)) => {
                model.tokenize_pipeline(sequence, scratch, output)
            }
            (Self::Unigram(model), PipelineModelScratch::Unigram(scratch)) => {
                model.tokenize_pipeline(sequence, scratch, output)
            }
            (Self::WordLevel(model), PipelineModelScratch::WordLevel(scratch)) => {
                model.tokenize_pipeline(sequence, scratch, output)
            }
            (Self::WordPiece(model), PipelineModelScratch::WordPiece(scratch)) => {
                model.tokenize_pipeline(sequence, scratch, output)
            }
            _ => unreachable!(),
        }
    }

    fn init_scratch(&self) -> Self::Scratch {
        match self {
            Self::BPE(bpe) => PipelineModelScratch::BPE(bpe.init_scratch()),
            Self::WordLevel(_) => Self::Scratch::WordLevel(()),
            Self::WordPiece(wordpiece) => Self::Scratch::WordPiece(wordpiece.init_scratch()),
            Self::Unigram(unigram) => Self::Scratch::Unigram(unigram.init_scratch()),
        }
    }
}

pub enum PipelineModelScratch {
    BPE(BpeScratch),
    WordLevel(()),
    WordPiece(WordPieceScratch),
    Unigram(UnigramScratch),
}

impl PipelineModelScratch {
    /// Whether this scratch was built for `model`'s kind (see
    /// [`PipelineModel::scratch_in`]).
    fn matches(&self, model: &PipelineModel) -> bool {
        matches!(
            (model, self),
            (PipelineModel::BPE(_), Self::BPE(_))
                | (PipelineModel::Unigram(_), Self::Unigram(_))
                | (PipelineModel::WordLevel(_), Self::WordLevel(_))
                | (PipelineModel::WordPiece(_), Self::WordPiece(_))
        )
    }
}

impl ModelScratch for PipelineModelScratch {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::bpe::BPE;
    use crate::models::wordpiece::WordPiece;
    use crate::pre_tokenizers::byte_level::ByteLevel;
    use crate::pre_tokenizers::sequence::Sequence;

    /// Serial oracle: encode one sequence with a fresh scratch, no pool.
    fn encode_one(tok: &PipelineTokenizer, input: &str) -> Result<Vec<PipelineToken>> {
        tok.encode_one_with(input, &mut EncodeState::new())
    }

    /// A chunk-safe pipeline: WordLevel model + `WhitespaceSplit` pre-tokenizer, no
    /// normalizer, no added tokens.
    fn chunk_safe_pipeline() -> PipelineTokenizer {
        use crate::models::wordlevel::WordLevelBuilder;
        use crate::pre_tokenizers::whitespace::WhitespaceSplit;
        use ahash::AHashMap;
        let vocab: AHashMap<String, u32> = ["aa", "bb", "cc", "<unk>"]
            .iter()
            .enumerate()
            .map(|(i, w)| ((*w).to_string(), i as u32))
            .collect();
        let model = WordLevelBuilder::new()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .build()
            .unwrap();
        let mut tok = Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(WhitespaceSplit));
        PipelineTokenizer::try_from(&tok).unwrap()
    }

    /// A large chunk-safe input is actually split into multiple chunks, and the chunked
    /// (parallel) encode is identical to the serial encode.
    #[test]
    fn intra_seq_splits_and_matches_serial() {
        let tok = chunk_safe_pipeline();
        assert!(
            tok.stride_boundary().is_some(),
            "wordlevel + WhitespaceSplit must be raw-chunkable"
        );

        let big = "aa bb cc\n".repeat(4000); // ~36 KB, newline-delimited
        assert!(big.len() > 2 * PARALLEL_MIN_TOTAL_BYTES);

        let strides = big.len().div_ceil(PARALLEL_MIN_TOTAL_BYTES);
        let mut tags = Vec::new();
        let chunks = (0..strides)
            .filter_map(|j| stride_range(&big, j, boundary_at_whitespace, &mut tags))
            .count();
        assert!(
            chunks > 1,
            "expected the large input to split, got {}",
            chunks
        );

        let ids = |v: Vec<PipelineToken>| v.iter().map(|t| t.id).collect::<Vec<_>>();
        let parallel = ids(tok.encode(big.as_str()).into_single().unwrap());
        let serial = ids(encode_one(&tok, big.as_str()).unwrap());
        assert_eq!(parallel, serial, "chunked encode must equal serial encode");

        // The owned (returnable) job must agree too — move the storage in.
        let owned = ids(tok.encode(big.clone()).into_single().unwrap());
        assert_eq!(owned, serial, "owned EncodeHandle must equal serial encode");
    }

    /// Whitespace-family strides cut *at* a whitespace character (space or
    /// newline) and tile the input exactly.
    #[test]
    fn stride_cuts_at_whitespace() {
        let input = "aa bb cc\n".repeat(4000);
        let chunks = assert_strides_partition(&input, boundary_at_whitespace, |bytes, start| {
            assert!(
                bytes[start].is_ascii_whitespace(),
                "cut must fall at a whitespace character"
            );
        });
        assert!(chunks > 1, "expected several chunks, got {}", chunks);
    }

    /// Every non-empty stride chunk must tile the input exactly (no gap, no
    /// overlap — adjacent workers must agree on every cut with no
    /// communication), with each chunk start satisfying the cut predicate.
    /// Returns the number of non-empty chunks.
    fn assert_strides_partition(
        input: &str,
        boundary: StrideBoundary,
        check_start: impl Fn(&[u8], usize),
    ) -> usize {
        let bytes = input.as_bytes();
        let mut tags = Vec::new();
        let mut covered = 0;
        let mut chunks = 0;
        for j in 0..input.len().div_ceil(PARALLEL_MIN_TOTAL_BYTES) {
            if let Some(range) = stride_range(input, j, boundary, &mut tags) {
                assert_eq!(range.start, covered, "chunks must tile the input");
                assert!(range.end > range.start);
                if range.start > 0 {
                    check_start(bytes, range.start);
                }
                covered = range.end;
                chunks += 1;
            }
        }
        assert_eq!(covered, input.len(), "chunks must cover the whole input");
        chunks
    }

    /// An input with no safe boundary at all degrades to stride 0 owning the
    /// whole input (serial), every other stride empty.
    #[test]
    fn strides_without_boundaries_degrade_to_whole_input() {
        let input = "a".repeat(4 * PARALLEL_MIN_TOTAL_BYTES);
        let mut tags = Vec::new();
        for boundary in [boundary_at_whitespace as StrideBoundary, boundary_fsm] {
            assert_eq!(
                stride_range(&input, 0, boundary, &mut tags),
                Some(0..input.len())
            );
            for j in 1..input.len().div_ceil(PARALLEL_MIN_TOTAL_BYTES) {
                assert_eq!(
                    stride_range(&input, j, boundary, &mut tags),
                    None,
                    "stride {}",
                    j
                );
            }
        }
    }

    /// `encode` drains via both faces (`wait_for_completion` bulk collect and the
    /// streaming `Iterator`), over a mixed batch (large splittable inputs + a small one), and
    /// both must equal the serial per-input encode in input order.
    fn assert_encode_matches_serial(tok: &PipelineTokenizer) {
        let a = "aa bb cc\n".repeat(3000); // ~27 KB, splits into several chunks
        let b = "bb cc aa\n".repeat(40); // small, stays one chunk
        let c = "cc aa bb\n".repeat(2000);
        let inputs: Vec<&str> = vec![a.as_str(), b.as_str(), c.as_str()];

        let ids = |v: Vec<PipelineToken>| v.iter().map(|t| t.id).collect::<Vec<_>>();
        let serial: Vec<Vec<u32>> = inputs
            .iter()
            .map(|s| ids(encode_one(tok, s).unwrap()))
            .collect();

        // Borrowed-slice sugar (copies in), bulk collect.
        let collected = tok
            .encode(&inputs[..])
            .wait_for_completion()
            .unwrap();
        let collected_ids: Vec<Vec<u32>> = collected.into_iter().map(ids).collect();
        assert_eq!(collected_ids, serial, "wait_for_completion != serial");

        // Borrowed-slice sugar, streaming iterator in input order.
        let streamed: Vec<Vec<u32>> = tok
            .encode(&inputs[..])
            .map(|r| ids(r.unwrap()))
            .collect();
        assert_eq!(streamed, serial, "streaming Iterator != serial");

        // Owned returnable job (the default API): the handle escapes the call and
        // is drained afterwards, both faces.
        let owned_inputs: Vec<String> = inputs.iter().map(|s| (*s).to_owned()).collect();
        let job = tok.encode(owned_inputs.clone());
        let owned: Vec<Vec<u32>> = job
            .wait_for_completion()
            .unwrap()
            .into_iter()
            .map(ids)
            .collect();
        assert_eq!(owned, serial, "owned wait_for_completion != serial");

        let job = tok.encode(owned_inputs.clone());
        let owned_streamed: Vec<Vec<u32>> = job.map(|r| ids(r.unwrap())).collect();
        assert_eq!(owned_streamed, serial, "owned Iterator != serial");

        // Dropping a job early cancels cleanly (no hang, no panic) and the pool
        // stays usable.
        let job = tok.encode(owned_inputs);
        drop(job);
        let again = ids(tok.encode(a.clone()).into_single().unwrap());
        assert_eq!(
            again, serial[0],
            "pool must stay usable after a dropped job"
        );
    }

    #[test]
    fn encode_streams_and_matches_serial() {
        assert_encode_matches_serial(&chunk_safe_pipeline());
    }

    /// FSM-family cuts on mixed prose land at whitespace boundaries and tile the
    /// input exactly.
    #[test]
    fn stride_cuts_at_fsm_boundaries() {
        let input = "word, another.  且つ 更に\n".repeat(2000);
        let chunks = assert_strides_partition(&input, boundary_fsm, |bytes, start| {
            assert!(
                bytes[start].is_ascii_whitespace(),
                "cut must land at whitespace, got {:#x}",
                bytes[start]
            );
        });
        assert!(chunks > 1, "expected several chunks, got {}", chunks);
    }

    /// The newline-after-word cut is what parallelizes space-sparse scripts: a
    /// document of CJK lines (no ASCII spaces) still splits, because a newline
    /// following a CJK letter is a safe FSM cut. The previous character is a
    /// multibyte letter, so this also exercises the `Cont` walk-back.
    #[test]
    fn stride_cuts_cjk_at_newlines() {
        let input = "吾輩は猫である名前はまだ無い\n".repeat(3000); // ~120 KB, no spaces
        let chunks = assert_strides_partition(&input, boundary_fsm, |bytes, start| {
            assert_eq!(bytes[start], b'\n', "CJK cut must land at a newline");
        });
        assert!(
            chunks > 1,
            "space-sparse CJK input must still split, got {}",
            chunks
        );
    }

    /// The GPT-2 byte-level regex qualifies for `SpaceRun` raw cuts, both bare
    /// and in the llama-shaped `Sequence[Span, None]`. A whitespace-containing
    /// added token disables raw cutting only when it is `normalized` (matched
    /// after normalization, so invisible to rung 0's raw peel); a raw/special one
    /// is peeled by rung 0 first, so a stride can never bisect it.
    #[test]
    fn space_run_gating() {
        use crate::AddedToken;
        let split = || {
            SplitPretok::new(
                SplitPattern::Regex(GPT2_REGEX_STR.to_owned()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .unwrap()
        };
        let make = |token: Option<AddedToken>| {
            let mut tok = Tokenizer::new(crate::models::bpe::BPE::default());
            tok.with_pre_tokenizer(Some(split()));
            if let Some(t) = token {
                tok.add_tokens([t]).unwrap();
            }
            PipelineTokenizer::try_from(&tok).unwrap()
        };
        assert!(make(None).stride_boundary().is_some());
        // Raw special with whitespace: peeled by rung 0, so raw cuts still qualify.
        assert!(
            make(Some(AddedToken::from("<s> extra", true)))
                .stride_boundary()
                .is_some(),
            "a raw special is peeled by rung 0 and must not disable raw cuts"
        );
        // Normalized added token with whitespace: rung 0's raw peel can't see it,
        // so a stride could bisect it — raw cuts must be disabled.
        assert!(
            make(Some(AddedToken::from("aa bb", false).normalized(true)))
                .stride_boundary()
                .is_none(),
            "whitespace inside a normalized added token must disable raw cuts"
        );

        // The llama-3 shape: Sequence[Span(known regex), ByteLevel(no regex) -> None].
        let seq = PipelinePreTokenizer::Sequence(PipelineSequence::new(vec![
            PipelinePreTokenizer::Split(split()),
            PipelinePreTokenizer::None,
        ]));
        assert!(seq.stride_boundary().is_some());
        // A pre-tokenizer without a provable boundary must not qualify.
        let no_boundary = PipelinePreTokenizer::Punctuation(Punctuation::default());
        assert!(no_boundary.stride_boundary().is_none());
    }

    /// A large input under the GPT-2 regex pre-tokenizer takes the `SpaceRun`
    /// raw-cut path and stays id-identical to the serial encode. (WordPiece
    /// stands in for the model; the boundary behavior under test is the
    /// pre-tokenizer's.)
    #[test]
    fn space_run_split_matches_serial() {
        use crate::models::wordpiece::WordPieceBuilder;
        use ahash::AHashMap;
        let vocab: AHashMap<String, u32> = ["aa", "bb", "cc", "<unk>"]
            .iter()
            .enumerate()
            .map(|(i, w)| ((*w).to_string(), i as u32))
            .collect();
        let model = WordPieceBuilder::new()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .build()
            .unwrap();
        let mut tok = Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(
            SplitPretok::new(
                SplitPattern::Regex(GPT2_REGEX_STR.to_owned()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .unwrap(),
        ));
        let tok = PipelineTokenizer::try_from(&tok).unwrap();
        assert!(matches!(tok.plan(), ParallelPlan::SplitRaw(_)));

        let big = "aa bb,cc!  aa\tbb  cc\n\n".repeat(2000); // ~44 KB, mixed runs
        let mut tags = Vec::new();
        let chunks = (0..big.len().div_ceil(PARALLEL_MIN_TOTAL_BYTES))
            .filter_map(|j| stride_range(&big, j, boundary_fsm, &mut tags))
            .count();
        assert!(chunks > 1, "expected the input to split, got {}", chunks);

        let ids = |v: Vec<PipelineToken>| v.iter().map(|t| t.id).collect::<Vec<_>>();
        let serial = ids(encode_one(&tok, &big).unwrap());
        let by_ref = ids(tok.encode(big.as_str()).into_single().unwrap());
        assert_eq!(by_ref, serial, "SpaceRun != serial");
        let owned = ids(tok.encode(big).into_single().unwrap());
        assert_eq!(owned, serial, "SpaceRun owned != serial");
    }

    /// Gate refinement guard: llama-3 has 256 `normalized == false` reserved
    /// specials (`<|reserved_special_token_0|>`), each carrying an internal
    /// number-transition (`_→0`) cut. The old gate checked *all* added tokens and
    /// would disqualify raw striding (silent `SplitAtModel` regression); the
    /// refined gate only checks `normalized` tokens (rung 0 peels the raw ones),
    /// so llama-3 must keep `SplitRaw` with the number-transition boundary.
    #[test]
    fn byte_level_raw_specials_keep_split_raw() {
        let oracle = Tokenizer::from_file("../data/llama-3-tokenizer.json").unwrap();
        let tok = PipelineTokenizer::try_from(&oracle).unwrap();
        assert!(
            matches!(tok.plan(), ParallelPlan::SplitRaw(_)),
            "llama-3's raw-only specials must not disqualify SplitRaw"
        );
    }

    /// A non-chunk-safe config: regex `Span` pre-tokenizer (unknown regex, so
    /// no raw cut qualifies) + WordPiece model, so the plan must
    /// escalate to [`ParallelPlan::SplitAtModel`]. Optionally a `Lowercase`
    /// normalizer (safe per-char, but the pretok already forces the escalation)
    /// and a `<s>` special token.
    fn split_at_model_pipeline(lowercase: bool) -> PipelineTokenizer {
        use crate::models::wordpiece::WordPieceBuilder;
        use crate::normalizers::utils::Lowercase;
        use crate::AddedToken;
        use ahash::AHashMap;
        let vocab: AHashMap<String, u32> = ["aa", "bb", "cc", ".", "<unk>", "<s>"]
            .iter()
            .enumerate()
            .map(|(i, w)| ((*w).to_string(), i as u32))
            .collect();
        let model = WordPieceBuilder::new()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .build()
            .unwrap();
        let mut tok = Tokenizer::new(model);
        // Punctuation keeps whitespace inside runs, so no raw cut qualifies —
        // and it needs no regex backend (the default build has none).
        tok.with_pre_tokenizer(Some(Punctuation::default()));
        if lowercase {
            tok.with_normalizer(Some(Lowercase)).unwrap();
        }
        tok.add_special_tokens([AddedToken::from("<s>", true)])
            .unwrap();
        PipelineTokenizer::try_from(&tok).unwrap()
    }

    /// SplitAtModel: a large single input under a non-chunk-safe config must
    /// actually split into several parallel span-groups and stay id-identical
    /// to the serial encode, through both the scoped and the owned paths.
    #[test]
    fn split_at_model_matches_serial() {
        let tok = split_at_model_pipeline(false);
        assert!(matches!(tok.plan(), ParallelPlan::SplitAtModel));

        let big = "aa.bb.cc.".repeat(3000); // ~27 KB, punctuation-delimited
        let prefix = tok.model_prefix_chunk(&big, &big).unwrap();
        let groups = prefix
            .units
            .iter()
            .filter(|u| matches!(u, PrefixUnit::Group { .. }))
            .count();
        assert!(groups > 1, "expected several span-groups, got {}", groups);
        assert!(
            prefix.norm_bufs.is_empty(),
            "no normalizer -> no owned bufs"
        );

        let ids = |v: Vec<PipelineToken>| v.iter().map(|t| t.id).collect::<Vec<_>>();
        let serial = ids(encode_one(&tok, &big).unwrap());
        let by_ref = ids(tok.encode(big.as_str()).into_single().unwrap());
        assert_eq!(by_ref, serial, "SplitAtModel != serial");
        let owned = ids(tok.encode(big).into_single().unwrap());
        assert_eq!(owned, serial, "owned SplitAtModel != serial");
    }

    /// SplitAtModel with a rewriting normalizer (uppercase input + `Lowercase`
    /// forces `Cow::Owned` → the `Norm` buffer path) and an interleaved special
    /// token (preresolved units must keep their position in the stream).
    #[test]
    fn split_at_model_normalizer_and_specials_match_serial() {
        let tok = split_at_model_pipeline(true);
        assert!(matches!(tok.plan(), ParallelPlan::SplitAtModel));

        let half = "AA.BB.cc.".repeat(2000);
        let big = format!("{half}<s>{}", "CC.aa.BB.".repeat(2000));
        // Rung 0 peels `<s>` in the builder; the per-segment model prefix runs on
        // special-free text and (with `Lowercase`) must own its normalized buffer.
        let prefix = tok.model_prefix_chunk(&half, &half).unwrap();
        assert!(
            !prefix.norm_bufs.is_empty(),
            "lowercased text must live in owned norm bufs"
        );

        let ids = |v: Vec<PipelineToken>| v.iter().map(|t| t.id).collect::<Vec<_>>();
        let serial = ids(encode_one(&tok, &big).unwrap());
        let by_ref = ids(tok.encode(big.as_str()).into_single().unwrap());
        assert_eq!(by_ref, serial, "parallel != serial (norm + specials)");
        let owned = ids(tok.encode(big).into_single().unwrap());
        assert_eq!(owned, serial, "owned != serial (norm + specials)");
    }

    /// The `Stream` face of an owned job yields the same results, in input order.
    /// Polled manually with a noop waker (busy-poll is fine in a test — the pool
    /// workers progress independently of the polling).
    #[cfg(feature = "async")]
    #[test]
    fn encode_job_stream_matches_serial() {
        use futures_core::Stream;
        use std::task::{Context, Poll, Waker};

        let tok = chunk_safe_pipeline();
        let inputs: Vec<String> = vec![
            "aa bb cc\n".repeat(3000),
            "bb cc aa\n".repeat(40),
            "cc aa bb\n".repeat(2000),
        ];
        let ids = |v: Vec<PipelineToken>| v.iter().map(|t| t.id).collect::<Vec<_>>();
        let serial: Vec<Vec<u32>> = inputs
            .iter()
            .map(|s| ids(encode_one(&tok, s).unwrap()))
            .collect();

        let mut job = tok.encode(inputs);
        let mut cx = Context::from_waker(Waker::noop());
        let mut got: Vec<Vec<u32>> = Vec::new();
        loop {
            match Stream::poll_next(std::pin::Pin::new(&mut job), &mut cx) {
                Poll::Ready(Some(r)) => got.push(ids(r.unwrap())),
                Poll::Ready(None) => break,
                Poll::Pending => std::thread::yield_now(),
            }
        }
        assert_eq!(got, serial, "Stream != serial");
    }

    /// A boundary-sensitive normalizer (`Strip`) makes the config not chunk-safe, so a large
    /// input stays whole (one chunk) — no unsafe splitting.
    #[test]
    fn unsafe_normalizer_config_does_not_split() {
        use crate::models::wordlevel::WordLevelBuilder;
        use crate::normalizers::strip::Strip;
        use crate::pre_tokenizers::whitespace::WhitespaceSplit;
        use ahash::AHashMap;
        let vocab: AHashMap<String, u32> = [("aa", 0u32), ("<unk>", 1)]
            .iter()
            .map(|(w, i)| ((*w).to_string(), *i))
            .collect();
        let model = WordLevelBuilder::new()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .build()
            .unwrap();
        let mut tok = Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(WhitespaceSplit));
        tok.with_normalizer(Some(Strip::new(true, true))).unwrap();
        let pipe = PipelineTokenizer::try_from(&tok).unwrap();
        assert!(
            pipe.stride_boundary().is_none(),
            "Strip normalizer must disable raw-text chunking"
        );
        // WordLevel model: the plan cannot escalate to SplitAtModel either.
        assert!(
            matches!(pipe.plan(), ParallelPlan::WholeInput),
            "Strip + WordLevel must fall to batch-level parallelism only",
        );
    }

    /// Rung 0 (always first): a config no lower rung can cut — `WordLevel` (so no
    /// `SplitAtModel`) under a boundary-hostile `Strip` normalizer (so no
    /// `SplitRaw`/`SplitNormalized`) picks the `WholeInput` sub-rung. But a single
    /// large input dense with special tokens still parallelizes, because rung 0
    /// wraps every plan: the builder peels the specials (preresolved) and hands
    /// each inter-special text segment to a worker. Must stay byte-identical to
    /// the serial encode through both the borrowed and owned handles.
    #[test]
    fn rung0_specials_parallelize_whole_input() {
        use crate::models::wordlevel::WordLevelBuilder;
        use crate::normalizers::strip::Strip;
        use crate::pre_tokenizers::whitespace::WhitespaceSplit;
        use crate::AddedToken;
        use ahash::AHashMap;
        let vocab: AHashMap<String, u32> = [("aa", 0u32), ("bb", 1), ("cc", 2), ("<unk>", 3)]
            .iter()
            .map(|(w, i)| ((*w).to_string(), *i))
            .collect();
        let model = WordLevelBuilder::new()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .build()
            .unwrap();
        let mut tok = Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(WhitespaceSplit));
        tok.with_normalizer(Some(Strip::new(true, true))).unwrap();
        tok.add_special_tokens([AddedToken::from("<s>", true)]).unwrap();
        let tok = PipelineTokenizer::try_from(&tok).unwrap();
        assert!(
            matches!(tok.plan(), ParallelPlan::WholeInput),
            "Strip + WordLevel sub-rung must be WholeInput (rung 0 does the splitting)"
        );

        // ~24 KB, ~2000 interspersed specials → thousands of worker units.
        let big = "aa bb cc <s> ".repeat(2000);
        let ids = |v: Vec<PipelineToken>| v.iter().map(|t| t.id).collect::<Vec<_>>();
        let serial = ids(encode_one(&tok, &big).unwrap());
        let by_ref = ids(tok.encode(big.as_str()).into_single().unwrap());
        assert_eq!(by_ref, serial, "rung-0 split (borrowed) != serial");
        let owned = ids(tok.encode(big.clone()).into_single().unwrap());
        assert_eq!(owned, serial, "rung-0 split (owned) != serial");
    }

    /// Rung 2 (`SplitNormalized`): a non-boundary-preserving normalizer
    /// (`Prepend`) rules out `SplitRaw`, but the `WhitespaceSplit` pre-tokenizer
    /// can cut the normalized text. The config serial-normalizes then
    /// parallelizes pre-tokenize + model, and must stay byte-identical to the
    /// serial encode (through both the borrowed and owned handles).
    #[test]
    fn split_normalized_matches_serial() {
        use crate::models::wordpiece::WordPieceBuilder;
        use crate::normalizers::prepend::Prepend;
        use crate::pre_tokenizers::whitespace::WhitespaceSplit;
        use ahash::AHashMap;
        let vocab: AHashMap<String, u32> = ["▁aa", "aa", "bb", "cc", "<unk>"]
            .iter()
            .enumerate()
            .map(|(i, w)| ((*w).to_string(), i as u32))
            .collect();
        let model = WordPieceBuilder::new()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .build()
            .unwrap();
        let mut tok = Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(WhitespaceSplit));
        tok.with_normalizer(Some(Prepend::new("▁".to_string()))).unwrap();
        let tok = PipelineTokenizer::try_from(&tok).unwrap();
        assert!(
            matches!(tok.plan(), ParallelPlan::SplitNormalized(_)),
            "Prepend (unsafe normalizer) + WhitespaceSplit + WordPiece must pick rung 2"
        );
        let big = "aa bb cc\n".repeat(4000); // ~36 KB, forces striding
        let ids = |v: Vec<PipelineToken>| v.iter().map(|t| t.id).collect::<Vec<_>>();
        let serial = ids(encode_one(&tok, &big).unwrap());
        let by_ref = ids(tok.encode(big.as_str()).into_single().unwrap());
        assert_eq!(by_ref, serial, "SplitNormalized (borrowed) != serial");
        let owned = ids(tok.encode(big.clone()).into_single().unwrap());
        assert_eq!(owned, serial, "SplitNormalized (owned) != serial");
    }

    struct FixedMatcher(Vec<((usize, usize), u32)>);
    impl PipelinePatternMatcher for FixedMatcher {
        fn extract_next(
            &self,
            _bytes: &[u8],
            search_offset: usize,
            _normalized: bool,
        ) -> Option<((usize, usize), u32)> {
            self.0
                .iter()
                .find(|((start, _), _)| *start >= search_offset)
                .copied()
        }
    }

    #[test]
    fn segment_iterator_yields_text_and_specials_in_order() {
        let input = "aa<s>bb<s>cc";
        let matcher = FixedMatcher(vec![((2, 5), 0), ((7, 10), 1)]);

        let segments: Vec<_> = SpecialSegmentIterator::new(input, &matcher, false)
            .map(|segment| match segment {
                Segment::Text(text) => (Some(text), None),
                Segment::SpecialToken(id) => (None, Some(id)),
            })
            .collect();

        assert_eq!(
            segments,
            vec![
                (Some("aa"), None),
                (None, Some(0)),
                (Some("bb"), None),
                (None, Some(1)),
                (Some("cc"), None),
            ]
        );
    }

    // The three rejections below guard configs the pipeline would otherwise
    // encode with silently wrong ids (the byte-level vocab transform only
    // applies when ByteLevel is the model's direct input). Each test pins the
    // error message so an unrelated failure can't stand in for the guard.

    fn conversion_error(tok: &Tokenizer) -> String {
        PipelineTokenizer::try_from(tok).err().unwrap().to_string()
    }

    #[test]
    fn conversion_rejects_nested_sequence() {
        let mut tok = Tokenizer::new(BPE::default());
        tok.with_pre_tokenizer(Some(Sequence::new(vec![PreTokenizerWrapper::Sequence(
            Sequence::new(vec![PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit)]),
        )])));
        let err = conversion_error(&tok);
        assert!(err.contains("Nesting Sequence"), "{}", err);
    }

    #[test]
    fn conversion_rejects_byte_level_not_last_in_sequence() {
        let mut tok = Tokenizer::new(BPE::default());
        tok.with_pre_tokenizer(Some(Sequence::new(vec![
            PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, true, true)),
            PreTokenizerWrapper::WhitespaceSplit(WhitespaceSplit),
        ])));
        let err = conversion_error(&tok);
        assert!(err.contains("must be the last"), "{}", err);
    }

    #[test]
    fn conversion_rejects_byte_level_with_non_bpe_model() {
        let mut tok = Tokenizer::new(WordPiece::default());
        tok.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
        let err = conversion_error(&tok);
        assert!(err.contains("not supported with model"), "{}", err);
    }
}
