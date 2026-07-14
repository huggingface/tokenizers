use std::cell::RefCell;
use std::convert::TryInto;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex, PoisonError};
use std::{borrow::Cow, convert::TryFrom};

use atomsplit::classify::classify;

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
    /// Create a new iterator over [`Segment`] of the [`input`].
    /// This iterator will yield [`Segment`] in order.
    pub(crate) fn new(
        input: &'a str,
        pattern_matcher: &'b PatternMatcher,
        normalized: bool,
    ) -> Self {
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
/// [`EncodeJob`] keeps the tokenizer alive past the `encode` call by holding a
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
    /// Any added/special token whose content contains whitespace could span a
    /// raw-text cut (which falls at a whitespace boundary), so its presence
    /// disables raw chunking entirely. Computed once at build.
    added_token_contains_whitespace: bool,
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
pub struct EncodeState {
    pre_tokens: Vec<Span>,
    /// Model-specific heap buffers (merge heap, word symbols, candidate string,
    /// …), lazily built for the model kind this thread last encoded with and
    /// re-initialized on a kind switch — thread-local state is shared across
    /// tokenizers on a thread. Deliberately *not* cleared by [`reset`](Self::reset):
    /// persistence across sequences and encode calls is the point.
    model_scratch: Option<PipelineModelScratch>,
}

impl EncodeState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear the scratch for reuse on the next sequence, keeping capacity.
    pub fn reset(&mut self) {
        self.pre_tokens.clear();
    }
}

/// Total input bytes below which `encode` runs inline on the calling thread (no pool):
/// under this, the pool's dispatch/wakeup cost outweighs the parallelism gain. The same
/// threshold sets the target chunk size when a large input is split, so it is the single
/// "unit of parallel work" knob. Bench-tunable.
const PARALLEL_MIN_TOTAL_BYTES: usize = 8 * 1024;

/// Interim stand-in for the `fast_split` `classify` pass. The real one tags every input
/// byte with its category in one SIMD/scalar pass; this stub keeps the same shape
/// (`classify(text, tags)` — one tag byte per input byte) but only distinguishes newlines,
/// which is enough to derive chunk boundaries for the parallel encode path. The whole
/// module is throwaway: it is replaced when `fast_split::classify` is wired in.
mod classify {
    /// Any non-newline byte.
    pub const OTHER: u8 = 0;
    /// A `\n` or `\r` byte.
    pub const NEWLINE: u8 = 1;

    /// Fill `tags` (length must equal `text.len()`) with one category byte per input byte.
    pub fn classify(text: &[u8], tags: &mut [u8]) {
        debug_assert_eq!(text.len(), tags.len());
        for (b, t) in text.iter().zip(tags.iter_mut()) {
            *t = if *b == b'\n' || *b == b'\r' {
                NEWLINE
            } else {
                OTHER
            };
        }
    }
}

/// A unit of scheduled work: a byte sub-range of input `seq_id`. A small input
/// is a single chunk; a large, safely-splittable one becomes several (cut at
/// newline boundaries), all tagged with the same `seq_id` so results reassemble
/// per input. Ranges (not `&str`s) so the owned encode path can store them
/// without self-reference.
struct Chunk {
    seq_id: usize,
    range: Range<usize>,
}

/// How a single large input may be split for intra-input parallelism — the
/// escalation ladder: chunk the raw text if the config allows it; otherwise run
/// the pipeline up to the model serially and parallelize the model; otherwise
/// rely on batch-level parallelism only. Never wrong, only less parallel.
/// Where a raw-text cut may fall such that no pre-token (and no added token)
/// can span it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RawCut {
    /// Just *after* a newline: for whitespace-delimiting pre-tokenizers a
    /// newline is a dropped hard delimiter.
    Newline,
    /// At a non-whitespace→space boundary, the space going to the right chunk:
    /// for the byte-level GPT regex family, a leading space binds to the
    /// following token (` ?\p{L}+`) and no token ever contains a
    /// non-whitespace→space transition, so the cut is match-preserving.
    SpaceRun,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ParallelPlan {
    /// Cut raw text at safe boundaries; each chunk runs the full pipeline.
    SplitRaw(RawCut),
    /// Frame + normalize + pre-tokenize serially on the caller; parallelize the
    /// model over ~[`PARALLEL_MIN_TOTAL_BYTES`]-sized groups of pre-token spans.
    /// Valid for every config (each pre-token is modeled independently).
    SplitAtModel,
    /// No intra-input parallelism.
    WholeInput,
}

/// Serial-prefix output of the `SplitAtModel` rung for **one input**: the frame
/// (special tokens), normalization and pre-tokenization are done; what remains
/// is model work, packaged as span-groups. Everything is index/range-based (no
/// borrows), so both the scoped and the owned paths can consume it.
struct ModelPrefix {
    /// Ordered pipeline units for this input.
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
    Group { text: PrefixText, spans: Range<usize> },
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

/// A scheduled unit of work for the scoped path: the `idx`-th piece of input
/// `seq`.
struct Item<'a> {
    seq: usize,
    idx: usize,
    work: Work<'a>,
}

/// What a worker does with one claimed unit.
enum Work<'a> {
    /// Run the full pipeline over this text (`SplitRaw` chunks and
    /// `WholeInput` units).
    Full(&'a str),
    /// Run only the model over these pre-token spans of `text` — the
    /// `SplitAtModel` rung: frame/normalize/pre-tokenize already ran serially
    /// on the caller. Spans are ranges into `text`.
    Spans { text: &'a str, spans: &'a [Span] },
}

impl Work<'_> {
    /// Approximate work size in input bytes (the LPT sort key).
    fn cost(&self) -> usize {
        match self {
            Work::Full(text) => text.len(),
            Work::Spans { spans, .. } => spans
                .last()
                .map(|last| (last.end - spans[0].start) as usize)
                .unwrap_or(0),
        }
    }
}

thread_local! {
    /// Per-thread reusable encode scratch: pool drainers, inline callers and the
    /// caller-assist path all keep their buffers warm across encode calls
    /// (reset, never realloc'd) — pool threads persist for the process, so this
    /// is the cache-warmth contract of [`EncodeState`].
    static SCRATCH: RefCell<EncodeState> = RefCell::new(EncodeState::new());
}

/// One-or-many borrowed input sequences for [`PipelineTokenizer::encode`],
/// held on the stack — wrapping a single `&str` allocates nothing.
pub enum EncodeBatch<'a> {
    One([&'a str; 1]),
    Many(&'a [&'a str]),
}

impl<'a> EncodeBatch<'a> {
    pub fn as_slice(&self) -> &[&'a str] {
        match self {
            Self::One(one) => one,
            Self::Many(many) => many,
        }
    }
}

/// Accepts one *or many* sequences for [`PipelineTokenizer::encode`], borrowing
/// them (zero-copy, zero-alloc). A single `&str`/`&String` becomes a one-element
/// batch; a slice or vec is borrowed as-is.
pub trait EncodeInputs<'a> {
    fn into_batch(self) -> EncodeBatch<'a>;
}

impl<'a> EncodeInputs<'a> for &'a str {
    fn into_batch(self) -> EncodeBatch<'a> {
        EncodeBatch::One([self])
    }
}
impl<'a> EncodeInputs<'a> for &'a String {
    fn into_batch(self) -> EncodeBatch<'a> {
        EncodeBatch::One([self.as_str()])
    }
}
impl<'a> EncodeInputs<'a> for &'a [&'a str] {
    fn into_batch(self) -> EncodeBatch<'a> {
        EncodeBatch::Many(self)
    }
}
impl<'a> EncodeInputs<'a> for &'a Vec<&'a str> {
    fn into_batch(self) -> EncodeBatch<'a> {
        EncodeBatch::Many(self.as_slice())
    }
}

/// A per-chunk result travelling from a worker back to the handle:
/// `(input index, chunk index within that input, tokens-or-error)`.
type ChunkMsg = (usize, usize, Result<Vec<PipelineToken>>);

/// Handle over an in-progress or completed encode, observed only inside the
/// closure passed to [`PipelineTokenizer::encode`] — the scope that guarantees
/// every worker borrow of the inputs is dead before `encode` returns. Results
/// are surfaced in **input order**.
///
/// - [`wait_for_completion`](Self::wait_for_completion) blocks and returns every input's
///   token ids.
/// - `impl Iterator` yields one `Result<Vec<PipelineToken>>` per input, in input order, so a
///   caller can consume results as they become ready.
/// - [`into_single`](Self::into_single) is single-input sugar.
///
/// Two states, one surface:
/// - `Ready` — fully computed already (the inline path, below the cost gate).
/// - `Streaming` — backed by a channel that pool drainers fill concurrently.
///   The handle also carries a [`ScopedAssist`]: while waiting for results, the
///   calling thread claims and encodes chunks itself instead of idling (and is
///   the liveness backstop if pool threads are scarce).
pub struct EncodeHandle<'s> {
    inner: HandleInner<ScopedAssist<'s>>,
}

/// Caller-assist for the scoped path: the handle claims chunks off the scope's
/// shared cursor and encodes them with the caller's thread-local scratch.
struct ScopedAssist<'s> {
    tokenizer: &'s PipelineTokenizer,
    items: &'s [Item<'s>],
    cursor: &'s AtomicUsize,
    add_special_tokens: bool,
    tx: mpsc::Sender<ChunkMsg>,
}

impl Assist for ScopedAssist<'_> {
    fn assist_one(&self) -> bool {
        SCRATCH.with(|st| {
            self.tokenizer.claim_one(
                self.items,
                self.cursor,
                self.add_special_tokens,
                &self.tx,
                &mut st.borrow_mut(),
            )
        })
    }
}

enum HandleInner<A> {
    /// Fully computed, drained in input order.
    Ready(std::vec::IntoIter<Result<Vec<PipelineToken>>>),
    /// Being filled by workers over a channel; reassembled in input order on the fly.
    Streaming(StreamState<A>),
}

/// A way for the waiting consumer thread to claim and encode one pending unit of
/// work itself instead of idling (caller-assist). Implemented by the scoped
/// path's [`ScopedAssist`] and the owned path's job core.
trait Assist {
    /// Claim and run one unit; `false` when nothing is left to claim.
    fn assist_one(&self) -> bool;
}

/// Reassembly state for the streaming handle: buffers out-of-order chunk arrivals and yields
/// each input once all of its chunks are in, in input order.
struct StreamState<A> {
    rx: mpsc::Receiver<ChunkMsg>,
    /// `slots[seq][chunk_idx]` — a chunk's tokens once received; the row length
    /// is the number of chunks expected for that input.
    slots: Vec<Vec<Option<Vec<PipelineToken>>>>,
    /// chunks received so far per input.
    filled: Vec<usize>,
    /// first error seen per input (if any).
    err: Vec<Option<super::Error>>,
    /// next input index to emit (input-order cursor).
    next_yield: usize,
    /// Claim work on the calling thread instead of idling (rayon-scoped path has
    /// none — its callers can't steal from `in_place_scope`).
    assist: Option<A>,
}

impl<A: Assist> StreamState<A> {
    fn new(rx: mpsc::Receiver<ChunkMsg>, counts: Vec<usize>) -> Self {
        let slots: Vec<Vec<Option<Vec<PipelineToken>>>> =
            counts.iter().map(|&n| (0..n).map(|_| None).collect()).collect();
        let filled = vec![0; counts.len()];
        let err = counts.iter().map(|_| None).collect();
        Self {
            rx,
            slots,
            filled,
            err,
            next_yield: 0,
            assist: None,
        }
    }

    fn with_assist(mut self, assist: A) -> Self {
        self.assist = Some(assist);
        self
    }

    fn absorb(&mut self, (seq, idx, res): ChunkMsg) {
        match res {
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
    fn next_in_order(&mut self) -> Option<Result<Vec<PipelineToken>>> {
        loop {
            if self.next_yield >= self.slots.len() {
                return None;
            }
            if let Some(out) = self.try_emit_next() {
                return Some(out);
            }
            // Absorb anything already delivered without blocking.
            match self.rx.try_recv() {
                Ok(msg) => {
                    self.absorb(msg);
                    continue;
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => return Some(self.disconnected()),
            }
            // Nothing ready: claim a unit of work ourselves rather than idling.
            if let Some(assist) = &self.assist {
                if assist.assist_one() {
                    continue;
                }
                // Cursor exhausted: drop the assist (releasing its `Sender` clone,
                // if it holds one).
                self.assist = None;
                continue;
            }
            match self.rx.recv() {
                Ok(msg) => self.absorb(msg),
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
        core: Option<&JobCore>,
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
                    self.absorb(msg);
                    continue;
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    return Poll::Ready(Some(self.disconnected()))
                }
            }
            let Some(core) = core else {
                // No core to wake us (cannot happen for a streaming job) — yield.
                cx.waker().wake_by_ref();
                return Poll::Pending;
            };
            // Register-then-recheck: a send that raced the registration is caught
            // by the second `try_recv`; a send after it wakes the stored waker.
            core.set_waker(cx.waker());
            match self.rx.try_recv() {
                Ok(msg) => {
                    self.absorb(msg);
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

impl<'s> EncodeHandle<'s> {
    fn ready(results: Vec<Result<Vec<PipelineToken>>>) -> Self {
        Self {
            inner: HandleInner::Ready(results.into_iter()),
        }
    }

    fn streaming(state: StreamState<ScopedAssist<'s>>) -> Self {
        Self {
            inner: HandleInner::Streaming(state),
        }
    }

    /// Block until all inputs are encoded; return per-input token lists in input order. Fails
    /// on the first input that errored.
    pub fn wait_for_completion(self) -> Result<Vec<Vec<PipelineToken>>> {
        self.collect()
    }

    /// Convenience for a single-input encode: return the sole result (or the first, if called
    /// on a multi-input handle).
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

impl Iterator for EncodeHandle<'_> {
    type Item = Result<Vec<PipelineToken>>;
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            HandleInner::Ready(it) => it.next(),
            HandleInner::Streaming(st) => st.next_in_order(),
        }
    }
}

/// Input storage an [`EncodeJob`] keeps alive for its whole lifetime — the
/// contract that makes the returnable handle sound with no scoped lifetimes:
/// implementors promise every `&str` returned by [`get`](Self::get) stays valid
/// and unchanged for as long as the value lives. Owned (`String`, `Vec<String>`)
/// and refcounted (`Arc<str>`) storage qualify trivially; bindings can wrap
/// refcounted foreign strings (e.g. a `Py<PyString>` keep-alive over Python's
/// stable UTF-8 buffer) for zero-copy owned encodes.
pub trait StableInputs: Send + Sync + 'static {
    fn len(&self) -> usize;
    fn get(&self, i: usize) -> &str;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl StableInputs for String {
    fn len(&self) -> usize {
        1
    }
    fn get(&self, _i: usize) -> &str {
        self
    }
}
impl StableInputs for Vec<String> {
    fn len(&self) -> usize {
        self.as_slice().len()
    }
    fn get(&self, i: usize) -> &str {
        &self[i]
    }
}
impl StableInputs for Arc<str> {
    fn len(&self) -> usize {
        1
    }
    fn get(&self, _i: usize) -> &str {
        self
    }
}
impl StableInputs for Vec<Arc<str>> {
    fn len(&self) -> usize {
        self.as_slice().len()
    }
    fn get(&self, i: usize) -> &str {
        &self[i]
    }
}

/// Conversion into [`StableInputs`] for [`PipelineTokenizer::encode`]. Owned and
/// refcounted inputs convert for free (a move / refcount bump); `&str`-family
/// inputs pay **one copy** at the boundary. Callers who cannot give up ownership
/// and cannot afford the copy should use
/// [`encode_scoped`](PipelineTokenizer::encode_scoped) instead (zero-copy,
/// closure-scoped).
pub trait IntoStableInputs {
    type Storage: StableInputs;
    fn into_stable(self) -> Self::Storage;
}

impl IntoStableInputs for String {
    type Storage = String;
    fn into_stable(self) -> String {
        self
    }
}
impl IntoStableInputs for Vec<String> {
    type Storage = Vec<String>;
    fn into_stable(self) -> Vec<String> {
        self
    }
}
impl IntoStableInputs for Arc<str> {
    type Storage = Arc<str>;
    fn into_stable(self) -> Arc<str> {
        self
    }
}
impl IntoStableInputs for Vec<Arc<str>> {
    type Storage = Vec<Arc<str>>;
    fn into_stable(self) -> Vec<Arc<str>> {
        self
    }
}
impl IntoStableInputs for &str {
    type Storage = String;
    fn into_stable(self) -> String {
        self.to_owned()
    }
}
impl IntoStableInputs for &String {
    type Storage = String;
    fn into_stable(self) -> String {
        self.clone()
    }
}
impl IntoStableInputs for &[&str] {
    type Storage = Vec<String>;
    fn into_stable(self) -> Vec<String> {
        self.iter().map(|s| (*s).to_owned()).collect()
    }
}
impl IntoStableInputs for &Vec<&str> {
    type Storage = Vec<String>;
    fn into_stable(self) -> Vec<String> {
        self.iter().map(|s| (*s).to_owned()).collect()
    }
}

/// One owned encode call's worth of work, shared with the pool workers via
/// `Arc`. Fully safe: the job owns its storage and a tokenizer handle, and the
/// chunks are byte ranges into the storage — nothing borrows the caller, so the
/// job outlives the `encode` call freely. Dropping the [`EncodeJob`] sets
/// `cancelled`; workers stop claiming, in-flight chunks finish into the closed
/// channel, and the last `Arc` holder releases the storage.
struct JobCore {
    storage: Box<dyn StableInputs>,
    /// Ordered worker units, (seq, idx)-tagged for reassembly.
    units: Vec<OwnedUnit>,
    /// Owned normalized segments for `SplitAtModel` units
    /// ([`PrefixText::Norm`] points here).
    norm_bufs: Vec<String>,
    /// Flat pool of pre-token spans for `SplitAtModel` units.
    spans: Vec<Span>,
    tokenizer: PipelineTokenizer,
    add_special_tokens: bool,
    cursor: AtomicUsize,
    cancelled: AtomicBool,
    tx: mpsc::Sender<ChunkMsg>,
    /// Wakes an async consumer after each delivered chunk; blocking consumers
    /// are woken by the channel itself.
    waker: Mutex<Option<std::task::Waker>>,
}

/// One owned work unit: the `idx`-th piece of input `seq`.
struct OwnedUnit {
    seq: usize,
    idx: usize,
    work: OwnedWork,
}

enum OwnedWork {
    /// Full pipeline over this byte range of input `seq`.
    Full(Range<usize>),
    /// Model-only over a span-group (`SplitAtModel`): `text` resolves against
    /// input `seq` (`Raw`) or `JobCore::norm_bufs` (`Norm`); `spans` indexes
    /// `JobCore::spans`.
    Spans { text: PrefixText, spans: Range<usize> },
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
            OwnedWork::Full(range) => {
                let text = &self.storage.get(seq)[range.clone()];
                self.tokenizer
                    .encode_one_with(text, self.add_special_tokens, scratch)
            }
            OwnedWork::Spans { text, spans } => {
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
        let _ = self.tx.send((seq, unit.idx, res));
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
struct OwnedAssist(Arc<JobCore>);

impl Assist for OwnedAssist {
    fn assist_one(&self) -> bool {
        SCRATCH.with(|st| self.0.run_one(&mut st.borrow_mut()))
    }
}

/// An owned, escapable encode — what [`PipelineTokenizer::encode`] returns.
/// Workers keep encoding in the background while the caller holds this; results
/// surface in **input order** through:
/// - the blocking `Iterator` (which also *assists*: the calling thread claims
///   and encodes pending chunks instead of idling),
/// - [`wait_for_completion`](Self::wait_for_completion) /
///   [`into_single`](Self::into_single),
/// - `impl futures_core::Stream` with the `async` feature (poll-based, does not
///   assist — workers make the progress).
///
/// Dropping the job cancels all unclaimed work; chunks already being encoded
/// finish into the void. Worker panics surface as the affected input's `Err`.
pub struct EncodeJob {
    /// `None` for a `Ready` (already-complete) job.
    core: Option<Arc<JobCore>>,
    inner: HandleInner<OwnedAssist>,
}

impl EncodeJob {
    fn ready(results: Vec<Result<Vec<PipelineToken>>>) -> Self {
        Self {
            core: None,
            inner: HandleInner::Ready(results.into_iter()),
        }
    }

    fn streaming(core: Arc<JobCore>, rx: mpsc::Receiver<ChunkMsg>, counts: Vec<usize>) -> Self {
        let state = StreamState::new(rx, counts).with_assist(OwnedAssist(Arc::clone(&core)));
        Self {
            core: Some(core),
            inner: HandleInner::Streaming(state),
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

impl Iterator for EncodeJob {
    type Item = Result<Vec<PipelineToken>>;
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            HandleInner::Ready(it) => it.next(),
            HandleInner::Streaming(st) => st.next_in_order(),
        }
    }
}

impl Drop for EncodeJob {
    fn drop(&mut self) {
        // Cancel whatever hasn't been claimed. Harmless after a full drain (the
        // cursor is already exhausted); on an early drop it stops the workers
        // from encoding into the void.
        if let Some(core) = &self.core {
            core.cancel();
        }
    }
}

#[cfg(feature = "async")]
impl futures_core::Stream for EncodeJob {
    type Item = Result<Vec<PipelineToken>>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match &mut this.inner {
            HandleInner::Ready(it) => std::task::Poll::Ready(it.next()),
            HandleInner::Streaming(st) => st.poll_next_in_order(this.core.as_deref(), cx),
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
        let added_token_contains_whitespace = added_tokens
            .iter()
            .any(|(_, t)| t.content.contains(char::is_whitespace));
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
                added_token_contains_whitespace,
            }),
        })
    }
}

impl PipelineTokenizer {
    /// Stage gates for [`encode_upto`](Self::encode_upto), in execution order. Each
    /// level runs every stage up to and including itself; `STAGE_MODEL` is a full
    /// encode. `STAGE_FRAME` is the special-token scan + iteration only (the "other"
    /// slice in the decomposition).
    pub const STAGE_FRAME: u8 = 0;
    pub const STAGE_NORMALIZE: u8 = 1;
    pub const STAGE_SPLIT: u8 = 2;
    pub const STAGE_MODEL: u8 = 3;

    pub fn get_model(&self) -> &PipelineModel {
        &self.inner.model
    }

    /// Encode one or many **borrowed** sequences, zero-copy; `f` observes the
    /// results through an [`EncodeHandle`] while workers may still be encoding.
    ///
    /// Pass a `&str` for one sequence or a `&[&str]` / `&Vec<&str>` for many (see
    /// [`EncodeInputs`]). Results surface in input order; consume them inside `f`
    /// via `wait_for_completion`, the streaming `Iterator`, or `into_single`.
    ///
    /// The scoped-closure shape is what keeps inputs borrowed (`&'a str`, zero-copy)
    /// while worker threads read them: like `std::thread::scope` (and how smol's
    /// `Executor<'a>` is actually driven), all worker borrows are discharged before
    /// `encode_scoped` returns, and there is no returnable handle to `mem::forget`
    /// your way past. If you can hand over (or share) ownership of the inputs,
    /// prefer [`encode`](Self::encode) — same runtime, returnable handle.
    ///
    /// Special tokens are matched in two passes:
    ///  1. on the raw input,
    ///  2. then on each segment after normalization
    ///
    /// This way, special / added tokens declared on raw or normalized text are both caught.
    /// The remaining text is pre-tokenized and run through the model span by span.
    ///
    /// Scheduling: below [`PARALLEL_MIN_TOTAL_BYTES`] of total work (or when
    /// segmentation yields a lone chunk) everything runs inline on the calling thread
    /// with warm thread-local scratch — no pool, no channel. Otherwise the inputs are
    /// [`segment`](Self::segment)ed into a flat list of chunks behind one shared
    /// claim cursor, drained by tasks on the process-global library-private rayon
    /// pool (see [`pool`]) and by the handle itself (caller-assist). A single big
    /// input and a batch of small ones ride the same runtime — the big input just
    /// produces more chunks.
    ///
    /// `&PipelineTokenizer` is shared across worker threads — see the `Send + Sync`
    /// guard next to the struct definition. A panic on a worker resurfaces here after
    /// the scope joins (the affected input also reports an `Err`, so `f` never
    /// hangs).
    ///
    /// todo: post-processing; fused boundary detection via the real `fast_split`
    /// classify (this uses the interim newline-only [`classify`] stub).
    pub fn encode_scoped<'a, I, R>(
        &self,
        inputs: I,
        add_special_tokens: bool,
        f: impl FnOnce(EncodeHandle<'_>) -> R,
    ) -> R
    where
        I: EncodeInputs<'a>,
    {
        let batch = inputs.into_batch();
        let inputs = batch.as_slice();

        // Cost gate: below the work threshold, run inline (no pool, no channel).
        let total_bytes: usize = inputs.iter().map(|s| s.len()).sum();
        if total_bytes < PARALLEL_MIN_TOTAL_BYTES {
            return f(EncodeHandle::ready(self.encode_serial(inputs, add_special_tokens)));
        }

        let pool::Backend::Rayon(p) = pool::backend() else {
            return f(EncodeHandle::ready(self.encode_serial(inputs, add_special_tokens)));
        };

        // Pick the ladder rung and build the (seq, idx)-tagged work list.
        // `prefixes` must outlive `items`: `Spans` units borrow its buffers.
        let mut counts = vec![0usize; inputs.len()];
        let mut items: Vec<Item> = Vec::new();
        let mut preresolved: Vec<ChunkMsg> = Vec::new();
        let mut prefixes: Vec<(usize, ModelPrefix)> = Vec::new();

        match self.plan() {
            ParallelPlan::SplitRaw(cut) => {
                for c in self.segment(inputs, cut) {
                    let idx = counts[c.seq_id];
                    counts[c.seq_id] += 1;
                    items.push(Item {
                        seq: c.seq_id,
                        idx,
                        work: Work::Full(&inputs[c.seq_id][c.range]),
                    });
                }
            }
            // Only worth its serial prefix when the batch alone can't feed the
            // pool (few, large inputs) — bench-tunable heuristic.
            ParallelPlan::SplitAtModel if inputs.len() < p.current_num_threads() => {
                for (seq, &input) in inputs.iter().enumerate() {
                    if input.len() >= 2 * PARALLEL_MIN_TOTAL_BYTES {
                        // A prefix error falls through to a Full unit, whose
                        // worker will surface the same error for this input.
                        if let Ok(prefix) = self.model_prefix(input) {
                            prefixes.push((seq, prefix));
                            continue;
                        }
                    }
                    let idx = counts[seq];
                    counts[seq] += 1;
                    items.push(Item {
                        seq,
                        idx,
                        work: Work::Full(input),
                    });
                }
                // Second phase: `prefixes` storage is settled, borrows are safe.
                for (seq, prefix) in &prefixes {
                    for unit in &prefix.units {
                        let idx = counts[*seq];
                        counts[*seq] += 1;
                        match unit {
                            PrefixUnit::Special(t) => {
                                preresolved.push((*seq, idx, Ok(vec![PipelineToken { id: *t }])))
                            }
                            PrefixUnit::Group { text, spans } => {
                                let text: &str = match text {
                                    PrefixText::Raw(r) => &inputs[*seq][r.clone()],
                                    PrefixText::Norm { buf, range } => {
                                        &prefix.norm_bufs[*buf][range.clone()]
                                    }
                                };
                                items.push(Item {
                                    seq: *seq,
                                    idx,
                                    work: Work::Spans {
                                        text,
                                        spans: &prefix.spans[spans.clone()],
                                    },
                                });
                            }
                        }
                    }
                }
            }
            _ => {
                for (seq, &input) in inputs.iter().enumerate() {
                    counts[seq] = 1;
                    items.push(Item {
                        seq,
                        idx: 0,
                        work: Work::Full(input),
                    });
                }
            }
        }

        // Too little schedulable work to be worth the pool.
        if items.len() < 2 && preresolved.is_empty() {
            return f(EncodeHandle::ready(self.encode_serial(inputs, add_special_tokens)));
        }

        // LPT: claim the largest units first so no giant chunk lands last
        // (straggler tail); the (seq, idx) tags keep reassembly order-free.
        items.sort_by_key(|item| std::cmp::Reverse(item.work.cost()));

        self.encode_rayon(p, &items, counts, preresolved, add_special_tokens, f)
    }

    /// Encode one or many sequences, returning an [`EncodeJob`] — an owned,
    /// escapable handle: workers keep encoding in the background while the caller
    /// holds it, drains it (blocking `Iterator` / [`wait_for_completion`]), or polls
    /// it as a `Stream` (with the `async` feature). **This is the default API.**
    ///
    /// The job is `'static`: it holds the input storage (see [`IntoStableInputs`] —
    /// moving a `String`/`Vec<String>` or sharing an `Arc<str>` is zero-copy;
    /// passing `&str` pays one copy at the boundary) and a cheap clone of this
    /// tokenizer handle. Chunks are byte *ranges* into that storage, so there is no
    /// unsafe and no join barrier on this path — dropping the job cancels unclaimed
    /// work, and the last worker releases the storage.
    ///
    /// Same runtime and semantics as [`encode_scoped`](Self::encode_scoped): cost
    /// gate below [`PARALLEL_MIN_TOTAL_BYTES`] (the job comes back already
    /// complete), chunk-safe segmentation, input-order results, worker panics
    /// surface as that input's `Err`.
    ///
    /// [`wait_for_completion`]: EncodeJob::wait_for_completion
    pub fn encode<I: IntoStableInputs>(&self, inputs: I, add_special_tokens: bool) -> EncodeJob {
        let storage = inputs.into_stable();
        let n_inputs = storage.len();
        let refs: Vec<&str> = (0..n_inputs).map(|i| storage.get(i)).collect();

        // Cost gate: small work is encoded right here; the job comes back Ready.
        let total_bytes: usize = refs.iter().map(|s| s.len()).sum();
        if total_bytes < PARALLEL_MIN_TOTAL_BYTES {
            return EncodeJob::ready(self.encode_serial(&refs, add_special_tokens));
        }
        let p = match pool::backend() {
            pool::Backend::Inline => {
                return EncodeJob::ready(self.encode_serial(&refs, add_special_tokens))
            }
            pool::Backend::Rayon(p) => p,
        };

        // Pick the ladder rung and build the (seq, idx)-tagged unit list.
        // Everything is range-based, so nothing borrows `refs`/`storage`.
        let mut counts = vec![0usize; n_inputs];
        let mut units: Vec<OwnedUnit> = Vec::new();
        let mut norm_bufs: Vec<String> = Vec::new();
        let mut spans: Vec<Span> = Vec::new();
        let mut preresolved: Vec<ChunkMsg> = Vec::new();

        match self.plan() {
            ParallelPlan::SplitRaw(cut) => {
                for c in self.segment(&refs, cut) {
                    let idx = counts[c.seq_id];
                    counts[c.seq_id] += 1;
                    units.push(OwnedUnit {
                        seq: c.seq_id,
                        idx,
                        work: OwnedWork::Full(c.range),
                    });
                }
            }
            // Only worth its serial prefix when the batch alone can't feed the
            // pool (few, large inputs) — bench-tunable heuristic.
            ParallelPlan::SplitAtModel if n_inputs < p.current_num_threads() => {
                for (seq, &input) in refs.iter().enumerate() {
                    if input.len() >= 2 * PARALLEL_MIN_TOTAL_BYTES {
                        // A prefix error falls through to a Full unit, whose
                        // worker will surface the same error for this input.
                        if let Ok(prefix) = self.model_prefix(input) {
                            // Merge this input's prefix into the job-wide pools.
                            let buf_base = norm_bufs.len();
                            let span_base = spans.len();
                            norm_bufs.extend(prefix.norm_bufs);
                            spans.extend_from_slice(&prefix.spans);
                            for unit in prefix.units {
                                let idx = counts[seq];
                                counts[seq] += 1;
                                match unit {
                                    PrefixUnit::Special(t) => preresolved.push((
                                        seq,
                                        idx,
                                        Ok(vec![PipelineToken { id: t }]),
                                    )),
                                    PrefixUnit::Group { text, spans: sr } => {
                                        let text = match text {
                                            PrefixText::Raw(r) => PrefixText::Raw(r),
                                            PrefixText::Norm { buf, range } => PrefixText::Norm {
                                                buf: buf + buf_base,
                                                range,
                                            },
                                        };
                                        units.push(OwnedUnit {
                                            seq,
                                            idx,
                                            work: OwnedWork::Spans {
                                                text,
                                                spans: sr.start + span_base..sr.end + span_base,
                                            },
                                        });
                                    }
                                }
                            }
                            continue;
                        }
                    }
                    let idx = counts[seq];
                    counts[seq] += 1;
                    units.push(OwnedUnit {
                        seq,
                        idx,
                        work: OwnedWork::Full(0..input.len()),
                    });
                }
            }
            _ => {
                for (seq, &input) in refs.iter().enumerate() {
                    counts[seq] = 1;
                    units.push(OwnedUnit {
                        seq,
                        idx: 0,
                        work: OwnedWork::Full(0..input.len()),
                    });
                }
            }
        }

        // Too little schedulable work to be worth the pool.
        if units.len() < 2 && preresolved.is_empty() {
            return EncodeJob::ready(self.encode_serial(&refs, add_special_tokens));
        }
        drop(refs);

        // LPT: claim the largest units first so no giant chunk lands last
        // (straggler tail); the (seq, idx) tags keep reassembly order-free.
        units.sort_by_key(|unit| {
            std::cmp::Reverse(match &unit.work {
                OwnedWork::Full(range) => range.len(),
                OwnedWork::Spans { spans: sr, .. } => {
                    let group = &spans[sr.clone()];
                    group
                        .last()
                        .map(|last| (last.end - group[0].start) as usize)
                        .unwrap_or(0)
                }
            })
        });

        let (tx, rx) = mpsc::channel::<ChunkMsg>();
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
            add_special_tokens,
            cursor: AtomicUsize::new(0),
            cancelled: AtomicBool::new(false),
            tx,
            waker: Mutex::new(None),
        });

        // Spawn one cursor-drainer per potential worker ('static: each holds an
        // Arc of the core). Drainers that find the cursor exhausted are cheap
        // no-ops.
        let drainers = n_units.min(p.current_num_threads());
        for _ in 0..drainers {
            let core = Arc::clone(&core);
            p.spawn(move || {
                SCRATCH.with(|st| {
                    let scratch = &mut *st.borrow_mut();
                    while core.run_one(scratch) {}
                })
            });
        }

        EncodeJob::streaming(core, rx, counts)
    }

    /// Scoped-path scheduling on the library-private rayon pool: the chunks sit
    /// behind one shared claim cursor, drained by `min(chunks, threads)`
    /// scope-spawned drainer tasks (warm TLS scratch) **and** by the caller's
    /// handle (caller-assist) — the same shared-cursor job model as the owned
    /// path. `in_place_scope` joins every drainer before returning, so the
    /// borrowed chunks never escape.
    fn encode_rayon<R>(
        &self,
        p: &rayon::ThreadPool,
        items: &[Item<'_>],
        counts: Vec<usize>,
        preresolved: Vec<ChunkMsg>,
        add_special_tokens: bool,
        f: impl FnOnce(EncodeHandle<'_>) -> R,
    ) -> R {
        let (tx, rx) = mpsc::channel::<ChunkMsg>();
        // Units the serial prefix already resolved (special tokens) go straight
        // into the channel; reassembly absorbs them like any other chunk.
        for msg in preresolved {
            let _ = tx.send(msg);
        }
        let cursor = AtomicUsize::new(0);
        let cursor = &cursor;
        p.in_place_scope(|s| {
            let drainers = items.len().min(p.current_num_threads());
            for _ in 0..drainers {
                let tx = tx.clone();
                s.spawn(move |_| {
                    SCRATCH.with(|st| {
                        let scratch = &mut *st.borrow_mut();
                        while self.claim_one(items, cursor, add_special_tokens, &tx, scratch) {}
                    })
                });
            }
            let assist = ScopedAssist {
                tokenizer: self,
                items,
                cursor,
                add_special_tokens,
                tx,
            };
            f(EncodeHandle::streaming(
                StreamState::new(rx, counts).with_assist(assist),
            ))
        })
    }

    /// Claim and encode one chunk off `cursor`; `false` when the cursor is
    /// exhausted or the receiver is gone (handle consumed early — stop working).
    /// A panicking chunk is delivered as its input's `Err` and the drainer keeps
    /// going, so the consumer never hangs on a chunk that will not arrive.
    fn claim_one(
        &self,
        items: &[Item<'_>],
        cursor: &AtomicUsize,
        add_special_tokens: bool,
        tx: &mpsc::Sender<ChunkMsg>,
        scratch: &mut EncodeState,
    ) -> bool {
        let i = cursor.fetch_add(1, Ordering::Relaxed);
        if i >= items.len() {
            return false;
        }
        let item = &items[i];
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match item.work {
            Work::Full(text) => self.encode_one_with(text, add_special_tokens, scratch),
            Work::Spans { text, spans } => self.encode_spans(text, spans, scratch),
        }))
        .unwrap_or_else(|_| Err("encode worker panicked".into()));
        tx.send((item.seq, item.idx, res)).is_ok()
    }

    /// Model-only kernel for the `SplitAtModel` rung: tokenize each pre-token
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
    fn encode_serial(
        &self,
        inputs: &[&str],
        add_special_tokens: bool,
    ) -> Vec<Result<Vec<PipelineToken>>> {
        SCRATCH.with(|st| {
            let state = &mut *st.borrow_mut();
            inputs
                .iter()
                .map(|&input| self.encode_one_with(input, add_special_tokens, state))
                .collect()
        })
    }

    /// Encode a single sequence with a fresh scratch buffer (serial test oracle).
    #[cfg(test)]
    fn encode_one(&self, input: &str, add_special_tokens: bool) -> Result<Vec<PipelineToken>> {
        let mut state = EncodeState::new();
        self.encode_one_with(input, add_special_tokens, &mut state)
    }

    /// Encode a single sequence, reusing the caller's [`EncodeState`] scratch.
    fn encode_one_with(
        &self,
        input: &str,
        _add_special_tokens: bool,
        state: &mut EncodeState,
    ) -> Result<Vec<PipelineToken>> {
        state.reset();
        // ~4.3 input bytes per token measured on English corpora; /4 is a
        // conservative reserve that avoids most growth reallocations.
        let mut output = Vec::with_capacity(input.len() / 4);
        let scratch = self.inner.model.scratch_in(&mut state.model_scratch);
        self.encode_generic::<{ Self::STAGE_MODEL }>(
            input,
            &mut state.pre_tokens,
            scratch,
            &mut output,
        )?;
        Ok(output)
    }

    /// Span one-or-many inputs into a flat, ordered list of [`Chunk`]s. A large input is
    /// cut at `cut` boundaries (see [`raw_cut`](Self::raw_cut)) into
    /// ~[`PARALLEL_MIN_TOTAL_BYTES`]-sized pieces; small inputs stay whole. Each chunk keeps
    /// its `seq_id` so results reassemble per input.
    fn segment(&self, inputs: &[&str], cut: RawCut) -> Vec<Chunk> {
        let mut chunks = Vec::with_capacity(inputs.len());
        let mut tags = Vec::new();
        for (seq_id, &input) in inputs.iter().enumerate() {
            // Only bother splitting inputs big enough for at least two target-sized pieces.
            if input.len() >= 2 * PARALLEL_MIN_TOTAL_BYTES {
                let mut start = 0;
                for cut in Self::chunk_cuts(input, &mut tags, cut) {
                    chunks.push(Chunk {
                        seq_id,
                        range: start..cut,
                    });
                    start = cut;
                }
                chunks.push(Chunk {
                    seq_id,
                    range: start..input.len(),
                });
            } else {
                chunks.push(Chunk {
                    seq_id,
                    range: 0..input.len(),
                });
            }
        }
        chunks
    }

    /// Byte offsets at which to cut `input`, at `cut`-kind boundaries spaced roughly
    /// [`PARALLEL_MIN_TOTAL_BYTES`] apart.
    ///
    /// NOTE (MS-B): this scans the whole input on the caller and then *discards*
    /// the classification, so the (future `fast_split`) pre-tokenizer would
    /// classify the same bytes again on the worker. The fused MS-B executor
    /// replaces this central cut list entirely: workers claim byte offsets,
    /// classify their own chunk into `EncodeState` scratch, snap the boundary
    /// from those tags, and feed the *same* tags to pretok — one classify per
    /// byte, parallelized. See PARALLEL_ENCODE_PLAN.md § MS-B.
    ///
    /// [`RawCut::Newline`] cuts fall *after* a newline (a dropped delimiter for the
    /// whitespace-delimiting pre-tokenizers that rung permits, so cut placement
    /// doesn't change tokens); it uses the interim newline-only [`classify`] pass —
    /// the shape `fast_split` will slot into, with `tags` as caller-owned scratch.
    /// [`RawCut::SpaceRun`] cuts fall *at* an ASCII non-whitespace→space boundary
    /// (the space starts the right chunk, where the GPT regex family binds it to the
    /// following token); the previous byte is required to be ASCII printable so a
    /// cut can never land beside a multibyte character.
    fn chunk_cuts(input: &str, tags: &mut Vec<u8>, cut: RawCut) -> Vec<usize> {
        let target = PARALLEL_MIN_TOTAL_BYTES;
        let mut cuts = Vec::new();
        let mut goal = target;
        match cut {
            RawCut::Newline => {
                tags.clear();
                tags.resize(input.len(), classify::OTHER);
                classify::classify(input.as_bytes(), tags);
                for (i, &tag) in tags.iter().enumerate() {
                    // Once past the target, cut just after the next newline (a 1-byte char).
                    if i >= goal && tag == classify::NEWLINE {
                        let pos = i + 1;
                        if pos < input.len() {
                            cuts.push(pos);
                            goal = pos + target;
                        }
                    }
                }
            }
            RawCut::SpaceRun => {
                let bytes = input.as_bytes();
                for i in 1..bytes.len() {
                    if i >= goal && bytes[i] == b' ' && matches!(bytes[i - 1], 0x21..=0x7E) {
                        cuts.push(i);
                        goal = i + target;
                    }
                }
            }
        }
        cuts
    }

    /// How this config may split a single large input for intra-input
    /// parallelism (the escalation ladder, `PARALLEL_RUNTIME_DESIGN.md` §8):
    /// cheapest safe rung wins.
    fn plan(&self) -> ParallelPlan {
        if let Some(cut) = self.raw_cut() {
            ParallelPlan::SplitRaw(cut)
        } else if matches!(self.inner.model, PipelineModel::WordLevel(_)) {
            // A WordLevel "model step" is one hash lookup per pre-token — the
            // serial pre-tokenization prefix would dwarf the parallelized part.
            ParallelPlan::WholeInput
        } else {
            ParallelPlan::SplitAtModel
        }
    }

    /// Whether — and where — the config lets raw text be cut and each piece
    /// encoded independently with the same result as the whole. Requires a
    /// per-character normalizer (or none), no whitespace-absorbing or
    /// whitespace-containing added tokens, and a pre-tokenizer with a provable
    /// hard boundary: whitespace-delimiting ([`RawCut::Newline`]) or a
    /// known-safe byte-level regex ([`RawCut::SpaceRun`]). The real
    /// classify/boundary substrate (MS-B) will replace this heuristic.
    fn raw_cut(&self) -> Option<RawCut> {
        let norm_ok = self.inner.normalizer.as_ref().is_none_or(Self::norm_chunk_safe);
        if !norm_ok
            || self.inner.added_vocabulary.has_affix_strip()
            || self.inner.added_token_contains_whitespace
        {
            return None;
        }
        match &self.inner.pre_tokenizer {
            PipelinePreTokenizer::Whitespace(_)
            | PipelinePreTokenizer::WhitespaceSplit(_)
            | PipelinePreTokenizer::Bert(_) => Some(RawCut::Newline),
            pt if Self::space_cut_safe(pt) => Some(RawCut::SpaceRun),
            _ => None,
        }
    }

    /// Whether `pt` provably never yields a pre-token spanning a
    /// non-whitespace→space boundary — the invariant behind
    /// [`RawCut::SpaceRun`]. Conservative allowlist of the byte-level GPT
    /// regex family (each listed pattern has been checked branch by branch: no
    /// alternative can match a non-whitespace character directly followed by a
    /// space *inside* one token). The llama/GPT-4 pre-tokenizer shape
    /// `Sequence[Span(regex), ByteLevel(use_regex: false) → None]` is
    /// recognized too.
    fn space_cut_safe(pt: &PipelinePreTokenizer) -> bool {
        /// GPT-2's regex plus the llama-3 / cl100k-style variant.
        const SPACE_SAFE_REGEXES: &[&str] = &[
            GPT2_REGEX_STR,
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        ];
        match pt {
            PipelinePreTokenizer::Split(s) => {
                !s.invert
                    && matches!(s.behavior, SplitDelimiterBehavior::Isolated)
                    && matches!(
                        &s.pattern,
                        SplitPattern::Regex(r) if SPACE_SAFE_REGEXES.contains(&r.as_str())
                    )
            }
            PipelinePreTokenizer::Sequence(seq) => match seq.members() {
                [only] => Self::space_cut_safe(only),
                [split, PipelinePreTokenizer::None] => Self::space_cut_safe(split),
                _ => false,
            },
            _ => false,
        }
    }

    /// Run the serial prefix of the [`ParallelPlan::SplitAtModel`] rung over one
    /// input: frame → normalize → frame(normalized) → pre-tokenize, packaging
    /// the remaining model work as ~[`PARALLEL_MIN_TOTAL_BYTES`]-sized
    /// span-groups. Mirrors [`encode_generic`](Self::encode_generic) stage by
    /// stage — the two must agree on unit order for reassembly to be
    /// byte-identical to the serial encode.
    fn model_prefix(&self, input: &str) -> Result<ModelPrefix> {
        let mut prefix = ModelPrefix {
            units: Vec::new(),
            norm_bufs: Vec::new(),
            spans: Vec::new(),
        };
        let mut pre_tokens: Vec<Span> = Vec::new();
        for segment in SpecialSegmentIterator::new(input, &self.inner.added_vocabulary, false) {
            match segment {
                Segment::SpecialToken(token) => prefix.units.push(PrefixUnit::Special(token)),
                Segment::Text(chunk) => {
                    let normalized: Cow<str> = match &self.inner.normalizer {
                        Some(normalizer) => normalizer.normalize(chunk)?,
                        None => Cow::Borrowed(chunk),
                    };
                    let owned = matches!(normalized, Cow::Owned(_));
                    for seg in
                        SpecialSegmentIterator::new(&normalized, &self.inner.added_vocabulary, true)
                    {
                        match seg {
                            Segment::SpecialToken(token) => {
                                prefix.units.push(PrefixUnit::Special(token))
                            }
                            Segment::Text(nchunk) => {
                                pre_tokens.clear();
                                self.inner.pre_tokenizer.pre_tokenize(nchunk, &mut pre_tokens)?;
                                if pre_tokens.is_empty() {
                                    continue;
                                }
                                // Where this segment's text will live once the
                                // prefix is finalized (`normalized` is pushed to
                                // `norm_bufs` below, so its future index is the
                                // current length).
                                let base = if owned {
                                    offset_within(&normalized, nchunk)
                                } else {
                                    // Borrowed all the way down: nchunk is a
                                    // subslice of the raw input.
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
                                    let span_range =
                                        prefix.spans.len()..prefix.spans.len() + (e - g);
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
                }
            }
        }
        Ok(prefix)
    }

    /// Whether a normalizer preserves the newline-boundary chunking invariant: it must be
    /// per-character (no edge-strip, prefix-insert, or cross-boundary rules). A `Sequence`
    /// is safe iff every member is.
    fn norm_chunk_safe(n: &NormalizerWrapper) -> bool {
        match n {
            NormalizerWrapper::NFC(_)
            | NormalizerWrapper::NFD(_)
            | NormalizerWrapper::NFKC(_)
            | NormalizerWrapper::NFKD(_)
            | NormalizerWrapper::Lowercase(_)
            | NormalizerWrapper::BertNormalizer(_)
            | NormalizerWrapper::StripAccents(_) => true,
            NormalizerWrapper::Sequence(seq) => seq.as_ref().iter().all(Self::norm_chunk_safe),
            // Strip, Prepend, Replace, Precompiled, Nmt, ByteLevel: may cross boundaries.
            _ => false,
        }
    }

    /// Single source of truth for the encode pipeline, generic over how many stages
    /// run. `STAGE` is a **const generic**, so `if STAGE >= …` folds at compile time and
    /// the disabled stages are compiled out — the full specialization ([`STAGE_MODEL`],
    /// which [`encode`](Self::encode) calls) is branchless and identical to a
    /// hand-written full pipeline, while the benchmark drives lower `STAGE` values to
    /// time each stage's marginal cost (the ablation ladder), e.g.
    /// `model = t(MODEL) − t(SPLIT)`. No runtime gate, no `Instant` in the loop.
    ///
    /// [`STAGE_MODEL`]: Self::STAGE_MODEL
    ///
    /// `output` and the `pre_tokens` scratch are caller-owned so a benchmark can reuse
    /// them across calls and observe both buffers to anchor the ablation levels — the
    /// library itself stays free of any `black_box`/timing artifact.
    #[doc(hidden)] // public only so `examples/fixture_bench.rs` can drive partial stages
    pub fn encode_generic<const STAGE: u8>(
        &self,
        input: &str,
        pre_tokens: &mut Vec<Span>,
        scratch: &mut PipelineModelScratch,
        output: &mut Vec<PipelineToken>,
    ) -> Result<()> {
        // First, we extract all special tokens from the non-normalized input
        for segment in SpecialSegmentIterator::new(input, &self.inner.added_vocabulary, false) {
            match segment {
                Segment::SpecialToken(token) => {
                    output.push(PipelineToken { id: token });
                }
                Segment::Text(chunk) => {
                    let normalized: Cow<str> = if STAGE >= Self::STAGE_NORMALIZE {
                        match &self.inner.normalizer {
                            Some(normalizer) => normalizer.normalize(chunk)?,
                            None => Cow::Borrowed(chunk),
                        }
                    } else {
                        Cow::Borrowed(chunk)
                    };

                    // Extract special tokens from the normalized input
                    for segment in
                        SpecialSegmentIterator::new(&normalized, &self.inner.added_vocabulary, true)
                    {
                        match segment {
                            Segment::SpecialToken(token) => {
                                output.push(PipelineToken { id: token });
                            }
                            Segment::Text(normalized_chunk) => {
                                if STAGE >= Self::STAGE_SPLIT {
                                    // Pre-tokenize the chunk of normalized text
                                    pre_tokens.clear();
                                    self.inner.pre_tokenizer
                                        .pre_tokenize(normalized_chunk, pre_tokens)?;
                                    if STAGE >= Self::STAGE_MODEL {
                                        // Tokenize each chunk
                                        for pre_token in pre_tokens.iter() {
                                            self.inner.model.tokenize_pipeline(
                                                &normalized_chunk[pre_token.range()],
                                                scratch,
                                                output,
                                            )?;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            };
        }
        Ok(())
    }
}

/// What [`split`] does with each split it forms
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SplitPolicy {
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
pub fn split<C: Copy + PartialEq>(
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
pub fn split_delimiter(
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
pub fn split_matches(
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
    /// thread-local [`EncodeState`] is shared across tokenizers on a thread.
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
        assert_eq!(
            tok.raw_cut(),
            Some(RawCut::Newline),
            "wordlevel + WhitespaceSplit must be newline-chunk-safe"
        );

        let big = "aa bb cc\n".repeat(4000); // ~36 KB, newline-delimited
        assert!(big.len() > 2 * PARALLEL_MIN_TOTAL_BYTES);

        let chunks = tok.segment(&[big.as_str()], RawCut::Newline);
        assert!(
            chunks.len() > 1,
            "expected the large input to split, got {} chunk(s)",
            chunks.len()
        );

        let ids = |v: Vec<PipelineToken>| v.iter().map(|t| t.id).collect::<Vec<_>>();
        let parallel = ids(tok
            .encode_scoped(big.as_str(), false, |h| h.into_single())
            .unwrap());
        let serial = ids(tok.encode_one(big.as_str(), false).unwrap());
        assert_eq!(parallel, serial, "chunked encode must equal serial encode");

        // The owned (returnable) job must agree too — move the storage in.
        let owned = ids(tok.encode(big.clone(), false).into_single().unwrap());
        assert_eq!(owned, serial, "owned EncodeJob must equal serial encode");
    }

    /// `chunk_cuts` cuts a multi-line input just after newlines, spaced ~target apart, never
    /// at end-of-input.
    #[test]
    fn chunk_cuts_at_newlines() {
        let input = "aa bb cc\n".repeat(4000);
        let mut tags = Vec::new();
        let cuts = PipelineTokenizer::chunk_cuts(&input, &mut tags, RawCut::Newline);
        assert!(!cuts.is_empty(), "expected at least one cut");
        for &c in &cuts {
            assert!(c < input.len(), "cut must be within the input");
            assert_eq!(
                input.as_bytes()[c - 1],
                b'\n',
                "cut must fall just after a newline"
            );
        }
        for w in cuts.windows(2) {
            assert!(
                w[1] - w[0] >= PARALLEL_MIN_TOTAL_BYTES,
                "cuts spaced >= target apart"
            );
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
            .map(|s| ids(tok.encode_one(s, false).unwrap()))
            .collect();

        // Scoped, bulk collect.
        let collected = tok
            .encode_scoped(&inputs, false, |h| h.wait_for_completion())
            .unwrap();
        let collected_ids: Vec<Vec<u32>> = collected.into_iter().map(ids).collect();
        assert_eq!(collected_ids, serial, "wait_for_completion != serial");

        // Scoped, streaming iterator drained inside the scope in input order.
        let streamed: Vec<Vec<u32>> =
            tok.encode_scoped(&inputs, false, |h| h.map(|r| ids(r.unwrap())).collect());
        assert_eq!(streamed, serial, "streaming Iterator != serial");

        // Owned returnable job (the default API): the handle escapes the call and
        // is drained afterwards, both faces.
        let owned_inputs: Vec<String> = inputs.iter().map(|s| (*s).to_owned()).collect();
        let job = tok.encode(owned_inputs.clone(), false);
        let owned: Vec<Vec<u32>> = job.wait_for_completion().unwrap().into_iter().map(ids).collect();
        assert_eq!(owned, serial, "owned wait_for_completion != serial");

        let job = tok.encode(owned_inputs.clone(), false);
        let owned_streamed: Vec<Vec<u32>> = job.map(|r| ids(r.unwrap())).collect();
        assert_eq!(owned_streamed, serial, "owned Iterator != serial");

        // Dropping a job early cancels cleanly (no hang, no panic) and the pool
        // stays usable.
        let job = tok.encode(owned_inputs, false);
        drop(job);
        let again = ids(tok.encode(a.clone(), false).into_single().unwrap());
        assert_eq!(again, serial[0], "pool must stay usable after a dropped job");
    }

    #[test]
    fn encode_streams_and_matches_serial() {
        assert_encode_matches_serial(&chunk_safe_pipeline());
    }

    /// `SpaceRun` cuts land only at ASCII non-whitespace→space boundaries,
    /// spaced ~target apart, never beside a multibyte character.
    #[test]
    fn chunk_cuts_at_space_runs() {
        let input = "word, another.  且つ 更に\n".repeat(2000);
        let mut tags = Vec::new();
        let cuts = PipelineTokenizer::chunk_cuts(&input, &mut tags, RawCut::SpaceRun);
        assert!(!cuts.is_empty(), "expected at least one cut");
        let bytes = input.as_bytes();
        for &c in &cuts {
            assert_eq!(bytes[c], b' ', "cut must land on a space");
            assert!(
                matches!(bytes[c - 1], 0x21..=0x7E),
                "byte before a cut must be ASCII non-whitespace"
            );
        }
    }

    /// The GPT-2 byte-level regex qualifies for `SpaceRun` raw cuts, both bare
    /// and in the llama-shaped `Sequence[Span, None]`; a whitespace-containing
    /// added token disables raw cutting entirely.
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
        let make = |with_ws_token: bool| {
            let mut tok = Tokenizer::new(crate::models::bpe::BPE::default());
            tok.with_pre_tokenizer(Some(split()));
            if with_ws_token {
                tok.add_special_tokens([AddedToken::from("<s> extra", true)])
                    .unwrap();
            }
            PipelineTokenizer::try_from(&tok).unwrap()
        };
        assert_eq!(make(false).raw_cut(), Some(RawCut::SpaceRun));
        assert_eq!(
            make(true).raw_cut(),
            None,
            "whitespace inside an added token must disable raw cuts"
        );

        // The llama-3 shape: Sequence[Span(known regex), ByteLevel(no regex) -> None].
        let seq = PipelinePreTokenizer::Sequence(PipelineSequence::new(vec![
            PipelinePreTokenizer::Split(split()),
            PipelinePreTokenizer::None,
        ]));
        assert!(PipelineTokenizer::space_cut_safe(&seq));
        // A pre-tokenizer outside the allowlist must not qualify.
        let not_allowlisted = PipelinePreTokenizer::Punctuation(Punctuation::default());
        assert!(!PipelineTokenizer::space_cut_safe(&not_allowlisted));
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
        assert_eq!(tok.plan(), ParallelPlan::SplitRaw(RawCut::SpaceRun));

        let big = "aa bb,cc!  aa\tbb  cc\n\n".repeat(2000); // ~44 KB, mixed runs
        let chunks = tok.segment(&[big.as_str()], RawCut::SpaceRun);
        assert!(chunks.len() > 1, "expected the input to split");

        let ids = |v: Vec<PipelineToken>| v.iter().map(|t| t.id).collect::<Vec<_>>();
        let serial = ids(tok.encode_one(&big, false).unwrap());
        let scoped = ids(tok
            .encode_scoped(big.as_str(), false, |h| h.into_single())
            .unwrap());
        assert_eq!(scoped, serial, "SpaceRun scoped != serial");
        let owned = ids(tok.encode(big, false).into_single().unwrap());
        assert_eq!(owned, serial, "SpaceRun owned != serial");
    }

    /// A non-chunk-safe config: regex `Span` pre-tokenizer (unknown regex, so
    /// no raw cut qualifies) + WordPiece model, so the ladder must
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
        assert_eq!(tok.plan(), ParallelPlan::SplitAtModel);

        let big = "aa.bb.cc.".repeat(3000); // ~27 KB, punctuation-delimited
        let prefix = tok.model_prefix(&big).unwrap();
        let groups = prefix
            .units
            .iter()
            .filter(|u| matches!(u, PrefixUnit::Group { .. }))
            .count();
        assert!(groups > 1, "expected several span-groups, got {}", groups);
        assert!(prefix.norm_bufs.is_empty(), "no normalizer -> no owned bufs");

        let ids = |v: Vec<PipelineToken>| v.iter().map(|t| t.id).collect::<Vec<_>>();
        let serial = ids(tok.encode_one(&big, false).unwrap());
        let scoped = ids(tok
            .encode_scoped(big.as_str(), false, |h| h.into_single())
            .unwrap());
        assert_eq!(scoped, serial, "scoped SplitAtModel != serial");
        let owned = ids(tok.encode(big, false).into_single().unwrap());
        assert_eq!(owned, serial, "owned SplitAtModel != serial");
    }

    /// SplitAtModel with a rewriting normalizer (uppercase input + `Lowercase`
    /// forces `Cow::Owned` → the `Norm` buffer path) and an interleaved special
    /// token (preresolved units must keep their position in the stream).
    #[test]
    fn split_at_model_normalizer_and_specials_match_serial() {
        let tok = split_at_model_pipeline(true);
        assert_eq!(tok.plan(), ParallelPlan::SplitAtModel);

        let big = format!(
            "{}<s>{}",
            "AA.BB.cc.".repeat(2000),
            "CC.aa.BB.".repeat(2000)
        );
        let prefix = tok.model_prefix(&big).unwrap();
        assert!(
            !prefix.norm_bufs.is_empty(),
            "lowercased text must live in owned norm bufs"
        );
        assert!(
            prefix
                .units
                .iter()
                .any(|u| matches!(u, PrefixUnit::Special(_))),
            "the <s> special must be a preresolved unit"
        );

        let ids = |v: Vec<PipelineToken>| v.iter().map(|t| t.id).collect::<Vec<_>>();
        let serial = ids(tok.encode_one(&big, false).unwrap());
        let scoped = ids(tok
            .encode_scoped(big.as_str(), false, |h| h.into_single())
            .unwrap());
        assert_eq!(scoped, serial, "scoped != serial (norm + specials)");
        let owned = ids(tok.encode(big, false).into_single().unwrap());
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
            .map(|s| ids(tok.encode_one(s, false).unwrap()))
            .collect();

        let mut job = tok.encode(inputs, false);
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
            pipe.raw_cut().is_none(),
            "Strip normalizer must disable raw-text chunking"
        );
        // WordLevel model: the ladder cannot escalate to SplitAtModel either.
        assert_eq!(
            pipe.plan(),
            ParallelPlan::WholeInput,
            "Strip + WordLevel must fall to batch-level parallelism only",
        );
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
