//! Parallel encode runtime for [`PipelineTokenizer`].
//!
//! [`PipelineTokenizer::encode`] returns an owned [`EncodeHandle`] immediately;
//! workers encode in the background. Results surface through the handle's two
//! faces: the blocking `Iterator`, yielding `(input index, result)` in
//! completion order — each input the moment it finishes (and *assisting* — the
//! caller drains pending work instead of idling) — and
//! [`EncodeHandle::wait_for_completion`], which blocks and returns all results
//! collected back into input order.
//!
//! **Job model.** Work is split into `Item`s behind one atomic claim cursor in
//! `JobCore`; rayon pool tasks and the consuming thread both drain it, so there
//! is no scheduler — just a cursor and caller-assist. There is no result
//! channel: each worker writes its unit into a shared write-once `Slot` and
//! decrements the input's `pending` counter; the consumer reads an input's slots
//! once `pending` hits zero. Per-worker `EncodeState` scratch persists
//! (thread-local, reset-not-realloc) for cache warmth. Panics are caught per
//! unit and delivered as that input's `Err`.
//!
//! **Where the work is split.** Specials are peeled first (the outermost stage,
//! so a cut there is always safe; ~0.1% of encode, and it keeps strided segments
//! special-free). Each special-free **segment** is then split at the earliest
//! pipeline stage its config allows (the `ParallelPlan` ladder). Earlier = more
//! parallel; never wrong, only less parallel.
//!
//! **Fork safety** lives in `pool`: a `pthread_atfork` child abandons the
//! stale pool without touching it and lazily rebuilds.

use std::cell::{RefCell, UnsafeCell};
use std::convert::TryInto;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use rayon::prelude::*;
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
use crate::processors::bert::BertProcessing;
use crate::processors::roberta::RobertaProcessing;
use crate::{
    ModelWrapper, PostProcessorWrapper, PreTokenizerWrapper, Tokenizer,
    normalizers::{NormalizerWrapper, metaspace::MetaspaceNormalizer},
    pre_tokenizers::{
        bert::BertPreTokenizer,
        delimiter::CharDelimiterSplit,
        digits::Digits,
        fixed_length::FixedLength,
        metaspace,
        punctuation::Punctuation,
        sequence::PipelineSequence,
        split::{Split as SplitPretok, SplitPattern},
        unicode_scripts::UnicodeScripts,
        whitespace::{Whitespace, WhitespaceSplit},
    },
    processors::template::Piece,
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
        static TAGS: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    }
    let n = bytes.len();
    TAGS.with(|cell| {
        let tags = &mut *cell.borrow_mut();
        if tags.len() < n {
            tags.resize(n, 0); // grow-only: after the largest segment, no realloc / no re-zeroing
        }
        classify(bytes, &mut tags[..n]);
        // The fsm writes its spans straight into `out`. It used to fill a thread-local buffer
        // that was then copied here, which cost a second `Span` write for every pre-token -- and
        // a buffer as large as the input, since the fsm's worst case is one span per byte.
        // `out` is caller-owned scratch reused across calls, so it is the same memory either way.
        let base = out.len();
        out.reserve(n + 1);
        // SAFETY: `reserve` gives `n + 1` slots past `base`, which is the fsm's worst case (one
        // span per byte, plus one). `Span` is `Copy` with no drop glue, so uninitialised slots are
        // sound to hand over, and `set_len` below only counts the `k` the fsm actually wrote.
        let k = unsafe {
            fsm(
                bytes,
                &tags[..n],
                std::slice::from_raw_parts_mut(out.as_mut_ptr().add(base), n + 1),
            )
        };
        debug_assert!(k <= n + 1);
        // SAFETY: the fsm wrote `k <= n + 1` spans from `base`.
        unsafe { out.set_len(base + k) };
    });
}

/// [`classify_into_spans`] for a splitter that works off bitstreams: it needs two `u64` bitmaps
/// (token starts, and the flags its scalar escapes key on) alongside the tags.
pub(crate) fn classify_into_spans_bits(
    bytes: &[u8],
    split: impl FnOnce(&[u8], &[u8], &mut [u64], &mut [u64], &mut [Span]) -> usize,
    out: &mut Vec<Span>,
) {
    thread_local! {
        static SCRATCH: RefCell<(Vec<u8>, Vec<u64>, Vec<u64>)> =
            const { RefCell::new((Vec::new(), Vec::new(), Vec::new())) };
    }
    let n = bytes.len();
    if n == 0 {
        return;
    }
    SCRATCH.with(|cell| {
        let (tags, starts, flags) = &mut *cell.borrow_mut();
        if tags.len() < n {
            tags.resize(n, 0);
        }
        let words = n.div_ceil(64) + 1;
        if starts.len() < words {
            starts.resize(words, 0);
            flags.resize(words, 0);
        }
        classify(bytes, &mut tags[..n]);
        let base = out.len();
        out.reserve(n + 1);
        // SAFETY: as in `classify_into_spans` -- `reserve` covers the splitter's worst case of one
        // span per byte plus one, `Span` has no drop glue, and `set_len` counts only what was written.
        let k = unsafe {
            split(
                bytes,
                &tags[..n],
                &mut starts[..words],
                &mut flags[..words],
                std::slice::from_raw_parts_mut(out.as_mut_ptr().add(base), n + 1),
            )
        };
        debug_assert!(k <= n + 1);
        // SAFETY: the splitter wrote `k <= n + 1` spans from `base`.
        unsafe { out.set_len(base + k) };
    });
}

pub trait Normalizer {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>>;
}

/// Runs `normalizers` in order, each one seeing what the one before it produced.
///
/// A normalizer returns a [`Cow`] (copy-on-write): a borrow when it had nothing to change, an owned
/// `String` when it rewrote the text.
///
/// Chaining them needs care, because an owned `String` produced halfway through the chain is local to this function.
/// The next normalizer may hand back a borrow of that locally owned [`String`], and that borrow cannot outlive the `String`.
///
/// Text no normalizer touches is never copied: it stays a borrow of `input` throughout.
pub(crate) fn normalize_all<'a, N: Normalizer>(
    normalizers: &[N],
    input: &'a str,
) -> Result<Cow<'a, str>> {
    let mut cow: Cow<'a, str> = Cow::Borrowed(input);
    for normalizer in normalizers {
        cow = match cow {
            // Still `input` itself, which outlives us: pass it straight on.
            Cow::Borrowed(s) => normalizer.normalize(s)?,
            Cow::Owned(s) => {
                let out = match normalizer.normalize(&s)? {
                    // Rewritten again: keep the new `String`, drop ours.
                    Cow::Owned(o) => Some(o),
                    // Handed `s` back untouched: keep the `String` we already own.
                    Cow::Borrowed(b) if b.as_ptr() == s.as_ptr() && b.len() == s.len() => None,
                    // A borrow into `s` (for example, a substring of `s`): copy it out before `s` is dropped.
                    Cow::Borrowed(b) => Some(b.to_owned()),
                };
                Cow::Owned(out.unwrap_or(s))
            }
        };
    }
    Ok(cow)
}

/// One normalization step of a [`PipelineTokenizer`]. Not every step comes from the config's
/// `normalizer` field: a `Metaspace` pre-tokenizer contributes one too, see
/// [`PipelineTokenizer::try_from`].
// `NormalizerWrapper` is the big variant, and there are only ever a couple of these per tokenizer.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
enum PipelineNormalizer {
    /// The `normalizer` field of the config, as-is.
    Declared(NormalizerWrapper),
    /// The text-rewriting half of a `Metaspace` pre-tokenizer.
    Metaspace(MetaspaceNormalizer),
}

impl Normalizer for PipelineNormalizer {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        match self {
            Self::Declared(normalizer) => normalizer.normalize(input),
            Self::Metaspace(normalizer) => normalizer.normalize(input),
        }
    }
}

impl PipelineNormalizer {
    /// Whether cutting the input and normalizing the halves separately gives what
    /// normalizing the whole would -- the precondition for the `Raw` parallel plan.
    fn preserves_stride_boundaries(&self) -> bool {
        match self {
            Self::Declared(normalizer) => normalizer.preserves_stride_boundaries(),
            // Metaspace rewrites spaces to `▁` and may prepend one, so a cut on
            // whitespace stops being a cut on whitespace: the `Normalized` plan
            // (serial normalize, then stride) takes these instead.
            Self::Metaspace(_) => false,
        }
    }
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
    /// Return the [`StrideBoundary`] predicate for this pre-tokenizer, if it has one.
    /// The predicate determines the safe cutting points in the input, such that:
    /// ```text
    /// let (a, b) = input.split(input.len() / 2);
    /// pre_tokenize(input) == pre_tokenize(a) + pre_tokenize(b)
    /// ```
    pub(crate) fn stride_boundary(&self) -> Option<StrideBoundary> {
        match self {
            // Whitespace-delimiting: whitespace is a dropped delimiter, so any
            // whitespace character is a safe cut (see [`boundary_at_whitespace`]).
            Self::Whitespace(_) | Self::WhitespaceSplit(_) | Self::Bert(_) => {
                Some(boundary_at_whitespace)
            }
            // atomsplit-FSM byte-level families: cut at a space after non-ws, or
            // a newline after a word character (see [`boundary_fsm`]).
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

/// A post-processor compiled to a prefix and a suffix (slices of token IDs)
/// The prefix and suffix are respectively prepended and appended to the sequence encoding:
/// The default (both slices empty) is the no-post-processor case.
/// Processors that don't reduce to such a frame are rejected at conversion.
///
/// Example:
/// ```text
///     PipelinePostProcessor {
///         prefix: vec![100].into_boxed_slice(),
///         suffix: vec![101, 102].into_boxed_slice()
///     };
///
///     [CLS] The quick Brown fox  [SEP]
///     <100>|  <3> <4> <19> <67> | <101> <102>
///   prefix |  sequence encoding | suffix
/// ```
///
#[derive(Debug, Default)]
pub struct PipelinePostProcessor {
    prefix: Box<[PipelineToken]>,
    suffix: Box<[PipelineToken]>,
}

impl TryFrom<&PostProcessorWrapper> for PipelinePostProcessor {
    type Error = crate::Error;

    fn try_from(value: &PostProcessorWrapper) -> Result<Self> {
        match value {
            PostProcessorWrapper::Bert(BertProcessing {
                cls: (_, cls_id),
                sep: (_, sep_id),
            }) => Ok(Self {
                prefix: vec![PipelineToken { id: *cls_id }].into_boxed_slice(),
                suffix: vec![PipelineToken { id: *sep_id }].into_boxed_slice(),
            }),
            PostProcessorWrapper::Roberta(RobertaProcessing {
                cls: (_, cls_id),
                sep: (_, sep_id),
                ..
            }) => Ok(Self {
                prefix: vec![PipelineToken { id: *cls_id }].into_boxed_slice(),
                suffix: vec![PipelineToken { id: *sep_id }].into_boxed_slice(),
            }),
            PostProcessorWrapper::Template(pp) => {
                // todo: handle pair template
                let mut prefix = vec![];
                let mut suffix = vec![];
                let mut seen_sequence = false;
                for piece in pp.single.iter_pieces() {
                    match piece {
                        Piece::Sequence { .. } => {
                            if seen_sequence {
                                return Err(
                                    "post-processor not supported: Template `single` references the sequence more than once"
                                        .into(),
                                );
                            }
                            seen_sequence = true;
                        }
                        Piece::SpecialToken {
                            id: token_string, ..
                        } => {
                            let special = pp.get_special_tokens().0.get(token_string).ok_or_else(|| {
                                format!(
                                    "post-processor not supported: Template references unknown special token `{token_string}`"
                                )
                            })?;
                            let token_ids = special.ids().iter().map(|&id| PipelineToken { id });
                            if seen_sequence {
                                suffix.extend(token_ids);
                            } else {
                                prefix.extend(token_ids);
                            }
                        }
                    }
                }
                if !seen_sequence {
                    return Err(
                        "post-processor not supported: Template `single` does not reference the sequence"
                            .into(),
                    );
                }
                Ok(Self {
                    prefix: prefix.into_boxed_slice(),
                    suffix: suffix.into_boxed_slice(),
                })
            }
            PostProcessorWrapper::ByteLevel(_) => Ok(Self::default()),
            PostProcessorWrapper::Sequence(sequence) => {
                // Each member wraps the previous members' output, so later members end up
                // outermost: prefix accumulates in reverse member order, suffix in forward.
                let items = sequence
                    .as_ref()
                    .iter()
                    .map(PipelinePostProcessor::try_from)
                    .collect::<Result<Vec<_>>>()?;
                let prefix: Vec<_> = items
                    .iter()
                    .rev()
                    .flat_map(|item| item.prefix.iter().copied())
                    .collect();
                let suffix: Vec<_> = items
                    .iter()
                    .flat_map(|item| item.suffix.iter().copied())
                    .collect();
                Ok(Self {
                    prefix: prefix.into_boxed_slice(),
                    suffix: suffix.into_boxed_slice(),
                })
            }
        }
    }
}

/// An output token. Carries only the vocabulary `id`, since offsets and the token
/// string are dropped, which is all an encode-only caller needs.
#[derive(Debug, Clone, Copy)]
pub struct PipelineToken {
    pub id: u32,
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
    /// Ordinary text still to be (optionally) normalized, pre-tokenized and run through the model.
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

/// Main tokenizer struct
///
/// Thread-safe because immutable after construction
/// and cheap clones thanks to arc wrapped internals
#[derive(Clone)]
pub struct PipelineTokenizer {
    inner: Arc<PipelineInner>,
}

struct PipelineInner {
    /// Identifies this tokenizer to a thread's [`EncodeState`], see
    /// [`EncodeState::scratch_for`]. Arc clones share it -- they share the model.
    id: u64,
    added_vocabulary: BucketAddedVocabulary,
    normalizers: Vec<PipelineNormalizer>,
    pre_tokenizer: PipelinePreTokenizer,
    model: PipelineModel,
    post_processor: PipelinePostProcessor,
    /// Whether some `normalized`-form added token can't survive a stride cut —
    /// either its content carries an internal [`StrideBoundary`] cut, or it is an
    /// affix (`lstrip`/`rstrip`) token whose absorbed adjacent whitespace a cut
    /// could split. Its presence disables `Raw`/`Normalized`. Only
    /// `normalized` tokens count — the special split peels raw-matched ones
    /// before any striding. Precomputed at build; `plan()` reads it per `encode`,
    /// so it must not re-scan the vocab.
    normalized_added_token_blocks_stride: bool,
}

// comptime verification that PipelineTokenizer is Send + Sync
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PipelineTokenizer>();
};

/// Reusable per-encode scratch, kept alive across sequences processed by the
/// same worker thread (`reset`, never realloc'd).
#[derive(Default)]
pub(crate) struct EncodeState {
    pre_tokens: Vec<Span>,
    /// Model-specific heap buffers. Deliberately not cleared by `reset`:
    /// persistence across sequences and encode calls is the point.
    model_scratch: Option<PipelineModelScratch>,
    /// Which tokenizer `model_scratch` was built for, see [`Self::scratch_for`].
    owner: Option<u64>,
}

impl EncodeState {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Clear the scratch for reuse on the next sequence, keeping capacity.
    pub(crate) fn reset(&mut self) {
        self.pre_tokens.clear();
    }

    /// The model scratch for `inner`, rebuilt whenever this thread last encoded with a
    /// *different* tokenizer.
    ///
    /// Matching on the model's kind is not enough: the scratch carries a word cache keyed
    /// on the word's bytes alone, so lending it to another BPE would answer with that
    /// tokenizer's ids. One `EncodeState` is shared by every tokenizer on the thread, so
    /// identity is what has to match.
    // ponytail: alternating two tokenizers on one thread rebuilds (and so re-warms) the
    // scratch every call. Keep a small per-owner map if that pattern ever shows up.
    /// Returns the pre-token buffer alongside it, so a caller that needs both gets two
    /// disjoint borrows out of one check.
    fn scratch_for(
        &mut self,
        id: u64,
        model: &PipelineModel,
    ) -> (&mut Vec<Span>, &mut PipelineModelScratch) {
        if self.owner != Some(id) {
            self.model_scratch = Some(model.init_scratch());
            self.owner = Some(id);
        }
        (&mut self.pre_tokens, self.model_scratch.as_mut().unwrap())
    }
}

/// Source of [`PipelineInner::id`]: a fresh value per built tokenizer, never reused, so
/// an [`EncodeState`] can tell "same tokenizer" from "a new one at the same address".
static NEXT_PIPELINE_ID: AtomicU64 = AtomicU64::new(0);

/// Total input bytes below which `encode` runs inline on the calling thread (no pool):
/// under this, the pool's dispatch/wakeup cost outweighs the parallelism gain. The same
/// threshold sets the stride size for raw-cut chunking and the span-group target for
/// `Pretokenized`, so it is the single "unit of parallel work" knob. Bench-tunable —
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
fn stride_range(text: &str, index: usize, boundary: StrideBoundary) -> Option<Range<usize>> {
    const STRIDE: usize = PARALLEL_MIN_TOTAL_BYTES;
    let len = text.len();
    let start = if index == 0 {
        0
    } else {
        boundary_in_window(text, index * STRIDE, ((index + 1) * STRIDE).min(len), boundary)?
    };
    let mut window = index + 1;
    let end = loop {
        let lo = window * STRIDE;
        if lo >= len {
            break len;
        }
        if let Some(b) = boundary_in_window(text, lo, ((window + 1) * STRIDE).min(len), boundary) {
            break b;
        }
        window += 1;
    };
    (start < end).then_some(start..end)
}

/// A safe-cut predicate for one pre-tokenizer family, expressed over the
/// `atomsplit` **tag stream** (the same SIMD [`classify`] substrate the FSM
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

/// [`StrideBoundary`] for whitespace-delimiting pre-tokenizers (`Whitespace`,
/// `WhitespaceSplit`, `Bert`): any whitespace char is a safe cut — these
/// splitters drop whitespace and never emit a token spanning it, so each side
/// drops its own run. The cut joins the right chunk and stays on a char boundary
/// (a ws lead is `WS`, its continuation `Cont`).
fn boundary_at_whitespace(window_tags: &[u8]) -> Option<usize> {
    (1..window_tags.len()).find(|&i| in_mask(window_tags[i], mask::WS))
}

/// Side-independent cut for the atomsplit-FSM byte-level families (gpt2 / cl100k
/// / o200k / deepseek), with the cut char `cur` joining the right chunk. Four
/// safe `prev`→`cur` transitions (each verified branch-by-branch against
/// `atomsplit::regexes`): non-ws→whitespace, word ([`NEWLINE_PREV`])→newline,
/// non-number ([`NUM_START_PREV`])→number, number→letter. The number transitions
/// can fall inside a special, so they are sound *only* because specials are
/// peeled before striding (a strided segment is special-free); normalized-form
/// added tokens still need the stride guard.
fn safe_fsm_cut(prev: u8, cur: u8) -> bool {
    (in_mask(cur, WS_NON_NEWLINE) && in_mask(prev, NON_WS_PREV))
        || (in_mask(cur, mask::NEWLINE) && in_mask(prev, NEWLINE_PREV))
        || (in_mask(cur, mask::NUMBER) && in_mask(prev, NUM_START_PREV))
        || (in_mask(cur, mask::LETTER) && in_mask(prev, mask::NUMBER))
}

/// [`StrideBoundary`] for the atomsplit-FSM byte-level families: the first
/// [`safe_fsm_cut`] transition in the window. The previous character is read
/// through [`prev_char_tag`], so every branch also fires after a multibyte
/// character (e.g. CJK).
fn boundary_fsm(window_tags: &[u8]) -> Option<usize> {
    (1..window_tags.len()).find(|&i| safe_fsm_cut(prev_char_tag(window_tags, i), window_tags[i]))
}

/// Whether `token`'s own bytes contain a cut of `boundary` strictly inside it —
/// i.e. a stride could split this token and each half would mis-frame it. Used
/// by `normalized_added_token_blocks_stride` to disqualify striding for a `normalized`
/// added token whose content carries an internal cut (e.g. a `_→0` number
/// transition). The special split peels raw-matched tokens before striding, so
/// only normalized-form tokens reach this check.
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
/// Block edges are snapped to char boundaries ([`classify`] must see complete
/// characters). The snap is a pure function of the text, so adjacent workers
/// still agree; the extra bytes a snap adds are continuation bytes, which tag
/// as `Cont` and can never satisfy a boundary predicate.
fn boundary_in_window(text: &str, lo: usize, hi: usize, boundary: StrideBoundary) -> Option<usize> {
    // Per-thread classify scratch: overwritten (clear + resize + classify) on
    // every block before it is read, so nothing carries across calls — it is
    // kept only to reuse the allocation. Distinct from the pre-tokenizer's own
    // classify scratch (`classify_into_spans`); the two never share a buffer.
    thread_local! {
        static TAGS: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    }
    const BLOCK: usize = 1024;
    debug_assert!(1 <= lo && lo <= hi && hi <= text.len());
    TAGS.with(|cell| {
        let tags = &mut *cell.borrow_mut();
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
    })
}

/// How one special-free segment is split, chosen per config (the special split
/// already peeled the specials). The pipeline splits at the earliest stage the
/// config allows; earlier is more parallel. Never wrong, only less parallel.
#[derive(Clone, Copy, Debug)]
enum ParallelPlan {
    /// Stride the raw text; each chunk runs the full pipeline (`encode_one_with`).
    /// Needs a boundary-preserving normalizer.
    Raw(StrideBoundary),
    /// Normalize the segment serially, then stride the normalized buffer and
    /// parallelize pre-tokenize + model (`encode_normalized`). The serial
    /// normalize takes the normalizer out of the equation, so this needs only a
    /// splittable pre-tokenizer. Used when the normalizer rules out `Raw`.
    Normalized(StrideBoundary),
    /// Normalize + pre-tokenize serially, then parallelize the model over
    /// ~`PARALLEL_MIN_TOTAL_BYTES`-sized span-groups. Valid for any config.
    Pretokenized,
    /// No intra-input split.
    Whole,
}

/// One special-free segment reduced to model-ready work by the `Pretokenized`
/// plan: normalize, frame normalized specials, and pre-tokenize are done; the
/// model stage remains, packaged as span-groups. Index/range-based, no borrows.
struct PretokenizedSegment {
    /// Ordered pipeline units for this segment.
    units: Vec<SegmentUnit>,
    /// The owned normalized segment ([`SegmentText::Norm`] points here) — `Some`
    /// only when the normalizer rewrote this chunk. One chunk is normalized once,
    /// so there is at most one buffer (all `Norm` units point at it).
    norm_buf: Option<String>,
    /// Flat pool of pre-token spans; groups reference sub-ranges of it.
    spans: Vec<Span>,
}

enum SegmentUnit {
    /// A matched special/added token
    Special(u32),
    /// A group of consecutive pre-token spans over one text segment.
    Group {
        text: SegmentText,
        spans: Range<usize>,
    },
}

/// Where a span-group's text lives. Spans are relative to the resolved text.
enum SegmentText {
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
    /// is the cache-warmth contract of [`EncodeState`].
    static SCRATCH: RefCell<EncodeState> = RefCell::new(EncodeState::new());
}

/// A write-once result cell for one unit. Exactly one worker writes it — the one
/// the cursor handed the unit to — then publishes by decrementing the input's
/// `pending` counter. The consumer reads it only after seeing `pending == 0`
/// (acquire), which happens-after every write. That single-writer / published-
/// read contract is what makes the `unsafe impl Sync` sound.
struct Slot(UnsafeCell<Option<Result<Vec<PipelineToken>>>>);
unsafe impl Sync for Slot {}
impl Slot {
    fn empty() -> Self {
        Self(UnsafeCell::new(None))
    }
    /// SAFETY: caller holds the sole claim on this unit (via the cursor).
    unsafe fn set(&self, tokens: Result<Vec<PipelineToken>>) {
        // SAFETY: as above -- the caller's claim makes us the sole writer.
        unsafe { *self.0.get() = Some(tokens) };
    }
    /// Consumer-side take, valid once the owning input has completed.
    fn take(&self) -> Option<Result<Vec<PipelineToken>>> {
        // SAFETY: input complete; the single consumer is the only reader now.
        unsafe { &mut *self.0.get() }.take()
    }
}

enum HandleInner {
    /// Fully computed; yields `(input index, result)` in input order.
    Ready(std::iter::Enumerate<std::vec::IntoIter<Result<Vec<PipelineToken>>>>),
    /// Filled by pool workers writing shared slots; the calling thread assists
    /// and surfaces each input in completion order the moment its chunks are in.
    Streaming {
        core: Arc<JobCore>,
        state: StreamState,
    },
}

/// Completion-order streaming cursor over a job's shared slots. No channel and no
/// per-unit messages: the calling thread assists the pool (claims and encodes
/// pending units) until the shared cursor is drained, and surfaces each input
/// in completion order (via `completed_order`) the moment its chunks are all in.
struct StreamState {
    /// Next completion slot to surface — an index into
    /// [`JobCore::completed_order`], not an input index: inputs are yielded in
    /// the order they *finish*.
    next_k: usize,
    /// Total inputs.
    n: usize,
    /// Set once the cursor is exhausted: nothing left to assist, so the blocking
    /// faces stop trying to claim and just wait on the last in-flight units.
    assist_done: bool,
}

impl StreamState {
    fn new(n: usize) -> Self {
        Self {
            next_k: 0,
            n,
            assist_done: false,
        }
    }

    /// Surface the next input to *complete* (completion order), assisting the
    /// pool while chunks are still in flight. `None` once every input is out.
    /// The worker that finishes an input logs its `seq` in `completed_order`; we
    /// read that log in order, so a fast input never waits behind a slow
    /// earlier-index one.
    fn next_completed(&mut self, core: &JobCore) -> Option<(usize, Result<Vec<PipelineToken>>)> {
        if self.next_k >= self.n {
            return None;
        }
        loop {
            let seq = core.completed_order[self.next_k].load(Ordering::Acquire);
            if seq != NOT_DONE {
                self.next_k += 1;
                return Some((seq, core.take_result(seq)));
            }
            // the caller has to wait for its results, so instead of spinning idle or parking the thread,
            // we do some work which is ~equivalent to waiting: added benefit of not having
            // to add extra machinery to wake the caller or implement the spin loop; we simply
            // reuse our worker's job function
            if !self.assist_done {
                if SCRATCH.with(|st| core.run_one(&mut st.borrow_mut())) {
                    continue;
                }
                self.assist_done = true;
            }
            std::hint::spin_loop();
        }
    }
}

/// Sentinel for an unfilled [`JobCore::completed_order`] slot.
const NOT_DONE: usize = usize::MAX;

/// Input storage an [`EncodeHandle`] keeps alive for its whole lifetime — the
/// contract that makes the returnable handle sound with no scoped lifetimes:
/// implementors promise every `&str` returned by `get` stays valid
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
/// inputs convert for free via move; `&str`-family inputs pay one copy at
/// the boundary
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
/// One owned encode call's worth of work, shared with the pool workers via
/// `Arc`. Fully safe: the job owns its storage and a tokenizer handle, and the
/// chunks are byte ranges into the storage — nothing borrows the caller, so the
/// job outlives the `encode` call freely. Dropping the [`EncodeHandle`] sets
/// `cancelled`; workers stop claiming, in-flight chunks finish into their slots,
/// and the last `Arc` holder releases the storage.
/// Isolates a hot atomic on its own cache line. `JobCore.cursor` is `fetch_add`'d
/// once per unit by every worker; without padding it could share a line with the
/// read-only fields workers also read per unit, so each claim would invalidate
/// their caches (false sharing).
#[repr(align(64))]
struct CachePadded<T>(T);

struct JobCore {
    storage: Box<dyn Inputs>,
    /// Ordered worker units, (seq, idx)-tagged for reassembly.
    units: Vec<Item>,
    /// Owned normalized segments for `Pretokenized` units
    /// ([`SegmentText::Norm`] points here).
    norm_bufs: Vec<String>,
    /// Flat pool of pre-token spans for `Pretokenized` units.
    spans: Vec<Span>,
    tokenizer: PipelineTokenizer,
    /// Whether [`take_result`](JobCore::take_result) frames each input with the
    /// post-processor's special tokens. Per input, so it can only be applied there.
    add_special_tokens: bool,
    cursor: CachePadded<AtomicUsize>,
    cancelled: AtomicBool,
    /// `slots[seq][idx]` — one write-once result cell per chunk of each input.
    slots: Vec<Vec<Slot>>,
    /// `pending[seq]` — chunks of input `seq` not yet written; zero ⇒ the input
    /// is complete and its slots are safe to read. Distinct counters per input,
    /// so worker completions don't contend on one line (only the cursor does).
    pending: Vec<AtomicUsize>,
    /// Completion log: the worker that drives an input's `pending` to zero
    /// claims the next `completed_slot` and stores its `seq` here, so the
    /// consumer can surface inputs in completion order. `NOT_DONE` until filled.
    completed_order: Vec<AtomicUsize>,
    /// Claim counter for the next free `completed_order` slot.
    completed_slot: CachePadded<AtomicUsize>,
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
        /// The special-free segment (byte range in the source) this stride tiles.
        /// Strides index from the segment start, so each segment is strided
        /// independently of the specials around it.
        seg: Range<usize>,
    },
}

impl Region {
    /// Tile a special-free `seg` (byte range within its source) into work
    /// regions: fixed ~`PARALLEL_MIN_TOTAL_BYTES` fused strides (workers
    /// boundary-snap them), or a single `Full` region when the segment is too
    /// small to be worth striding. The shared "stride-or-whole" decision for the
    /// raw (`Raw`) and normalized (`Normalized`) paths.
    fn tile(seg: Range<usize>, boundary: StrideBoundary) -> impl Iterator<Item = Region> {
        let strided = seg.len() >= 2 * PARALLEL_MIN_TOTAL_BYTES;
        let n = if strided {
            seg.len().div_ceil(PARALLEL_MIN_TOTAL_BYTES)
        } else {
            1
        };
        (0..n).map(move |index| {
            if strided {
                Region::Stride {
                    index,
                    boundary,
                    seg: seg.clone(),
                }
            } else {
                Region::Full(seg.clone())
            }
        })
    }

    /// Resolve to an absolute byte range within `text`. `None` when a stride's
    /// window has no boundary — that stride yields an empty result, keeping the
    /// per-input chunk count static for reassembly. A stride resolves against its
    /// own `seg` sub-slice, then shifts the result back to `text` coordinates.
    fn resolve(&self, text: &str) -> Option<Range<usize>> {
        match self {
            Region::Full(r) => Some(r.clone()),
            Region::Stride {
                index,
                boundary,
                seg,
            } => stride_range(&text[seg.clone()], *index, *boundary)
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
    /// already-normalized buffer `JobCore::norm_bufs[buf]` (`Normalized`).
    Normalized { buf: usize, region: Region },
    /// Model-only over a span-group (the `Pretokenized` plan): `text` resolves
    /// against input `seq` ([`SegmentText::Raw`]) or `JobCore::norm_bufs`
    /// ([`SegmentText::Norm`]); `spans` indexes `JobCore::spans`.
    Pretokenized {
        text: SegmentText,
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
        let i = self.cursor.0.fetch_add(1, Ordering::Relaxed);
        if i >= self.units.len() {
            return false;
        }
        let unit = &self.units[i];
        let seq = unit.seq;
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match &unit.work {
            Work::Raw(region) => {
                let text = self.storage.get(seq);
                match region.resolve(text) {
                    Some(range) => self.tokenizer.encode_one_with(&text[range], scratch),
                    None => Ok(Vec::new()),
                }
            }
            Work::Normalized { buf, region } => {
                let text = &self.norm_bufs[*buf];
                match region.resolve(text) {
                    Some(range) => {
                        let mut output = Vec::with_capacity(range.len() / 4);
                        self.tokenizer
                            .encode_normalized(&text[range], scratch, &mut output)
                            .map(|()| output)
                    }
                    None => Ok(Vec::new()),
                }
            }
            Work::Pretokenized { text, spans } => {
                let text = match text {
                    SegmentText::Raw(r) => &self.storage.get(seq)[r.clone()],
                    SegmentText::Norm { buf, range } => &self.norm_bufs[*buf][range.clone()],
                };
                self.tokenizer
                    .encode_spans(text, &self.spans[spans.clone()], scratch)
            }
        }))
        .unwrap_or_else(|_| Err("encode worker panicked".into()));
        // Publish: write our slot, then decrement the input's pending count.
        // SAFETY: the cursor handed unit `i` to this thread alone, so we are the
        // sole writer of `slots[seq][idx]`.
        unsafe { self.slots[seq][unit.idx].set(res) };
        // AcqRel so the thread that drives `pending` to zero has acquired every
        // other chunk's write (release sequence on `pending`); it then logs the
        // completion, whose release publishes the whole input to the consumer.
        if self.pending[seq].fetch_sub(1, Ordering::AcqRel) == 1 {
            self.mark_done(seq);
        }
        true
    }

    /// Log `seq` in `completed_order` — called once, by whoever drives its
    /// `pending` to zero (a worker, or the builder for all-special / empty
    /// inputs). Claims the next completion slot and releases `seq` into it.
    fn mark_done(&self, seq: usize) {
        let pos = self.completed_slot.0.fetch_add(1, Ordering::Relaxed);
        self.completed_order[pos].store(seq, Ordering::Release);
    }

    /// Concatenate input `seq`'s chunks in idx order, or its first error. Called
    /// once per input by the single consumer, after `pending[seq] == 0`. A large
    /// many-chunk input (a strided single document) would spend most of its
    /// wall-time in this copy while the pool idles, so it is committed in
    /// parallel: disjoint offsets → disjoint writes into the output buffer.
    fn take_result(&self, seq: usize) -> Result<Vec<PipelineToken>> {
        self.take_chunks(seq)
            .map(|tokens| self.tokenizer.frame(tokens, self.add_special_tokens))
    }

    /// [`take_result`](Self::take_result) without the post-processor framing.
    fn take_chunks(&self, seq: usize) -> Result<Vec<PipelineToken>> {
        let row = &self.slots[seq];
        // One chunk (small / whole-unit inputs): hand back its Vec, no concat.
        if row.len() == 1 {
            return row[0].take().unwrap_or(Ok(Vec::new()));
        }
        // Move each chunk out (cheap — just Vec headers); first error wins.
        let mut chunks: Vec<Vec<PipelineToken>> = Vec::with_capacity(row.len());
        for slot in row {
            match slot.take() {
                Some(Ok(t)) => chunks.push(t),
                Some(Err(e)) => return Err(e),
                None => chunks.push(Vec::new()),
            }
        }
        let total: usize = chunks.iter().map(Vec::len).sum();

        // Small enough (or no pool): serial concat.
        const PAR_COMMIT_MIN: usize = 64 * 1024;
        let Some(pool) = (total >= PAR_COMMIT_MIN).then(pool::rayon).flatten() else {
            let mut out = Vec::with_capacity(total);
            for c in &chunks {
                out.extend_from_slice(c);
            }
            return Ok(out);
        };

        // Parallel scatter. Prefix-sum offsets, then each chunk copies into its
        // own disjoint window of the output.
        let mut offsets = Vec::with_capacity(chunks.len());
        let mut acc = 0usize;
        for c in &chunks {
            offsets.push(acc);
            acc += c.len();
        }
        let mut out: Vec<PipelineToken> = Vec::with_capacity(total);
        let base = out.as_mut_ptr() as usize;
        pool.install(|| {
            chunks
                .par_iter()
                .zip(offsets.par_iter())
                .for_each(|(chunk, &off)| {
                    // SAFETY: offsets are a prefix sum of the chunk lengths, so
                    // the windows are disjoint and all lie within `total`
                    // (reserved above); no two tasks touch the same element.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            chunk.as_ptr(),
                            (base as *mut PipelineToken).add(off),
                            chunk.len(),
                        );
                    }
                });
        });
        // SAFETY: the scatter wrote every element of `0..total`.
        unsafe { out.set_len(total) };
        Ok(out)
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

/// An owned, escapable encode — what [`PipelineTokenizer::encode`] returns.
/// Workers keep encoding in the background while the caller holds this; results
/// surface through:
/// - the blocking `Iterator`, yielding `(input index, result)` in **completion
///   order** — each input the moment it finishes (and *assisting*: the calling
///   thread claims and encodes pending chunks instead of idling),
/// - [`EncodeHandle::wait_for_completion`], which blocks and returns all results
///   collected back into **input order**.
///
/// Dropping the job cancels all unclaimed work; chunks already being encoded
/// finish into slots nobody reads. Worker panics surface as the input's `Err`.
pub struct EncodeHandle {
    inner: HandleInner,
}

impl EncodeHandle {
    fn ready(results: Vec<Result<Vec<PipelineToken>>>) -> Self {
        Self {
            inner: HandleInner::Ready(results.into_iter().enumerate()),
        }
    }

    fn streaming(core: Arc<JobCore>, n: usize) -> Self {
        Self {
            inner: HandleInner::Streaming {
                core,
                state: StreamState::new(n),
            },
        }
    }

    /// Number of inputs the handle will process
    fn len(&self) -> usize {
        match &self.inner {
            HandleInner::Ready(it) => it.len(),
            HandleInner::Streaming { state, .. } => state.n,
        }
    }

    /// Block until all inputs are encoded, results are ordered in input order.
    /// Fails fast: returns the first error it receives.
    pub fn wait_for_completion(self) -> Result<Vec<Vec<PipelineToken>>> {
        // XXX: `Vec::new` does not allocate when capacity == 0
        let mut out: Vec<Vec<PipelineToken>> = vec![Vec::new(); self.len()];
        for (seq, res) in self {
            out[seq] = res?;
        }
        Ok(out)
    }
}

impl Iterator for EncodeHandle {
    /// `(input index, that input's tokens or error)`. The index is the position
    /// in the input batch (always `0` for a single input); results arrive in
    /// completion order — each input the moment its chunks are all in.
    type Item = (usize, Result<Vec<PipelineToken>>);
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            HandleInner::Ready(it) => it.next(),
            HandleInner::Streaming { core, state } => state.next_completed(core),
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

impl TryFrom<&Tokenizer> for PipelineTokenizer {
    type Error = super::Error;

    /// Build a pipeline from an existing [`Tokenizer`], cloning its components.
    ///
    /// The base [`Tokenizer`] carries the legacy `crate::AddedVocabulary`; the pipeline uses the
    /// fast bucket `BucketAddedVocabulary`, so we rebuild it from the tokenizer's added tokens.
    /// Adding them in id order preserves ids (tokens present in the model reuse their model id, the
    /// rest keep their dense order), so the pipeline emits the same ids as the reference tokenizer.
    fn try_from(tok: &Tokenizer) -> Result<Self> {
        let mut normalizers = Vec::new();
        if let Some(declared) = tok.get_normalizer() {
            normalizers.push(PipelineNormalizer::Declared(declared.clone()));
        }

        // A `Metaspace` pre-tokenizer does two jobs at once: it writes `▁` delimiters into the text,
        // then cuts on them. The pipeline keeps rewriting and cutting apart, so we rebuild it as a
        // normalizer plus a `Split`. That normalizer runs after the declared one, matching the order
        // the config asks for: the whole normalizer first, then the pre-tokenizer.
        let pre_tokenizer = match metaspace::to_normalizer_and_split(tok.get_pre_tokenizer()) {
            Some((metaspace_normalizer, split)) => {
                // One shift this brings: added tokens flagged `normalized` are matched against text
                // that already carries the delimiters, so such a token containing a space would stop
                // matching. The t5 and albert configs we test have no normalized added token at all.
                normalizers.push(PipelineNormalizer::Metaspace(metaspace_normalizer));
                PipelinePreTokenizer::Split(split)
            }
            // Every other pre-tokenizer converts on its own.
            None => tok
                .get_pre_tokenizer()
                .cloned()
                .map(TryInto::try_into)
                .transpose()?
                .unwrap_or(PipelinePreTokenizer::None),
        };

        let legacy_av = tok.get_added_vocabulary();
        let mut added_tokens: Vec<_> = legacy_av.get_added_tokens_decoder().iter().collect();
        added_tokens.sort_by_key(|(id, _)| **id);
        // A stride cut must never land inside — or in the affix-stripped
        // whitespace of — an added token (each half would then mis-frame it).
        // The special split always peels the *raw*-matched (`normalized == false`) tokens
        // before any striding, so those are safe regardless of the boundary.
        // Only `normalized`-form tokens survive into strided text; disqualify
        // striding if any of them either (a) carries an internal cut of the
        // pre-tokenizer's boundary (e.g. a `_→0` number transition), or (b) is an
        // affix token (`lstrip`/`rstrip`) whose absorbed adjacent-whitespace run a
        // whitespace-boundary cut could split. Precomputed once: this feeds the
        // per-`encode` `plan()` path, so it must not iterate the vocab per call.
        let normalized_added_token_blocks_stride =
            pre_tokenizer.stride_boundary().is_some_and(|boundary| {
                added_tokens
                    .iter()
                    .filter(|(_, t)| t.normalized)
                    .any(|(_, t)| {
                        t.lstrip || t.rstrip || has_internal_boundary(&t.content, boundary)
                    })
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
                id: NEXT_PIPELINE_ID.fetch_add(1, Ordering::Relaxed),
                added_vocabulary,
                normalizers,
                pre_tokenizer,
                model,
                post_processor: tok
                    .get_post_processor()
                    .map(PipelinePostProcessor::try_from)
                    .transpose()?
                    .unwrap_or_default(),
                normalized_added_token_blocks_stride,
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

    /// The normalization steps, in execution order. More than one when a `Metaspace`
    /// pre-tokenizer contributed its text-rewriting half, see [`PipelineTokenizer::try_from`].
    pub fn get_normalizer(&self) -> Option<&NormalizerWrapper> {
        self.inner.normalizers.iter().find_map(|n| match n {
            PipelineNormalizer::Declared(normalizer) => Some(normalizer),
            PipelineNormalizer::Metaspace(_) => None,
        })
    }

    pub fn get_pre_tokenizer(&self) -> &PipelinePreTokenizer {
        &self.inner.pre_tokenizer
    }

    /// Stage gates for [`encode_generic`](Self::encode_generic), in execution order.
    /// Each level runs every stage up to and including itself; `STAGE_POSTPROCESS` is
    /// a full encode. `STAGE_FRAME` is the special-token scan + iteration only (the
    /// "other" slice in the decomposition).
    pub const STAGE_FRAME: u8 = 0;
    pub const STAGE_NORMALIZE: u8 = 1;
    pub const STAGE_SPLIT: u8 = 2;
    pub const STAGE_MODEL: u8 = 3;
    pub const STAGE_POSTPROCESS: u8 = 4;

    /// Decode token ids back to a `String`.
    ///
    /// Not implemented yet: the pipeline decode path is being built. It fails
    /// loud (rather than returning a plausible-but-wrong string) so the oracle
    /// test and the comparative benchmark report decode as *pending* instead of
    /// silently validating garbage. Implementing this flips the ignored
    /// `pipeline_decode_oracle` test on and lights up the decode charts.
    pub fn decode(&self, _ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
        Err("PipelineTokenizer::decode is not implemented yet".into())
    }

    /// Decode several id sequences at once, one `String` per input. Mirrors the
    /// released `decode_batch`; sequential (KISS), behavior-identical to a
    /// parallel map, since each [`decode`](Self::decode) is independent.
    pub fn decode_batch(
        &self,
        sentences: &[&[u32]],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        sentences
            .iter()
            .map(|ids| self.decode(ids, skip_special_tokens))
            .collect()
    }

    /// Incremental decode: feed ids one at a time via [`PipelineDecodeStream::step`].
    /// Same prefix-tracking scheme as the released `DecodeStream`, built on
    /// [`decode`](Self::decode), so it is correct exactly where `decode` is.
    pub fn decode_stream(&self, skip_special_tokens: bool) -> PipelineDecodeStream<'_> {
        PipelineDecodeStream {
            tokenizer: self,
            ids: Vec::new(),
            skip_special_tokens,
            prefix: String::new(),
            prefix_index: 0,
        }
    }

    /// One sequence on the calling thread, generic over how many stages run. `STAGE`
    /// is a **const generic**, so `if STAGE >= …` folds at compile time and the
    /// disabled stages are compiled out, so the full specialization
    /// ([`STAGE_POSTPROCESS`]) is branchless and identical to a hand-written full
    /// pipeline, while the benchmark drives lower `STAGE` values to time each stage's
    /// marginal cost (the ablation ladder), e.g. `model = t(MODEL) − t(SPLIT)`. No
    /// runtime gate, no `Instant` in the loop.
    ///
    /// [`STAGE_POSTPROCESS`]: Self::STAGE_POSTPROCESS
    ///
    /// `output` and the `pre_tokens` scratch are caller-owned so a benchmark can reuse
    /// them across calls and observe both buffers to anchor the ablation levels. The
    /// library itself stays free of any `black_box`/timing artifact.
    #[doc(hidden)] // public only so `examples/fixture_bench.rs` can drive partial stages
    pub fn encode_generic<const STAGE: u8>(
        &self,
        input: &str,
        add_special_tokens: bool,
        pre_tokens: &mut Vec<Span>,
        scratch: &mut PipelineModelScratch,
        output: &mut Vec<PipelineToken>,
    ) -> Result<()> {
        let PipelinePostProcessor { prefix, suffix } = &self.inner.post_processor;
        // Prepend prefix tokens, if any
        // todo: handle post-processing when encoding a pair of sequences (currently unsupported by the PipelineTokenizer)
        if add_special_tokens && STAGE >= Self::STAGE_POSTPROCESS {
            output.extend_from_slice(prefix);
        }
        // First, we extract all special tokens from the non-normalized input
        for segment in SpecialSegmentIterator::new(input, &self.inner.added_vocabulary, false) {
            match segment {
                Segment::SpecialToken(token) => output.push(PipelineToken { id: token }),
                Segment::Text(chunk) => {
                    let normalized: Cow<str> = if STAGE >= Self::STAGE_NORMALIZE {
                        normalize_all(&self.inner.normalizers, chunk)?
                    } else {
                        Cow::Borrowed(chunk)
                    };
                    // Extract special tokens from the normalized input
                    for segment in
                        SpecialSegmentIterator::new(&normalized, &self.inner.added_vocabulary, true)
                    {
                        match segment {
                            Segment::SpecialToken(token) => {
                                output.push(PipelineToken { id: token })
                            }
                            Segment::Text(normalized_chunk) => {
                                if STAGE >= Self::STAGE_SPLIT {
                                    pre_tokens.clear();
                                    self.inner
                                        .pre_tokenizer
                                        .pre_tokenize(normalized_chunk, pre_tokens)?;
                                    if STAGE >= Self::STAGE_MODEL {
                                        self.inner.model.tokenize_spans(
                                            normalized_chunk,
                                            pre_tokens,
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
        }
        // Append suffix tokens, if any
        if add_special_tokens && STAGE >= Self::STAGE_POSTPROCESS {
            output.extend_from_slice(suffix);
        }
        Ok(())
    }

    /// Encode one or many sequences, returning an [`EncodeHandle`]:
    /// workers keep encoding in the background while the caller
    /// holds it.
    ///
    /// The job is `'static`: it holds the input storage and a cheap clone of this
    /// tokenizer handle. Dropping the job cancels unclaimed
    /// work, and the last worker releases the storage.
    pub fn encode<I: IntoInputs>(&self, inputs: I) -> EncodeHandle {
        self.encode_with(inputs, true)
    }

    /// One sequence, start to finish on the calling thread: the same stages as
    /// [`encode`](Self::encode) — including the post-processor's framing — with no
    /// job, no pool and no handle. For a single short input that is all overhead;
    /// `encode` is the batched/parallel face of the same work.
    pub fn encode_one(&self, input: &str, add_special_tokens: bool) -> Result<Vec<PipelineToken>> {
        SCRATCH.with(|st| {
            let state = &mut *st.borrow_mut();
            self.encode_one_with(input, state)
                .map(|tokens| self.frame(tokens, add_special_tokens))
        })
    }

    /// [`encode`](Self::encode), with the post-processor's special-token framing
    /// (BOS/EOS) made optional. Framing is per *input*, so it is applied where an
    /// input's chunks are joined, not per chunk.
    pub fn encode_with<I: IntoInputs>(&self, inputs: I, add_special_tokens: bool) -> EncodeHandle {
        let storage = inputs.into_inputs();
        let n_inputs = storage.len();
        let refs: Vec<&str> = (0..n_inputs).map(|i| storage.get(i)).collect();

        let total_bytes: usize = refs.iter().map(|s| s.len()).sum();
        if total_bytes < PARALLEL_MIN_TOTAL_BYTES {
            return EncodeHandle::ready(self.encode_serial(&refs, add_special_tokens));
        }
        let Some(pool) = pool::rayon() else {
            return EncodeHandle::ready(self.encode_serial(&refs, add_special_tokens));
        };

        let mut counts = vec![0usize; n_inputs];
        let mut units: Vec<Item> = Vec::new();
        let mut norm_bufs: Vec<String> = Vec::new();
        let mut spans: Vec<Span> = Vec::new();
        let mut preresolved: Vec<(usize, usize, u32)> = Vec::new();

        let plan = self.plan();
        // `Normalized`/`Pretokenized` earn their serial prefix only for a
        // segment larger than a fair per-thread share of the batch. Below that,
        // batch parallelism balances it as one whole unit; the prefix would just
        // serialize work. This also splits a lone huge segment inside an
        // otherwise-large batch (a straggler otherwise). `Raw` never pays a
        // prefix, so it ignores this.
        let fair_share = total_bytes / pool.current_num_threads();

        for (seq, &input) in refs.iter().enumerate() {
            if input.len() < 2 * PARALLEL_MIN_TOTAL_BYTES {
                self.emit_full(seq, 0..input.len(), &mut counts, &mut units);
                continue;
            }
            for piece in SpecialSegmentIterator::new(input, &self.inner.added_vocabulary, false) {
                let segment = match piece {
                    Segment::SpecialToken(id) => {
                        let idx = counts[seq];
                        counts[seq] += 1;
                        preresolved.push((seq, idx, id));
                        continue;
                    }
                    Segment::Text(segment) => segment,
                };
                let base = offset_within(input, segment);
                let range = base..base + segment.len();
                let big_enough = segment.len() >= 2 * PARALLEL_MIN_TOTAL_BYTES;

                match &plan {
                    ParallelPlan::Raw(boundary) => {
                        for region in Region::tile(range.clone(), *boundary) {
                            let idx = counts[seq];
                            counts[seq] += 1;
                            units.push(Item {
                                seq,
                                idx,
                                work: Work::Raw(region),
                            });
                        }
                    }
                    ParallelPlan::Normalized(boundary)
                        if big_enough && segment.len() > fair_share =>
                    {
                        match normalize_all(&self.inner.normalizers, segment)
                            .map(Cow::into_owned)
                        {
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
                            Err(_) => self.emit_full(seq, range.clone(), &mut counts, &mut units),
                        }
                    }
                    ParallelPlan::Pretokenized if big_enough && segment.len() > fair_share => {
                        match self.pretokenize_segment(segment, input) {
                            Ok(pretokenized) => {
                                // Merge this segment's pretokenized units into the job pools.
                                // It has at most one norm buffer (local
                                // index 0); its job index is `buf_base`.
                                let buf_base = norm_bufs.len();
                                let span_base = spans.len();
                                norm_bufs.extend(pretokenized.norm_buf);
                                spans.extend_from_slice(&pretokenized.spans);
                                for unit in pretokenized.units {
                                    let idx = counts[seq];
                                    counts[seq] += 1;
                                    match unit {
                                        SegmentUnit::Special(t) => preresolved.push((seq, idx, t)),
                                        SegmentUnit::Group { text, spans: sr } => {
                                            // Norm buffers were merged at `buf_base`;
                                            // Raw offsets are already input-absolute.
                                            let text = match text {
                                                SegmentText::Norm { buf, range } => {
                                                    SegmentText::Norm {
                                                        buf: buf + buf_base,
                                                        range,
                                                    }
                                                }
                                                raw => raw,
                                            };
                                            units.push(Item {
                                                seq,
                                                idx,
                                                work: Work::Pretokenized {
                                                    text,
                                                    spans: sr.start + span_base..sr.end + span_base,
                                                },
                                            });
                                        }
                                    }
                                }
                            }
                            Err(_) => self.emit_full(seq, range.clone(), &mut counts, &mut units),
                        }
                    }
                    // `Whole`, or a prefix plan not worth it here (batch saturates
                    // the pool, or the segment is small): one whole-segment unit,
                    // full pipeline in the worker.
                    _ => self.emit_full(seq, range.clone(), &mut counts, &mut units),
                }
            }
        }

        // Too little schedulable work to be worth the pool. (Preresolved specials
        // need no worker, so only worker `units` count; falling back re-encodes
        // serially, which is byte-identical and cheap at this size.)
        if units.len() < 2 {
            return EncodeHandle::ready(self.encode_serial(&refs, add_special_tokens));
        }
        drop(refs);

        // LPT: claim the largest units first so no giant chunk lands last
        // (straggler tail); the (seq, idx) tags keep reassembly order-free.
        units.sort_by_key(|unit| {
            std::cmp::Reverse(match &unit.work {
                Work::Raw(region) => region.approx_len(),
                Work::Normalized { region, .. } => region.approx_len(),
                Work::Pretokenized { spans: sr, .. } => {
                    let group = &spans[sr.clone()];
                    group
                        .last()
                        .map(|last| (last.end - group[0].start) as usize)
                        .unwrap_or(0)
                }
            })
        });

        let n_units = units.len();
        let n_inputs = counts.len();
        // Per-(seq, idx) write-once cells, per-input pending counters, and the
        // completion log (one slot per input, filled as inputs finish).
        let slots: Vec<Vec<Slot>> = counts
            .iter()
            .map(|&c| (0..c).map(|_| Slot::empty()).collect())
            .collect();
        let pending: Vec<AtomicUsize> = counts.iter().map(|&c| AtomicUsize::new(c)).collect();
        let completed_order: Vec<AtomicUsize> =
            (0..n_inputs).map(|_| AtomicUsize::new(NOT_DONE)).collect();
        let core = Arc::new(JobCore {
            storage: Box::new(storage),
            units,
            norm_bufs,
            spans,
            tokenizer: self.clone(),
            add_special_tokens,
            cursor: CachePadded(AtomicUsize::new(0)),
            cancelled: AtomicBool::new(false),
            slots,
            pending,
            completed_order,
            completed_slot: CachePadded(AtomicUsize::new(0)),
        });
        // Specials were resolved in the serial prefix; write their cells now
        // (single thread, before any worker runs) and retire them from
        // `pending`. An all-special input completes here and is logged.
        for (seq, idx, id) in preresolved {
            // SAFETY: no worker has spawned yet; this thread is the sole writer.
            unsafe { core.slots[seq][idx].set(Ok(vec![PipelineToken { id }])) };
            if core.pending[seq].fetch_sub(1, Ordering::Relaxed) == 1 {
                core.mark_done(seq);
            }
        }
        // Empty inputs (no chunks) are complete from the start.
        for seq in 0..n_inputs {
            if counts[seq] == 0 {
                core.mark_done(seq);
            }
        }

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

        EncodeHandle::streaming(core, n_inputs)
    }

    /// Model-only kernel for the `Pretokenized` plan: tokenize each pre-token
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
        let (_, scratch) = state.scratch_for(self.inner.id, &self.inner.model);
        self.inner
            .model
            .tokenize_spans(text, spans, scratch, &mut output)?;
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
                .map(|&input| {
                    self.encode_one_with(input, state)
                        .map(|tokens| self.frame(tokens, add_special_tokens))
                })
                .collect()
        })
    }

    /// Wrap one *input*'s tokens in the post-processor's special-token framing.
    /// Per input, never per chunk: a strided input's chunks are framed once, after
    /// they are joined back together.
    fn frame(&self, tokens: Vec<PipelineToken>, add_special_tokens: bool) -> Vec<PipelineToken> {
        let PipelinePostProcessor { prefix, suffix } = &self.inner.post_processor;
        if !add_special_tokens || (prefix.is_empty() && suffix.is_empty()) {
            return tokens;
        }
        let mut framed = Vec::with_capacity(prefix.len() + tokens.len() + suffix.len());
        framed.extend_from_slice(prefix);
        framed.extend_from_slice(&tokens);
        framed.extend_from_slice(suffix);
        framed
    }

    /// The post-normalize suffix of the pipeline: frame (special tokens on
    /// already-normalized text) → pre-tokenize → model, appending to `output`.
    /// Workers use this for the `Normalized` plan (normalization already ran);
    /// it is also the shared tail of `encode_one_with`.
    /// Reuses the caller's [`EncodeState`] scratch.
    fn encode_normalized(
        &self,
        normalized: &str,
        state: &mut EncodeState,
        output: &mut Vec<PipelineToken>,
    ) -> Result<()> {
        let (pre_tokens, scratch) = state.scratch_for(self.inner.id, &self.inner.model);
        for segment in SpecialSegmentIterator::new(normalized, &self.inner.added_vocabulary, true) {
            match segment {
                Segment::SpecialToken(token) => output.push(PipelineToken { id: token }),
                Segment::Text(chunk) => {
                    pre_tokens.clear();
                    self.inner.pre_tokenizer.pre_tokenize(chunk, pre_tokens)?;
                    self.inner
                        .model
                        .tokenize_spans(chunk, pre_tokens, scratch, output)?;
                }
            }
        }
        Ok(())
    }

    /// The full single-sequence kernel: frame (special tokens on raw text) →
    /// normalize → `encode_normalized`, reusing the
    /// caller's [`EncodeState`] scratch. The entry point for a worker holding a raw
    /// region (the `Raw` and `Whole` plans); span-group model work goes through
    /// `encode_spans` instead.
    ///
    /// Note: the post-processor (special-token framing like BOS/EOS) is not
    /// wired yet; when it lands it applies per input after chunk concatenation.
    fn encode_one_with(&self, input: &str, state: &mut EncodeState) -> Result<Vec<PipelineToken>> {
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
                    let normalized = normalize_all(&self.inner.normalizers, chunk)?;
                    self.encode_normalized(&normalized, state, &mut output)?;
                }
            }
        }
        Ok(output)
    }

    /// How this config splits each special-free segment (specials are peeled
    /// first). The cheapest safe split wins.
    fn plan(&self) -> ParallelPlan {
        if let Some(boundary) = self.stride_boundary() {
            // `Raw`: no serial prefix, so always worth it when available.
            ParallelPlan::Raw(boundary)
        } else if matches!(self.inner.model, PipelineModel::WordLevel(_)) {
            // WordLevel's model + pre-tokenize are cheap, so a serial prefix
            // would dwarf the parallel part. Lean on batch parallelism instead.
            ParallelPlan::Whole
        } else if let Some(boundary) = self.normalized_stride_boundary() {
            // Normalizer rules out a raw cut, but the pre-tokenizer can cut the
            // (serially) normalized text.
            ParallelPlan::Normalized(boundary)
        } else {
            ParallelPlan::Pretokenized
        }
    }

    /// The pre-tokenizer's cut boundary for **already-normalized** text, gated
    /// only on the added vocabulary (the `Normalized` plan normalizes serially
    /// first, so the normalizer is out of the equation). The one remaining hazard
    /// is a `normalized`-form added token — which the special split's raw pass
    /// can't remove — split by a stride; the precomputed
    /// `normalized_added_token_blocks_stride`
    /// covers it (raw-matched tokens are already peeled).
    fn normalized_stride_boundary(&self) -> Option<StrideBoundary> {
        if self.inner.normalized_added_token_blocks_stride {
            return None;
        }
        self.inner.pre_tokenizer.stride_boundary()
    }

    /// Whether — and where — the config lets **raw** text be cut and each piece
    /// encoded independently with the same result as the whole (the `Raw` plan).
    /// Composes the per-stage split-safety: the normalizer must preserve
    /// boundaries ([`NormalizerWrapper::preserves_stride_boundaries`]) *and* the
    /// added vocabulary + pre-tokenizer must permit a cut
    /// (`normalized_stride_boundary`). If the
    /// normalizer doesn't preserve boundaries, the `Normalized` plan takes over
    /// after a serial normalize.
    fn stride_boundary(&self) -> Option<StrideBoundary> {
        let norm_ok = self
            .inner
            .normalizers
            .iter()
            .all(PipelineNormalizer::preserves_stride_boundaries);
        if !norm_ok {
            return None;
        }
        self.normalized_stride_boundary()
    }

    /// Emit one whole-segment `Work::Raw` unit (full pipeline in the worker) for
    /// the byte `range` of input `seq`. Shared fallback for `Whole` and for the
    /// prefix plans when their serial prefix isn't worth running.
    fn emit_full(
        &self,
        seq: usize,
        range: Range<usize>,
        counts: &mut [usize],
        units: &mut Vec<Item>,
    ) {
        let idx = counts[seq];
        counts[seq] += 1;
        units.push(Item {
            seq,
            idx,
            work: Work::Raw(Region::Full(range)),
        });
    }

    /// Serial prefix of the `ParallelPlan::Pretokenized` plan over one
    /// special-free segment (specials already peeled): normalize → frame
    /// normalized specials → pre-tokenize, packaging the model work as
    /// ~`PARALLEL_MIN_TOTAL_BYTES`-sized span-groups. `input` is the whole source
    /// the borrowed ([`SegmentText::Raw`]) offsets resolve against; `segment` is a
    /// subslice of it. Mirrors the walker in `encode_one_with`
    /// stage by stage — the two must agree on unit order for reassembly to be
    /// byte-identical to the serial encode.
    fn pretokenize_segment(&self, segment: &str, input: &str) -> Result<PretokenizedSegment> {
        let mut pretokenized = PretokenizedSegment {
            units: Vec::new(),
            norm_buf: None,
            spans: Vec::new(),
        };
        let mut pre_tokens: Vec<Span> = Vec::new();
        let normalized = normalize_all(&self.inner.normalizers, segment)?;
        let owned = matches!(normalized, Cow::Owned(_));
        for piece in SpecialSegmentIterator::new(&normalized, &self.inner.added_vocabulary, true) {
            match piece {
                Segment::SpecialToken(token) => pretokenized.units.push(SegmentUnit::Special(token)),
                Segment::Text(ntext) => {
                    pre_tokens.clear();
                    self.inner
                        .pre_tokenizer
                        .pre_tokenize(ntext, &mut pre_tokens)?;
                    if pre_tokens.is_empty() {
                        continue;
                    }
                    // Where this piece's text lives. When owned, all groups
                    // point at the single `norm_buf` (local index 0, rebased to
                    // the job pool at merge); when borrowed, at the raw input.
                    let base = if owned {
                        offset_within(&normalized, ntext)
                    } else {
                        // Borrowed all the way down: ntext is a subslice of the
                        // raw input.
                        offset_within(input, ntext)
                    };
                    let text_range = base..base + ntext.len();
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
                        let span_range = pretokenized.spans.len()..pretokenized.spans.len() + (e - g);
                        pretokenized.spans.extend_from_slice(&pre_tokens[g..e]);
                        let text = if owned {
                            // Local index 0 — the one `norm_buf`; merge rebases it.
                            SegmentText::Norm {
                                buf: 0,
                                range: text_range.clone(),
                            }
                        } else {
                            SegmentText::Raw(text_range.clone())
                        };
                        pretokenized.units.push(SegmentUnit::Group {
                            text,
                            spans: span_range,
                        });
                        g = e;
                    }
                }
            }
        }
        if let Cow::Owned(s) = normalized {
            pretokenized.norm_buf = Some(s);
        }
        Ok(pretokenized)
    }
}

/// Streaming decoder over a [`PipelineTokenizer`]; see [`PipelineTokenizer::decode_stream`].
pub struct PipelineDecodeStream<'tok> {
    tokenizer: &'tok PipelineTokenizer,
    ids: Vec<u32>,
    skip_special_tokens: bool,
    prefix: String,
    prefix_index: usize,
}

impl PipelineDecodeStream<'_> {
    /// Push one id and return the text it completes, or `None` while a multi-token
    /// (or multi-byte) unit is still forming. Ids past the emitted prefix are kept
    /// as decode context so cross-token decoders (byte-level, WordPiece `##`, …)
    /// see the same input a one-shot [`decode`](PipelineTokenizer::decode) would.
    pub fn step(&mut self, id: u32) -> Result<Option<String>> {
        if self.prefix.is_empty() && !self.ids.is_empty() {
            let new_prefix = self.tokenizer.decode(&self.ids, self.skip_special_tokens)?;
            if !new_prefix.ends_with('\u{fffd}') {
                self.prefix = new_prefix;
                self.prefix_index = self.ids.len();
            }
        }

        self.ids.push(id);
        let string = self.tokenizer.decode(&self.ids, self.skip_special_tokens)?;
        if string.len() > self.prefix.len() && !string.ends_with('\u{fffd}') {
            if !string.starts_with(&self.prefix) {
                return Err(format!(
                    "decode stream: {string:?} does not extend prefix {:?}",
                    self.prefix
                )
                .into());
            }
            let new_text = string[self.prefix.len()..].to_string();
            let new_prefix_index = self.ids.len() - self.prefix_index;
            self.ids = self.ids.split_off(self.prefix_index);
            self.prefix = self.tokenizer.decode(&self.ids, self.skip_special_tokens)?;
            self.prefix_index = new_prefix_index;
            Ok(Some(new_text))
        } else {
            Ok(None)
        }
    }
}

/// What `split` does with each split it forms
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SplitPolicy {
    /// Drop it, emit no split
    Remove,
    /// Emit it whole as one split
    Keep,
    /// Emit each character as its own split
    Isolate,
}

/// Splits `text` into same-class groups, emitting each as a `Span`
/// according to its `SplitPolicy`.
///
/// [`classify`] maps each char to a small `Copy + Eq` class, the current
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
        if let Some(p) = prev
            && (p != c || policy(c) == SplitPolicy::Isolate)
        {
            if policy(p) != SplitPolicy::Remove {
                out.push(Span {
                    start,
                    end: i as u32,
                });
            }
            start = i as u32;
        }
        prev = Some(c);
    }

    if let Some(p) = prev
        && policy(p) != SplitPolicy::Remove
    {
        out.push(Span {
            start,
            end: text.len() as u32,
        });
    }
}

/// Splits `text` around a single delimiter predicate, honoring the full
/// `SplitDelimiterBehavior` contract. The pipeline-side equivalent of
/// `NormalizedString::split(pattern, behavior)` for a char predicate.
///
/// The three non-merging behaviors reduce to a `SplitPolicy` on the delimiter
/// class and reuse `split`. The two merge variants are their own single pass:
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
        if d { delim_policy } else { SplitPolicy::Keep }
    });
}

/// Applies a `SplitDelimiterBehavior` to a match segmentation and appends the
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

    // (offsets, should_remove), mirroring `NormalizedString::split`.
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

    /// Every pre-token of a chunk at once.
    ///
    /// The pipeline has the whole span list before the model runs, so handing them over one at a
    /// time buys nothing and costs a virtual call, a slice, a `Result` and an output capacity
    /// check per pre-token -- on English that is one round trip per 5.8 bytes. The default is the
    /// loop it replaces, so a model only overrides this if it has something to hoist out of it.
    fn tokenize_spans(
        &self,
        chunk: &str,
        spans: &[Span],
        scratch: &mut Self::Scratch,
        output: &mut Vec<PipelineToken>,
    ) -> Result<()> {
        for span in spans {
            self.tokenize_pipeline(&chunk[span.range()], scratch, output)?;
        }
        Ok(())
    }

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

impl Model for PipelineModel {
    type Scratch = PipelineModelScratch;

    fn tokenize_spans(
        &self,
        chunk: &str,
        spans: &[Span],
        scratch: &mut Self::Scratch,
        output: &mut Vec<PipelineToken>,
    ) -> Result<()> {
        match (self, scratch) {
            (Self::BPE(model), PipelineModelScratch::BPE(scratch)) => {
                model.tokenize_spans(chunk, spans, scratch, output)
            }
            (Self::Unigram(model), PipelineModelScratch::Unigram(scratch)) => {
                model.tokenize_spans(chunk, spans, scratch, output)
            }
            (Self::WordPiece(model), PipelineModelScratch::WordPiece(scratch)) => {
                model.tokenize_spans(chunk, spans, scratch, output)
            }
            (Self::WordLevel(model), PipelineModelScratch::WordLevel(scratch)) => {
                model.tokenize_spans(chunk, spans, scratch, output)
            }
            _ => Err("pipeline model and scratch are of different kinds".into()),
        }
    }

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

    /// Test-only convenience: drain a single-input handle to its one result.
    /// (Not on the public API — throughput callers use the streaming `Iterator`.)
    trait IntoSingle {
        fn into_single(self) -> Result<Vec<PipelineToken>>;
    }
    impl IntoSingle for EncodeHandle {
        fn into_single(self) -> Result<Vec<PipelineToken>> {
            self.wait_for_completion()
                .map(|all| all.into_iter().next().unwrap_or_default())
        }
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
        let chunks = (0..strides)
            .filter_map(|j| stride_range(&big, j, boundary_at_whitespace))
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
        let mut covered = 0;
        let mut chunks = 0;
        for j in 0..input.len().div_ceil(PARALLEL_MIN_TOTAL_BYTES) {
            if let Some(range) = stride_range(input, j, boundary) {
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
        for boundary in [boundary_at_whitespace as StrideBoundary, boundary_fsm] {
            assert_eq!(stride_range(&input, 0, boundary), Some(0..input.len()));
            for j in 1..input.len().div_ceil(PARALLEL_MIN_TOTAL_BYTES) {
                assert_eq!(
                    stride_range(&input, j, boundary),
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
        let collected = tok.encode(&inputs[..]).wait_for_completion().unwrap();
        let collected_ids: Vec<Vec<u32>> = collected.into_iter().map(ids).collect();
        assert_eq!(collected_ids, serial, "wait_for_completion != serial");

        // Borrowed-slice sugar, streaming iterator: yields (index, result) in
        // completion order; scattering by index reproduces the batch.
        let mut streamed = vec![Vec::new(); inputs.len()];
        for (seq, r) in tok.encode(&inputs[..]) {
            streamed[seq] = ids(r.unwrap());
        }
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
        let mut owned_streamed = vec![Vec::new(); owned_inputs.len()];
        for (seq, r) in job {
            owned_streamed[seq] = ids(r.unwrap());
        }
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
    /// after normalization, so invisible to the special split's raw pass); a raw/special one
    /// is peeled first, so a stride can never bisect it.
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
        // Raw special with whitespace: peeled by the special split, so raw cuts still qualify.
        assert!(
            make(Some(AddedToken::from("<s> extra", true)))
                .stride_boundary()
                .is_some(),
            "a raw special is peeled by the special split and must not disable raw cuts"
        );
        // Normalized added token with whitespace: the special split's raw pass can't see it,
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
        assert!(matches!(tok.plan(), ParallelPlan::Raw(_)));

        let big = "aa bb,cc!  aa\tbb  cc\n\n".repeat(2000); // ~44 KB, mixed runs
        let chunks = (0..big.len().div_ceil(PARALLEL_MIN_TOTAL_BYTES))
            .filter_map(|j| stride_range(&big, j, boundary_fsm))
            .count();
        assert!(chunks > 1, "expected the input to split, got {}", chunks);

        let ids = |v: Vec<PipelineToken>| v.iter().map(|t| t.id).collect::<Vec<_>>();
        let serial = ids(encode_one(&tok, &big).unwrap());
        let by_ref = ids(tok.encode(big.as_str()).into_single().unwrap());
        assert_eq!(by_ref, serial, "SpaceRun != serial");
        let owned = ids(tok.encode(big).into_single().unwrap());
        assert_eq!(owned, serial, "SpaceRun owned != serial");
    }

    /// Directly settles "peeling specials means a stride can't split one" — it
    /// holds only for *raw* (`normalized == false`) tokens. The special split's raw pass
    /// (`SpecialSegmentIterator(.., false)`) searches the non-normalized vocab, so
    /// a `normalized`-form token is invisible to it and stays inside a text
    /// segment; it is peeled only later, *per stride*, by the worker. That is the
    /// residual the `normalized_added_token_blocks_stride` gate exists for.
    #[test]
    fn special_peel_misses_normalized_tokens() {
        use crate::AddedToken;
        let mut tok = Tokenizer::new(crate::models::bpe::BPE::default());
        tok.add_tokens([AddedToken::from("mask", false).normalized(true)])
            .unwrap();
        let pipe = PipelineTokenizer::try_from(&tok).unwrap();
        let av = &pipe.inner.added_vocabulary;
        let input = "aa mask bb";

        // Raw pass (normalized = false): the token is NOT peeled.
        let raw_special = SpecialSegmentIterator::new(input, av, false)
            .any(|s| matches!(s, Segment::SpecialToken(_)));
        assert!(
            !raw_special,
            "the raw pass must not see a normalized-form token"
        );
        // Post-normalization pass (normalized = true): now it IS peeled.
        let norm_special = SpecialSegmentIterator::new(input, av, true)
            .any(|s| matches!(s, Segment::SpecialToken(_)));
        assert!(norm_special, "the normalized pass must peel it");
    }

    /// Affix-strip gating after the special split. A *raw* (`normalized == false`)
    /// `lstrip`/`rstrip` token is peeled by the special split with its whitespace absorbed,
    /// so raw cuts still qualify. A `normalized`-form affix token can't be peeled
    /// before striding, and a whitespace-boundary cut could split its absorbed
    /// run, so it must disable raw cuts. (The old `has_affix_strip` gate rejected
    /// *both*, and re-scanned the vocab on every `encode`.)
    #[test]
    fn affix_strip_gating() {
        use crate::AddedToken;
        let make = |token: AddedToken| {
            let mut tok = Tokenizer::new(crate::models::bpe::BPE::default());
            tok.with_pre_tokenizer(Some(
                SplitPretok::new(
                    SplitPattern::Regex(GPT2_REGEX_STR.to_owned()),
                    SplitDelimiterBehavior::Isolated,
                    false,
                )
                .unwrap(),
            ));
            tok.add_tokens([token]).unwrap();
            PipelineTokenizer::try_from(&tok).unwrap()
        };
        assert!(
            make(
                AddedToken::from("<mask>", true)
                    .normalized(false)
                    .lstrip(true)
            )
            .stride_boundary()
            .is_some(),
            "a raw lstrip token is peeled by the special split and must not disable raw cuts"
        );
        assert!(
            make(
                AddedToken::from("<mask>", false)
                    .normalized(true)
                    .lstrip(true)
            )
            .stride_boundary()
            .is_none(),
            "a normalized lstrip token must disable raw cuts"
        );
    }

    /// A raw (`normalized == false`) `lstrip` token in a large input: the special split peels
    /// it (absorbing the preceding whitespace) and the surrounding segments stride;
    /// the result must stay byte-identical to the serial encode.
    #[test]
    fn raw_affix_token_strides_byte_identical() {
        use crate::AddedToken;
        let mut tok = Tokenizer::new(crate::models::bpe::BPE::default());
        tok.with_pre_tokenizer(Some(
            SplitPretok::new(
                SplitPattern::Regex(GPT2_REGEX_STR.to_owned()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .unwrap(),
        ));
        tok.add_tokens([AddedToken::from("<mask>", true)
            .normalized(false)
            .lstrip(true)])
            .unwrap();
        let pipe = PipelineTokenizer::try_from(&tok).unwrap();
        assert!(
            matches!(pipe.plan(), ParallelPlan::Raw(_)),
            "raw affix token must keep Raw"
        );
        // Two ~20 KB segments around one lstrip token → both stride.
        let big = format!("{} <mask> {}", "word ".repeat(4000), "word ".repeat(4000));
        let ids = |v: Vec<PipelineToken>| v.iter().map(|t| t.id).collect::<Vec<_>>();
        let serial = ids(encode_one(&pipe, &big).unwrap());
        let par = ids(pipe.encode(big.as_str()).into_single().unwrap());
        assert_eq!(par, serial, "raw affix striding != serial");
    }

    /// Gate refinement guard: llama-3 has 256 `normalized == false` reserved
    /// specials (`<|reserved_special_token_0|>`), each carrying an internal
    /// number-transition (`_→0`) cut. The old gate checked *all* added tokens and
    /// would disqualify raw striding (silent `Pretokenized` regression); the
    /// refined gate only checks `normalized` tokens (the special split peels the raw ones),
    /// so llama-3 must keep `Raw` with the number-transition boundary.
    #[test]
    fn byte_level_raw_specials_keep_split_raw() {
        let oracle = Tokenizer::from_file("../data/llama-3-tokenizer.json").unwrap();
        let tok = PipelineTokenizer::try_from(&oracle).unwrap();
        assert!(
            matches!(tok.plan(), ParallelPlan::Raw(_)),
            "llama-3's raw-only specials must not disqualify Raw"
        );
    }

    /// A non-chunk-safe config: regex `Span` pre-tokenizer (unknown regex, so
    /// no raw cut qualifies) + WordPiece model, so the plan must
    /// escalate to `ParallelPlan::Pretokenized`. Optionally a `Lowercase`
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

    /// Pretokenized: a large single input under a non-chunk-safe config must
    /// actually split into several parallel span-groups and stay id-identical
    /// to the serial encode, through both the scoped and the owned paths.
    #[test]
    fn split_at_model_matches_serial() {
        let tok = split_at_model_pipeline(false);
        assert!(matches!(tok.plan(), ParallelPlan::Pretokenized));

        let big = "aa.bb.cc.".repeat(3000); // ~27 KB, punctuation-delimited
        let pretokenized = tok.pretokenize_segment(&big, &big).unwrap();
        let groups = pretokenized
            .units
            .iter()
            .filter(|u| matches!(u, SegmentUnit::Group { .. }))
            .count();
        assert!(groups > 1, "expected several span-groups, got {}", groups);
        assert!(pretokenized.norm_buf.is_none(), "no normalizer -> no owned buf");

        let ids = |v: Vec<PipelineToken>| v.iter().map(|t| t.id).collect::<Vec<_>>();
        let serial = ids(encode_one(&tok, &big).unwrap());
        let by_ref = ids(tok.encode(big.as_str()).into_single().unwrap());
        assert_eq!(by_ref, serial, "Pretokenized != serial");
        let owned = ids(tok.encode(big).into_single().unwrap());
        assert_eq!(owned, serial, "owned Pretokenized != serial");
    }

    /// Pretokenized with a rewriting normalizer (uppercase input + `Lowercase`
    /// forces `Cow::Owned` → the `Norm` buffer path) and an interleaved special
    /// token (preresolved units must keep their position in the stream).
    #[test]
    fn split_at_model_normalizer_and_specials_match_serial() {
        let tok = split_at_model_pipeline(true);
        assert!(matches!(tok.plan(), ParallelPlan::Pretokenized));

        let half = "AA.BB.cc.".repeat(2000);
        let big = format!("{half}<s>{}", "CC.aa.BB.".repeat(2000));
        // The special split peels `<s>` in the builder; the per-segment model prefix runs on
        // special-free text and (with `Lowercase`) must own its normalized buffer.
        let pretokenized = tok.pretokenize_segment(&half, &half).unwrap();
        assert!(
            pretokenized.norm_buf.is_some(),
            "lowercased text must live in the owned norm buf"
        );

        let ids = |v: Vec<PipelineToken>| v.iter().map(|t| t.id).collect::<Vec<_>>();
        let serial = ids(encode_one(&tok, &big).unwrap());
        let by_ref = ids(tok.encode(big.as_str()).into_single().unwrap());
        assert_eq!(by_ref, serial, "parallel != serial (norm + specials)");
        let owned = ids(tok.encode(big).into_single().unwrap());
        assert_eq!(owned, serial, "owned != serial (norm + specials)");
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
        // WordLevel model: the plan cannot escalate to Pretokenized either.
        assert!(
            matches!(pipe.plan(), ParallelPlan::Whole),
            "Strip + WordLevel must fall to batch-level parallelism only",
        );
    }

    /// A config no plan can split — `WordLevel` (so no `Pretokenized`) under a
    /// boundary-hostile `Strip` normalizer (so no `Raw`/`Normalized`) picks
    /// `Whole`. A single large input dense with special tokens still
    /// parallelizes: the special split (always first) peels the specials
    /// (preresolved) and hands each inter-special segment to a worker. Must stay
    /// byte-identical to the serial encode, both borrowed and owned handles.
    #[test]
    fn specials_parallelize_whole_input() {
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
        tok.add_special_tokens([AddedToken::from("<s>", true)])
            .unwrap();
        let tok = PipelineTokenizer::try_from(&tok).unwrap();
        assert!(
            matches!(tok.plan(), ParallelPlan::Whole),
            "Strip + WordLevel plan must be Whole (the special split does the splitting)"
        );

        // ~24 KB, ~2000 interspersed specials → thousands of worker units.
        let big = "aa bb cc <s> ".repeat(2000);
        let ids = |v: Vec<PipelineToken>| v.iter().map(|t| t.id).collect::<Vec<_>>();
        let serial = ids(encode_one(&tok, &big).unwrap());
        let by_ref = ids(tok.encode(big.as_str()).into_single().unwrap());
        assert_eq!(by_ref, serial, "special-split (borrowed) != serial");
        let owned = ids(tok.encode(big.clone()).into_single().unwrap());
        assert_eq!(owned, serial, "special-split (owned) != serial");
    }

    /// The `Normalized` plan: a non-boundary-preserving normalizer
    /// (`Prepend`) rules out `Raw`, but the `WhitespaceSplit` pre-tokenizer
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
        tok.with_normalizer(Some(Prepend::new("▁".to_string())))
            .unwrap();
        let tok = PipelineTokenizer::try_from(&tok).unwrap();
        assert!(
            matches!(tok.plan(), ParallelPlan::Normalized(_)),
            "Prepend (unsafe normalizer) + WhitespaceSplit + WordPiece must pick `Normalized`"
        );
        let big = "aa bb cc\n".repeat(4000); // ~36 KB, forces striding
        let ids = |v: Vec<PipelineToken>| v.iter().map(|t| t.id).collect::<Vec<_>>();
        let serial = ids(encode_one(&tok, &big).unwrap());
        let by_ref = ids(tok.encode(big.as_str()).into_single().unwrap());
        assert_eq!(by_ref, serial, "Normalized (borrowed) != serial");
        let owned = ids(tok.encode(big.clone()).into_single().unwrap());
        assert_eq!(owned, serial, "Normalized (owned) != serial");
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

    /// Test the literal only replace and splits can be run without the fancy-regex feature
    #[cfg(not(feature = "fancy-regex"))]
    #[test]
    fn string_pattern_config_loads_and_encodes_with_no_regex_backend() {
        let normalizer: NormalizerWrapper =
            serde_json::from_str(r#"{"type":"Replace","pattern":{"String":" "},"content":"▁"}"#)
                .unwrap();
        let pre_tokenizer: PreTokenizerWrapper = serde_json::from_str(
            r#"{"type":"Split","pattern":{"String":"▁"},"behavior":"MergedWithPrevious","invert":false}"#,
        )
        .unwrap();

        let mut tok = wordlevel_tokenizer(vec![("<unk>", 0), ("hello▁", 1), ("world", 2)], None);
        tok.with_normalizer(Some(normalizer)).unwrap();
        tok.with_pre_tokenizer(Some(pre_tokenizer));

        let ids: Vec<u32> = PipelineTokenizer::try_from(&tok)
            .unwrap()
            .encode_one("hello world", false)
            .unwrap()
            .iter()
            .map(|t| t.id)
            .collect();
        // Not the unk id: both the `Replace` and the `Split` really ran on the literal path.
        assert_eq!(ids, [1, 2]);
        assert_pipeline_matches_reference(&tok, "hello world");
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

    fn wordlevel_tokenizer(
        vocab: Vec<(&str, u32)>,
        post_processor: Option<PostProcessorWrapper>,
    ) -> Tokenizer {
        use crate::models::wordlevel::WordLevel;
        use crate::pre_tokenizers::whitespace::Whitespace;

        let unk = vocab[0].0.to_string();
        let vocab: ahash::AHashMap<String, u32> =
            vocab.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token(unk)
            .build()
            .unwrap();
        let mut tok = Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(Whitespace));
        tok.with_post_processor(post_processor);
        tok
    }

    fn assert_pipeline_matches_reference(tok: &Tokenizer, input: &str) {
        let pipeline = PipelineTokenizer::try_from(tok).unwrap();
        for add_special_tokens in [false, true] {
            let expected = tok
                .encode(input, add_special_tokens)
                .unwrap()
                .get_ids()
                .to_vec();
            let got: Vec<u32> = pipeline
                .encode_one(input, add_special_tokens)
                .unwrap()
                .iter()
                .map(|t| t.id)
                .collect();
            assert_eq!(expected, got, "add_special_tokens={add_special_tokens}");
        }
    }

    #[test]
    fn pipeline_runs_bert_post_processor_matching_reference() {
        use crate::processors::bert::BertProcessing;

        let tok = wordlevel_tokenizer(
            vec![("[CLS]", 0), ("[SEP]", 1), ("hello", 2), ("world", 3)],
            Some(PostProcessorWrapper::Bert(BertProcessing::new(
                ("[SEP]".to_string(), 1),
                ("[CLS]".to_string(), 0),
            ))),
        );
        assert_pipeline_matches_reference(&tok, "hello world");

        let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
        let ids = |enc: Vec<PipelineToken>| enc.iter().map(|t| t.id).collect::<Vec<_>>();
        assert_eq!(
            ids(pipeline.encode_one("hello world", true).unwrap()),
            vec![0, 2, 3, 1]
        );
        assert_eq!(
            ids(pipeline.encode_one("hello world", false).unwrap()),
            vec![2, 3]
        );
    }

    #[test]
    fn pipeline_runs_roberta_post_processor_matching_reference() {
        use crate::processors::roberta::RobertaProcessing;

        let tok = wordlevel_tokenizer(
            vec![("<s>", 0), ("</s>", 1), ("hello", 2), ("world", 3)],
            Some(PostProcessorWrapper::Roberta(RobertaProcessing::new(
                ("</s>".to_string(), 1),
                ("<s>".to_string(), 0),
            ))),
        );
        assert_pipeline_matches_reference(&tok, "hello world");
    }

    #[test]
    fn pipeline_runs_template_post_processor_matching_reference() {
        use crate::processors::template::TemplateProcessing;

        let tok = wordlevel_tokenizer(
            vec![("[CLS]", 0), ("[SEP]", 1), ("hello", 2), ("world", 3)],
            Some(PostProcessorWrapper::Template(
                TemplateProcessing::builder()
                    .try_single("[CLS] $0 [SEP]")
                    .unwrap()
                    .special_tokens(vec![("[CLS]", 0u32), ("[SEP]", 1u32)])
                    .build()
                    .unwrap(),
            )),
        );
        assert_pipeline_matches_reference(&tok, "hello world");
    }

    #[test]
    fn pipeline_bytelevel_post_processor_is_noop() {
        use crate::pre_tokenizers::byte_level::ByteLevel;

        let tok = wordlevel_tokenizer(
            vec![("hello", 2), ("world", 3)],
            Some(PostProcessorWrapper::ByteLevel(ByteLevel::default())),
        );
        assert_pipeline_matches_reference(&tok, "hello world");
    }

    #[test]
    fn pipeline_sequence_composes_later_member_as_outermost() {
        use crate::processors::bert::BertProcessing;
        use crate::processors::sequence::Sequence as ProcSequence;

        let tok = wordlevel_tokenizer(
            vec![
                ("A", 100),
                ("B", 101),
                ("C", 102),
                ("D", 103),
                ("hello", 2),
                ("world", 3),
            ],
            Some(PostProcessorWrapper::Sequence(ProcSequence::new(vec![
                PostProcessorWrapper::Bert(BertProcessing::new(
                    ("B".to_string(), 101),
                    ("A".to_string(), 100),
                )),
                PostProcessorWrapper::Bert(BertProcessing::new(
                    ("D".to_string(), 103),
                    ("C".to_string(), 102),
                )),
            ]))),
        );
        assert_pipeline_matches_reference(&tok, "hello world");

        let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
        let ids: Vec<u32> = pipeline
            .encode_one("hello world", true)
            .unwrap()
            .iter()
            .map(|t| t.id)
            .collect();
        assert_eq!(ids, vec![102, 100, 2, 3, 101, 103]);
    }

    #[test]
    fn pipeline_sequence_bytelevel_then_template_matches_reference() {
        use crate::pre_tokenizers::byte_level::ByteLevel;
        use crate::processors::sequence::Sequence as ProcSequence;
        use crate::processors::template::TemplateProcessing;

        let tok = wordlevel_tokenizer(
            vec![("<|begin_of_text|>", 0), ("hello", 2), ("world", 3)],
            Some(PostProcessorWrapper::Sequence(ProcSequence::new(vec![
                PostProcessorWrapper::ByteLevel(ByteLevel::default()),
                PostProcessorWrapper::Template(
                    TemplateProcessing::builder()
                        .try_single("<|begin_of_text|> $0")
                        .unwrap()
                        .special_tokens(vec![("<|begin_of_text|>", 0u32)])
                        .build()
                        .unwrap(),
                ),
            ]))),
        );
        assert_pipeline_matches_reference(&tok, "hello world");
    }

    #[test]
    fn conversion_rejects_template_referencing_sequence_twice() {
        use crate::processors::template::TemplateProcessing;

        let tok = wordlevel_tokenizer(
            vec![("hello", 2), ("world", 3)],
            Some(PostProcessorWrapper::Template(
                TemplateProcessing::builder()
                    .try_single("$0 $0")
                    .unwrap()
                    .build()
                    .unwrap(),
            )),
        );
        let err = conversion_error(&tok);
        assert!(err.contains("not supported"), "{}", err);
    }

    #[test]
    fn conversion_rejects_template_without_sequence_piece() {
        use crate::processors::template::TemplateProcessing;

        let tok = wordlevel_tokenizer(
            vec![("[CLS]", 0), ("hello", 2), ("world", 3)],
            Some(PostProcessorWrapper::Template(
                TemplateProcessing::builder()
                    .try_single("[CLS]")
                    .unwrap()
                    .special_tokens(vec![("[CLS]", 0u32)])
                    .build()
                    .unwrap(),
            )),
        );
        let err = conversion_error(&tok);
        assert!(err.contains("not supported"), "{}", err);
    }

    #[test]
    fn conversion_rejects_template_with_unknown_special_token() {
        // Deserializing straight from JSON skips `TemplateProcessingBuilder::validate`,
        // so this (unlike the builder) can reach the pipeline with a dangling reference.
        let json = r#"{
            "type":"TemplateProcessing",
            "single":[
                {"SpecialToken":{"id":"[CLS]","type_id":0}},
                {"Sequence":{"id":"A","type_id":0}}
            ],
            "pair":[{"Sequence":{"id":"A","type_id":0}}],
            "special_tokens":{}
        }"#;
        let processor: crate::processors::template::TemplateProcessing =
            serde_json::from_str(json).unwrap();

        let tok = wordlevel_tokenizer(
            vec![("hello", 2), ("world", 3)],
            Some(PostProcessorWrapper::Template(processor)),
        );
        let err = conversion_error(&tok);
        assert!(err.contains("not supported"), "{}", err);
    }

    #[test]
    fn conversion_rejects_sequence_containing_unsupported_member() {
        use crate::processors::sequence::Sequence as ProcSequence;
        use crate::processors::template::TemplateProcessing;

        let tok = wordlevel_tokenizer(
            vec![("hello", 2), ("world", 3)],
            Some(PostProcessorWrapper::Sequence(ProcSequence::new(vec![
                PostProcessorWrapper::Template(
                    TemplateProcessing::builder()
                        .try_single("$0 $0")
                        .unwrap()
                        .build()
                        .unwrap(),
                ),
            ]))),
        );
        let err = conversion_error(&tok);
        assert!(err.contains("not supported"), "{}", err);
    }

    /// A BPE pipeline that merges "hello" into the single id 7.
    fn hello_pipeline() -> PipelineTokenizer {
        use crate::models::bpe::{BpeBuilder, Merges, Vocab};

        let vocab: Vocab = [
            ("h", 0u32),
            ("e", 1),
            ("l", 2),
            ("o", 3),
            ("he", 4),
            ("hel", 5),
            ("hell", 6),
            ("hello", 7),
        ]
        .into_iter()
        .map(|(s, i)| (s.to_string(), i))
        .collect();
        let merges: Merges = vec![
            ("h".to_string(), "e".to_string()),
            ("he".to_string(), "l".to_string()),
            ("hel".to_string(), "l".to_string()),
            ("hell".to_string(), "o".to_string()),
        ];
        let bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, merges)
            .build()
            .unwrap();
        PipelineTokenizer::try_from(&Tokenizer::new(bpe)).unwrap()
    }

    // The pool exists so ONE `&self` tokenizer can be shared across rayon workers. Encode
    // the same input from thousands of threads through a single instance; each must get a
    // private scratch and produce the sequential result. Two threads sharing a scratch
    // would corrupt some of them. This only compiles if `PipelineTokenizer: Sync`,
    // which the pool has to preserve.
    #[test]
    fn encode_shared_across_threads() {
        use rayon::prelude::*;

        let pipeline = hello_pipeline();

        let want: Vec<u32> = pipeline
            .encode_one("hello", false)
            .unwrap()
            .iter()
            .map(|t| t.id)
            .collect();
        assert_eq!(want, vec![7]);

        let all_match = (0..10_000u32).into_par_iter().all(|_| {
            pipeline
                .encode_one("hello", false)
                .unwrap()
                .iter()
                .map(|t| t.id)
                .collect::<Vec<_>>()
                == want
        });
        assert!(all_match);
    }

    // The scratch is thread-local and reset-not-rebuilt, so it has to still know the
    // words of the last encode: a cache emptied between calls would never hit. This is
    // what makes the word cache worth having at all.
    //
    // (The old pool-size test went away with the pool: there is one scratch per thread
    // by construction now, so "does not pile up" is not observable — or violable.)
    #[test]
    fn the_word_cache_outlives_the_encode_call() {
        let pipeline = hello_pipeline();
        pipeline.encode_one("hello", false).unwrap();

        SCRATCH.with(|st| {
            let state = &mut *st.borrow_mut();
            let (_, scratch) = state.scratch_for(pipeline.inner.id, &pipeline.inner.model);
            let PipelineModelScratch::BPE(bpe) = scratch else {
                panic!("a BPE pipeline encodes with a BPE scratch");
            };
            let cache = bpe.word_cache.as_mut().expect("BPE encodes with a cache");
            assert_eq!(cache.lookup(b"hello").hit(), Some(&[7u32][..]));
        });
    }
}
