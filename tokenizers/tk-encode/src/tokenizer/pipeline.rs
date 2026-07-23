use std::cell::RefCell;
use std::convert::TryInto;
use std::ops::Range;
use std::{borrow::Cow, convert::TryFrom};

use atomsplit::classify::{classify, Atoms};
use atomsplit::fsm::Span;

use crate::models::bpe::PipelineBPE;
use crate::models::unigram::Unigram;
use crate::models::wordlevel::WordLevel;
use crate::models::wordpiece::PipelineWordPiece;
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

use super::{Result, SplitDelimiterBehavior};

/// A pre-token split, a range into the input text. `repr(C)` and layout-identical to
/// `atomsplit::fsm::Span` (`(u32, u32)`), so the FSM can write final `Split`s straight into a
/// caller buffer with no scratch->out copy (see [`classify_into_spans`]).
#[derive(Copy, Clone)]
#[repr(C)]
pub struct Split {
    pub start: u32,
    pub end: u32,
}

impl Split {
    #[inline]
    pub fn range(self) -> Range<usize> {
        self.start as usize..self.end as usize
    }
}

/// Classify `bytes` and run `fsm` (an atomsplit FSM / class-runs recipe) into a **thread-local scratch**,
/// appending the resulting spans to `out`. The scratch (tags + spans) grows to the largest segment seen
/// and is reused across calls — so pre-tokenizing many small segments (e.g. a special-token-dense input,
/// where the special scan carves the text into many tiny pieces) pays no per-segment `vec![…]` alloc.
/// `fsm` writes spans into `&mut [Span]` (len ≥ text len) and returns the count. Byte-identical to a
/// fresh `vec![0u8; n]` + `vec![(0,0); n+1]` per call — just without the allocations.
pub(crate) fn classify_into_spans(
    bytes: &[u8],
    fsm: impl FnOnce(&[u8], &[u8], &mut [Span]) -> usize,
    out: &mut Vec<Split>,
) {
    thread_local! {
        static SCRATCH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    }
    let n = bytes.len();
    SCRATCH.with(|cell| {
        let tags = &mut *cell.borrow_mut();
        if tags.len() < n {
            tags.resize(n, 0); // grow-only: after the largest segment, no realloc / no re-zeroing
        }
        classify::<Atoms>(bytes, &mut tags[..n]);
        // The FSM writes its `Span`s (== `Split`, repr(C) 2×u32) directly into `out`'s spare
        // capacity — no `spans` scratch, no scratch->out copy (that copy was ~55% of split on
        // Latin: ~250k tuple->struct moves per MB). One write, then `set_len`.
        let base = out.len();
        out.reserve(n + 1);
        // SAFETY: reserved n+1 slots; `Split` and `Span` share layout, so the spare capacity is a
        // valid `&mut [Span]`; the FSM fills `[0, k)` and we only expose those via `set_len`.
        let spare = unsafe {
            std::slice::from_raw_parts_mut(out.as_mut_ptr().add(base) as *mut Span, n + 1)
        };
        let k = fsm(bytes, &tags[..n], spare);
        // SAFETY: FSM returned `k <= n+1` initialized spans, all in-bounds valid `Split`s.
        unsafe { out.set_len(base + k) };
    });
}

/// Pack a pre-token's first ≤15 bytes into a `u128` (length in the top byte), by ONE unaligned
/// 16-byte load off `bytes[start..]` — the bytes the classify pass just read, so they're hot. A
/// span in the buffer's last 15 bytes (rare, once per input) falls back to a zero-padded copy.
#[inline(always)]
fn pack_key(bytes: &[u8], start: usize, len: usize) -> u128 {
    // 0 = "not packable" (empty, or >15 bytes): the cache must byte-verify these, because a packed
    // key only holds the first 15 bytes — two long pre-tokens sharing a 15-byte prefix would alias.
    if len == 0 || len > 15 {
        return 0;
    }
    let v: u128 = if start + 16 <= bytes.len() {
        // SAFETY: `start + 16 <= len` — 16 readable bytes; unaligned load is always valid.
        unsafe { (bytes.as_ptr().add(start) as *const u128).read_unaligned() }
    } else {
        let mut b = [0u8; 16];
        b[..len].copy_from_slice(&bytes[start..start + len]);
        u128::from_le_bytes(b)
    };
    let mask: u128 = (1u128 << (8 * len)) - 1;
    (v & mask) | ((len as u128) << 120)
}

/// Hash the packed key with the hardware CRC32 instruction (~3 cycles) — the cheap hash gigatoken
/// derives during its classify pass. On non-aarch64, a multiplicative finalizer.
#[inline(always)]
pub(crate) fn crc_key_hash(key: u128) -> u64 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: CRC is baseline on the aarch64 stable ABI targets we build (Apple Silicon etc.).
        unsafe {
            use std::arch::aarch64::__crc32cd;
            __crc32cd(__crc32cd(0, key as u64), (key >> 64) as u64) as u64
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut h = (key as u64).wrapping_mul(0xff51afd7ed558ccd)
            ^ ((key >> 64) as u64).wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^ (h >> 33)
    }
}

/// Like [`classify_into_spans`] but ALSO fills `hashes` with each span's cache hash, derived in the
/// same pass (pack the span's bytes → CRC). This is the classify→cache fusion: the model's cache
/// probe then reuses this hash instead of re-hashing the pre-token bytes cold at lookup time.
pub(crate) fn classify_into_spans_keyed(
    bytes: &[u8],
    fsm: impl FnOnce(&[u8], &[u8], &mut [Span]) -> usize,
    out: &mut Vec<Split>,
    keys: &mut Vec<u128>,
) {
    let base = out.len();
    classify_into_spans(bytes, fsm, out);
    keys.reserve(out.len() - base);
    for s in &out[base..] {
        let (start, len) = (s.start as usize, (s.end - s.start) as usize);
        keys.push(pack_key(bytes, start, len));
    }
}

pub trait Normalizer {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>>;
}

/// Range-based pre-tokenization: yields spans into the input rather than owned
/// substrings, so the pipeline can pre-tokenize without allocating.
pub trait PreTokenizer {
    /// Split `text` into pre-tokens, appending to `out`. Ranges are into `text`.
    fn pre_tokenize(&self, text: &str, out: &mut Vec<Split>) -> Result<()>;

    /// Split, and where possible ALSO append each pre-token's cache hash to `hashes` (fused key
    /// derivation — see [`classify_into_spans_keyed`]). Default: no hashes (the model then hashes
    /// pre-token bytes itself). Only the byte-level GPT split path overrides this. A run either
    /// fills `hashes` 1:1 with the spans it appended, or appends none — never partially.
    fn pre_tokenize_keyed(
        &self,
        text: &str,
        out: &mut Vec<Split>,
        _keys: &mut Vec<u128>,
    ) -> Result<()> {
        self.pre_tokenize(text, out)
    }
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
    fn pre_tokenize(&self, text: &str, out: &mut Vec<Split>) -> Result<()> {
        match self {
            Self::None => {
                out.push(Split {
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

    fn pre_tokenize_keyed(
        &self,
        text: &str,
        out: &mut Vec<Split>,
        keys: &mut Vec<u128>,
    ) -> Result<()> {
        // Only the GPT byte-level Split path fuses keys; the rest use the default (no keys).
        match self {
            Self::Split(pretok) => pretok.pre_tokenize_keyed(text, out, keys),
            other => other.pre_tokenize(text, out),
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
pub struct PipelineTokenizer {
    added_vocabulary: BucketAddedVocabulary,
    normalizer: Option<NormalizerWrapper>,
    pre_tokenizer: PipelinePreTokenizer,
    model: PipelineModel,
    _post_processor: Option<PostProcessorWrapper>,
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
            added_vocabulary,
            normalizer: tok.get_normalizer().cloned(),
            pre_tokenizer,
            model,
            _post_processor: tok.get_post_processor().cloned(),
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

    /// Encode `input` into token ids.
    ///
    /// Special tokens are matched in two passes:
    ///  1. on the raw input,
    ///  2. then on each segment after normalization
    ///
    /// This way, special / added tokens declared on raw or normalized text are both caught.
    /// The remaining text is pre-tokenized and run through the model span by span.
    ///
    /// todo: wire the post-processing
    pub fn encode(&self, input: &str, _add_special_tokens: bool) -> Result<Vec<PipelineToken>> {
        let mut output = Vec::new();
        let mut pre_tokens = Vec::new();
        self.encode_generic::<{ Self::STAGE_MODEL }>(input, &mut output, &mut pre_tokens)?;
        Ok(output)
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
        output: &mut Vec<PipelineToken>,
        pre_tokens: &mut Vec<Split>,
    ) -> Result<()> {
        // Per-pre-token packed keys, derived by the split (fused). Reused across segments.
        let mut pre_keys: Vec<u128> = Vec::new();
        // First, we extract all special tokens from the non-normalized input
        for segment in SpecialSegmentIterator::new(input, &self.added_vocabulary, false) {
            match segment {
                Segment::SpecialToken(token) => {
                    output.push(PipelineToken { id: token });
                }
                Segment::Text(chunk) => {
                    let normalized: Cow<str> = if STAGE >= Self::STAGE_NORMALIZE {
                        match &self.normalizer {
                            Some(normalizer) => normalizer.normalize(chunk)?,
                            None => Cow::Borrowed(chunk),
                        }
                    } else {
                        Cow::Borrowed(chunk)
                    };

                    // Extract special tokens from the normalized input
                    for segment in
                        SpecialSegmentIterator::new(&normalized, &self.added_vocabulary, true)
                    {
                        match segment {
                            Segment::SpecialToken(token) => {
                                output.push(PipelineToken { id: token });
                            }
                            Segment::Text(normalized_chunk) => {
                                if STAGE >= Self::STAGE_SPLIT {
                                    // Pre-tokenize the chunk; the split also derives per-pre-token
                                    // cache hashes here (fused) so the model probe skips re-hashing.
                                    pre_tokens.clear();
                                    pre_keys.clear();
                                    self.pre_tokenizer.pre_tokenize_keyed(
                                        normalized_chunk,
                                        pre_tokens,
                                        &mut pre_keys,
                                    )?;
                                    if STAGE >= Self::STAGE_MODEL {
                                        // keys carried 1:1 with spans, or none → model hashes bytes
                                        let keyed = pre_keys.len() == pre_tokens.len();
                                        for (i, pre_token) in pre_tokens.iter().enumerate() {
                                            self.model.tokenize_pipeline(
                                                &normalized_chunk[pre_token.range()],
                                                output,
                                                keyed.then(|| pre_keys[i]),
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

/// Splits `text` into same-class groups, emitting each as a [`Split`]
/// according to its [`SplitPolicy`].
///
/// `classify` maps each char to a small `Copy + Eq` class, the current
/// split ends whenever the class changes (or on every char of an `Isolate`
/// class), and `policy` decides what becomes of it. Ranges are byte offsets
/// into `text`.
#[inline(always)]
pub fn split<C: Copy + PartialEq>(
    text: &str,
    out: &mut Vec<Split>,
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
                    out.push(Split {
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
            out.push(Split {
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
    out: &mut Vec<Split>,
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
                    out.push(Split { start, end });
                    start = end;
                }
            }
            if (start as usize) < text.len() {
                out.push(Split {
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
                        out.push(Split { start, end: i });
                    }
                    start = i;
                }
            }
            if (start as usize) < text.len() {
                out.push(Split {
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
    out: &mut Vec<Split>,
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
            out.push(Split {
                start: start as u32,
                end: end as u32,
            });
        }
    }
}

pub trait Model {
    /// Tokenize one pre-token into `output`. `carried_hash` is the pre-token's cache hash when the
    /// split derived it (fused); models that cache may use it instead of re-hashing, others ignore.
    fn tokenize_pipeline(
        &self,
        sequence: &str,
        output: &mut Vec<PipelineToken>,
        carried_key: Option<u128>,
    ) -> Result<()>;
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
    fn tokenize_pipeline(
        &self,
        sequence: &str,
        output: &mut Vec<PipelineToken>,
        carried_key: Option<u128>,
    ) -> Result<()> {
        match self {
            Self::BPE(model) => model.tokenize_pipeline(sequence, output, carried_key),
            Self::Unigram(model) => model.tokenize_pipeline(sequence, output, carried_key),
            Self::WordLevel(model) => model.tokenize_pipeline(sequence, output, carried_key),
            Self::WordPiece(model) => model.tokenize_pipeline(sequence, output, carried_key),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::bpe::BPE;
    use crate::models::wordpiece::WordPiece;
    use crate::pre_tokenizers::byte_level::ByteLevel;
    use crate::pre_tokenizers::sequence::Sequence;

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
