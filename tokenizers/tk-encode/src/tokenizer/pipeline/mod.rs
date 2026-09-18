use std::collections::BTreeMap;
use std::sync::Arc;

#[cfg(feature = "unigram")]
use crate::models::unigram::{Unigram, UnigramScratch};
#[cfg(feature = "wordlevel")]
use crate::models::wordlevel::WordLevel;
#[cfg(feature = "wordpiece")]
use crate::models::wordpiece::{PipelineWordPiece, WordPieceScratch};
use crate::{
    DecoderRuntime, PaddingParams,
    models::bpe::{BpeScratch, PipelineBPE},
    pipeline::scratch_pool::{EncodeScratch, ScratchGuard, ScratchPool},
    tokenizer::Decoder as _,
    utils::padding::pad_flat,
    vocab::bucket_added_vocabulary::AddedVocabulary as BucketAddedVocabulary,
};
#[cfg(feature = "parallelism")]
pub use parallel::PARALLEL_MIN_BYTES;

use super::Result;

#[cfg(feature = "parallelism")]
mod parallel;
mod scratch_pool;

pub use scratch_pool::ModelScratch;

pub use bitsplit::Span;
pub mod encode_options;
pub use encode_options::{EncodeOptions, Override};

mod normalizer;
mod post_processor;
mod pre_tokenizer;
mod token_sink;

pub use normalizer::{Normalizer, NormalizerChain, PipelineNormalizer, normalize_all};
pub use post_processor::{PipelinePostProcessor, Template};
pub use pre_tokenizer::{
    PipelinePreTokenizer, PreTokenizer, PreTokenizerScratch, SplitPolicy, split, split_delimiter,
    split_matches,
};
pub use token_sink::{TokenSink, TokenSlot};

/// Below this many tokens, laying a batch out across threads costs more than doing it here.
///
/// The pool's fixed cost is tens of microseconds; a batch of 100 short documents is two. Measured
/// on a 20k-document batch, where the parallel split is twice as fast, and on a 100-document one,
/// where it was three times slower.
pub(crate) const PARALLEL_LAYOUT_MIN_TOKENS: usize = 64 * 1024;

/// An output token. Carries only the vocabulary `id`, since offsets and the token
/// string are dropped, which is all an encode-only caller needs.
///
/// TODO: For RC0 this is abolutely fine. For v1, we need an enum, Token or Encoding, which can
/// both be outputed by the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct PipelineToken(u32);

impl PipelineToken {
    /// The vocabulary id this token stands for.
    pub const fn id(self) -> u32 {
        self.0
    }

    /// `n` zeroed tokens, for free.
    ///
    /// `vec![0u32; n]` is `alloc_zeroed`, which the allocator answers with pages that are already
    /// zero -- no write at all. `vec![PipelineToken::from(0); n]` misses that specialisation and
    /// writes every slot: 0.14 ms per 8M tokens against 0.001 ms.
    pub fn zeroed(n: usize) -> Vec<Self> {
        let zeroed: Vec<u32> = vec![0; n];
        let mut zeroed = std::mem::ManuallyDrop::new(zeroed);
        // SAFETY: `PipelineToken` is `#[repr(transparent)]` over `u32`, so the two have the same
        // size, alignment and validity, and the allocation transfers unchanged.
        unsafe {
            Vec::from_raw_parts(
                zeroed.as_mut_ptr().cast::<Self>(),
                zeroed.len(),
                zeroed.capacity(),
            )
        }
    }

    /// The ids of a whole slice, without copying it.
    ///
    /// Lets a caller hand a batch's ids to something that wants `u32` -- numpy, the Python
    /// bindings -- without walking the slice to rebuild it as a `Vec<u32>`, which for a batch of
    /// short documents is the dominant cost of returning the batch at all.
    pub const fn ids_of(tokens: &[Self]) -> &[u32] {
        // SAFETY: `PipelineToken` is `#[repr(transparent)]` over `u32` (see the attribute above),
        // so a `[PipelineToken]` and a `[u32]` of the same length have identical layout.
        unsafe { std::slice::from_raw_parts(tokens.as_ptr().cast::<u32>(), tokens.len()) }
    }
}

impl From<u32> for PipelineToken {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<PipelineToken> for u32 {
    fn from(value: PipelineToken) -> Self {
        value.0
    }
}

/// Compares a token against a bare id, so a caller can assert against `[u32]`
/// without mapping the ids out first.
impl PartialEq<u32> for PipelineToken {
    fn eq(&self, id: &u32) -> bool {
        self.0 == *id
    }
}

impl PartialEq<PipelineToken> for u32 {
    fn eq(&self, token: &PipelineToken) -> bool {
        *self == token.0
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
    /// input_offset is the start position of `text` in the input
    Text { text: &'a str, input_offset: usize },
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
///         Segment::Text { text, input_offset } => { /* tokenize this chunk */ }
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
            let input_offset = self.offset;
            self.offset = end;
            if !before_token.is_empty() {
                // The iterator returns segments in order: we need to return the chunk of text and then the special token.
                // Store the special token to return in the next call and return a [`Segment::Text`]
                self.pending = Some(token);
                return Some(Segment::Text {
                    text: before_token,
                    input_offset,
                });
            } else {
                return Some(Segment::SpecialToken(token));
            }
        }
        let input_offset = self.offset;
        self.offset = self.input.len();
        Some(Segment::Text {
            text: remaining_input,
            input_offset,
        })
    }
}

struct TokenizerInner {
    added_vocabulary: BucketAddedVocabulary,
    normalizers: Vec<PipelineNormalizer>,
    pre_tokenizer: PipelinePreTokenizer,
    model: PipelineModel,
    post_processor: PipelinePostProcessor,
    decoder: Option<DecoderRuntime>,
    /// Lowest id owned by the added vocabulary, or `u32::MAX` when there is none.
    /// Allows to skip the added vocabulary lookup if the token id is lower than this value.
    added_id_min: u32,
    /// Which token plays which role (`"eos_token"` -> `"</s>"`), so a `tokenizer.json` can carry
    /// the special-token metadata that used to need a separate `tokenizer_config.json`. Empty
    /// when the config declares none. `BTreeMap` so the writer emits a stable key order.
    role_to_token: BTreeMap<String, String>,
    /// Padding configuration. [`EncodeOptions::padding`] overrides it per call.
    padding: Option<PaddingParams>,
    scratch_pool: ScratchPool,
}

/// Experimental encode-only pipeline built from a [`Tokenizer`]. Runs the same
/// stages over borrowed ranges to avoid the reference path's allocations.
#[derive(Clone)]
pub struct PipelineTokenizer {
    inner: Arc<TokenizerInner>,
}

// comptime verification that PipelineTokenizer is Send + Sync
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PipelineTokenizer>();
};

impl PipelineTokenizer {
    /// This is the new "constructor" we expose.
    ///
    /// `added_vocabulary` must already have had its tokens replayed *against the concrete model and
    /// in id order*, because `add_tokens` reuses a model id when the token is already in the
    /// vocabulary; doing it later, or out of order, moves ids silently.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        added_vocabulary: BucketAddedVocabulary,
        normalizers: Vec<PipelineNormalizer>,
        pre_tokenizer: PipelinePreTokenizer,
        model: PipelineModel,
        post_processor: PipelinePostProcessor,
        decoder: Option<DecoderRuntime>,
        role_to_token: BTreeMap<String, String>,
        padding: Option<PaddingParams>,
    ) -> Self {
        let added_id_min = added_vocabulary
            .get_added_tokens_decoder()
            .keys()
            .copied()
            .min()
            .unwrap_or(u32::MAX);
        Self {
            inner: Arc::new(TokenizerInner {
                added_vocabulary,
                normalizers,
                pre_tokenizer,
                model,
                post_processor,
                decoder,
                added_id_min,
                role_to_token,
                padding,
                scratch_pool: ScratchPool::new(),
            }),
        }
    }

    pub fn resolve_padding<'a>(
        &'a self,
        padding_override: &'a Override<PaddingParams>,
    ) -> Option<&'a PaddingParams> {
        match padding_override {
            Override::InheritConfig => self.inner.padding.as_ref(),
            Override::Off => None,
            Override::With(params) => Some(params),
        }
    }
    // TODO: resolve_truncation
}
#[derive(Clone)]
pub enum Input {
    Single(String),
    Pair(String, String),
}

/// One document to encode: a sequence, and for a pair the second one.
///
/// Borrowed, because encoding is a blocking call -- the text only has to outlive it.
#[derive(Clone, Copy)]
pub struct Document<'a> {
    pub text: &'a str,
    pub pair: Option<&'a str>,
}

impl<'a> From<&'a str> for Document<'a> {
    fn from(text: &'a str) -> Self {
        Self { text, pair: None }
    }
}

#[derive(Clone)]
pub enum Inputs {
    Single(Input),
    Batch(Vec<Input>),
}

impl Inputs {
    fn as_slice(&self) -> &[Input] {
        match self {
            Self::Single(s) => std::slice::from_ref(s),
            Self::Batch(b) => b,
        }
    }

    /// The batch as borrowed documents, which is what the encode path takes.
    fn as_documents(&self) -> Vec<Document<'_>> {
        self.as_slice()
            .iter()
            .map(|input| match input {
                Input::Single(text) => Document { text, pair: None },
                Input::Pair(text, pair) => Document {
                    text,
                    pair: Some(pair),
                },
            })
            .collect()
    }
}

impl<'a> IntoIterator for &'a Inputs {
    type Item = &'a Input;
    type IntoIter = std::slice::Iter<'a, Input>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl From<String> for Inputs {
    fn from(s: String) -> Self {
        Self::Single(Input::Single(s))
    }
}

impl From<&str> for Inputs {
    fn from(s: &str) -> Self {
        Self::Single(Input::Single(s.to_owned()))
    }
}

impl From<&String> for Inputs {
    fn from(s: &String) -> Self {
        Self::Single(Input::Single(s.to_owned()))
    }
}

impl From<Vec<String>> for Inputs {
    fn from(b: Vec<String>) -> Self {
        Self::Batch(b.into_iter().map(Input::Single).collect())
    }
}

/// Above this many inputs, the copy below is worth handing to the pool.
#[cfg(feature = "parallelism")]
const PARALLEL_CONVERT_MIN: usize = 1024;

impl From<&[&str]> for Inputs {
    fn from(b: &[&str]) -> Self {
        // `Input` owns its text, so a borrowed batch is copied before anything else can start --
        // and it was copied on the calling thread, ahead of every worker. On a 20k-line batch that
        // measured 19% of the whole encode, which by itself capped the speedup however many
        // threads were free. The copies are independent, so the pool can do them.
        #[cfg(feature = "parallelism")]
        if b.len() >= PARALLEL_CONVERT_MIN
            && let Some(pool) = crate::utils::parallelism::pool()
        {
            use rayon::prelude::*;
            return pool.install(|| {
                Self::Batch(
                    b.par_iter()
                        .map(|s| Input::Single((*s).to_owned()))
                        .collect(),
                )
            });
        }
        Self::Batch(b.iter().map(|s| Input::Single((*s).to_owned())).collect())
    }
}

impl From<Vec<&str>> for Inputs {
    fn from(b: Vec<&str>) -> Self {
        Self::Batch(b.into_iter().map(|s| Input::Single(s.to_owned())).collect())
    }
}

impl From<(String, String)> for Inputs {
    fn from(p: (String, String)) -> Self {
        Self::Single(Input::Pair(p.0, p.1))
    }
}

impl From<(&str, &str)> for Inputs {
    fn from(p: (&str, &str)) -> Self {
        Self::Single(Input::Pair(p.0.to_owned(), p.1.to_owned()))
    }
}

impl From<(&String, &String)> for Inputs {
    fn from(p: (&String, &String)) -> Self {
        Self::Single(Input::Pair(p.0.to_owned(), p.1.to_owned()))
    }
}

impl From<Vec<(String, String)>> for Inputs {
    fn from(b: Vec<(String, String)>) -> Self {
        Self::Batch(b.into_iter().map(|p| Input::Pair(p.0, p.1)).collect())
    }
}

impl From<&[(&str, &str)]> for Inputs {
    fn from(b: &[(&str, &str)]) -> Self {
        Self::Batch(
            b.iter()
                .map(|(s1, s2)| Input::Pair((*s1).to_owned(), (*s2).to_owned()))
                .collect(),
        )
    }
}

/// What [`PipelineTokenizer::encode`] hands back, waited on with [`Self::wait`].
///
/// There is one encode path behind it and it has already finished by the time you hold this, so
/// `wait` is a move. The type stays because it is the shape an asynchronous encode returns: when
/// the batch path learns to hand back documents as they finish, callers do not change.
pub struct EncodeHandle {
    encodings: Result<Vec<Encoding>>,
}

impl EncodeHandle {
    /// Wait for the encode to finish. Returns one [`Encoding`] per document, in input order.
    pub fn wait(self) -> Result<Vec<Encoding>> {
        self.encodings
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Encoding {
    pub(crate) ids: Vec<PipelineToken>,
    /// `None` if every token has type id 0
    pub(crate) type_ids: Option<Vec<u8>>,
    /// `None` if the encoding is not padded (mask is all ones)
    pub(crate) attention_mask: Option<Vec<u8>>,
    /// Document starts when this holds a whole batch, CSR-style, `n_documents() + 1` entries: document `i` is
    /// `ids[offsets[i]..offsets[i + 1]]`. `None` for a single document, which is all of `ids`.
    ///
    /// One type rather than two, because a batch differs from a document only in knowing where
    /// the documents begin. [`PipelineTokenizer::encode`] hands back a `Vec` per document, which
    /// is nothing for a handful of long ones and the dominant cost for thousands of short ones:
    /// 20k chat lines cost 20k allocations plus the matching frees, on a different thread than
    /// allocated them. With offsets a whole batch costs a fixed handful, and it is the shape a
    /// serving stack wants anyway, having to assemble a contiguous batch for the model regardless.
    pub(crate) offsets: Option<Vec<u32>>,
    /// Set when the documents do not sit back to back -- the workers write into regions sized
    /// ahead of the encode, so what they do not use is a gap. `offsets` then holds each
    /// document's start and this its length, instead of CSR's "starts, and the next one's start".
    pub(crate) lengths: Option<Vec<u32>>,
}

impl Encoding {
    /// A batch the workers already laid out at a uniform stride, mask and all.
    #[cfg(feature = "parallelism")]
    pub(crate) fn padded(
        ids: Vec<PipelineToken>,
        attention_mask: Vec<u8>,
        offsets: Vec<u32>,
    ) -> Self {
        Self {
            ids,
            type_ids: None,
            attention_mask: Some(attention_mask),
            offsets: Some(offsets),
            lengths: None,
        }
    }

    /// A batch the workers laid out contiguously, document after document.
    #[cfg(feature = "parallelism")]
    pub(crate) fn batch(ids: Vec<PipelineToken>, offsets: Vec<u32>) -> Self {
        debug_assert_eq!(
            offsets.last().copied(),
            Some(ids.len() as u32),
            "[BUG] offsets must end at the id count"
        );
        Self {
            ids,
            type_ids: None,
            attention_mask: None,
            offsets: Some(offsets),
            lengths: None,
        }
    }

    /// Lay per-document encodings out as one batch, for a template the flat path cannot frame.
    pub fn is_empty(&self) -> bool {
        self.ids.len() == 0
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn ids(&self) -> &[PipelineToken] {
        &self.ids
    }

    pub fn type_ids(&self) -> Option<&[u8]> {
        self.type_ids.as_deref()
    }

    pub fn attention_mask(&self) -> Option<&[u8]> {
        self.attention_mask.as_deref()
    }

    /// How many documents this holds. `1` unless it came from a batch path.
    pub fn n_documents(&self) -> usize {
        match (&self.offsets, &self.lengths) {
            (_, Some(lengths)) => lengths.len(),
            (Some(offsets), None) => offsets.len().saturating_sub(1),
            (None, None) => 1,
        }
    }

    /// Where document `i` sits in the flat buffers, or `None` when out of range.
    ///
    /// `type_ids` and `attention_mask` run the length of `ids`, so a caller reading one document
    /// out of a batch needs the range, not just the ids.
    pub fn document_range(&self, i: usize) -> Option<std::ops::Range<usize>> {
        match (&self.offsets, &self.lengths) {
            (Some(starts), Some(lengths)) => {
                let start = *starts.get(i)? as usize;
                Some(start..start + *lengths.get(i)? as usize)
            }
            (Some(offsets), None) => Some(*offsets.get(i)? as usize..*offsets.get(i + 1)? as usize),
            (None, _) => (i == 0).then_some(0..self.ids.len()),
        }
    }

    /// Ids of document `i`, or `None` when out of range.
    pub fn document(&self, i: usize) -> Option<&[PipelineToken]> {
        self.ids.get(self.document_range(i)?)
    }

    /// How many ids document `i` holds. `0` when out of range.
    pub fn document_len(&self, i: usize) -> usize {
        self.document_range(i).map_or(0, |range| range.len())
    }

    /// The width every document occupies when they all have the same one, which is what lets a
    /// batch be read as a rectangular `(documents, stride)` array rather than one at a time.
    ///
    /// `None` for a ragged batch: unpadded, or padded to a fixed length that some document
    /// already exceeds, since padding never truncates.
    pub fn stride(&self) -> Option<usize> {
        let documents = self.n_documents();
        let first = self.document_len(0);
        (documents > 0 && (1..documents).all(|i| self.document_len(i) == first)).then_some(first)
    }

    /// The batch split into one [`Encoding`] per document.
    ///
    /// Slicing, not re-encoding: each document copies its own run out of the shared buffer. A
    /// caller that can read the batch as it stands should do that instead.
    pub fn into_documents(self) -> Vec<Self> {
        let cut = |i: usize| {
            let range = self.document_range(i).expect("[BUG] document out of range");
            Self {
                ids: self.ids[range.clone()].to_vec(),
                type_ids: self.type_ids.as_ref().map(|t| t[range.clone()].to_vec()),
                attention_mask: self.attention_mask.as_ref().map(|m| m[range].to_vec()),
                offsets: None,
                lengths: None,
            }
        };
        let documents = self.n_documents();

        // One allocation per document, so for a batch of short ones this is the same shape of
        // work as the encode and wants the same threads.
        #[cfg(feature = "parallelism")]
        if self.ids.len() >= PARALLEL_LAYOUT_MIN_TOKENS
            && let Some(pool) = crate::parallelism::pool()
        {
            use rayon::prelude::*;
            return pool.install(|| (0..documents).into_par_iter().map(cut).collect());
        }
        (0..documents).map(cut).collect()
    }

    /// Each document's ids in turn.
    pub fn documents(&self) -> impl Iterator<Item = &[PipelineToken]> {
        (0..self.n_documents()).filter_map(|i| self.document(i))
    }

    /// Document starts, when this holds a batch. See the field.
    pub fn offsets(&self) -> Option<&[u32]> {
        self.offsets.as_deref()
    }
}
impl PipelineTokenizer {
    pub fn get_model(&self) -> &PipelineModel {
        &self.inner.model
    }

    pub fn get_pre_tokenizer(&self) -> &PipelinePreTokenizer {
        &self.inner.pre_tokenizer
    }

    pub fn get_post_processor(&self) -> &PipelinePostProcessor {
        &self.inner.post_processor
    }

    /// The whole flattened normalizer chain, in the order it runs.
    ///
    /// This is for a writer, which needs the members themselves. A config `Sequence` was
    /// flattened on the way in, so what comes back is the concatenation, not the nesting.
    pub fn get_normalizers(&self) -> &[PipelineNormalizer] {
        &self.inner.normalizers
    }

    /// The decoder, if the config declared one.
    pub fn get_decoder(&self) -> Option<&DecoderRuntime> {
        self.inner.decoder.as_ref()
    }

    /// The added vocabulary, whose `get_added_tokens_decoder` is the `added_tokens` array.
    pub fn get_added_vocabulary(&self) -> &BucketAddedVocabulary {
        &self.inner.added_vocabulary
    }

    /// Which token plays which role, as the config declared it. Empty when it declared none.
    pub fn get_role_to_token(&self) -> &BTreeMap<String, String> {
        &self.inner.role_to_token
    }

    /// The token a role points at, e.g. `get_token_for_role("eos_token")`.
    pub fn get_token_for_role(&self, role: &str) -> Option<&str> {
        self.inner.role_to_token.get(role).map(String::as_str)
    }

    pub fn get_padding(&self) -> Option<&PaddingParams> {
        self.inner.padding.as_ref()
    }

    /// Encode `input` into token ids.
    ///
    /// Special tokens are matched in two passes:
    ///  1. on the raw input,
    ///  2. then on each segment after normalization
    ///
    /// This way, special / added tokens declared on raw or normalized text are both caught.
    /// The remaining text is pre-tokenized and run through the model span by span.
    /// Returns one [`Encoding`] per document, in input order. A caller that wants the whole batch
    /// as one buffer should use [`Self::encode_batch_flat`] and skip the split.
    pub fn encode(&self, inputs: impl Into<Inputs>, options: &EncodeOptions) -> EncodeHandle {
        EncodeHandle {
            encodings: self.encode_now(inputs.into(), options),
        }
    }

    fn encode_now(&self, inputs: Inputs, options: &EncodeOptions) -> Result<Vec<Encoding>> {
        Ok(self
            .encode_documents(&inputs.as_documents(), options)?
            .into_documents())
    }

    /// Encode a batch into one contiguous id buffer.
    ///
    /// Every document is framed by its template and appended to one buffer, with `offsets`
    /// recording where each begins. Nothing is allocated per document, which is what a batch of
    /// short ones otherwise spends its time on.
    ///
    /// Padding applies to the finished buffer: the offsets already say how long every document
    /// is, so it is one pass at the padded stride rather than a reallocation per document.
    pub fn encode_documents(
        &self,
        documents: &[Document<'_>],
        options: &EncodeOptions,
    ) -> Result<Encoding> {
        let padding = self.resolve_padding(&options.padding);
        self.lay_out(documents, options.add_special_tokens, padding)
    }

    /// Encode borrowed texts, the common case of [`Self::encode_documents`].
    pub fn encode_batch_flat(&self, inputs: &[&str], options: &EncodeOptions) -> Result<Encoding> {
        let documents: Vec<Document<'_>> = inputs.iter().copied().map(Document::from).collect();
        self.encode_documents(&documents, options)
    }

    /// Encode every document into one buffer, padded if asked.
    ///
    /// The pool pads as it lays the batch out, so only the serial route needs a pass of its own.
    fn lay_out(
        &self,
        documents: &[Document<'_>],
        add_special_tokens: bool,
        padding: Option<&PaddingParams>,
    ) -> Result<Encoding> {
        let post_processor = &self.inner.post_processor;
        let template = |document: &Document<'_>| match document.pair {
            Some(_) => &post_processor.pair,
            None => &post_processor.single,
        };
        // Type ids need a buffer of their own; most templates tag every token 0 and need none.
        let tagged = documents
            .iter()
            .any(|document| template(document).has_type_ids());

        #[cfg(feature = "parallelism")]
        if !tagged
            && documents.len() > 1
            && documents.iter().all(|document| document.pair.is_none())
        {
            let total: usize = documents.iter().map(|d| d.text.len()).sum();
            if total >= parallel::PARALLEL_MIN_BYTES
                && let Some(batch) = parallel::encode_flat(
                    self,
                    documents,
                    &post_processor.single,
                    add_special_tokens,
                    padding,
                )?
            {
                return Ok(batch);
            }
        }

        let mut ids = Vec::with_capacity(
            documents
                .iter()
                .map(|document| document.text.len())
                .sum::<usize>()
                / 4,
        );
        let mut type_ids = tagged.then(Vec::new);
        let mut offsets = Vec::with_capacity(documents.len() + 1);
        let mut scratch = self.scratch();
        for document in documents {
            offsets.push(ids.len() as u32);
            self.frame(
                *document,
                template(document),
                add_special_tokens,
                &mut scratch,
                &mut ids,
                type_ids.as_mut(),
            )?;
        }
        offsets.push(ids.len() as u32);
        let mut batch = Encoding {
            ids,
            type_ids,
            attention_mask: None,
            offsets: Some(offsets),
            lengths: None,
        };
        if let Some(params) = padding {
            pad_flat(&mut batch, params)?;
        }
        Ok(batch)
    }

    /// One scratch, for a caller that will encode many documents with it.
    fn scratch(&self) -> ScratchGuard<'_> {
        self.inner.scratch_pool.get(&self.inner.model)
    }

    /// One document into `ids`, framed by its template: `prefix A infix? B? suffix`.
    ///
    /// The only place framing happens, for the serial loop and the parallel one alike.
    /// `type_ids` is filled alongside when the template tags anything -- which is why the
    /// sequence lengths are taken here rather than recovered from the offsets afterwards, since
    /// a pair needs to know where A ends.
    pub(crate) fn frame<S: TokenSink>(
        &self,
        document: Document<'_>,
        template: &Template,
        add_special_tokens: bool,
        scratch: &mut EncodeScratch,
        ids: &mut S,
        mut type_ids: Option<&mut Vec<u8>>,
    ) -> Result<()> {
        fn specials<S: TokenSink>(
            add: bool,
            run: &[(PipelineToken, u8)],
            ids: &mut S,
            type_ids: &mut Option<&mut Vec<u8>>,
        ) {
            if !add {
                return;
            }
            ids.extend(run.iter().map(|&(id, _)| id));
            if let Some(out) = type_ids {
                Extend::extend(*out, run.iter().map(|&(_, type_id)| type_id));
            }
        }

        specials(add_special_tokens, &template.prefix, ids, &mut type_ids);
        let start = ids.len();
        self.encode_sequence_into(document.text, 0, scratch, ids)?;
        if let Some(out) = type_ids.as_mut() {
            out.resize(out.len() + (ids.len() - start), template.a_type_id);
        }

        if let Some(pair) = document.pair {
            specials(add_special_tokens, &template.infix, ids, &mut type_ids);
            let start = ids.len();
            self.encode_sequence_into(pair, 0, scratch, ids)?;
            if let Some(out) = type_ids.as_mut() {
                let type_id = template.b_type_id.unwrap_or(template.a_type_id);
                out.resize(out.len() + (ids.len() - start), type_id);
            }
        }

        specials(add_special_tokens, &template.suffix, ids, &mut type_ids);
        Ok(())
    }

    fn encode_sequence_into<S: TokenSink>(
        &self,
        input: &str,
        offset: usize,
        scratch: &mut EncodeScratch,
        output: &mut S,
    ) -> Result<()> {
        // First, we extract all special tokens from the non-normalized input
        for segment in SpecialSegmentIterator::new(input, &self.inner.added_vocabulary, false) {
            match segment {
                Segment::SpecialToken(token) => {
                    output.push(PipelineToken::from(token));
                }
                Segment::Text {
                    text: chunk,
                    input_offset,
                } => {
                    let normalized =
                        normalize_all(&self.inner.normalizers, chunk, offset + input_offset)?;

                    // Extract special tokens from the normalized input
                    for segment in
                        SpecialSegmentIterator::new(&normalized, &self.inner.added_vocabulary, true)
                    {
                        match segment {
                            Segment::SpecialToken(token) => {
                                output.push(PipelineToken::from(token));
                            }
                            Segment::Text {
                                text: normalized_chunk,
                                ..
                            } => {
                                // A [`Span`] holds `u32` offsets, which breaks the `PreTokenizer` contract the
                                // `str::get_unchecked` below relies on.
                                // Normalization can grow the text (`Metaspace` widens every space to a 3-byte delimiter),
                                // which is why this is checked here and not on `input`.
                                if normalized_chunk.len() > u32::MAX as usize {
                                    return Err(format!(
                                        "sequence too long to pre-tokenize: {} bytes after normalization, the limit is {}",
                                        normalized_chunk.len(),
                                        u32::MAX
                                    )
                                    .into());
                                }
                                // Pre-tokenize the chunk of normalized text
                                let EncodeScratch {
                                    model: model_scratch,
                                    pre_tokens,
                                    split: pre_tokenizer_scratch,
                                } = &mut *scratch;
                                pre_tokens.clear();
                                self.inner.pre_tokenizer.pre_tokenize(
                                    normalized_chunk,
                                    pre_tokenizer_scratch,
                                    pre_tokens,
                                )?;
                                let _ = output.room(pre_tokens.len());
                                #[cfg(debug_assertions)]
                                for pre_token in pre_tokens.iter() {
                                    let range = pre_token.range();
                                    debug_assert!(
                                        range.start <= range.end
                                            && normalized_chunk.is_char_boundary(range.start)
                                            && normalized_chunk.is_char_boundary(range.end),
                                        "{:?} broke the PreTokenizer contract: emitted {pre_token:?} for {normalized_chunk:?}",
                                        self.inner.pre_tokenizer,
                                    );
                                }
                                // The whole span list at once; see Model::tokenize_spans.
                                // SAFETY: `PreTokenizer` guarantees every span is a valid range of
                                // `normalized_chunk`, which is what lets the model slice it unchecked.
                                self.inner.model.tokenize_spans(
                                    normalized_chunk,
                                    pre_tokens,
                                    model_scratch,
                                    output,
                                )?;
                            }
                        }
                    }
                }
            };
        }
        Ok(())
    }

    /// [`Self::encode_sequence_into`] into a fresh buffer, for the callers that want one back.
    /// Encode `input`, appending its ids to `out`.
    ///
    /// The entry point that allocates nothing per call: no `Encoding`, no `Vec<Encoding>` from the
    /// handle, and no copy out of either. [`Self::encode`] is this plus those wrappers, and an
    /// encode-only caller in a loop -- a server, a benchmark -- wants this one.
    ///
    /// Falls back to the general path when the post-processor actually has something to add
    /// around the sequence, since that has to assemble a whole `Encoding` anyway.
    pub fn encode_into(
        &self,
        input: &str,
        options: &EncodeOptions,
        out: &mut Vec<PipelineToken>,
    ) -> Result<()> {
        let mut scratch = self.inner.scratch_pool.get(&self.inner.model);
        // A template that adds nothing frames to exactly the sequence, so there is no special
        // case for it: `frame` appends the same ids either way.
        self.frame(
            Document::from(input),
            &self.inner.post_processor.single,
            options.add_special_tokens,
            &mut scratch,
            out,
            None,
        )
    }

    /// Decode token ids back to a `String`.
    ///
    /// Two routes, picked by what the model's vocab store actually holds:
    ///
    /// * **byte-level BPE** -- [`byte_level::transform_vocab`] already replaced every entry with
    ///   its decoded raw bytes when the model was built, so decoding is a concatenation of
    ///   borrowed slices. See [`Self::decode_byte_level`].
    /// * **everything else** -- the store holds the token strings as written, so the configured
    ///   [`DecoderWrapper`] still has to invert whatever the pre-tokenizer did. Same shape as the
    ///   released `Tokenizer::decode`.
    ///
    /// [`byte_level::transform_vocab`]: crate::utils::byte_level::transform_vocab
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        // Irrefutable in a `bpe`-only build, where `PipelineModel` has exactly one variant, and
        // nightly lints a leading irrefutable pattern in a let chain. Nesting the `if` instead
        // would trade this for `collapsible_if` on every build that has more than one model.
        #[allow(irrefutable_let_patterns)]
        if let PipelineModel::BPE(bpe) = &self.inner.model
            && bpe.is_byte_level()
        {
            return Ok(self.decode_byte_level(bpe, ids, skip_special_tokens));
        }
        let tokens = ids
            .iter()
            .filter_map(|&id| {
                if id >= self.inner.added_id_min {
                    self.inner
                        .added_vocabulary
                        .simple_id_to_token(id)
                        .or_else(|| self.inner.model.id_to_token(id))
                        .filter(|token| {
                            !skip_special_tokens
                                || !self.inner.added_vocabulary.is_special_token(token)
                        })
                } else {
                    self.inner.model.id_to_token(id)
                }
            })
            .collect::<Vec<_>>();

        match &self.inner.decoder {
            Some(decoder) => decoder.decode(tokens),
            None => Ok(tokens.join(" ")),
        }
    }

    /// Decode for a byte-level BPE, whose vocab entries are already decoded raw bytes.
    fn decode_byte_level(
        &self,
        bpe: &PipelineBPE,
        ids: &[u32],
        skip_special_tokens: bool,
    ) -> String {
        // Byte-level tokens average ~4 bytes
        let mut out: Vec<u8> = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            if id >= self.inner.added_id_min
                && let Some(token) = self.inner.added_vocabulary.simple_id_to_token(id)
            {
                if !skip_special_tokens || !self.inner.added_vocabulary.is_special_token(&token) {
                    out.extend_from_slice(token.as_bytes());
                }
                continue;
            }
            if let Some(bytes) = bpe.id_to_token_bytes(id) {
                out.extend_from_slice(bytes);
            }
        }
        match String::from_utf8(out) {
            Ok(decoded) => decoded,
            Err(invalid) => String::from_utf8_lossy(invalid.as_bytes()).into_owned(),
        }
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

pub trait Model {
    type Scratch: ModelScratch;

    fn tokenize_pipeline<S: TokenSink>(
        &self,
        sequence: &str,
        scratch: &mut Self::Scratch,
        output: &mut S,
    ) -> Result<()>;

    /// Every pre-token of a chunk in one call.
    ///
    /// The pipeline has the whole span list before the model runs, so handing them over one at a
    /// time bought nothing and cost a virtual call, a slice, a `Result` and an output capacity
    /// check per pre-token -- on English that is one round trip per ~5 bytes.
    ///
    /// The default is the loop it replaces, so a model only overrides this if it has per-chunk
    /// work to hoist out of the loop.
    fn tokenize_spans<S: TokenSink>(
        &self,
        chunk: &str,
        spans: &[Span],
        scratch: &mut Self::Scratch,
        output: &mut S,
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
    #[cfg(feature = "unigram")]
    Unigram(Unigram),
    #[cfg(feature = "wordlevel")]
    WordLevel(WordLevel),
    #[cfg(feature = "wordpiece")]
    WordPiece(PipelineWordPiece),
}

impl PipelineModel {
    /// `id -> token` used by [`PipelineTokenizer::decode`]
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        match self {
            Self::BPE(model) => model.id_to_token(id),
            #[cfg(feature = "unigram")]
            Self::Unigram(model) => model.id_to_token(id),
            #[cfg(feature = "wordlevel")]
            Self::WordLevel(model) => model.id_to_token(id),
            #[cfg(feature = "wordpiece")]
            Self::WordPiece(model) => model.id_to_token(id),
        }
    }
}

/// A set of buffers and other state the model needs to encode efficiently,
/// reused among calls to [`PipelineTokenizer::encode`].
///
/// Each model gets its own variant.
impl Model for PipelineModel {
    type Scratch = PipelineModelScratch;

    fn tokenize_pipeline<S: TokenSink>(
        &self,
        sequence: &str,
        scratch: &mut Self::Scratch,
        output: &mut S,
    ) -> Result<()> {
        match (self, scratch) {
            (Self::BPE(model), PipelineModelScratch::BPE(scratch)) => {
                model.tokenize_pipeline(sequence, scratch, output)
            }
            #[cfg(feature = "unigram")]
            (Self::Unigram(model), PipelineModelScratch::Unigram(scratch)) => {
                model.tokenize_pipeline(sequence, scratch, output)
            }
            #[cfg(feature = "wordlevel")]
            (Self::WordLevel(model), PipelineModelScratch::WordLevel(scratch)) => {
                model.tokenize_pipeline(sequence, scratch, output)
            }
            #[cfg(feature = "wordpiece")]
            (Self::WordPiece(model), PipelineModelScratch::WordPiece(scratch)) => {
                model.tokenize_pipeline(sequence, scratch, output)
            }
            _ => unreachable!(),
        }
    }

    fn tokenize_spans<S: TokenSink>(
        &self,
        chunk: &str,
        spans: &[Span],
        scratch: &mut Self::Scratch,
        output: &mut S,
    ) -> Result<()> {
        match (self, scratch) {
            (Self::BPE(model), PipelineModelScratch::BPE(scratch)) => {
                model.tokenize_spans(chunk, spans, scratch, output)
            }
            #[cfg(feature = "unigram")]
            (Self::Unigram(model), PipelineModelScratch::Unigram(scratch)) => {
                model.tokenize_spans(chunk, spans, scratch, output)
            }
            #[cfg(feature = "wordlevel")]
            (Self::WordLevel(model), PipelineModelScratch::WordLevel(scratch)) => {
                model.tokenize_spans(chunk, spans, scratch, output)
            }
            #[cfg(feature = "wordpiece")]
            (Self::WordPiece(model), PipelineModelScratch::WordPiece(scratch)) => {
                model.tokenize_spans(chunk, spans, scratch, output)
            }
            _ => unreachable!(),
        }
    }

    fn init_scratch(&self) -> Self::Scratch {
        match self {
            Self::BPE(bpe) => PipelineModelScratch::BPE(bpe.init_scratch()),
            #[cfg(feature = "wordlevel")]
            Self::WordLevel(_) => Self::Scratch::WordLevel(()),
            #[cfg(feature = "wordpiece")]
            Self::WordPiece(wordpiece) => Self::Scratch::WordPiece(wordpiece.init_scratch()),
            #[cfg(feature = "unigram")]
            Self::Unigram(unigram) => Self::Scratch::Unigram(unigram.init_scratch()),
        }
    }
}

#[derive(Default)]
pub enum PipelineModelScratch {
    BPE(BpeScratch),
    #[cfg(feature = "wordlevel")]
    WordLevel(()),
    #[cfg(feature = "wordpiece")]
    WordPiece(WordPieceScratch),
    #[cfg(feature = "unigram")]
    Unigram(UnigramScratch),
    /// We need a default value to be able to use [`mem::take`] in [`ScratchGuard::drop`]
    #[default]
    None,
}

impl ModelScratch for PipelineModelScratch {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PaddingStrategy;

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
                Segment::Text { text, .. } => (Some(text), None),
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

    #[test]
    fn padding_defaults_to_none() {
        let pipeline = hello_pipeline();

        assert!(pipeline.get_padding().is_none());
    }

    #[test]
    fn from_parts_padding_is_visible_through_get_padding() {
        let pipeline = hello_pipeline_with_padding(PaddingParams {
            pad_id: 42,
            ..PaddingParams::default()
        });

        assert_eq!(pipeline.get_padding().unwrap().pad_id, 42);
    }

    // "hhello" tokenizes to 2 ids and "hello" to 1 (both merge down to `hello`, but the leading
    // "h" in "hhello" has nothing left to merge with once "hello" is taken), which is exactly the
    // uneven batch padding exists for.
    #[test]
    fn wait_leaves_a_batch_unpadded_with_no_padding_config() {
        let pipeline = hello_pipeline();

        let encodings = pipeline
            .encode(vec!["hhello", "hello"], &EncodeOptions::no_specials())
            .wait()
            .unwrap();

        assert_eq!(encodings[0].len(), 2);
        assert_eq!(encodings[1].len(), 1);
    }

    #[test]
    fn wait_applies_the_tokenizers_configured_padding() {
        let pipeline = hello_pipeline_with_padding(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..PaddingParams::default()
        });

        let encodings = pipeline
            .encode(vec!["hhello", "hello"], &EncodeOptions::no_specials())
            .wait()
            .unwrap();

        assert_eq!(encodings[0].len(), 2);
        assert_eq!(encodings[1].len(), 2);
    }

    #[test]
    fn override_with_replaces_the_tokenizers_configured_padding() {
        let pipeline = hello_pipeline_with_padding(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..PaddingParams::default()
        });
        let options = EncodeOptions {
            padding: Override::With(PaddingParams {
                strategy: PaddingStrategy::Fixed(5),
                ..PaddingParams::default()
            }),
            ..EncodeOptions::no_specials()
        };

        let encodings = pipeline
            .encode(vec!["hhello", "hello"], &options)
            .wait()
            .unwrap();

        assert!(encodings.iter().all(|e| e.len() == 5));
    }

    #[test]
    fn override_off_turns_off_the_tokenizers_configured_padding() {
        let pipeline = hello_pipeline_with_padding(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..PaddingParams::default()
        });
        let options = EncodeOptions {
            padding: Override::Off,
            ..EncodeOptions::no_specials()
        };

        let encodings = pipeline
            .encode(vec!["hhello", "hello"], &options)
            .wait()
            .unwrap();

        assert_eq!(encodings[0].len(), 2);
        assert_eq!(encodings[1].len(), 1);
    }

    fn hello_bpe() -> PipelineBPE {
        use crate::models::bpe::{BpeConfig, Merges, Vocab};

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
        PipelineBPE::from_config(BpeConfig {
            vocab,
            merges,
            ..BpeConfig::default()
        })
        .unwrap()
    }

    fn hello_pipeline() -> PipelineTokenizer {
        PipelineTokenizer::from_parts(
            BucketAddedVocabulary::new(),
            Vec::new(),
            PipelinePreTokenizer::None,
            PipelineModel::BPE(hello_bpe()),
            PipelinePostProcessor::default(),
            None,
            Default::default(),
            None,
        )
    }

    fn hello_pipeline_with_padding(padding: PaddingParams) -> PipelineTokenizer {
        PipelineTokenizer::from_parts(
            BucketAddedVocabulary::new(),
            Vec::new(),
            PipelinePreTokenizer::None,
            PipelineModel::BPE(hello_bpe()),
            PipelinePostProcessor::default(),
            None,
            Default::default(),
            Some(padding),
        )
    }
}
