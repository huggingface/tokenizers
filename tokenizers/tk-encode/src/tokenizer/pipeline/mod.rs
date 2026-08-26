use std::collections::BTreeMap;
use std::iter::Enumerate;
use std::sync::Arc;
use std::vec::IntoIter;

#[cfg(feature = "unigram")]
use crate::models::unigram::{Unigram, UnigramScratch};
#[cfg(feature = "wordlevel")]
use crate::models::wordlevel::WordLevel;
#[cfg(feature = "wordpiece")]
use crate::models::wordpiece::{PipelineWordPiece, WordPieceScratch};
use crate::{
    DecoderRuntime,
    models::bpe::{BpeScratch, PipelineBPE},
    pipeline::scratch_pool::{EncodeScratch, ScratchPool},
    tokenizer::Decoder as _,
    vocab::bucket_added_vocabulary::AddedVocabulary as BucketAddedVocabulary,
};
#[cfg(feature = "parallelism")]
pub use parallel::PARALLEL_MIN_BYTES;
#[cfg(feature = "parallelism")]
use parallel::StreamingIter;

use super::Result;

#[cfg(feature = "parallelism")]
mod parallel;
mod scratch_pool;

pub use scratch_pool::ModelScratch;

pub use bitsplit::Span;

mod normalizer;
mod post_processor;
mod pre_tokenizer;

pub use normalizer::{Normalizer, NormalizerChain, PipelineNormalizer, normalize_all};
pub use post_processor::{PipelinePostProcessor, Seq, Slice, Template, build_slices, compose};
pub use pre_tokenizer::{
    PipelinePreTokenizer, PreTokenizer, PreTokenizerScratch, SplitPolicy, split, split_delimiter,
    split_matches,
};

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
        full_input: &str,
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
                .extract_next(self.input, self.offset, self.normalized)
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
    pub fn from_parts(
        added_vocabulary: BucketAddedVocabulary,
        normalizers: Vec<PipelineNormalizer>,
        pre_tokenizer: PipelinePreTokenizer,
        model: PipelineModel,
        post_processor: PipelinePostProcessor,
        decoder: Option<DecoderRuntime>,
        role_to_token: BTreeMap<String, String>,
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
                scratch_pool: ScratchPool::new(),
            }),
        }
    }
}
#[derive(Clone)]
pub enum Input {
    Single(String),
    Pair(String, String),
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

    fn len(&self) -> usize {
        self.as_slice().len()
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

impl From<&[&str]> for Inputs {
    fn from(b: &[&str]) -> Self {
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

enum HandleState {
    Blocking(Enumerate<IntoIter<Result<Encoding>>>),
    #[cfg(feature = "parallelism")]
    Streaming(StreamingIter),
}

pub struct EncodeHandle {
    state: HandleState,
}

impl EncodeHandle {
    /// Fully computed results, for the serial case
    fn blocking(results: Vec<Result<Encoding>>) -> Self {
        Self {
            state: HandleState::Blocking(results.into_iter().enumerate()),
        }
    }

    #[cfg(feature = "parallelism")]
    fn streaming(it: StreamingIter) -> Self {
        Self {
            state: HandleState::Streaming(it),
        }
    }

    fn len(&self) -> usize {
        match &self.state {
            HandleState::Blocking(it) => it.len(),
            #[cfg(feature = "parallelism")]
            HandleState::Streaming(it) => it.len(),
        }
    }
}

impl EncodeHandle {
    /// Wait for all scheduled encoding to finish
    ///
    /// Returns in input order
    pub fn wait(self) -> Result<Vec<Encoding>> {
        // XXX: `Vec::new` does not allocate anything when capacity == 0, so creating empty
        // Encodings should not allocate anything either
        let mut out = vec![Encoding::empty(); self.len()];
        for (seq, res) in self {
            out[seq] = res?;
        }
        Ok(out)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Encoding {
    ids: Vec<PipelineToken>,
    type_ids: Option<Vec<u8>>,
}

impl Encoding {
    fn empty() -> Self {
        Self {
            ids: Vec::new(),
            type_ids: None,
        }
    }

    fn new(ids: Vec<PipelineToken>, type_ids: Option<Vec<u8>>) -> Self {
        debug_assert!(type_ids.as_ref().is_none_or(|t| t.len() == ids.len()));
        Self { ids, type_ids }
    }
}

impl Encoding {
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
}
/// Iterator yields results in completion order
pub struct HandleIter {
    handle: EncodeHandle,
}

impl Iterator for HandleIter {
    type Item = (usize, Result<Encoding>);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.handle.state {
            HandleState::Blocking(it) => it.next(),
            #[cfg(feature = "parallelism")]
            HandleState::Streaming(it) => it.next(),
        }
    }
}

impl IntoIterator for EncodeHandle {
    type Item = (usize, Result<Encoding>);
    type IntoIter = HandleIter;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter { handle: self }
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

    /// Encode `input` into token ids.
    ///
    /// Special tokens are matched in two passes:
    ///  1. on the raw input,
    ///  2. then on each segment after normalization
    ///
    /// This way, special / added tokens declared on raw or normalized text are both caught.
    /// The remaining text is pre-tokenized and run through the model span by span.
    pub fn encode(&self, inputs: impl Into<Inputs>, add_special_tokens: bool) -> EncodeHandle {
        let inputs = inputs.into();
        assert!(
            inputs.len() < usize::MAX,
            "we use usize::MAX as a sentinel value for the completion queue, we don't support batches larger than that"
        );
        #[cfg(not(feature = "parallelism"))]
        return EncodeHandle::blocking(self.encode_serial(inputs, add_special_tokens));

        #[cfg(feature = "parallelism")]
        parallel::encode(self, inputs, add_special_tokens)
    }

    fn encode_serial(&self, inputs: Inputs, add_special_tokens: bool) -> Vec<Result<Encoding>> {
        let mut scratch = self.inner.scratch_pool.get(&self.inner.model);
        match inputs {
            Inputs::Single(input) => {
                vec![self.encode_one(input, add_special_tokens, &mut scratch)]
            }
            Inputs::Batch(batch) => {
                let mut output = Vec::with_capacity(batch.len());
                for input in batch {
                    output.push(self.encode_one(input, add_special_tokens, &mut scratch));
                }
                output
            }
        }
    }

    fn encode_one(
        &self,
        input: Input,
        add_special_tokens: bool,
        scratch: &mut EncodeScratch,
    ) -> Result<Encoding> {
        match input {
            Input::Single(seq) => {
                let toks = self.encode_sequence_with(&seq, scratch)?;
                Ok(self.post_process(toks, None, add_special_tokens)?)
            }
            Input::Pair(s1, s2) => {
                let a = self.encode_sequence_with(&s1, scratch)?;
                let b = self.encode_sequence_with(&s2, scratch)?;
                Ok(self.post_process(a, Some(b), add_special_tokens)?)
            }
        }
    }

    fn post_process(
        &self,
        s1: Vec<PipelineToken>,
        s2: Option<Vec<PipelineToken>>,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        let pp = &self.inner.post_processor;
        let template = if s2.is_some() { &pp.pair } else { &pp.single };

        // Fast path: when nothing has to be woven around the sequence, the sequence already *is*
        // the encoding, and the loop below is a second full pass over every token plus a second
        // allocation to hold it. That pass is invisible on sparse text and the dominant added cost
        // on token-dense text, which is where it showed up: the regression against
        // `poc/target-encode-clean` tracked tokens-per-byte, not bytes.
        //
        // Taken only when the template would reproduce `s1` exactly: one sequence, it is `A`, no
        // type ids to interleave, and any `Specials` slices are being skipped anyway.
        if s2.is_none() && !template.has_type_ids {
            let mut sequences = 0usize;
            let reproduces_s1 = template.slices.iter().all(|slice| match slice {
                Slice::Specials { .. } => !add_special_tokens,
                Slice::Sequence { seq: Seq::A, .. } => {
                    sequences += 1;
                    true
                }
                Slice::Sequence { seq: Seq::B, .. } => false,
            });
            if reproduces_s1 && sequences == 1 {
                return Ok(Encoding::new(s1, None));
            }
        }

        let seq_len = s1.len() + s2.as_ref().map_or(0, Vec::len);
        let cap = template.n_special + seq_len;

        let (mut a, mut b) = (Some(s1), s2);
        let mut ids = Vec::with_capacity(cap);
        let mut type_ids = template.has_type_ids.then(|| Vec::with_capacity(cap));

        for slice in &template.slices {
            match slice {
                Slice::Specials { tokens, type_id } => {
                    if !add_special_tokens {
                        continue;
                    }
                    ids.extend_from_slice(tokens);
                    if let Some(tids) = type_ids.as_mut() {
                        tids.resize(tids.len() + tokens.len(), *type_id);
                    }
                }
                Slice::Sequence { seq, type_id } => {
                    let tokens = match seq {
                        Seq::A => a.take(),
                        Seq::B => b.take(),
                    }
                    .ok_or("[BUG] valid template should guarantee each referenced sequence is provided exactly once")?;
                    if let Some(tids) = type_ids.as_mut() {
                        tids.resize(tids.len() + tokens.len(), *type_id);
                    }
                    ids.extend(tokens);
                }
            }
        }

        Ok(Encoding::new(ids, type_ids))
    }

    /// Encode one sequence, appending its ids to `output`.
    ///
    /// Takes the buffer rather than returning one, so a caller that already owns somewhere to put
    /// the ids -- [`Self::encode_into`] -- pays neither an allocation nor a copy for them.
    fn encode_sequence_into(
        &self,
        input: &str,
        scratch: &mut EncodeScratch,
        output: &mut Vec<PipelineToken>,
    ) -> Result<()> {
        // First, we extract all special tokens from the non-normalized input
        for segment in SpecialSegmentIterator::new(input, &self.inner.added_vocabulary, false) {
            match segment {
                Segment::SpecialToken(token) => {
                    output.push(PipelineToken::from(token));
                }
                Segment::Text { text: chunk, .. } => {
                    let normalized = normalize_all(&self.inner.normalizers, chunk)?;

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
                                output.reserve(pre_tokens.len());
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
    fn encode_sequence_with(
        &self,
        input: &str,
        scratch: &mut EncodeScratch,
    ) -> Result<Vec<PipelineToken>> {
        let mut output = Vec::with_capacity(input.len() / 4);
        self.encode_sequence_into(input, scratch, &mut output)?;
        Ok(output)
    }

    /// Encode `input`, appending its ids to `out`.
    ///
    /// The entry point that allocates nothing per call: no `Encoding`, no `Vec<Encoding>` from the
    /// handle, and no copy out of either. [`Self::encode`] is this plus those wrappers, and an
    /// encode-only caller in a loop -- a server, a benchmark -- wants this one.
    ///
    /// Falls back to the general path when the post-processor actually has something to weave
    /// around the sequence, since that has to assemble a whole `Encoding` anyway.
    pub fn encode_into(
        &self,
        input: &str,
        add_special_tokens: bool,
        out: &mut Vec<PipelineToken>,
    ) -> Result<()> {
        let mut scratch = self.inner.scratch_pool.get(&self.inner.model);
        let template = &self.inner.post_processor.single;
        let mut sequences = 0usize;
        // The same question `post_process` asks: would the template just reproduce the sequence?
        let reproduces_sequence = !template.has_type_ids
            && template.slices.iter().all(|slice| match slice {
                Slice::Specials { .. } => !add_special_tokens,
                Slice::Sequence { seq: Seq::A, .. } => {
                    sequences += 1;
                    true
                }
                Slice::Sequence { seq: Seq::B, .. } => false,
            })
            && sequences == 1;
        if reproduces_sequence {
            return self.encode_sequence_into(input, &mut scratch, out);
        }
        let encoding = self.encode_one(
            Input::Single(input.to_owned()),
            add_special_tokens,
            &mut scratch,
        )?;
        out.extend_from_slice(encoding.ids());
        Ok(())
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

    fn tokenize_pipeline(
        &self,
        sequence: &str,
        scratch: &mut Self::Scratch,
        output: &mut Vec<PipelineToken>,
    ) -> Result<()>;

    /// Every pre-token of a chunk in one call.
    ///
    /// The pipeline has the whole span list before the model runs, so handing them over one at a
    /// time bought nothing and cost a virtual call, a slice, a `Result` and an output capacity
    /// check per pre-token -- on English that is one round trip per ~5 bytes.
    ///
    /// The default is the loop it replaces, so a model only overrides this if it has per-chunk
    /// work to hoist out of the loop.
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

    struct FixedMatcher(Vec<((usize, usize), u32)>);
    impl PipelinePatternMatcher for FixedMatcher {
        fn extract_next(
            &self,
            _s: &str,
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
}
