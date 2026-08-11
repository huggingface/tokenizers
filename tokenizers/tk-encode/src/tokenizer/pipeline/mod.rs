use std::convert::TryInto;
use std::iter::Enumerate;
use std::vec::IntoIter;
use std::{borrow::Cow, convert::TryFrom};

use crate::models::bpe::{BpeScratch, PipelineBPE};
use crate::models::unigram::{Unigram, UnigramScratch};
use crate::models::wordlevel::WordLevel;
use crate::models::wordpiece::{PipelineWordPiece, WordPieceScratch};
use crate::pipeline::scratch_pool::{EncodeScratch, ScratchPool};
use crate::processors::bert::BertProcessing;
use crate::processors::roberta::RobertaProcessing;
use crate::utils::byte_level::GPT2_REGEX_STR;
use crate::vocab::bucket_added_vocabulary::{
    AddedToken as BucketAddedToken, AddedVocabulary as BucketAddedVocabulary,
};
use crate::{
    DecoderWrapper, ModelWrapper, PostProcessorWrapper, PreTokenizerWrapper, Tokenizer,
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
    tokenizer::{Decoder as _, Model as _},
};

use super::{Result, SplitDelimiterBehavior};

use atomsplit::classify::classify;
pub use atomsplit::fsm::Span;

mod scratch_pool;

pub use scratch_pool::ModelScratch;

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

/// Range-based pre-tokenization: yields spans into the input rather than owned
/// substrings, so the pipeline can pre-tokenize without allocating.
///
/// # Safety
///
/// [`PreTokenizer::pre_tokenize`] produces [`Span`] objects that are later consumed by [`PipelineTokenizer::encode_sequence`].
/// For performance reasons, [`PipelineTokenizer::encode_sequence`] turns every span into a `&str` with [`str::get_unchecked`].
/// As a consequence, every implementation of this trait *MUST* ensure the following invariants for each [`Span`]:
///
/// * `span.end <= text.len()`: the span is inside the text
/// * `span.start <= span.end`: the span is not reversed
/// * `text.is_char_boundary(start)` and `text.is_char_boundary(end)`: the start and end offset must be UTF-8 char boundaries
/// * The offsets are relative to the `text`.
pub unsafe trait PreTokenizer {
    /// Split `text` into pre-tokens, appending to `out`. Ranges are into `text`.
    /// `scratch` holds the working buffers, see [`PreTokenizerScratch`].
    fn pre_tokenize(
        &self,
        text: &str,
        scratch: &mut PreTokenizerScratch,
        out: &mut Vec<Span>,
    ) -> Result<()>;
}

/// The working buffers a [`PreTokenizer`] needs to split a text into pre-tokens.
#[derive(Default)]
pub struct PreTokenizerScratch {
    /// One [`atomsplit`] atom tag per input byte, what [`atomsplit::classify::classify`] writes
    /// and the FSMs read.
    tags: Vec<u8>,
    /// An intermediate buffer in which the FSM writes the Span before they get appended to the output
    spans: Vec<Span>,
    /// The two buffers a [`PipelineSequence`] reads and writes as each of its children subdivides
    /// the spans the child before it produced. Handed out by [`Self::take_pair`] rather than
    /// borrowed, so the sequence can pass the rest of the scratch to its children while it holds
    /// them.
    pair: [Vec<Span>; 2],
}

impl PreTokenizerScratch {
    /// Tag every byte of `bytes` with its [`atomsplit`] atom class, run `fsm` over the tags, and
    /// append the spans to `out`.
    pub fn split_on_tags(
        &mut self,
        bytes: &[u8],
        fsm: impl FnOnce(&[u8], &[u8], &mut [Span]) -> usize,
        out: &mut Vec<Span>,
    ) {
        let n = bytes.len();
        let Self { tags, spans, .. } = self;
        if tags.len() < n {
            tags.resize(n, 0);
        }
        if spans.len() < n + 1 {
            spans.resize(n + 1, Span::default());
        }
        // Assign a tag to each byte
        classify(bytes, &mut tags[..n]);
        // Run the fsm on the tags to determine where to cut
        let k = fsm(bytes, &tags[..n], &mut spans[..n + 1]);
        // Copy spans to output
        out.extend_from_slice(&spans[..k]);
    }

    /// Run `fsm` over `bytes` and append the spans it cut to `out`, for an FSM that scans the bytes
    /// itself and so has no use for the atom tags. Skips the [`classify`]
    pub fn split_on_bytes(
        &mut self,
        bytes: &[u8],
        fsm: impl FnOnce(&[u8], &mut [Span]) -> usize,
        out: &mut Vec<Span>,
    ) {
        let n = bytes.len();
        let spans = &mut self.spans;
        if spans.len() < n + 1 {
            spans.resize(n + 1, Span::default());
        }
        let k = fsm(bytes, &mut spans[..n + 1]);
        out.extend_from_slice(&spans[..k]);
    }

    /// Take the sequence's two buffers, leaving empty ones behind.
    pub fn take_pair(&mut self) -> [Vec<Span>; 2] {
        std::mem::take(&mut self.pair)
    }

    /// Give back what [`Self::take_pair`] handed out, so the next sequence reuses the allocations.
    pub fn put_pair(&mut self, pair: [Vec<Span>; 2]) {
        self.pair = pair;
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

// SAFETY: every arm but `None` forwards to another `PreTokenizer`, which upholds the contract itself.
// `None` emits the single span covering all of `text`, whose ends are `0` and `text.len()`.
unsafe impl PreTokenizer for PipelinePreTokenizer {
    fn pre_tokenize(
        &self,
        text: &str,
        scratch: &mut PreTokenizerScratch,
        out: &mut Vec<Span>,
    ) -> Result<()> {
        match self {
            Self::None => {
                out.push(Span {
                    start: 0,
                    end: text.len() as u32,
                });
                Ok(())
            }
            Self::Bert(pretok) => pretok.pre_tokenize(text, scratch, out),
            Self::Delimiter(pretok) => pretok.pre_tokenize(text, scratch, out),
            Self::Digits(pretok) => pretok.pre_tokenize(text, scratch, out),
            Self::FixedLength(pretok) => pretok.pre_tokenize(text, scratch, out),
            Self::Punctuation(pretok) => pretok.pre_tokenize(text, scratch, out),
            Self::Sequence(pretok) => pretok.pre_tokenize(text, scratch, out),
            Self::Split(pretok) => pretok.pre_tokenize(text, scratch, out),
            Self::UnicodeScripts(pretok) => pretok.pre_tokenize(text, scratch, out),
            Self::Whitespace(pretok) => pretok.pre_tokenize(text, scratch, out),
            Self::WhitespaceSplit(pretok) => pretok.pre_tokenize(text, scratch, out),
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
// todo: the frame is all the post-processing the pipeline does today. What the post-processors in
// `crate::processors` did and this does not, in rough order of how much it matters:
//
//  - Pairs. `PipelineTokenizer::encode` frames each side of an `Inputs::Pair` on its own, so a
//    BERT pair comes out as two `[CLS] A [SEP]` sequences instead of one `[CLS] A [SEP] B [SEP]`.
//    `RobertaProcessing` puts a `[SEP]` on both sides of its pair and `TemplateProcessing` keeps
//    a second template for it, so neither reduces to one frame.
//  - `type_ids`. BERT numbers the pair 1, Roberta forces every token to 0, and every template
//    piece carries its own `type_id`. A `PipelineToken` holds an id and nothing else.
//  - The templates the conversion below refuses (a `single` referencing the sequence twice) and
//    the field it ignores (the `id` on `Piece::Sequence`, so `$B` reads as `$A`).
//  - `trim_offsets`, on both `ByteLevel` and `RobertaProcessing`. There are no offsets to trim
//    while the pipeline computes none, so both are parsed and then ignored.
//  - Everything the released crate's `Encoding` carries besides the ids: the token strings,
//    `offsets`, `words`, `special_tokens_mask`, `attention_mask`, and the token range recorded per
//    input sequence. That last one is what `char_to_token`, `word_to_tokens` and `token_to_word`
//    look up, and the post-processors were careful to leave the frame tokens outside it. Those
//    three are the reason callers pick this crate over a plain BPE encoder, so dropping the ids-only
//    shape is a decision to revisit, not a settled one.
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
                prefix: vec![PipelineToken::from(*cls_id)].into_boxed_slice(),
                suffix: vec![PipelineToken::from(*sep_id)].into_boxed_slice(),
            }),
            PostProcessorWrapper::Roberta(RobertaProcessing {
                cls: (_, cls_id),
                sep: (_, sep_id),
                ..
            }) => Ok(Self {
                prefix: vec![PipelineToken::from(*cls_id)].into_boxed_slice(),
                suffix: vec![PipelineToken::from(*sep_id)].into_boxed_slice(),
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
                            let token_ids = special.ids().iter().copied().map(PipelineToken::from);
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
/// stages over borrowed ranges to avoid the reference path's allocations.
// todo: truncation and padding are features this crate used to have and now has nowhere to put.
// Both rewrote the per-token arrays that went out with the reference `Encoding`, and the pipeline
// never had them, so a `tokenizer.json` asking for either is parsed and then ignored (there is a
// test pinning that in `crate::tokenizer::serialization`). Building them back means:
//
//  - Truncation config: `max_length`, `stride`, a `direction` (left or right) and, for a pair, a
//    strategy picking which side gives up tokens (longest first, first only, second only).
//  - A `stride` above 0 does not just cut the tail off, it emits overlapping windows of the
//    sequence. The released crate returns the first window and hangs the rest off it as
//    `overflowing`, which a flat `Vec<PipelineToken>` cannot express. Decide where they go before
//    writing any of the cutting logic.
//  - `max_length` counts the frame. The released crate subtracts `PostProcessor::added_tokens`
//    from it first, and refuses a `stride` that no longer fits in what is left.
//  - Padding config: a strategy (pad every sequence to a fixed length, or to the longest in the
//    batch), a `direction`, `pad_to_multiple_of`, `pad_id`, `pad_type_id` and `pad_token`. Padding
//    to the batch longest needs every sequence in hand at once, so it belongs where
//    [`PipelineTokenizer::encode`] already holds the whole batch, not in `encode_sequence`.
//  - Padding is the only thing that makes an attention mask worth returning: it is all ones until
//    pad tokens show up. Whatever ends up carrying the ids has to carry that mask alongside.
pub struct PipelineTokenizer {
    added_vocabulary: BucketAddedVocabulary,
    normalizers: Vec<PipelineNormalizer>,
    pre_tokenizer: PipelinePreTokenizer,
    model: PipelineModel,
    post_processor: PipelinePostProcessor,
    decoder: Option<DecoderWrapper>,
    /// Lowest id owned by the added vocabulary, or `u32::MAX` when there is none.
    /// Allows to skip the added vocabulary lookup if the token id is lower than this value.
    added_id_min: u32,
    scratch_pool: ScratchPool,
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

        let added_id_min = added_vocabulary
            .get_added_tokens_decoder()
            .keys()
            .copied()
            .min()
            .unwrap_or(u32::MAX);

        Ok(Self {
            added_vocabulary,
            normalizers,
            pre_tokenizer,
            model,
            post_processor: tok
                .get_post_processor()
                .map(PipelinePostProcessor::try_from)
                .transpose()?
                .unwrap_or_default(),
            decoder: tok.get_decoder().cloned(),
            added_id_min,
            scratch_pool: ScratchPool::new(),
        })
    }
}

pub enum Inputs {
    Single(String),
    Pair(String, String),
    Batch(Vec<String>),
    PairBatch(Vec<(String, String)>),
}

impl From<String> for Inputs {
    fn from(s: String) -> Self {
        Inputs::Single(s)
    }
}

impl From<&str> for Inputs {
    fn from(s: &str) -> Self {
        Inputs::Single(s.to_owned())
    }
}

impl From<&String> for Inputs {
    fn from(s: &String) -> Self {
        Inputs::Single(s.to_owned())
    }
}

impl From<Vec<String>> for Inputs {
    fn from(b: Vec<String>) -> Self {
        Inputs::Batch(b)
    }
}

impl From<&[&str]> for Inputs {
    fn from(b: &[&str]) -> Self {
        Inputs::Batch(b.iter().map(|s| (*s).to_owned()).collect())
    }
}

impl From<(String, String)> for Inputs {
    fn from(p: (String, String)) -> Self {
        Inputs::Pair(p.0, p.1)
    }
}

impl From<(&str, &str)> for Inputs {
    fn from(p: (&str, &str)) -> Self {
        Inputs::Pair(p.0.to_owned(), p.1.to_owned())
    }
}

impl From<(&String, &String)> for Inputs {
    fn from(p: (&String, &String)) -> Self {
        Inputs::Pair(p.0.to_owned(), p.1.to_owned())
    }
}

impl From<Vec<(String, String)>> for Inputs {
    fn from(b: Vec<(String, String)>) -> Self {
        Inputs::PairBatch(b)
    }
}

impl From<&[(&str, &str)]> for Inputs {
    fn from(b: &[(&str, &str)]) -> Self {
        Inputs::PairBatch(
            b.iter()
                .map(|(s1, s2)| ((*s1).to_owned(), (*s2).to_owned()))
                .collect(),
        )
    }
}

impl From<Vec<&str>> for Inputs {
    fn from(b: Vec<&str>) -> Self {
        Inputs::Batch(b.into_iter().map(|s| s.to_owned()).collect())
    }
}

enum HandleState {
    Blocking(Enumerate<IntoIter<Result<Vec<PipelineToken>>>>),
    // TODO:
    // Streaming
}

pub struct EncodeHandle {
    state: HandleState,
}

impl EncodeHandle {
    /// Fully computed results, for the serial case
    fn blocking(results: Vec<Result<Vec<PipelineToken>>>) -> Self {
        Self {
            state: HandleState::Blocking(results.into_iter().enumerate()),
        }
    }

    fn len(&self) -> usize {
        match &self.state {
            HandleState::Blocking(it) => it.len(),
        }
    }
}

impl EncodeHandle {
    /// Wait for all scheduled encoding to finish
    ///
    /// Returns in input order
    pub fn wait(self) -> Result<Vec<Vec<PipelineToken>>> {
        // XXX: `Vec::new` does not allocate anything when capacity == 0
        let mut out = vec![Vec::new(); self.len()];
        for (seq, res) in self {
            out[seq] = res?;
        }
        Ok(out)
    }
}

/// Iterator yields results in completion order
pub struct HandleIter {
    handle: EncodeHandle,
}

impl Iterator for HandleIter {
    type Item = (usize, Result<Vec<PipelineToken>>);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.handle.state {
            HandleState::Blocking(it) => it.next(),
        }
    }
}

impl IntoIterator for EncodeHandle {
    type Item = (usize, Result<Vec<PipelineToken>>);
    type IntoIter = HandleIter;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter { handle: self }
    }
}

impl PipelineTokenizer {
    pub fn get_model(&self) -> &PipelineModel {
        &self.model
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

        match inputs {
            Inputs::Single(s) => {
                let output = self.encode_sequence(&s, add_special_tokens);
                EncodeHandle::blocking(vec![output])
            }
            // TODO: proper post-processor logic, this is temporary
            Inputs::Pair(s1, s2) => {
                let p1 = self.encode_sequence(&s1, add_special_tokens);
                let p2 = self.encode_sequence(&s2, add_special_tokens);
                EncodeHandle::blocking(vec![p1, p2])
            }
            Inputs::Batch(b) => {
                let mut output = Vec::with_capacity(b.len());
                for seq in b {
                    output.push(self.encode_sequence(&seq, add_special_tokens));
                }
                EncodeHandle::blocking(output)
            }
            // TODO: proper post-processor logic, this is temporary
            Inputs::PairBatch(pb) => {
                let mut output = Vec::with_capacity(pb.len() * 2);
                for (s1, s2) in pb {
                    output.push(self.encode_sequence(&s1, add_special_tokens));
                    output.push(self.encode_sequence(&s2, add_special_tokens));
                }
                EncodeHandle::blocking(output)
            }
        }
    }

    pub fn encode_sequence(
        &self,
        input: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<PipelineToken>> {
        let mut output = Vec::with_capacity(input.len() / 4);
        let mut scratch = self.scratch_pool.get(&self.model);
        let PipelinePostProcessor { prefix, suffix } = &self.post_processor;
        // Prepend prefix tokens, if any
        // todo: handle post-processing when encoding a pair of sequences (currently unsupported by the PipelineTokenizer)
        if add_special_tokens {
            output.extend_from_slice(prefix);
        }
        // First, we extract all special tokens from the non-normalized input
        for segment in SpecialSegmentIterator::new(input, &self.added_vocabulary, false) {
            match segment {
                Segment::SpecialToken(token) => {
                    output.push(PipelineToken::from(token));
                }
                Segment::Text(chunk) => {
                    let normalized = normalize_all(&self.normalizers, chunk)?;

                    // Extract special tokens from the normalized input
                    for segment in
                        SpecialSegmentIterator::new(&normalized, &self.added_vocabulary, true)
                    {
                        match segment {
                            Segment::SpecialToken(token) => {
                                output.push(PipelineToken::from(token));
                            }
                            Segment::Text(normalized_chunk) => {
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
                                self.pre_tokenizer.pre_tokenize(
                                    normalized_chunk,
                                    pre_tokenizer_scratch,
                                    pre_tokens,
                                )?;
                                output.reserve(pre_tokens.len());
                                for pre_token in pre_tokens {
                                    let range = pre_token.range();
                                    debug_assert!(
                                        range.start <= range.end
                                            && normalized_chunk.is_char_boundary(range.start)
                                            && normalized_chunk.is_char_boundary(range.end),
                                        "{:?} broke the PreTokenizer contract: emitted {pre_token:?} for {normalized_chunk:?}",
                                        self.pre_tokenizer,
                                    );
                                    // SAFETY: `PreTokenizer` guarantees every span is a valid range of `normalized_chunk`
                                    let sequence = unsafe { normalized_chunk.get_unchecked(range) };
                                    self.model.tokenize_pipeline(
                                        sequence,
                                        model_scratch,
                                        &mut output,
                                    )?;
                                }
                            }
                        }
                    }
                }
            };
        }
        // Append suffix tokens, if any
        if add_special_tokens {
            output.extend_from_slice(suffix);
        }
        Ok(output)
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
        if let PipelineModel::BPE(bpe) = &self.model
            && bpe.is_byte_level()
        {
            return Ok(self.decode_byte_level(bpe, ids, skip_special_tokens));
        }
        let tokens = ids
            .iter()
            .filter_map(|&id| {
                if id >= self.added_id_min {
                    self.added_vocabulary
                        .simple_id_to_token(id)
                        .or_else(|| self.model.id_to_token(id))
                        .filter(|token| {
                            !skip_special_tokens || !self.added_vocabulary.is_special_token(token)
                        })
                } else {
                    self.model.id_to_token(id)
                }
            })
            .collect::<Vec<_>>();

        match &self.decoder {
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
            if id >= self.added_id_min
                && let Some(token) = self.added_vocabulary.simple_id_to_token(id)
            {
                if !skip_special_tokens || !self.added_vocabulary.is_special_token(&token) {
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
        if d { delim_policy } else { SplitPolicy::Keep }
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
    /// `id -> token`, for the decoder-chain route in [`PipelineTokenizer::decode`].
    fn id_to_token(&self, id: u32) -> Option<String> {
        match self {
            Self::BPE(model) => model.id_to_token(id),
            Self::Unigram(model) => model.id_to_token(id),
            Self::WordLevel(model) => model.id_to_token(id),
            Self::WordPiece(model) => model.id_to_token(id),
        }
    }
}

/// A set of buffers and other state the model needs to encode efficiently,
/// reused among calls to [`PipelineTokenizer::encode`].
///
/// Each model gets its own variant.
#[derive(Default)]
pub enum PipelineModelScratch {
    BPE(BpeScratch),
    WordLevel(()),
    WordPiece(WordPieceScratch),
    Unigram(UnigramScratch),
    /// We need a default value to be able to use [`mem::take`] in [`ScratchGuard::drop`]
    #[default]
    None,
}

impl ModelScratch for PipelineModelScratch {}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::bpe::BPE;
    use crate::models::wordpiece::WordPiece;
    use crate::pre_tokenizers::byte_level::ByteLevel;
    use crate::pre_tokenizers::sequence::Sequence;
    fn variant_name(pre_tokenizer: &PipelinePreTokenizer) -> &'static str {
        match pre_tokenizer {
            PipelinePreTokenizer::Bert(_) => "Bert",
            PipelinePreTokenizer::Delimiter(_) => "Delimiter",
            PipelinePreTokenizer::Digits(_) => "Digits",
            PipelinePreTokenizer::FixedLength(_) => "FixedLength",
            PipelinePreTokenizer::Punctuation(_) => "Punctuation",
            PipelinePreTokenizer::Sequence(_) => "Sequence",
            PipelinePreTokenizer::Split(_) => "Split",
            PipelinePreTokenizer::UnicodeScripts(_) => "UnicodeScripts",
            PipelinePreTokenizer::Whitespace(_) => "Whitespace",
            PipelinePreTokenizer::WhitespaceSplit(_) => "WhitespaceSplit",
            PipelinePreTokenizer::None => "None",
        }
    }

    const HOSTILE: &[&str] = &[
        "",
        " ",
        "\r",
        "\r\n\n",
        "hello world",
        "café naïve",
        "a\u{0301}b\u{0301}\u{0301}c",
        "中文分词。ひらがな 한글",
        "😀👍🏽 👨‍👩‍👧‍👦 x",
        "\u{FEFF}leading bom",
        "مرحبا\u{200F} العربية",
        "नरेंद्र मोदी",
        "hello▁world▁▁",
        "a1b22c333d4444 42 ½²¼ Ⅷ",
        "!!!...?! a, b; c",
        "  \t  trailing   ",
        "Ⓘ\u{200D}x_y a-b'c",
    ];

    /// Every variant must produce spans that [`PipelineTokenizer::encode_sequence`] can hand to `str::get_unchecked`
    #[test]
    fn every_pre_tokenizer_emits_sliceable_spans() {
        use crate::pre_tokenizers::digits::Digits;
        use crate::pre_tokenizers::fixed_length::FixedLength;
        use crate::pre_tokenizers::punctuation::Punctuation;
        use crate::pre_tokenizers::split::SplitPattern;
        use SplitDelimiterBehavior::*;

        let literal_split = |pattern: &str, behavior| {
            SplitPretok::new(SplitPattern::String(pattern.to_owned()), behavior, false).unwrap()
        };
        // The gpt2 regex is recognized, so this routes to `fsm_byte_level` with no regex backend.
        let gpt2_split = SplitPretok::new(
            SplitPattern::Regex(GPT2_REGEX_STR.to_owned()),
            Isolated,
            false,
        )
        .unwrap();

        let cases = vec![
            PipelinePreTokenizer::Bert(BertPreTokenizer),
            PipelinePreTokenizer::Delimiter(CharDelimiterSplit::new(' ')),
            PipelinePreTokenizer::Delimiter(CharDelimiterSplit::new('▁')),
            PipelinePreTokenizer::Digits(Digits::new(true)),
            PipelinePreTokenizer::Digits(Digits::new(false)),
            PipelinePreTokenizer::FixedLength(FixedLength::new(3)),
            PipelinePreTokenizer::FixedLength(FixedLength::new(0)),
            PipelinePreTokenizer::Punctuation(Punctuation::new(Removed)),
            PipelinePreTokenizer::Punctuation(Punctuation::new(Isolated)),
            PipelinePreTokenizer::Punctuation(Punctuation::new(Contiguous)),
            PipelinePreTokenizer::Punctuation(Punctuation::new(MergedWithPrevious)),
            PipelinePreTokenizer::Punctuation(Punctuation::new(MergedWithNext)),
            PipelinePreTokenizer::Sequence(PipelineSequence::new(vec![
                PipelinePreTokenizer::WhitespaceSplit(WhitespaceSplit),
                PipelinePreTokenizer::Punctuation(Punctuation::new(Isolated)),
            ])),
            PipelinePreTokenizer::Split(gpt2_split),
            PipelinePreTokenizer::Split(literal_split("▁", MergedWithPrevious)),
            PipelinePreTokenizer::Split(literal_split("▁", MergedWithNext)),
            PipelinePreTokenizer::Split(literal_split(" ", Removed)),
            PipelinePreTokenizer::UnicodeScripts(UnicodeScripts::new()),
            PipelinePreTokenizer::Whitespace(Whitespace),
            PipelinePreTokenizer::WhitespaceSplit(WhitespaceSplit),
            PipelinePreTokenizer::None,
        ];

        let mut covered: Vec<&str> = cases.iter().map(variant_name).collect();
        covered.sort_unstable();
        covered.dedup();
        assert_eq!(
            covered,
            [
                "Bert",
                "Delimiter",
                "Digits",
                "FixedLength",
                "None",
                "Punctuation",
                "Sequence",
                "Split",
                "UnicodeScripts",
                "Whitespace",
                "WhitespaceSplit",
            ],
            "every PipelinePreTokenizer variant needs a case above",
        );

        let mut scratch = PreTokenizerScratch::default();
        let mut spans = Vec::new();
        for pre_tokenizer in &cases {
            for text in HOSTILE {
                spans.clear();
                pre_tokenizer
                    .pre_tokenize(text, &mut scratch, &mut spans)
                    .unwrap();
                for span in &spans {
                    let range = span.range();
                    assert!(
                        range.start <= range.end,
                        "{} reversed {span:?} on {text:?}",
                        variant_name(pre_tokenizer),
                    );
                    assert!(
                        text.is_char_boundary(range.start) && text.is_char_boundary(range.end),
                        "{} cut {span:?} off a character boundary of {text:?}",
                        variant_name(pre_tokenizer),
                    );
                    // What the pipeline does with the span. Slicing checked here is the point: it
                    // panics on exactly the ranges `get_unchecked` would turn into undefined behavior.
                    assert!(text.get(range).is_some());
                }
            }
        }
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

        let encoded = PipelineTokenizer::try_from(&tok)
            .unwrap()
            .encode("hello world", false)
            .wait()
            .unwrap();
        // Not the unk id: both the `Replace` and the `Split` really ran on the literal path.
        assert_eq!(*encoded.first().unwrap(), [1, 2]);
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

    fn pipeline_ids(pipeline: &PipelineTokenizer, input: &str) -> Vec<u32> {
        pipeline
            .encode(input, false)
            .wait()
            .unwrap()
            .remove(0)
            .iter()
            .map(|t| t.id())
            .collect()
    }

    // A single `&self` tokenizer is meant to be shared across rayon workers, and the scratch it
    // hands each of them carries the pre-token spans of the chunk being encoded. So the workers
    // must not be able to reach each other's: every thread encodes a different input here, and
    // has to get back the answer that input produces on its own.
    //
    // The inputs disagree on both how many tokens they produce and which, so another input's
    // spans cannot yield the right ids by luck. This also only compiles if
    // `PipelineTokenizer: Sync`.
    #[test]
    fn concurrent_encodes_of_different_inputs_stay_independent() {
        use rayon::prelude::*;

        let tok = wordlevel_tokenizer(vec![("<unk>", 0), ("hello", 1), ("world", 2)], None);
        let pipeline = PipelineTokenizer::try_from(&tok).unwrap();

        let inputs = [
            "hello".to_string(),
            "hello world".to_string(),
            "world hello world".to_string(),
            "hello world ".repeat(25),
        ];
        let want: Vec<Vec<u32>> = inputs
            .iter()
            .map(|input| pipeline_ids(&pipeline, input))
            .collect();
        assert_eq!(
            want,
            vec![vec![1], vec![1, 2], vec![2, 1, 2], [1, 2].repeat(25)]
        );

        let all_match = (0..10_000usize).into_par_iter().all(|i| {
            let case = i % inputs.len();
            pipeline_ids(&pipeline, &inputs[case]) == want[case]
        });
        assert!(all_match);
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
}
