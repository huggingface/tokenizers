use std::convert::TryInto;
use std::iter::Enumerate;
use std::sync::Arc;
use std::vec::IntoIter;
use std::{borrow::Cow, convert::TryFrom};

use crate::{
    DecoderWrapper, ModelWrapper, PostProcessorWrapper, PreTokenizerWrapper, Tokenizer,
    models::{
        bpe::{BpeScratch, PipelineBPE},
        unigram::{Unigram, UnigramScratch},
        wordlevel::WordLevel,
        wordpiece::{PipelineWordPiece, WordPieceScratch},
    },
    normalizers::{NormalizerWrapper, metaspace::MetaspaceNormalizer},
    pipeline::scratch_pool::{EncodeScratch, ScratchPool},
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
    processors::{
        bert::BertProcessing,
        roberta::RobertaProcessing,
        template::{Piece, Sequence, Tokens},
    },
    tokenizer::{Decoder as _, Model as _},
    utils::byte_level::GPT2_REGEX_STR,
    vocab::bucket_added_vocabulary::{
        AddedToken as BucketAddedToken, AddedVocabulary as BucketAddedVocabulary,
    },
};
#[cfg(feature = "parallelism")]
use parallel::StreamingIter;

use super::{Result, SplitDelimiterBehavior};

#[cfg(feature = "parallelism")]
mod parallel;
mod scratch_pool;

pub use scratch_pool::ModelScratch;

use atomsplit::classify::classify;
pub use atomsplit::fsm::Span;

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
#[derive(Debug)]
pub struct PipelinePostProcessor {
    single: Template,
    pair: Template,
}

#[derive(Clone, Debug)]
enum Slice {
    Specials {
        tokens: Box<[PipelineToken]>,
        type_id: u8,
    },
    Sequence {
        seq: Seq,
        type_id: u8,
    },
}

#[derive(Clone, Copy, Debug)]
enum Seq {
    A,
    B,
}

#[derive(Debug)]
struct Template {
    slices: Box<[Slice]>,
    n_special: usize,
    has_type_ids: bool,
}

impl Template {
    fn new(slices: Vec<Slice>) -> Self {
        let n_special = slices
            .iter()
            .map(|s| {
                if let Slice::Specials { tokens, .. } = s {
                    tokens.len()
                } else {
                    0
                }
            })
            .sum();
        let has_type_ids = slices.iter().any(|s| match s {
            Slice::Specials { type_id, .. } | Slice::Sequence { type_id, .. } => *type_id != 0,
        });
        Self {
            slices: slices.into_boxed_slice(),
            n_special,
            has_type_ids,
        }
    }
}

fn build_slices(pieces: &[Piece], specials: &Tokens, is_pair: bool) -> Result<Vec<Slice>> {
    let (mut seen_a, mut seen_b) = (false, false);
    let mut slices = Vec::new();
    for piece in pieces {
        match piece {
            Piece::Sequence {
                id: Sequence::A,
                type_id,
            } => {
                if seen_a {
                    return Err(
                        "not supported: template references sequence A more than once".into(),
                    );
                }
                seen_a = true;
                slices.push(Slice::Sequence {
                    seq: Seq::A,
                    type_id: u8::try_from(*type_id)
                        .map_err(|_| "not supported: type_id out of range")?,
                });
            }
            Piece::Sequence {
                id: Sequence::B,
                type_id,
            } => {
                if seen_b {
                    return Err(
                        "not supported: template references sequence B more than once".into(),
                    );
                }
                seen_b = true;
                slices.push(Slice::Sequence {
                    seq: Seq::B,
                    type_id: u8::try_from(*type_id)
                        .map_err(|_| "not supported: type_id out of range")?,
                });
            }
            Piece::SpecialToken {
                id: token_string,
                type_id,
            } => {
                let special = specials.0.get(token_string).ok_or_else(|| {
                    format!("not supported: unknown special token: `{token_string}`")
                })?;
                slices.push(Slice::Specials {
                    tokens: special
                        .ids()
                        .iter()
                        .map(|&id| PipelineToken::from(id))
                        .collect(),
                    type_id: u8::try_from(*type_id)
                        .map_err(|_| "not supported: type_id out of range")?,
                });
            }
        }
    }
    if !seen_a {
        return Err("not supported: template does not reference sequence A".into());
    }
    if is_pair && !seen_b {
        return Err("not supported: pair template does not reference sequence B".into());
    }
    if !is_pair && seen_b {
        return Err(
            "not supported: single template references sequence B (it should only refer to A)"
                .into(),
        );
    }
    Ok(slices)
}

impl TryFrom<&PostProcessorWrapper> for PipelinePostProcessor {
    type Error = crate::Error;

    fn try_from(value: &PostProcessorWrapper) -> Result<Self> {
        fn one(id: u32, tid: u8) -> Slice {
            Slice::Specials {
                tokens: Box::new([PipelineToken::from(id)]),
                type_id: tid,
            }
        }
        fn multi(ids: &[u32], tid: u8) -> Slice {
            Slice::Specials {
                tokens: ids.iter().map(|&id| PipelineToken::from(id)).collect(),
                type_id: tid,
            }
        }
        use Seq::{A, B};
        let sq = |seq, type_id| Slice::Sequence { seq, type_id };

        match value {
            PostProcessorWrapper::Bert(BertProcessing {
                cls: (_, cls_id),
                sep: (_, sep_id),
            }) => Ok(Self {
                single: Template::new(vec![one(*cls_id, 0), sq(A, 0), one(*sep_id, 0)]),
                pair: Template::new(vec![
                    one(*cls_id, 0),
                    sq(A, 0),
                    one(*sep_id, 0),
                    sq(B, 1),
                    one(*sep_id, 1),
                ]),
            }),
            PostProcessorWrapper::Roberta(RobertaProcessing {
                cls: (_, cls_id),
                sep: (_, sep_id),
                ..
            }) => Ok(Self {
                single: Template::new(vec![one(*cls_id, 0), sq(A, 0), one(*sep_id, 0)]),
                pair: Template::new(vec![
                    one(*cls_id, 0),
                    sq(A, 0),
                    multi(&[*sep_id, *sep_id], 0),
                    sq(B, 0),
                    one(*sep_id, 0),
                ]),
            }),
            PostProcessorWrapper::Template(pp) => Ok(Self {
                single: Template::new(build_slices(
                    pp.single.as_slice(),
                    pp.get_special_tokens(),
                    false,
                )?),
                pair: Template::new(build_slices(
                    pp.pair.as_slice(),
                    pp.get_special_tokens(),
                    true,
                )?),
            }),
            PostProcessorWrapper::ByteLevel(_) => Ok(Self {
                single: Template::new(vec![sq(A, 0)]),
                pair: Template::new(vec![sq(A, 0), sq(B, 1)]),
            }),
            PostProcessorWrapper::Sequence(sequence) => {
                let members = sequence
                    .as_ref()
                    .iter()
                    .map(Self::try_from)
                    .collect::<Result<Vec<_>>>()?;
                Ok(Self {
                    single: compose(members.iter().map(|m| &m.single))?,
                    pair: compose(members.iter().map(|m| &m.pair))?,
                })
            }
        }
    }
}

fn is_sequence(s: &Slice) -> bool {
    matches!(s, Slice::Sequence { .. })
}

/// Split into three groups, depending on the template's content:
/// - [<before>], [SeqA<middle>SeqB], [<after>]
/// - [<before>], [SeqA], [<after>]
fn into_parts(template: &Template) -> Result<(Vec<Slice>, Vec<Slice>, Vec<Slice>)> {
    let s = &template.slices;
    let (Some(first), Some(last)) = (
        s.iter().position(is_sequence),
        s.iter().rposition(is_sequence),
    ) else {
        return Err("not supported: could not find any sequence in post processor template".into());
    };
    Ok((
        s[..first].to_vec(),
        s[first..=last].to_vec(),
        s[last + 1..].to_vec(),
    ))
}

/// An identity core arranges the sequences exactly as the defaults: `$A` alone, or `$A $B` with
/// the default type ids (0 then 1). Only such a core is transparent and safe to drop when composing
/// a Sequence. Any other all-sequence core reorders or retags, so it is a real arrangement.
fn is_identity_core(core: &[Slice]) -> bool {
    matches!(
        core,
        [Slice::Sequence {
            seq: Seq::A,
            type_id: 0
        }] | [
            Slice::Sequence {
                seq: Seq::A,
                type_id: 0
            },
            Slice::Sequence {
                seq: Seq::B,
                type_id: 1
            }
        ]
    )
}

fn compose<'a>(templates: impl Iterator<Item = &'a Template>) -> Result<Template> {
    let parts = templates.map(into_parts).collect::<Result<Vec<_>>>()?;

    let mut core = None;
    for (_, c, _) in &parts {
        if !is_identity_core(c) && core.replace(c.clone()).is_some() {
            return Err(
                "post processor Sequence with multiple sequence referencing members is not supported".into(),
            );
        }
    }

    let core = core
        .or_else(|| parts.first().map(|(_, c, _)| c.clone()))
        .ok_or("empty Sequence post processor is not supported")?;

    let mut slices = Vec::new();
    for (prefix, _, _) in parts.iter().rev() {
        slices.extend(prefix.iter().cloned());
    }
    slices.extend(core);
    for (_, _, suffix) in parts {
        slices.extend(suffix.iter().cloned());
    }
    Ok(Template::new(slices))
}

impl Default for PipelinePostProcessor {
    fn default() -> Self {
        Self {
            single: Template::new(vec![Slice::Sequence {
                seq: Seq::A,
                type_id: 0,
            }]),
            pair: Template::new(vec![
                Slice::Sequence {
                    seq: Seq::A,
                    type_id: 0,
                },
                Slice::Sequence {
                    seq: Seq::B,
                    type_id: 1,
                },
            ]),
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
    decoder: Option<DecoderWrapper>,
    /// Lowest id owned by the added vocabulary, or `u32::MAX` when there is none.
    /// Allows to skip the added vocabulary lookup if the token id is lower than this value.
    added_id_min: u32,
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
            inner: Arc::new(TokenizerInner {
                added_vocabulary,
                normalizers,
                pre_tokenizer,
                model,
                post_processor: tok
                    .get_post_processor()
                    .map(PipelinePostProcessor::try_from)
                    .transpose()?
                    .unwrap_or_else(PipelinePostProcessor::default),
                decoder: tok.get_decoder().cloned(),
                added_id_min,
                scratch_pool: ScratchPool::new(),
            }),
        })
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

    pub fn into_parts(self) -> EncodingParts {
        EncodingParts {
            ids: self.ids,
            type_ids: self.type_ids,
        }
    }
}

pub struct EncodingParts {
    pub ids: Vec<PipelineToken>,
    pub type_ids: Option<Vec<u8>>,
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
        match inputs {
            Inputs::Single(input) => {
                vec![self.encode_one(input, add_special_tokens)]
            }
            Inputs::Batch(batch) => {
                let mut output = Vec::with_capacity(batch.len());
                for input in batch {
                    output.push(self.encode_one(input, add_special_tokens));
                }
                output
            }
        }
    }

    fn encode_one(&self, input: Input, add_special_tokens: bool) -> Result<Encoding> {
        match input {
            Input::Single(seq) => {
                let toks = self.encode_sequence(&seq)?;
                Ok(self.post_process(toks, None, add_special_tokens))
            }
            Input::Pair(s1, s2) => {
                let a = self.encode_sequence(&s1)?;
                let b = self.encode_sequence(&s2)?;
                Ok(self.post_process(a, Some(b), add_special_tokens))
            }
        }
    }

    fn post_process(
        &self,
        s1: Vec<PipelineToken>,
        s2: Option<Vec<PipelineToken>>,
        add_special_tokens: bool,
    ) -> Encoding {
        let pp = &self.inner.post_processor;
        let template = if s2.is_some() { &pp.pair } else { &pp.single };

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
                    .expect("[BUG] valid template should guarantee each referenced sequence is provided exactly once");
                    if let Some(tids) = type_ids.as_mut() {
                        tids.resize(tids.len() + tokens.len(), *type_id);
                    }
                    ids.extend(tokens);
                }
            }
        }

        Encoding::new(ids, type_ids)
    }

    pub fn encode_sequence(&self, input: &str) -> Result<Vec<PipelineToken>> {
        let mut output = Vec::with_capacity(input.len() / 4);
        let mut scratch = self.inner.scratch_pool.get(&self.inner.model);
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
                                for pre_token in pre_tokens {
                                    let range = pre_token.range();
                                    debug_assert!(
                                        range.start <= range.end
                                            && normalized_chunk.is_char_boundary(range.start)
                                            && normalized_chunk.is_char_boundary(range.end),
                                        "{:?} broke the PreTokenizer contract: emitted {pre_token:?} for {normalized_chunk:?}",
                                        self.inner.pre_tokenizer,
                                    );
                                    // SAFETY: `PreTokenizer` guarantees every span is a valid range of `normalized_chunk`
                                    let sequence = unsafe { normalized_chunk.get_unchecked(range) };
                                    self.inner.model.tokenize_pipeline(
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
    #[cfg(feature = "parallelism")]
    use crate::{parallelism::set_num_threads, pipeline::parallel::PARALLEL_MIN_BYTES};
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
        assert_eq!(*encoded.first().unwrap().ids(), [1, 2]);
        assert_pipeline_matches_reference(&tok, "hello world");
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

    #[cfg(feature = "parallelism")]
    fn pipeline_ids(pipeline: &PipelineTokenizer, input: &str) -> Vec<u32> {
        pipeline
            .encode(input, false)
            .wait()
            .unwrap()
            .remove(0)
            .ids()
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
    #[cfg(feature = "parallelism")]
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

    fn assert_pipeline_matches_reference(tok: &Tokenizer, input: &str) {
        let pipeline = PipelineTokenizer::try_from(tok).unwrap();
        for add_special_tokens in [false, true] {
            let expected = tok
                .encode(input, add_special_tokens)
                .unwrap()
                .get_ids()
                .to_vec();
            let got: Vec<u32> = pipeline
                .encode(input, add_special_tokens)
                .wait()
                .unwrap()
                .first()
                .unwrap()
                .ids
                .iter()
                .map(|t| t.id())
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
        let ids = |enc: Vec<Encoding>| {
            enc.first()
                .unwrap()
                .ids
                .iter()
                .map(|t| t.id())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            ids(pipeline.encode("hello world", true).wait().unwrap()),
            vec![0, 2, 3, 1]
        );
        assert_eq!(
            ids(pipeline.encode("hello world", false).wait().unwrap()),
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
        use crate::processors::sequence::Sequence as ProcSequence;
        use crate::processors::template::TemplateProcessing;

        let member = |prefix: &str, suffix: &str, p_id: u32, s_id: u32| {
            TemplateProcessing::builder()
                .try_single(format!("{prefix} $A {suffix}"))
                .unwrap()
                .try_pair(format!("{prefix} $A $B:1 {suffix}:1"))
                .unwrap()
                .special_tokens(vec![(prefix, p_id), (suffix, s_id)])
                .build()
                .unwrap()
        };
        let tok = wordlevel_tokenizer(
            vec![
                ("[X]", 100),
                ("[Y]", 101),
                ("[P]", 102),
                ("[Q]", 103),
                ("hello", 2),
                ("world", 3),
            ],
            Some(PostProcessorWrapper::Sequence(ProcSequence::new(vec![
                PostProcessorWrapper::Template(member("[X]", "[Y]", 100, 101)),
                PostProcessorWrapper::Template(member("[P]", "[Q]", 102, 103)),
            ]))),
        );

        let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
        let ids: Vec<u32> = pipeline
            .encode("hello world", true)
            .wait()
            .unwrap()
            .first()
            .unwrap()
            .ids
            .iter()
            .map(|t| t.id())
            .collect();
        assert_eq!(ids, vec![102, 100, 2, 3, 101, 103]);
    }

    #[test]
    fn conversion_rejects_sequence_with_two_arranging_members() {
        use crate::processors::bert::BertProcessing;
        use crate::processors::sequence::Sequence as ProcSequence;

        let tok = wordlevel_tokenizer(
            vec![("A", 100), ("B", 101), ("hello", 2), ("world", 3)],
            Some(PostProcessorWrapper::Sequence(ProcSequence::new(vec![
                PostProcessorWrapper::Bert(BertProcessing::new(
                    ("B".to_string(), 101),
                    ("A".to_string(), 100),
                )),
                PostProcessorWrapper::Bert(BertProcessing::new(
                    ("B".to_string(), 101),
                    ("A".to_string(), 100),
                )),
            ]))),
        );
        let err = conversion_error(&tok);
        assert!(err.contains("not supported"), "{}", err);
    }

    #[test]
    fn roberta_pair_without_specials_keeps_type_ids_zero() {
        use crate::processors::roberta::RobertaProcessing;

        // RoBERTa tags both pair sides type 0. `add_special_tokens = false` must suppress only the
        // special tokens, not fall back to the default A=0/B=1 tagging.
        let tok = wordlevel_tokenizer(
            vec![
                ("<s>", 0),
                ("</s>", 1),
                ("hello", 2),
                ("world", 3),
                ("foo", 4),
            ],
            Some(PostProcessorWrapper::Roberta(RobertaProcessing::new(
                ("</s>".to_string(), 1),
                ("<s>".to_string(), 0),
            ))),
        );
        let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
        let batch = pipeline
            .encode(("hello world", "foo"), false)
            .wait()
            .unwrap();
        let enc = batch.first().unwrap();

        assert!(
            enc.type_ids().is_none_or(|t| t.iter().all(|&x| x == 0)),
            "expected all-zero type ids, got {:?}",
            enc.type_ids()
        );
        let expected = tok.encode(("hello world", "foo"), false).unwrap();
        let ids: Vec<u32> = enc.ids().iter().map(|t| t.id()).collect();
        assert_eq!(expected.get_ids(), ids.as_slice());
    }

    #[test]
    fn sequence_keeps_reordering_member_core() {
        use crate::pre_tokenizers::byte_level::ByteLevel;
        use crate::processors::sequence::Sequence as ProcSequence;
        use crate::processors::template::TemplateProcessing;

        // ByteLevel has an identity core (safe to drop); the template reorders the pair to `$B $A`.
        // Compose must keep the reordering core, not discard it as trivial.
        let reorder = TemplateProcessing::builder()
            .try_single("$A")
            .unwrap()
            .try_pair("$B $A")
            .unwrap()
            .build()
            .unwrap();
        let tok = wordlevel_tokenizer(
            vec![("hello", 2), ("world", 3)],
            Some(PostProcessorWrapper::Sequence(ProcSequence::new(vec![
                PostProcessorWrapper::ByteLevel(ByteLevel::default()),
                PostProcessorWrapper::Template(reorder),
            ]))),
        );
        let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
        let batch = pipeline.encode(("hello", "world"), false).wait().unwrap();
        let ids: Vec<u32> = batch
            .first()
            .unwrap()
            .ids()
            .iter()
            .map(|t| t.id())
            .collect();
        // `$B $A` => world (3) before hello (2)
        assert_eq!(ids, vec![3, 2]);
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

    #[cfg(feature = "parallelism")]
    use std::sync::{Mutex, PoisonError};
    #[cfg(feature = "parallelism")]
    static LOCK: Mutex<()> = Mutex::new(());

    #[cfg(feature = "parallelism")]
    fn assert_parallel_matches(
        tokenizer: &PipelineTokenizer,
        inputs: Inputs,
        add_special_tokens: bool,
    ) {
        let _g = LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        set_num_threads(1);
        let serial = tokenizer
            .encode(inputs.clone(), add_special_tokens)
            .wait()
            .unwrap();
        for n in [2, 4, 8] {
            set_num_threads(n);
            for _ in 0..3 {
                let par = tokenizer
                    .encode(inputs.clone(), add_special_tokens)
                    .wait()
                    .unwrap();
                assert_eq!(par, serial);
            }
        }
        set_num_threads(0);
    }

    #[cfg(feature = "parallelism")]
    fn repeat_to(phrase: &str, min_bytes: usize) -> String {
        phrase.repeat(min_bytes / phrase.len() + 1)
    }

    #[cfg(feature = "parallelism")]
    #[test]
    fn parallel_matches_serial_batch_identity() {
        let tok = wordlevel_tokenizer(vec![("<unk>", 0), ("hello", 1), ("world", 2)], None);
        let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
        let inputs = Inputs::from(vec!["hello world".to_string(); 1000]);
        for add in [false, true] {
            assert_parallel_matches(&pipeline, inputs.clone(), add);
        }
    }

    #[cfg(feature = "parallelism")]
    #[test]
    fn parallel_matches_serial_batch_bert() {
        use crate::processors::bert::BertProcessing;
        let tok = wordlevel_tokenizer(
            vec![("[CLS]", 0), ("[SEP]", 1), ("hello", 2), ("world", 3)],
            Some(PostProcessorWrapper::Bert(BertProcessing::new(
                ("[SEP]".to_string(), 1),
                ("[CLS]".to_string(), 0),
            ))),
        );
        let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
        let inputs = Inputs::from(vec!["hello world".to_string(); 1000]);
        for add in [false, true] {
            assert_parallel_matches(&pipeline, inputs.clone(), add);
        }
    }

    #[cfg(feature = "parallelism")]
    #[test]
    fn parallel_matches_serial_pairs_bert() {
        use crate::processors::bert::BertProcessing;
        let tok = wordlevel_tokenizer(
            vec![("[CLS]", 0), ("[SEP]", 1), ("hello", 2), ("world", 3)],
            Some(PostProcessorWrapper::Bert(BertProcessing::new(
                ("[SEP]".to_string(), 1),
                ("[CLS]".to_string(), 0),
            ))),
        );
        let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
        let inputs = Inputs::from(vec![
            ("hello world".to_string(), "world hello".to_string());
            700
        ]);
        for add in [false, true] {
            assert_parallel_matches(&pipeline, inputs.clone(), add);
        }
    }

    #[cfg(feature = "parallelism")]
    #[test]
    fn parallel_matches_serial_batch_roberta() {
        use crate::processors::roberta::RobertaProcessing;
        let tok = wordlevel_tokenizer(
            vec![("<s>", 0), ("</s>", 1), ("hello", 2), ("world", 3)],
            Some(PostProcessorWrapper::Roberta(RobertaProcessing::new(
                ("</s>".to_string(), 1),
                ("<s>".to_string(), 0),
            ))),
        );
        let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
        let inputs = Inputs::from(vec!["hello world".to_string(); 1000]);
        for add in [false, true] {
            assert_parallel_matches(&pipeline, inputs.clone(), add);
        }
    }

    #[cfg(feature = "parallelism")]
    #[test]
    fn parallel_matches_serial_long_single_with_specials() {
        let mut tok = wordlevel_tokenizer(
            vec![("<unk>", 0), ("hello", 1), ("world", 2), ("<sep>", 3)],
            None,
        );
        tok.add_special_tokens([crate::tokenizer::AddedToken::from("<sep>", true)])
            .unwrap();
        let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
        let inputs = Inputs::from(repeat_to(
            "hello world <sep> ",
            2 * PARALLEL_MIN_BYTES + 4096,
        ));
        for add in [false, true] {
            assert_parallel_matches(&pipeline, inputs.clone(), add);
        }
    }

    #[cfg(feature = "parallelism")]
    #[test]
    fn parallel_matches_serial_mixed_batch_with_edges() {
        let tok = wordlevel_tokenizer(vec![("<unk>", 0), ("hello", 1), ("world", 2)], None);
        let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
        let mut batch = vec![
            String::new(),
            "hello".to_string(),
            "hello world".to_string(),
        ];
        batch.extend(vec!["hello world".to_string(); 1000]);
        assert_parallel_matches(&pipeline, Inputs::from(batch), false);
    }

    #[cfg(feature = "parallelism")]
    #[test]
    fn streaming_iterator_yields_each_seq_once() {
        let _g = LOCK.lock().unwrap();
        let tok = wordlevel_tokenizer(vec![("<unk>", 0), ("hello", 1), ("world", 2)], None);
        let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
        let inputs = Inputs::from(vec!["hello world".to_string(); 1000]);

        set_num_threads(1);
        let serial = pipeline.encode(inputs.clone(), false).wait().unwrap();

        set_num_threads(4);
        let mut streamed: Vec<Option<Encoding>> = vec![None; serial.len()];
        for (seq, res) in pipeline.encode(inputs, false) {
            assert!(
                streamed[seq].is_none(),
                "seq {seq} was yielded more than once"
            );
            streamed[seq] = Some(res.unwrap());
        }
        set_num_threads(0);

        let streamed: Vec<Encoding> = streamed
            .into_iter()
            .map(|e| e.expect("a seq was never yielded"))
            .collect();
        assert_eq!(streamed, serial);
    }

    #[cfg(feature = "parallelism")]
    #[test]
    fn streaming_handle_drop_after_partial_consume_is_clean() {
        let _g = LOCK.lock().unwrap();
        let tok = wordlevel_tokenizer(vec![("<unk>", 0), ("hello", 1), ("world", 2)], None);
        let pipeline = PipelineTokenizer::try_from(&tok).unwrap();
        let inputs = Inputs::from(vec!["hello world".to_string(); 1000]);

        set_num_threads(4);
        let mut it = pipeline.encode(inputs, false).into_iter();
        assert!(it.next().is_some());
        assert!(it.next().is_some());
        drop(it);
        set_num_threads(0);
    }
}
