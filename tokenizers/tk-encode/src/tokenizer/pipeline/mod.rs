use std::iter::Enumerate;
use std::sync::Arc;
use std::vec::IntoIter;
use std::{borrow::Cow, convert::TryFrom};

#[cfg(feature = "normalizers")]
use crate::normalizers::{
    bert::BertNormalizer,
    precompiled::PrecompiledNormalizer,
    strip::StripAccents,
    unicode::{NFC, NFD, NFKC, NFKD, Nmt},
};

#[cfg(feature = "unigram")]
use crate::models::unigram::{Unigram, UnigramScratch};
#[cfg(feature = "wordlevel")]
use crate::models::wordlevel::WordLevel;
#[cfg(feature = "wordpiece")]
use crate::models::wordpiece::{PipelineWordPiece, WordPieceScratch};
// `PipelineWordPiece::id_to_token` is inherent; Unigram and WordLevel get theirs from the trait.
#[cfg(any(feature = "unigram", feature = "wordlevel"))]
use crate::tokenizer::Model as _;

use crate::{
    DecoderRuntime,
    models::bpe::{BpeScratch, PipelineBPE},
    normalizers::{
        byte_level::ByteLevel as ByteLevelNormalizer, metaspace::MetaspaceNormalizer,
        prepend::Prepend, replace::Replace, strip::Strip, utils::Lowercase,
    },
    pipeline::scratch_pool::{EncodeScratch, ScratchPool},
    pre_tokenizers::{
        bert::BertPreTokenizer,
        delimiter::CharDelimiterSplit,
        digits::Digits,
        fixed_length::FixedLength,
        punctuation::Punctuation,
        sequence::PipelineSequence,
        split::Split as SplitPretok,
        whitespace::{Whitespace, WhitespaceSplit},
    },
    processors::template::{Piece, Sequence, Tokens},
    tokenizer::Decoder as _,
    vocab::bucket_added_vocabulary::AddedVocabulary as BucketAddedVocabulary,
};
#[cfg(feature = "parallelism")]
use parallel::StreamingIter;
// The differential "parallel == serial" tests moved to `tk-convert` (they need a `Tokenizer`
// to build from) and have to size an input past the threshold to reach the parallel path at all.
#[cfg(feature = "parallelism")]
pub use parallel::PARALLEL_MIN_BYTES;

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
// `pub` because the config layer's `Sequence` normalizer (a `Vec<NormalizerWrapper>`, so it lives
// in `tk-convert`) implements `pipeline::Normalizer` by running exactly this.
pub fn normalize_all<'a, N: Normalizer>(normalizers: &[N], input: &'a str) -> Result<Cow<'a, str>> {
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
/// `normalizer` field: a `Metaspace` pre-tokenizer contributes one too.
///
/// One variant per concrete normalizer, deliberately: naming a `NormalizerWrapper` here would make
/// every variant of it reachable — a match arm counts — and that reachability is most of what the
/// slim build exists to remove. Both readers therefore flatten a config `Sequence` into a `Vec` of
/// these rather than carrying the wrapper.
// `pub` because `tk-convert` builds these when it lowers a `NormalizerWrapper`.
#[derive(Debug)]
pub enum PipelineNormalizer {
    /// The text-rewriting half of a `Metaspace` pre-tokenizer.
    Metaspace(MetaspaceNormalizer),
    /// A literal `Replace`, built directly by the slim JSON reader.
    Replace(Replace),
    /// A `Prepend`, built directly by the slim JSON reader.
    Prepend(Prepend),
    // The rest are the remaining `NormalizerWrapper` variants, spelled out so the slim reader can
    // build each one without naming the wrapper. Only the four table-backed families sit behind
    // `normalizers`; the others carry no Unicode tables and are always available.
    Strip(Strip),
    Lowercase(Lowercase),
    ByteLevel(ByteLevelNormalizer),
    #[cfg(feature = "normalizers")]
    Bert(BertNormalizer),
    #[cfg(feature = "normalizers")]
    StripAccents(StripAccents),
    #[cfg(feature = "normalizers")]
    NFC(NFC),
    #[cfg(feature = "normalizers")]
    NFD(NFD),
    #[cfg(feature = "normalizers")]
    NFKC(NFKC),
    #[cfg(feature = "normalizers")]
    NFKD(NFKD),
    #[cfg(feature = "normalizers")]
    Nmt(Nmt),
    #[cfg(feature = "normalizers")]
    Precompiled(PrecompiledNormalizer),
}

/// A flattened normalizer chain. Applying the members in order is exactly what
/// the `Sequence` normalizer the slim reader flattened away did.
// `pub` for the same reason as `PipelineNormalizer`: `tk-convert` needs it to replay added
// tokens through a lowered chain.
pub struct NormalizerChain<'a>(pub &'a [PipelineNormalizer]);


impl Normalizer for NormalizerChain<'_> {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        normalize_all(self.0, input)
    }
}

impl Normalizer for PipelineNormalizer {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>> {
        match self {
            Self::Metaspace(normalizer) => normalizer.normalize(input),
            Self::Replace(normalizer) => normalizer.normalize(input),
            Self::Prepend(normalizer) => normalizer.normalize(input),
            Self::Strip(normalizer) => normalizer.normalize(input),
            Self::Lowercase(normalizer) => normalizer.normalize(input),
            Self::ByteLevel(normalizer) => normalizer.normalize(input),
            #[cfg(feature = "normalizers")]
            Self::Bert(normalizer) => normalizer.normalize(input),
            #[cfg(feature = "normalizers")]
            Self::StripAccents(normalizer) => normalizer.normalize(input),
            #[cfg(feature = "normalizers")]
            Self::NFC(normalizer) => normalizer.normalize(input),
            #[cfg(feature = "normalizers")]
            Self::NFD(normalizer) => normalizer.normalize(input),
            #[cfg(feature = "normalizers")]
            Self::NFKC(normalizer) => normalizer.normalize(input),
            #[cfg(feature = "normalizers")]
            Self::NFKD(normalizer) => normalizer.normalize(input),
            #[cfg(feature = "normalizers")]
            Self::Nmt(normalizer) => normalizer.normalize(input),
            #[cfg(feature = "normalizers")]
            Self::Precompiled(normalizer) => normalizer.normalize(input),
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
    #[cfg(feature = "unicode-scripts")]
    UnicodeScripts(crate::pre_tokenizers::unicode_scripts::UnicodeScripts),
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
            #[cfg(feature = "unicode-scripts")]
            Self::UnicodeScripts(pretok) => pretok.pre_tokenize(text, scratch, out),
            Self::Whitespace(pretok) => pretok.pre_tokenize(text, scratch, out),
            Self::WhitespaceSplit(pretok) => pretok.pre_tokenize(text, scratch, out),
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

impl PipelinePostProcessor {
    /// Assemble one from an already-built single and pair template.
    ///
    /// Exists so both readers stop at the same door: the slim reader builds the two templates from
    /// JSON, `tk-convert` builds them from a `PostProcessorWrapper`, and neither needs the
    /// fields.
    pub fn new(single: Template, pair: Template) -> Self {
        Self { single, pair }
    }

    /// The two templates back out again, for composing a `Sequence` post-processor out of the
    /// members it lowered to. Read-only: `compose` picks one of them, it does not edit them.
    pub fn templates(&self) -> (&Template, &Template) {
        (&self.single, &self.pair)
    }
}

// `Slice`, `Seq`, `Template` and the `build_slices`/`compose` helpers below are `pub` because the
// `PostProcessorWrapper` lowering moved to `tk-convert` and is written in terms of them.
// Keeping one set of rules beats a second copy: `compose`'s "at most one arranging member" rule and
// `build_slices`' template validation are the parts that decide ids.
#[derive(Clone, Debug)]
pub enum Slice {
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
pub enum Seq {
    A,
    B,
}

#[derive(Debug)]
pub struct Template {
    slices: Box<[Slice]>,
    n_special: usize,
    has_type_ids: bool,
}

impl Template {
    pub fn new(slices: Vec<Slice>) -> Self {
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

    /// The slices, for a writer that has to spell this template back out as a
    /// `TemplateProcessing`. `n_special` and `has_type_ids` are both derived from them, so this is
    /// the whole of the template's content.
    pub fn slices(&self) -> &[Slice] {
        &self.slices
    }
}

pub fn build_slices(pieces: &[Piece], specials: &Tokens, is_pair: bool) -> Result<Vec<Slice>> {
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

/// A pass-through template does nothing: no special tokens, and sequences in the default
/// arrangement (`$A`, or `$A $B` with the default type ids 0 then 1). Such a member is a no-op in a
/// Sequence and is dropped when composing. Anything else adds tokens or reorders/retags.
fn is_pass_through(slices: &[Slice]) -> bool {
    matches!(
        slices,
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

/// Compose the members of a Sequence post processor.
///
/// The reference applies each member in turn, and every member retags the whole previous output to
/// its own sequence type id. A static template cannot represent that chaining, so we only support
/// the representable case: at most one member that adds tokens or reorders/retags, wrapped by any
/// number of pass-through members (which are no-ops and dropped). More than one is rejected.
pub fn compose<'a>(templates: impl Iterator<Item = &'a Template>) -> Result<Template> {
    let templates = templates.collect::<Vec<_>>();
    let mut chosen: Option<&Template> = None;
    for template in &templates {
        if is_pass_through(&template.slices) {
            continue;
        }
        if chosen.replace(template).is_some() {
            return Err(
                "post processor Sequence with multiple sequence referencing members is not supported".into(),
            );
        }
    }
    let chosen = chosen
        .or_else(|| templates.first().copied())
        .ok_or("empty Sequence post processor is not supported")?;
    Ok(Template::new(chosen.slices.to_vec()))
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
    decoder: Option<DecoderRuntime>,
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

impl PipelineTokenizer {
    /// Assemble a pipeline from parts that are already lowered.
    ///
    /// The only constructor of a `TokenizerInner`, and therefore the only place `added_id_min` and
    /// the scratch pool are derived — both readers go through it: the slim JSON reader in
    /// [`from_json`](super::pipeline::from_json), and `tk-convert`'s lowering of a
    /// `Tokenizer`. Two copies of this is how the two paths drift.
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

    /// Whether any normalization step runs before the pre-tokenizer. An empty normalizer
    /// `Sequence` in the config counts as none, since it is elided on the way in.
    pub fn has_normalizer(&self) -> bool {
        !self.inner.normalizers.is_empty()
    }

    pub fn get_pre_tokenizer(&self) -> &PipelinePreTokenizer {
        &self.inner.pre_tokenizer
    }

    pub fn get_post_processor(&self) -> &PipelinePostProcessor {
        &self.inner.post_processor
    }

    /// The whole flattened normalizer chain, in the order it runs.
    ///
    /// [`Self::has_normalizer`] answers the only question the encode path asks; this is for a
    /// writer, which needs the members themselves. A config `Sequence` was flattened on the way in,
    /// so what comes back is the concatenation, not the nesting.
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
    #[cfg(feature = "unigram")]
    Unigram(Unigram),
    #[cfg(feature = "wordlevel")]
    WordLevel(WordLevel),
    #[cfg(feature = "wordpiece")]
    WordPiece(PipelineWordPiece),
}

impl PipelineModel {
    /// `id -> token`, for the decoder-chain route in [`PipelineTokenizer::decode`] and for a writer,
    /// which needs it to name the special tokens a post-processor template refers to by id.
    ///
    /// A byte-level BPE answers with its *decoded* bytes, lossily -- see
    /// [`PipelineBPE::id_to_token`]. That is the right answer for a printable special token, which
    /// byte-level leaves alone, and the wrong one for a token with a space in it. A writer should
    /// prefer the added vocabulary, where such tokens actually live.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pre_tokenizers::split::SplitPattern;
    use crate::utils::byte_level::GPT2_REGEX_STR;

    fn variant_name(pre_tokenizer: &PipelinePreTokenizer) -> &'static str {
        match pre_tokenizer {
            PipelinePreTokenizer::Bert(_) => "Bert",
            PipelinePreTokenizer::Delimiter(_) => "Delimiter",
            PipelinePreTokenizer::Digits(_) => "Digits",
            PipelinePreTokenizer::FixedLength(_) => "FixedLength",
            PipelinePreTokenizer::Punctuation(_) => "Punctuation",
            PipelinePreTokenizer::Sequence(_) => "Sequence",
            PipelinePreTokenizer::Split(_) => "Split",
            #[cfg(feature = "unicode-scripts")]
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

        // `mut` + a cfg'd push rather than a `#[cfg]` on a `vec!` element: attributes on
        // expression position are not stable.
        #[allow(unused_mut)]
        let mut cases = vec![
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
            PipelinePreTokenizer::Whitespace(Whitespace),
            PipelinePreTokenizer::WhitespaceSplit(WhitespaceSplit),
            PipelinePreTokenizer::None,
        ];
        #[cfg(feature = "unicode-scripts")]
        cases.push(PipelinePreTokenizer::UnicodeScripts(
            crate::pre_tokenizers::unicode_scripts::UnicodeScripts::new(),
        ));

        let mut covered: Vec<&str> = cases.iter().map(variant_name).collect();
        covered.sort_unstable();
        covered.dedup();
        #[allow(unused_mut)]
        let mut expected: Vec<&str> = vec![
            "Bert",
            "Delimiter",
            "Digits",
            "FixedLength",
            "None",
            "Punctuation",
            "Sequence",
            "Split",
            "Whitespace",
            "WhitespaceSplit",
        ];
        #[cfg(feature = "unicode-scripts")]
        expected.push("UnicodeScripts");
        expected.sort_unstable();
        assert_eq!(
            covered, expected,
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
