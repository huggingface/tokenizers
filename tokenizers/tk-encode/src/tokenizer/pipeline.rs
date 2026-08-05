use std::cell::RefCell;
use std::convert::TryInto;
use std::iter::Enumerate;
use std::mem;
use std::sync::{Mutex, PoisonError};
use std::vec::IntoIter;
use std::{borrow::Cow, convert::TryFrom};

use atomsplit::classify::classify;

use crate::models::bpe::{BpeScratch, PipelineBPE};
use crate::models::unigram::{Unigram, UnigramScratch};
use crate::models::wordlevel::WordLevel;
use crate::models::wordpiece::{PipelineWordPiece, WordPieceScratch};
use crate::processors::bert::BertProcessing;
use crate::processors::roberta::RobertaProcessing;
use crate::utils::byte_level::GPT2_REGEX_STR;
use crate::vocab::bucket_added_vocabulary::{
    AddedToken as BucketAddedToken, AddedVocabulary as BucketAddedVocabulary,
};
use crate::{
    ModelWrapper, PostProcessorWrapper, PreTokenizerWrapper, Token, Tokenizer,
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

use super::{Result, SplitDelimiterBehavior};

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
pub trait PreTokenizer {
    /// Split `text` into pre-tokens, appending to `out`. Ranges are into `text`.
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
/// stages over borrowed ranges to avoid the reference path's allocations.
pub struct PipelineTokenizer {
    added_vocabulary: BucketAddedVocabulary,
    normalizers: Vec<PipelineNormalizer>,
    pre_tokenizer: PipelinePreTokenizer,
    model: PipelineModel,
    post_processor: PipelinePostProcessor,
    scratch_pool: ScratchPool,
}

/// A pool of [`PipelineModelScratch`].
///
/// When calling [`PipelineTokenizer::encode`], an instance of [`PipelineModelScratch`] is taken out of this pool
/// and given to the tokenizer. When the encoding is done, the scratch buffer is returned to the pool and can be
/// reused by later calls.
///
/// The reusability matters because the scratch buffer may hold cache structures which are more useful when reused,
/// and less importantly it saves an extra allocation for an fresh buffer every time.
struct ScratchPool(Mutex<Vec<PipelineModelScratch>>);

impl ScratchPool {
    fn new() -> Self {
        Self(Mutex::new(Vec::new()))
    }

    /// Get a scratch buffer from the pool, wrapped in a [`ScratchGuard`].
    /// When the [`ScratchGuard`] gets dropped, the scratch buffer is pushed back to the pool.
    fn get<'a>(&'a self, model: &PipelineModel) -> ScratchGuard<'a> {
        // The Mutex lock is held just long enough to pop the scratch out of the pool
        let taken = self.0.lock().unwrap_or_else(PoisonError::into_inner).pop();
        ScratchGuard {
            // If there was no scratch buffer available in the pool, we build.a fresh one
            scratch: taken.unwrap_or_else(|| model.init_scratch()),
            pool: self,
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).len()
    }
}

/// A wrapper around [`PipelineModelScratch`].
/// Implements [`Deref`] and [`DerefMut`], so it behaves as [`PipelineModelScratch`].
///
/// When it gets dropped, it pushes [`Self::scratch`] back into the shared [`Self::pool`] so it can
/// get reused by a later call to [`PipelineTokenizer::encode`].
///
/// TODO @McPatate : The Mutex can create contention, to be replaced by a better access pattern
struct ScratchGuard<'a> {
    scratch: PipelineModelScratch,
    pool: &'a ScratchPool,
}

impl Drop for ScratchGuard<'_> {
    fn drop(&mut self) {
        // Steals the scratch buffer from self, replaces it with PipelineModelScratch::default()
        let scratch = mem::take(&mut self.scratch);
        // Push the scratch back in the pool
        self.pool
            .0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(scratch);
    }
}

impl std::ops::Deref for ScratchGuard<'_> {
    type Target = PipelineModelScratch;
    fn deref(&self) -> &PipelineModelScratch {
        &self.scratch
    }
}

impl std::ops::DerefMut for ScratchGuard<'_> {
    fn deref_mut(&mut self) -> &mut PipelineModelScratch {
        &mut self.scratch
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
    /// Stage gates for [`encode_generic`](Self::encode_generic), in execution order.
    /// Each level runs every stage up to and including itself; `STAGE_POSTPROCESS` is
    /// a full encode. `STAGE_FRAME` is the special-token scan + iteration only (the
    /// "other" slice in the decomposition).
    pub const STAGE_FRAME: u8 = 0;
    pub const STAGE_NORMALIZE: u8 = 1;
    pub const STAGE_SPLIT: u8 = 2;
    pub const STAGE_MODEL: u8 = 3;
    pub const STAGE_POSTPROCESS: u8 = 4;

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
                let output =
                    self.encode_generic::<{ Self::STAGE_POSTPROCESS }>(&s, add_special_tokens);
                EncodeHandle::blocking(vec![output])
            }
            // TODO: proper post-processor logic, this is temporary
            Inputs::Pair(s1, s2) => {
                let p1 =
                    self.encode_generic::<{ Self::STAGE_POSTPROCESS }>(&s1, add_special_tokens);
                let p2 =
                    self.encode_generic::<{ Self::STAGE_POSTPROCESS }>(&s2, add_special_tokens);
                EncodeHandle::blocking(vec![p1, p2])
            }
            Inputs::Batch(b) => {
                let mut output = Vec::with_capacity(b.len());
                for seq in b {
                    output.push(
                        self.encode_generic::<{ Self::STAGE_POSTPROCESS }>(
                            &seq,
                            add_special_tokens,
                        ),
                    );
                }
                EncodeHandle::blocking(output)
            }
            // TODO: proper post-processor logic, this is temporary
            Inputs::PairBatch(pb) => {
                let mut output = Vec::with_capacity(pb.len() * 2);
                for (s1, s2) in pb {
                    output.push(
                        self.encode_generic::<{ Self::STAGE_POSTPROCESS }>(&s1, add_special_tokens),
                    );
                    output.push(
                        self.encode_generic::<{ Self::STAGE_POSTPROCESS }>(&s2, add_special_tokens),
                    );
                }
                EncodeHandle::blocking(output)
            }
        }
    }

    /// Decode token ids back to a `String`.
    ///
    /// Not implemented yet — the pipeline decode path is being built. It fails
    /// loud (rather than returning a plausible-but-wrong string) so the oracle
    /// test and the comparative benchmark report decode as *pending* instead of
    /// silently validating garbage. Implementing this flips the ignored
    /// `pipeline_decode_oracle` test on and lights up the decode charts.
    pub fn decode(&self, _ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
        Err("PipelineTokenizer::decode is not implemented yet".into())
    }

    /// Decode several id sequences at once, one `String` per input. Mirrors the
    /// released `decode_batch`; sequential (KISS) — behavior-identical to a
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
    /// [`decode`](Self::decode) — so it is correct exactly where `decode` is.
    pub fn decode_stream(&self, skip_special_tokens: bool) -> PipelineDecodeStream<'_> {
        PipelineDecodeStream {
            tokenizer: self,
            ids: Vec::new(),
            skip_special_tokens,
            prefix: String::new(),
            prefix_index: 0,
        }
    }

    /// Single source of truth for the encode pipeline, generic over how many stages
    /// run. `STAGE` is a **const generic**, so `if STAGE >= …` folds at compile time and
    /// the disabled stages are compiled out — the full specialization
    /// ([`STAGE_POSTPROCESS`], which [`encode`](Self::encode) calls) is branchless and
    /// identical to a hand-written full pipeline, while the benchmark drives lower
    /// `STAGE` values to time each stage's marginal cost (the ablation ladder), e.g.
    /// `model = t(MODEL) − t(SPLIT)`. No runtime gate, no `Instant` in the loop.
    ///
    /// [`STAGE_POSTPROCESS`]: Self::STAGE_POSTPROCESS
    ///
    /// `output` and the `pre_tokens` scratch are caller-owned so a benchmark can reuse
    /// them across calls and observe both buffers to anchor the ablation levels — the
    /// library itself stays free of any `black_box`/timing artifact.
    #[doc(hidden)] // public only so `examples/fixture_bench.rs` can drive partial stages
    pub fn encode_generic<const STAGE: u8>(
        &self,
        input: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<PipelineToken>> {
        let mut pre_tokens = Vec::with_capacity(input.len() / 4);
        let mut output = Vec::with_capacity(input.len() / 4);
        let mut scratch = self.scratch_pool.get(&self.model);
        let PipelinePostProcessor { prefix, suffix } = &self.post_processor;
        // Prepend prefix tokens, if any
        // todo: handle post-processing when encoding a pair of sequences (currently unsupported by the PipelineTokenizer)
        if add_special_tokens && STAGE >= Self::STAGE_POSTPROCESS {
            output.extend_from_slice(prefix);
        }
        // First, we extract all special tokens from the non-normalized input
        for segment in SpecialSegmentIterator::new(input, &self.added_vocabulary, false) {
            match segment {
                Segment::SpecialToken(token) => {
                    output.push(PipelineToken { id: token });
                }
                Segment::Text(chunk) => {
                    let normalized: Cow<str> = if STAGE >= Self::STAGE_NORMALIZE {
                        normalize_all(&self.normalizers, chunk)?
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
                                    // Pre-tokenize the chunk of normalized text
                                    pre_tokens.clear();
                                    self.pre_tokenizer
                                        .pre_tokenize(normalized_chunk, &mut pre_tokens)?;
                                    if STAGE >= Self::STAGE_MODEL {
                                        // Tokenize each chunk
                                        for pre_token in pre_tokens.iter() {
                                            self.model.tokenize_pipeline(
                                                &normalized_chunk[pre_token.range()],
                                                &mut scratch,
                                                &mut output,
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
        // Append suffix tokens, if any
        if add_special_tokens && STAGE >= Self::STAGE_POSTPROCESS {
            output.extend_from_slice(suffix);
        }
        Ok(output)
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
    /// We need a default value to able to use [`mem::take`] in [`ScratchGuard::drop`]
    #[default]
    None,
}

impl ModelScratch for PipelineModelScratch {}

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
            .encode("hello world", false)
            .wait()
            .unwrap()
            .first()
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
                .encode(input, add_special_tokens)
                .wait()
                .unwrap()
                .first()
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
        let ids = |enc: Vec<Vec<PipelineToken>>| {
            enc.first()
                .unwrap()
                .iter()
                .map(|t| t.id)
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
            .encode("hello world", true)
            .wait()
            .unwrap()
            .first()
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
    // would corrupt some of them — and this only compiles if `PipelineTokenizer: Sync`,
    // which the pool has to preserve.
    #[test]
    fn encode_shared_across_threads() {
        use rayon::prelude::*;

        let pipeline = hello_pipeline();

        let want: Vec<u32> = pipeline
            .encode("hello", false)
            .wait()
            .unwrap()
            .remove(0)
            .iter()
            .map(|t| t.id)
            .collect();
        assert_eq!(want, vec![7]);

        let all_match = (0..10_000u32).into_par_iter().all(|_| {
            pipeline
                .encode("hello", false)
                .wait()
                .unwrap()
                .remove(0)
                .iter()
                .map(|t| t.id)
                .collect::<Vec<_>>()
                == want
        });
        assert!(all_match);
    }

    // Reusing scratches is the whole point of the pool, so it must not build one per call:
    // one thread encoding in a loop has to keep coming back to the same scratch, and a
    // burst of N threads must leave at most N behind for later calls to use.
    #[test]
    fn scratches_are_reused_rather_than_piling_up() {
        use std::sync::Barrier;

        let pipeline = hello_pipeline();
        for _ in 0..1000 {
            pipeline.encode("hello", false).wait().unwrap();
        }
        assert_eq!(pipeline.scratch_pool.len(), 1);

        let threads = 64;
        let all_holding = Barrier::new(threads);
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| {
                    let scratch = pipeline.scratch_pool.get(&pipeline.model);
                    all_holding.wait();
                    drop(scratch);
                });
            }
        });

        let after_burst = pipeline.scratch_pool.len();
        assert!(
            after_burst <= threads,
            "{after_burst} scratches kept for {threads} threads"
        );
        for _ in 0..1000 {
            pipeline.encode("hello", false).wait().unwrap();
        }
        assert_eq!(pipeline.scratch_pool.len(), after_burst);
    }

    // ScratchGuard::drop moves the scratch to the pool with mem::take, which leaves the
    // default None variant behind in the guard. The pool must end up holding the scratch
    // the model used: if the None leftover were pushed instead, the pool would still
    // count one scratch, and the next encode would take it and panic in
    // PipelineModel::tokenize_pipeline.
    #[test]
    fn the_pool_gets_back_the_used_scratch_not_the_none_leftover() {
        let pipeline = hello_pipeline();
        pipeline.encode("hello", false).wait().unwrap();
        assert_eq!(pipeline.scratch_pool.len(), 1);
        let scratch = pipeline.scratch_pool.get(&pipeline.model);
        assert!(
            matches!(*scratch, PipelineModelScratch::BPE(_)),
            "the pooled scratch is not the BPE scratch the model used"
        );
    }

    // Pooling is only worth it if a scratch keeps the state it built up across calls.
    // A fresh BpeScratch starts with an empty merge arena; encoding a longer sequence
    // reserves one entry per byte and so grows it. After a second, short encode the
    // pooled scratch must still be the grown one, not a fresh replacement built
    // somewhere along the way.
    #[test]
    fn a_reused_scratch_keeps_its_grown_buffers() {
        let pipeline = hello_pipeline();
        let long_input = "hello".repeat(50);
        pipeline.encode(&long_input, false).wait().unwrap();
        assert_eq!(pipeline.scratch_pool.len(), 1);

        pipeline.encode("hello", false).wait().unwrap();
        assert_eq!(pipeline.scratch_pool.len(), 1);

        let scratch = pipeline.scratch_pool.get(&pipeline.model);
        let PipelineModelScratch::BPE(bpe_scratch) = &*scratch else {
            panic!("the pooled scratch is not a BPE scratch");
        };
        assert!(
            bpe_scratch.queue.entries.capacity() >= long_input.len(),
            "the pooled scratch is not the one grown by the long input: \
             room for {} merge entries after a {}-byte input",
            bpe_scratch.queue.entries.capacity(),
            long_input.len()
        );
    }
}
