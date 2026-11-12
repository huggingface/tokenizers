use crate::pipeline::{self, PipelinePreTokenizer};
use crate::tokenizer::Result;
use crate::utils::byte_level::BYTES_CHAR_LOOKUP;

// The `Sequence` pre-tokenizer is a `Vec<PreTokenizerWrapper>`, so it lives in `tk-convert`
// with the wrapper it is parameterised by, together with the `TryFrom<Sequence>` that lowers it
// into the `PipelineSequence` below.

#[derive(Clone, Debug, PartialEq)]
pub struct PipelineSequence {
    pre_tokenizers: Vec<PipelinePreTokenizer>,
    byte_map: Option<usize>,
}

impl PipelineSequence {
    pub fn new(pre_tokenizers: Vec<PipelinePreTokenizer>) -> Self {
        Self {
            pre_tokenizers,
            byte_map: None,
        }
    }

    /// Insert a GPT-2 byte map before the member at `index`. Subsequent splitters
    /// operate on the mapped text. The tokenizer must use a BPE model with
    /// `byte_level: false` to consume that text.
    pub fn with_byte_map(mut self, index: usize) -> Result<Self> {
        if index >= self.pre_tokenizers.len() {
            return Err("a ByteMap must precede another pre-tokenizer in its Sequence".into());
        }
        self.byte_map = Some(index);
        Ok(self)
    }

    /// The index of the first splitter that operates on the mapped text.
    pub fn byte_map_index(&self) -> Option<usize> {
        self.byte_map
    }

    /// The members, in order. For a writer, which has to spell the `Sequence` back out.
    pub fn pre_tokenizers(&self) -> &[PipelinePreTokenizer] {
        &self.pre_tokenizers
    }

    /// Same recognition as [`crate::utils::is_deepseek`], on the converted children: the first three are
    /// Isolated, non-inverted `Split`s carrying deepseek's `[\p{N}{1,3}, CJK, big]` regexes (the trailing
    /// byte-map `ByteLevel` converts to `PipelinePreTokenizer::None`). Routes the whole split to one
    /// `fsm_deepseek` pass.
    // `pub` because the differential test that asserts deepseek's exact 3-`Split` sequence is
    // recognized lives in `tk-convert` now (it needs `PreTokenizerWrapper` to build the
    // legacy oracle it compares against).
    pub fn is_deepseek(&self) -> bool {
        if self.byte_map.is_some() {
            return false;
        }
        use crate::pre_tokenizers::split::SplitPattern;
        use crate::tokenizer::SplitDelimiterBehavior::Isolated;
        let regex = |i: usize| match self.pre_tokenizers.get(i) {
            Some(PipelinePreTokenizer::Split(s)) if s.behavior == Isolated && !s.invert => {
                match &s.pattern {
                    SplitPattern::Regex(r) => Some(r.as_str()),
                    SplitPattern::String(_) => None,
                }
            }
            _ => None,
        };
        matches!(
            (regex(0), regex(1), regex(2)),
            (Some(a), Some(b), Some(c)) if crate::utils::is_deepseek(a, b, c)
        )
    }

    pub(crate) fn with_mapped_spans(
        &self,
        text: &str,
        scratch: &mut pipeline::PreTokenizerScratch,
        spans: &mut Vec<pipeline::Span>,
        mut consume: impl FnMut(&str, &[pipeline::Span]) -> Result<()>,
    ) -> Result<()> {
        let index = self.byte_map.ok_or("the Sequence has no ByteMap")?;
        let (prefix, suffix) = self.pre_tokenizers.split_at(index);
        let mut pieces = Vec::new();
        split_children(prefix, text, scratch, &mut pieces)?;

        let table = &*BYTES_CHAR_LOOKUP;
        let mut mapped = String::new();
        for piece in pieces {
            mapped.clear();
            mapped.extend(text[piece.range()].bytes().map(|byte| table[byte as usize]));
            if mapped.len() > u32::MAX as usize {
                return Err("sequence too long to pre-tokenize after ByteMap".into());
            }
            // A suffix split can cut between two mapped bytes of one original UTF-8
            // character. Keep its ranges in the mapped string rather than rebasing them.
            spans.clear();
            split_children(suffix, &mapped, scratch, spans)?;
            consume(&mapped, spans)?;
        }
        Ok(())
    }
}

// SAFETY: a `Sequence` runs its children in order. A child splits the spans emitted by the previous child.
// `Sequence` is safe because:
// - all of its children are safe
// - offsets added by the sequence are correct and land on character boundaries
//
// The deepseek fast path has no children to run: it calls an `bitcannon` fsm, which splits only at
// character boundaries of `text`. See the `bitcannon` docs.
unsafe impl pipeline::PreTokenizer for PipelineSequence {
    /// Runs each child in turn, where every child subdivides the spans produced
    /// so far. A child sees only the text of a span (`&text[span]`) and returns
    /// offsets relative to it, which we rebase to absolute mirroring how the
    /// legacy path worked.
    fn pre_tokenize(
        &self,
        text: &str,
        scratch: &mut pipeline::PreTokenizerScratch,
        out: &mut Vec<pipeline::Span>,
    ) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        if self.byte_map.is_some() {
            return Err("a ByteMap Sequence needs an encoder that consumes the mapped text".into());
        }

        // deepseek's 3-Split composition → one native FSM pass (also lets the Sequence handle the
        // trailing byte-map ByteLevel, which the generic child loop can't range-split).
        if self.is_deepseek() {
            scratch.split_on_bits(
                text.as_bytes(),
                |t, tags, starts, _flags, _later, out| {
                    bitcannon::bitcannon_deepseek(t, tags, starts, out)
                },
                out,
            );
            return Ok(());
        }

        // Fuse Split+ByteLevel: a byte-map `ByteLevel` (use_regex=false) converts to
        // `PipelinePreTokenizer::None`, a pure identity pass. Skipping the `None`s collapses the
        // dominant `Sequence[Split(regex), ByteLevel]` archetype (~40% of Hub usage) to a lone child we
        // run straight into `out` — no double-buffer, no rebase, no redundant identity pass over every
        // token. Sequences with ≥2 real children (or none) fall through to the generic loop unchanged.
        let mut work = self
            .pre_tokenizers
            .iter()
            .filter(|c| !matches!(c, PipelinePreTokenizer::None));
        if let (Some(only), None) = (work.next(), work.next()) {
            return pipeline::PreTokenizer::pre_tokenize(only, text, scratch, out);
        }

        split_children(&self.pre_tokenizers, text, scratch, out)
    }
}

fn split_children(
    children: &[PipelinePreTokenizer],
    text: &str,
    scratch: &mut pipeline::PreTokenizerScratch,
    out: &mut Vec<pipeline::Span>,
) -> Result<()> {
    let [mut current, mut next] = scratch.take_pair();
    current.clear();
    current.push(pipeline::Span {
        start: 0,
        end: text.len() as u32,
    });

    for child in children {
        next.clear();
        for span in &current {
            let base = span.start;
            // The child appends span-relative spans straight into `next`;
            // rebase just those to absolute in place — no scratch buffer.
            let from = next.len();
            pipeline::PreTokenizer::pre_tokenize(child, &text[span.range()], scratch, &mut next)?;
            // FIXME: do we want to add an `offset` param to `pre_tokenize` so we don't have to
            // rebase?
            for s in &mut next[from..] {
                s.start += base;
                s.end += base;
            }
        }
        std::mem::swap(&mut current, &mut next);
    }

    // Every call the pipeline makes arrives with `out` empty, since `encode_sequence` clears
    // `pre_tokens` before pre-tokenizing: hand it the buffer the loop just filled instead of
    // copying. `current` takes `out`'s allocation in exchange and goes back to the scratch,
    // and both are pooled, so which buffer ends up where does not matter.
    if out.is_empty() {
        std::mem::swap(out, &mut current);
    } else {
        out.extend_from_slice(&current);
    }
    scratch.put_pair([current, next]);
    Ok(())
}
