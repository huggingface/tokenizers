use crate::pipeline::{self, PipelinePreTokenizer};
use crate::tokenizer::Result;

// The config-shaped `Sequence` pre-tokenizer, a `Vec<PreTokenizerWrapper>`, was deleted with the
// wrapper it was parameterised by, and so was the `TryFrom<Sequence>` that lowered it into the
// `PipelineSequence` below. `tk-serialize` builds one of these directly instead, and refuses a
// nested `Sequence` rather than flattening it.

#[derive(Clone, Debug, PartialEq)]
pub struct PipelineSequence {
    pre_tokenizers: Vec<PipelinePreTokenizer>,
}

impl PipelineSequence {
    pub fn new(pre_tokenizers: Vec<PipelinePreTokenizer>) -> Self {
        Self { pre_tokenizers }
    }

    /// The members, in order. For a writer, which has to spell the `Sequence` back out.
    pub fn pre_tokenizers(&self) -> &[PipelinePreTokenizer] {
        &self.pre_tokenizers
    }

    /// Same recognition as [`crate::utils::is_deepseek`], on the converted children: the first three are
    /// Isolated, non-inverted `Split`s carrying deepseek's `[\p{N}{1,3}, CJK, big]` regexes (the trailing
    /// byte-map `ByteLevel` converts to `PipelinePreTokenizer::None`). Routes the whole split to one
    /// `fsm_deepseek` pass.
    // `pub` because the differential test that asserted deepseek's exact 3-`Split` sequence is
    // recognized needed `PreTokenizerWrapper` to build the legacy oracle it compared against, and
    // went when the wrapper did.
    pub fn is_deepseek(&self) -> bool {
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
}

// SAFETY: a `Sequence` runs its children in order. A child splits the spans emitted by the previous child.
// `Sequence` is safe because:
// - all of its children are safe
// - offsets added by the sequence are correct and land on character boundaries
//
// The deepseek fast path has no children to run: it calls an `atomsplit` fsm, which splits only at
// character boundaries of `text`. See the `atomsplit::fsm` docs.
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

        // deepseek's 3-Split composition → one native FSM pass (also lets the Sequence handle the
        // trailing byte-map ByteLevel, which the generic child loop can't range-split).
        if self.is_deepseek() {
            scratch.split_on_tags(text.as_bytes(), atomsplit::fsm::fsm_deepseek, out);
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

        let [mut current, mut next] = scratch.take_pair();
        current.clear();
        current.push(pipeline::Span {
            start: 0,
            end: text.len() as u32,
        });

        for child in &self.pre_tokenizers {
            next.clear();
            for span in &current {
                let base = span.start;
                // The child appends span-relative spans straight into `next`;
                // rebase just those to absolute in place — no scratch buffer.
                let from = next.len();
                pipeline::PreTokenizer::pre_tokenize(
                    child,
                    &text[span.range()],
                    scratch,
                    &mut next,
                )?;
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
}
