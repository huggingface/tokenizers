use std::borrow::Cow;
use std::ops::Range;

use super::{
    EncodeOptions, EncodeScratch, Result, Segment, SpecialSegmentIterator, Token, Tokenizer,
    normalize_all,
};
use crate::vocab::bucket_added_vocabulary::WordNeighbors;

pub(super) fn encode(
    tokenizer: &Tokenizer,
    input: &str,
    untrusted_spans: &[Range<usize>],
    options: &EncodeOptions,
    scratch: &mut EncodeScratch,
) -> Result<Vec<Token>> {
    let mut encoder = SpannedEncoder {
        tokenizer,
        options,
        scratch,
        output: Vec::with_capacity(input.len() / 4),
        pending_trusted: None,
        before: None,
    };
    let mut offset = 0;
    let mut spans = untrusted_spans.iter().peekable();
    while let Some(span) = spans.next() {
        let mut end = span.end;
        while let Some(next) = spans.next_if(|next| next.start == end) {
            end = next.end;
        }
        encoder.trusted(input, offset..span.start)?;
        encoder.untrusted(&input[span.start..end], span.start)?;
        offset = end;
    }
    encoder.trusted(input, offset..input.len())?;
    encoder.flush_trusted(None)?;
    Ok(encoder.output)
}

struct SpannedEncoder<'text, 'call> {
    tokenizer: &'call Tokenizer,
    options: &'call EncodeOptions,
    scratch: &'call mut EncodeScratch,
    output: Vec<Token>,
    // A later piece can supply a word neighbor even when intervening text normalizes to nothing.
    pending_trusted: Option<Cow<'text, str>>,
    before: Option<char>,
}

impl<'text> SpannedEncoder<'text, '_> {
    fn trusted(&mut self, input: &'text str, range: Range<usize>) -> Result<()> {
        if range.is_empty() {
            return Ok(());
        }
        let inner = &self.tokenizer.inner;
        let matcher = inner.added_vocabulary.with_word_neighbors(WordNeighbors {
            before: input[..range.start].chars().next_back(),
            after: input[range.end..].chars().next(),
        });
        for segment in SpecialSegmentIterator::new(
            &input[range.clone()],
            &matcher,
            false,
            self.options.encode_special_tokens,
        ) {
            match segment {
                Segment::SpecialToken(id) => {
                    self.flush_trusted(None)?;
                    self.output.push(Token::from(id));
                    self.before = None;
                }
                Segment::Text { text, input_offset } => {
                    let normalized =
                        normalize_all(&inner.normalizers, text, range.start + input_offset)?;
                    if !normalized.is_empty() {
                        self.flush_trusted(normalized.chars().next())?;
                        self.pending_trusted = Some(normalized);
                    }
                }
            }
        }
        Ok(())
    }

    fn untrusted(&mut self, input: &'text str, offset: usize) -> Result<()> {
        let inner = &self.tokenizer.inner;
        let normalized = normalize_all(&inner.normalizers, input, offset)?;
        if normalized.is_empty() {
            return Ok(());
        }

        self.flush_trusted(normalized.chars().next())?;
        let token_start = self.output.len();
        self.tokenizer
            .encode_model(&normalized, self.scratch, &mut self.output)?;
        let unk_id = inner.model.unk_id();
        if let Some(id) = self.output[token_start..]
            .iter()
            .map(|token| token.id())
            .find(|&id| {
                id >= inner.added_id_min
                    && Some(id) != unk_id
                    && inner.added_vocabulary.contains_id(id)
            })
        {
            return Err(format!("model emitted added token ID {id} in untrusted text").into());
        }
        self.before = normalized.chars().next_back();
        Ok(())
    }

    fn flush_trusted(&mut self, after: Option<char>) -> Result<()> {
        if let Some(text) = self.pending_trusted.take() {
            let vocabulary = &self.tokenizer.inner.added_vocabulary;
            let matcher = vocabulary.with_word_neighbors(WordNeighbors {
                before: self.before,
                after,
            });
            self.tokenizer.encode_normalized(
                &text,
                &matcher,
                self.options,
                self.scratch,
                &mut self.output,
            )?;
            self.before = text.chars().next_back();
        }
        Ok(())
    }
}
