use std::borrow::Cow;
use std::fs::File;
use std::io::BufReader;

use tk_encode::pipeline::{self, PreTokenizerScratch, Segment, SpecialSegmentIterator};
use tk_encode::tokenizer::{Decoder, Model, TokenizerImpl};
use tk_encode::utils::iter::ResultShunt;
use tk_encode::utils::progress::{ProgressBar, ProgressStyle};
use tk_encode::vocab::bucket_added_vocabulary::{
    AddedToken as BucketAddedToken, AddedVocabulary as BucketAddedVocabulary,
};
use tk_encode::{LinesWithEnding, Result};

use crate::Trainer;

/// The added-token matcher [`pre_tokenize_for_training`] segments on, built from the
/// tokenizer's added vocabulary the way `PipelineTokenizer::try_from` builds its own.
///
/// Building it walks every added token, so callers hoist this out of the per-sequence loop.
fn added_vocabulary_matcher<M, N, PT, PP, D>(
    tokenizer: &TokenizerImpl<M, N, PT, PP, D>,
) -> Result<BucketAddedVocabulary>
where
    M: Model,
    N: pipeline::Normalizer,
    D: Decoder,
{
    let added_vocabulary = tokenizer.get_added_vocabulary();
    let mut added_tokens: Vec<_> = added_vocabulary.get_added_tokens_decoder().iter().collect();
    added_tokens.sort_by_key(|(id, _)| **id);

    let mut matcher = BucketAddedVocabulary::new();
    matcher.add_tokens(
        added_tokens.into_iter().map(|(_, token)| BucketAddedToken {
            content: token.content.clone(),
            single_word: token.single_word,
            lstrip: token.lstrip,
            rstrip: token.rstrip,
            normalized: token.normalized,
            special: token.special,
        }),
        tokenizer.get_model(),
        tokenizer.get_normalizer(),
    )?;
    matcher.set_encode_special_tokens(added_vocabulary.get_encode_special_tokens());
    Ok(matcher)
}

/// Normalize and pre-tokenize `sequence` into the words a [`Trainer`] consumes.
///
/// Reproduces what the removed `TokenizerImpl::pre_tokenize_for_training` fed the trainer, which
/// was `extract_and_normalize` + `do_pre_tokenize` + `get_splits`:
///
/// * added tokens are carved out of the raw input first, then out of each normalized chunk --
///   the same two passes `PipelineTokenizer::encode_sequence` runs;
/// * a matched added token becomes one word and goes no further. `PreTokenizedString::split`
///   skipped any split carrying tokens, so neither the normalizer nor the pre-tokenizer ever
///   saw it again;
/// * that word is the text the token *matched*, not the token's own content. `get_splits`
///   returned `split.normalized.get()`, so `lstrip`/`rstrip` whitespace pulled into the match
///   stays attached (`"   <s>"`, not `"<s>"`), and a `normalized = true` token contributes its
///   normalized form;
/// * empty splits were dropped, so no word is ever empty.
fn pre_tokenize_for_training<M, N, PT, PP, D>(
    tokenizer: &TokenizerImpl<M, N, PT, PP, D>,
    added_vocabulary: &BucketAddedVocabulary,
    sequence: &str,
) -> Result<Vec<String>>
where
    M: Model,
    N: pipeline::Normalizer,
    PT: pipeline::PreTokenizer,
    D: Decoder,
{
    let mut words = Vec::new();
    let mut scratch = PreTokenizerScratch::default();
    let mut spans = Vec::new();

    for segment in SpecialSegmentIterator::new(sequence, added_vocabulary, false) {
        let chunk = match segment {
            Segment::SpecialToken { text, .. } => {
                words.push(text.to_owned());
                continue;
            }
            Segment::Text(chunk) => chunk,
        };

        let normalized = match tokenizer.get_normalizer() {
            Some(normalizer) => normalizer.normalize(chunk)?,
            None => Cow::Borrowed(chunk),
        };

        for segment in SpecialSegmentIterator::new(&normalized, added_vocabulary, true) {
            let text = match segment {
                Segment::SpecialToken { text, .. } => {
                    words.push(text.to_owned());
                    continue;
                }
                Segment::Text(text) => text,
            };

            let Some(pre_tokenizer) = tokenizer.get_pre_tokenizer() else {
                words.push(text.to_owned());
                continue;
            };
            // A `Span` holds `u32` offsets, so a longer chunk would wrap them into spans that
            // still slice cleanly -- silently wrong words rather than an error.
            if text.len() > u32::MAX as usize {
                return Err(format!(
                    "sequence too long to pre-tokenize: {} bytes after normalization, the limit is {}",
                    text.len(),
                    u32::MAX
                )
                .into());
            }

            spans.clear();
            pre_tokenizer.pre_tokenize(text, &mut scratch, &mut spans)?;
            words.extend(
                spans
                    .iter()
                    .map(|span| &text[span.range()])
                    .filter(|word| !word.is_empty())
                    .map(str::to_owned),
            );
        }
    }

    Ok(words)
}

/// Adds the training entry points (`train` / `train_from_files`) onto any
/// `tk_encode` `TokenizerImpl`.
///
/// These used to be inherent methods on `TokenizerImpl`; they now live in
/// `tk-train` as an extension trait so the inference crate stays free of any
/// `Trainer` coupling. Bring this trait into scope to call them:
///
/// ```ignore
/// use tk_train::TokenizerTrainExt;
/// tokenizer.train_from_files(&mut trainer, files)?;
/// ```
pub trait TokenizerTrainExt<M> {
    /// Train our Model from files.
    fn train_from_files<T>(&mut self, trainer: &mut T, files: Vec<String>) -> Result<&mut Self>
    where
        T: Trainer<Model = M> + Sync;

    /// Train our Model, using the given Trainer and iterator.
    fn train<T, I, S>(&mut self, trainer: &mut T, sequences: I) -> Result<&mut Self>
    where
        T: Trainer<Model = M> + Sync,
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send;
}

impl<M, N, PT, PP, D> TokenizerTrainExt<M> for TokenizerImpl<M, N, PT, PP, D>
where
    M: Model + Send + Sync,
    N: pipeline::Normalizer + Send + Sync,
    PT: pipeline::PreTokenizer + Send + Sync,
    PP: Send + Sync,
    D: Decoder + Send + Sync,
{
    fn train_from_files<T>(&mut self, trainer: &mut T, files: Vec<String>) -> Result<&mut Self>
    where
        T: Trainer<Model = M> + Sync,
    {
        let mut len = 0;
        for file in files.iter() {
            len += File::open(file)
                .and_then(|f| f.metadata())
                .map(|m| m.len())?;
        }

        let max_read = 1_000_000;

        ResultShunt::process(
            files.into_iter().flat_map(|filename| {
                match File::open(filename) {
                    Ok(file) => {
                        let file = BufReader::with_capacity(max_read, file);
                        // We read new lines using this API instead of the Lines Iterator
                        // on purpose. We want to keep the `\n` and potential `\r` between each lines
                        // We use an iterator to be able to chain with par_bridge.
                        itertools::Either::Left(file.lines_with_ending())
                    }
                    Err(e) => itertools::Either::Right(std::iter::once(Err(e))),
                }
            }),
            |sequences| -> Result<()> {
                let progress = if trainer.should_show_progress() {
                    let progress = ProgressBar::new(len);
                    progress.set_style(
                        ProgressStyle::default_bar()
                            .template("[{elapsed_precise}] {msg:<30!} {wide_bar} {percent:>18!}%")
                            .expect("Invalid progress template"),
                    );
                    progress
                        .set_message(format!("Pre-processing files ({:.2} Mo)", len / 1_000_000));
                    Some(progress)
                } else {
                    None
                };

                let added_vocabulary = added_vocabulary_matcher(self)?;
                trainer.feed(
                    sequences.inspect(|s| {
                        if let Some(progress) = &progress {
                            progress.inc(s.len() as u64)
                        }
                    }),
                    |seq| pre_tokenize_for_training(self, &added_vocabulary, seq),
                )?;

                if let Some(pbar) = progress {
                    pbar.finish();
                }
                let special_tokens = trainer.train(self.get_model_mut())?;
                self.add_special_tokens(special_tokens)?;

                Ok(())
            },
        )??;
        Ok(self)
    }

    fn train<T, I, S>(&mut self, trainer: &mut T, sequences: I) -> Result<&mut Self>
    where
        T: Trainer<Model = M> + Sync,
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
    {
        let (lower, upper) = sequences.size_hint();
        let len = upper.unwrap_or(lower) as u64;
        let progress = if trainer.should_show_progress() {
            let progress = ProgressBar::new(len);
            progress.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {msg:<30!} {wide_bar} {pos:<9!}/{len:>9!}")
                    .expect("Invalid progress template"),
            );
            progress.set_message("Pre-processing sequences");
            Some(progress)
        } else {
            None
        };

        let added_vocabulary = added_vocabulary_matcher(self)?;
        trainer.feed(
            sequences.inspect(|_s| {
                if let Some(progress) = &progress {
                    progress.inc(1)
                }
            }),
            |seq| pre_tokenize_for_training(self, &added_vocabulary, seq),
        )?;
        if let Some(pbar) = progress {
            pbar.finish();
        }

        let special_tokens = trainer.train(self.get_model_mut())?;
        self.add_special_tokens(special_tokens)?;

        Ok(self)
    }
}
