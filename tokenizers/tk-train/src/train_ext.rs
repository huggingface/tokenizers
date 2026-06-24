use std::fs::File;
use std::io::BufReader;

use tk_encode::tokenizer::{
    Decoder, Model, Normalizer, PostProcessor, PreTokenizer, TokenizerImpl,
};
use tk_encode::utils::iter::ResultShunt;
use tk_encode::utils::progress::{ProgressBar, ProgressStyle};
use tk_encode::{LinesWithEnding, Result};

use crate::Trainer;

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
    N: Normalizer + Send + Sync,
    PT: PreTokenizer + Send + Sync,
    PP: PostProcessor + Send + Sync,
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

                trainer.feed(
                    sequences.inspect(|s| {
                        if let Some(progress) = &progress {
                            progress.inc(s.len() as u64)
                        }
                    }),
                    |seq| self.pre_tokenize_for_training(seq),
                )?;

                if let Some(pbar) = progress {
                    pbar.finish();
                }
                let special_tokens = trainer.train(&mut self.get_model_mut())?;
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

        trainer.feed(
            sequences.inspect(|_s| {
                if let Some(progress) = &progress {
                    progress.inc(1)
                }
            }),
            |seq| self.pre_tokenize_for_training(seq),
        )?;
        if let Some(pbar) = progress {
            pbar.finish();
        }

        let special_tokens = trainer.train(&mut self.get_model_mut())?;
        self.add_special_tokens(special_tokens)?;

        Ok(self)
    }
}
