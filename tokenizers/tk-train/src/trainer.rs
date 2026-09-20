use tk_encode::Result;
use tk_encode::vocab::bucket_added_vocabulary::AddedToken;

/// A `Trainer` has the responsibility to train a model. We feed it with lines/sentences
/// and then it can train the given `Model`.
pub trait Trainer {
    /// The model this trainer fills in.
    ///
    /// Deliberately *not* bound by `tk_encode::Model`: `PipelineBPE` is the only BPE left and it
    /// implements the pipeline's model trait, not that one -- the config-shaped `BPE` that did went
    /// with `tk-convert`'s strip. Nothing here calls a `Model` method on it anyway; a trainer only
    /// ever writes one.
    type Model: Sized;
    /// Whether we should show progress during the training.
    fn should_show_progress(&self) -> bool;
    /// The actual training method. This will return a new trained Model as well as a list
    /// of `special_tokens` to be added directly to the tokenizer along with the model.
    fn train(&self, model: &mut Self::Model) -> Result<Vec<AddedToken>>;
    /// Process an iterator of sequences, calling `process` for each of them in order to
    /// pre-process the said sequence as relevant.
    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync;
}
