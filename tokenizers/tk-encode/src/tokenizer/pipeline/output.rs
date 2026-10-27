use std::iter::Enumerate;
use std::vec::IntoIter;

#[cfg(feature = "parallelism")]
use super::parallel::StreamingIter;
use super::{EncodeOptions, PipelineToken, PipelineTokenizer, Result};
use crate::utils::padding::pad_windows;
use crate::{PaddingParams, pad_encodings};

/// What encoding one [`Input`](super::Input) produces: an [`Encoding`] from [`PipelineTokenizer::encode`], or
/// every window of it from [`PipelineTokenizer::encode_windows`].
///
/// It is `pub` to bound the public [`EncodeHandle`], but no path outside the crate names it, so
/// other crates cannot implement it.
pub trait EncodeOutput: Default + Send + Sized + 'static {
    /// Adds the post-processor's special tokens to the tokens of one input.
    fn post_process(
        tokenizer: &PipelineTokenizer,
        s1: Vec<PipelineToken>,
        s2: Option<Vec<PipelineToken>>,
        options: &EncodeOptions,
    ) -> Result<Self>;

    fn pad(outputs: &mut [Self], params: &PaddingParams);
}

impl EncodeOutput for Encoding {
    fn post_process(
        tokenizer: &PipelineTokenizer,
        s1: Vec<PipelineToken>,
        s2: Option<Vec<PipelineToken>>,
        options: &EncodeOptions,
    ) -> Result<Self> {
        tokenizer.post_process(s1, s2, options)
    }

    fn pad(outputs: &mut [Self], params: &PaddingParams) {
        pad_encodings(outputs, params)
    }
}

impl EncodeOutput for Vec<Encoding> {
    fn post_process(
        tokenizer: &PipelineTokenizer,
        s1: Vec<PipelineToken>,
        s2: Option<Vec<PipelineToken>>,
        options: &EncodeOptions,
    ) -> Result<Self> {
        tokenizer.post_process_windows(s1, s2, options)
    }

    fn pad(outputs: &mut [Self], params: &PaddingParams) {
        pad_windows(outputs, params)
    }
}

enum HandleState<T> {
    Blocking(Enumerate<IntoIter<Result<T>>>),
    #[cfg(feature = "parallelism")]
    Streaming(StreamingIter<T>),
}

pub struct EncodeHandle<T = Encoding> {
    state: HandleState<T>,
    padding: Option<PaddingParams>,
}

impl<T: EncodeOutput> EncodeHandle<T> {
    /// Fully computed results, for the serial case
    pub(crate) fn blocking(results: Vec<Result<T>>, padding: Option<PaddingParams>) -> Self {
        Self {
            state: HandleState::Blocking(results.into_iter().enumerate()),
            padding,
        }
    }

    #[cfg(feature = "parallelism")]
    pub(crate) fn streaming(it: StreamingIter<T>, padding: Option<PaddingParams>) -> Self {
        Self {
            state: HandleState::Streaming(it),
            padding,
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

impl<T: EncodeOutput> EncodeHandle<T> {
    /// Wait for all scheduled encoding to finish
    ///
    /// Returns in input order
    pub fn wait(mut self) -> Result<Vec<T>> {
        let padding = self.padding.take();
        // XXX: `Vec::new` does not allocate anything when capacity == 0, so creating empty
        // Encodings should not allocate anything either
        let mut out: Vec<T> = (0..self.len()).map(|_| T::default()).collect();
        for (seq, res) in self {
            out[seq] = res?;
        }
        if let Some(params) = padding {
            T::pad(&mut out, &params);
        }
        Ok(out)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Encoding {
    pub(crate) ids: Vec<PipelineToken>,
    /// `None` if every token has type id 0
    pub(crate) type_ids: Option<Vec<u8>>,
    /// `None` if the encoding is not padded (mask is all ones)
    pub(crate) attention_mask: Option<Vec<u8>>,
}

impl Encoding {
    pub fn new(
        ids: Vec<PipelineToken>,
        type_ids: Option<Vec<u8>>,
        attention_mask: Option<Vec<u8>>,
    ) -> Self {
        debug_assert!(type_ids.as_ref().is_none_or(|t| t.len() == ids.len()));
        Self {
            ids,
            type_ids,
            attention_mask,
        }
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

    pub fn attention_mask(&self) -> Option<&[u8]> {
        self.attention_mask.as_deref()
    }
}
/// Iterator yields results in completion order
pub struct HandleIter<T> {
    handle: EncodeHandle<T>,
}

impl<T: EncodeOutput> Iterator for HandleIter<T> {
    type Item = (usize, Result<T>);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.handle.state {
            HandleState::Blocking(it) => it.next(),
            #[cfg(feature = "parallelism")]
            HandleState::Streaming(it) => it.next(),
        }
    }
}

impl<T: EncodeOutput> IntoIterator for EncodeHandle<T> {
    type Item = (usize, Result<T>);
    type IntoIter = HandleIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter { handle: self }
    }
}
