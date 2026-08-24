//! The normalizer half of the pipeline: the [`Normalizer`] trait, the
//! [`PipelineNormalizer`] enum of concrete steps, and the chain that runs them.

use std::borrow::Cow;

#[cfg(feature = "normalizers")]
use crate::normalizers::{
    bert::BertNormalizer,
    precompiled::PrecompiledNormalizer,
    strip::StripAccents,
    unicode::{NFC, NFD, NFKC, NFKD, Nmt},
};
use crate::normalizers::{
    byte_level::ByteLevel as ByteLevelNormalizer, metaspace::MetaspaceNormalizer,
    prepend::Prepend, replace::Replace, strip::Strip, utils::Lowercase,
};
use crate::tokenizer::Result;

pub trait Normalizer {
    fn normalize<'a>(&self, input: &'a str) -> Result<Cow<'a, str>>;
}

/// One normalization step of a [`PipelineTokenizer`].
#[derive(Debug)]
pub enum PipelineNormalizer {
    /// This used to be a pretokenizer, it makes more sense as a normalizer.
    Metaspace(MetaspaceNormalizer),
    Replace(Replace),
    Prepend(Prepend),
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

// This replaces the previous normalizer sequence.
pub struct NormalizerChain<'a>(pub &'a [PipelineNormalizer]);

/// Runs `normalizers` in order, each one seeing what the one before it produced.
///
/// A normalizer returns a [`Cow`] (copy-on-write): a borrow when it had nothing to change, an owned
/// `String` when it rewrote the text.
///
/// Chaining them needs care, because an owned `String` produced halfway through the chain is local to this function.
/// The next normalizer may hand back a borrow of that locally owned [`String`], and that borrow cannot outlive the `String`.
///
/// Text no normalizer touches is never copied: it stays a borrow of `input` throughout.
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
