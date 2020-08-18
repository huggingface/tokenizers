use crate::{
    normalizer::Range, Encoding, NormalizedString, OffsetReferential, Offsets, Result, Token,
};

/// Wrapper for a subpart of a `NormalizedString`.
///
/// This Split contains the underlying `NormalizedString` as well as its offsets
/// in the original string. These offsets are in the `original` referential.
/// It also contains any `Token` associated to the current split
#[derive(Debug, Clone, PartialEq)]
pub struct Split {
    /// The underlying `NormalizedString`. Each SubString is represented by a `NormalizedString`
    /// and in the end we might be carrying a lot of SubString representing various parts of the
    /// original input string.
    normalized: NormalizedString,
    /// Optional Tokens associated to this Split
    tokens: Option<Vec<Token>>,
}

impl From<NormalizedString> for Split {
    fn from(n: NormalizedString) -> Self {
        Self {
            normalized: n,
            tokens: None,
        }
    }
}

impl From<(NormalizedString, Option<Vec<Token>>)> for Split {
    fn from(f: (NormalizedString, Option<Vec<Token>>)) -> Self {
        Self {
            normalized: f.0,
            tokens: f.1,
        }
    }
}

/// The `PreTokenizedString` is in charge of splitting an underlying string,
/// making sure everything is fine while doing so, and providing ways to normalize
/// and tokenize these splits.
/// Once everything has been normalized and tokenized, the `PreTokenizedString` is able
/// to build an `Encoding` with all the relevant offsets and word ids, relative to the
/// original string.
#[derive(Debug, Clone, PartialEq)]
pub struct PreTokenizedString {
    splits: Vec<Split>,
}

impl PreTokenizedString {
    /// Split the `PreTokenizedString` by providing a `split_fn` in charge of splitting
    /// each substring (`NormalizedString`) into multiple parts.
    ///
    /// `split_fn` takes a `NormalizedString` and is in charge of returning an iterator
    /// over the produced `NormalizedString`. `split_fn` is free of modifying these
    /// `NormalizedString` as relevant, as long as it respects the constraint stated below.
    ///
    /// There are only one constraint that *MUST* be respected:
    /// > The produced `NormalizedString`, if combined back together, must have the
    /// same `original` string as the original one given to `split_fn`. This concretely
    /// means that for the offset tracking to work as expected, `split_fn` must produce
    /// "splits" of the original string.
    pub fn split<F, U, R>(&mut self, mut split_fn: F) -> Result<()>
    where
        F: FnMut(usize, NormalizedString) -> Result<U>,
        U: IntoIterator<Item = R>,
        R: Into<Split>,
    {
        // new_splits is at least as big as self.splits
        let mut new_splits = Vec::with_capacity(self.splits.len());
        for (i, original_split) in self.splits.drain(..).enumerate() {
            if original_split.tokens.is_some() {
                new_splits.push(original_split);
                continue;
            }

            new_splits.extend(
                split_fn(i, original_split.normalized)?
                    .into_iter()
                    .filter_map(|split| {
                        let split: Split = split.into();
                        if split.normalized.is_empty() {
                            None
                        } else {
                            Some(split)
                        }
                    }),
            );
        }
        self.splits = new_splits;

        Ok(())
    }

    /// Normalized all the splits that do not have attached `Tokens`, using the provided
    /// `normalize` function.
    pub fn normalize<F>(&mut self, normalize: F) -> Result<()>
    where
        F: Fn(&mut NormalizedString) -> Result<()>,
    {
        for split in self.splits.iter_mut().filter(|s| s.tokens.is_none()) {
            normalize(&mut split.normalized)?;
        }
        Ok(())
    }

    /// Tokenize all the splits that do not have attached `Tokens`, using the provided
    /// `tokenize` function
    pub fn tokenize<F>(&mut self, tokenize: F) -> Result<()>
    where
        F: Fn(&NormalizedString) -> Result<Vec<Token>>,
    {
        for split in self.splits.iter_mut().filter(|s| s.tokens.is_none()) {
            split.tokens = Some(tokenize(&split.normalized)?);
        }

        Ok(())
    }

    /// Transform the current `PreTokenizedString` into an `Encoding`.
    ///
    /// If a `word_idx` is provided, any word in the generated `Encoding`
    /// will be set to this value. This is generally used with pre-tokenized
    /// input, that do not need the `PreTokenizedString` to generate word ids.
    ///
    /// This method will fail if some splits do not have associated `Token`.
    pub fn into_encoding(self, word_idx: Option<u32>, type_id: u32) -> Result<Encoding> {
        if self.splits.is_empty() {
            Ok(Encoding::default())
        } else if !self.splits.iter().all(|split| split.tokens.is_some()) {
            Err("Split has not been tokenized, call `PreTokenizedString::tokenize` first".into())
        } else {
            Ok(self
                .splits
                .into_iter()
                .enumerate()
                .flat_map(|(idx, split)| {
                    let normalized = split.normalized;
                    let offsets = normalized.offsets_original();
                    split.tokens.unwrap().into_iter().map(move |token| {
                        let converted_offsets = normalized
                            .convert_offsets(Range::Normalized(token.offsets.0..token.offsets.1))
                            .map_or(token.offsets, |range| {
                                (offsets.0 + range.start, offsets.0 + range.end)
                            });

                        (
                            token.id,
                            token.value,
                            converted_offsets,
                            if word_idx.is_some() {
                                word_idx
                            } else {
                                Some(idx as u32)
                            },
                            type_id,
                        )
                    })
                })
                .collect())
        }
    }

    /// Returns a list of splits, each of them being a slice of the normalized
    /// string, the associated offsets either in original or normalized
    /// referential, as well as the potention tokens
    pub fn get_splits(
        &self,
        offset_type: OffsetReferential,
    ) -> Vec<(&str, Offsets, &Option<Vec<Token>>)> {
        let mut offset = 0;
        self.splits
            .iter()
            .map(|split| {
                let offsets = match offset_type {
                    OffsetReferential::Original => split.normalized.offsets_original(),
                    OffsetReferential::Normalized => {
                        let len = split.normalized.len();
                        offset += len;
                        (offset - len, offset)
                    }
                };

                (split.normalized.get(), offsets, &split.tokens)
            })
            .collect()
    }
}

impl From<NormalizedString> for PreTokenizedString {
    fn from(s: NormalizedString) -> Self {
        let original_offsets = (0, s.len_original());
        Self {
            splits: vec![Split {
                normalized: s,
                tokens: None,
            }],
        }
    }
}

impl From<&str> for PreTokenizedString {
    fn from(s: &str) -> Self {
        let normalized: NormalizedString = s.into();
        normalized.into()
    }
}

impl From<String> for PreTokenizedString {
    fn from(s: String) -> Self {
        let normalized: NormalizedString = s.into();
        normalized.into()
    }
}

impl From<PreTokenizedString> for NormalizedString {
    fn from(p: PreTokenizedString) -> Self {
        p.splits.into_iter().map(|split| split.normalized).collect()
    }
}
