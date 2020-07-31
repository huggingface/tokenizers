use crate::{NormalizedString, OffsetReferential, Offsets, Result};

/// Wrapper for a subpart of a `NormalizedString`.
///
/// This SubString contains the underlying `NormalizedString` as well as its offsets
/// in the original string. These offsets are in the `original` referential
#[derive(Debug)]
pub struct SubString {
    /// The underlying `NormalizedString`. Each SubString is represented by a `NormalizedString`
    /// and in the end we might be carrying a lot of SubString representing various parts of the
    /// original input string.
    pub normalized: NormalizedString,
    /// Offsets of the `NormalizedString` in the `original` input string. These are useful to find
    /// the `original` offsets in the input string, as opposed to the `original` offsets in the
    /// sub-part of the input string represented by `NormalizedString`
    pub original_offsets: Offsets,
}

impl SubString {
    pub fn new(normalized: NormalizedString, original_offsets: Offsets) -> Self {
        Self {
            normalized,
            original_offsets,
        }
    }
}

/// sub strings, while ensuring that they form a coherend group. This let us keep
/// track of the offsets during the whole normalization and pre-tokenization steps.
#[derive(Debug)]
pub struct PreTokenizedString {
    parts: Vec<SubString>,
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
    pub fn split<F, U>(&mut self, mut split_fn: F) -> Result<()>
    where
        F: FnMut(usize, NormalizedString) -> Result<U>,
        U: IntoIterator<Item = NormalizedString>,
    {
        // new_parts is at least as big as self.parts
        let mut new_parts = Vec::with_capacity(self.parts.len());
        for (i, sub) in self.parts.drain(..).enumerate() {
            let original_len = sub.normalized.len_original();
            let original_offsets = sub.original_offsets;

            let mut new_len = 0;
            new_parts.extend(split_fn(i, sub.normalized)?.into_iter().map(|normalized| {
                let len = normalized.len_original();
                let start = original_offsets.0 + new_len;
                let end = original_offsets.0 + new_len + len;
                let new_s = SubString::new(normalized, (start, end));
                new_len += len;
                new_s
            }));
            if original_len != new_len {
                return Err(
                    "Split pre-tokenized string must represent the entire original string".into(),
                );
            }
        }
        self.parts = new_parts;

        Ok(())
    }

    pub fn iter(&self) -> std::slice::Iter<SubString> {
        self.into_iter()
    }

    /// Returns a list of normalized string and the associated offsets,
    /// either in original or normalized referential
    pub fn get_normalized(&self, offset_type: OffsetReferential) -> Vec<(&str, Offsets)> {
        let mut offset = 0;
        self.iter()
            .map(|sub| {
                let offsets = match offset_type {
                    OffsetReferential::Original => (
                        sub.original_offsets.0,
                        sub.original_offsets.0 + sub.normalized.len_original(),
                    ),
                    OffsetReferential::Normalized => {
                        let len = sub.normalized.len();
                        offset += len;
                        (offset - len, offset)
                    }
                };

                (sub.normalized.get(), offsets)
            })
            .collect()
    }

    /// Merge back to a NormalizedString
    pub fn into_merged(self) -> NormalizedString {
        let offsets = (
            0,
            self.parts
                .iter()
                .last()
                .map_or(0, |sub| sub.original_offsets.1),
        );
        let normalized: NormalizedString = self.into_iter().map(|sub| sub.normalized).collect();
        assert_eq!(offsets, (0, normalized.len_original()));
        normalized
    }
}

impl From<NormalizedString> for PreTokenizedString {
    fn from(s: NormalizedString) -> Self {
        let original_offsets = (0, s.len_original());
        Self {
            parts: vec![SubString {
                normalized: s,
                original_offsets,
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

impl IntoIterator for PreTokenizedString {
    type Item = SubString;
    type IntoIter = std::vec::IntoIter<SubString>;

    fn into_iter(self) -> Self::IntoIter {
        self.parts.into_iter()
    }
}

impl<'a> IntoIterator for &'a PreTokenizedString {
    type Item = &'a SubString;
    type IntoIter = std::slice::Iter<'a, SubString>;

    fn into_iter(self) -> Self::IntoIter {
        self.parts.iter()
    }
}
