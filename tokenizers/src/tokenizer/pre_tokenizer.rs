use crate::{NormalizedString, Offsets, Result};

pub struct SubString {
    /// The underlying `NormalizedString`. Each SubString is represented by a `NormalizedString`
    /// and in the end we might be carrying a lot of SubString representing various parts of the
    /// original input string.
    pub normalized: NormalizedString,
    /// Offsets of the `NormalizedString` in the `original` input string. These are useful to find
    /// the `original` offsets in the input string, as opposed to the `original` offsets in the
    /// sub-part of the input string represented by `NormalizedString`
    pub offsets: Offsets,
}

/// A `PreTokenizedString` takes care of splitting the input string in multiple `SubString`, while
/// ensuring that they form a coherend group. This let us keep track of the offsets during the whole
/// normalization and pre-tokenization steps.
pub struct PreTokenizedString {
    parts: Vec<SubString>,
}

impl PreTokenizedString {
    /// Split the `PreTokenizedString` by providing a `split_fn` in charge of splitting each
    /// substring (`NormalizedString`) into multiple parts.
    ///
    /// `split_fn` takes a `NormalizedString` and is in charge of returning an iterator over
    /// the produced `NormalizedString`. `split_fn` is free of modifying these `NormalizedString`
    /// as relevant.
    ///
    /// There are only one constraint that *MUST* be respected:
    /// > The produced `NormalizedString`, if combined together, must have the same `original` string
    /// as the original one given to `split_fn`. This concretely means that for the offset tracking
    /// to work as expected, `split_fn` must produce "splits" of the original string.
    pub fn split<F, U>(&mut self, split_fn: F) -> Result<()>
    where
        F: FnMut(usize, NormalizedString) -> U,
        U: IntoIterator<Item = NormalizedString>,
    {
        todo!()
    }

    pub fn iter(&self) -> std::slice::Iter<SubString> {
        self.into_iter()
    }

    /// Merge back to a NormalizedString
    pub fn into_merged(self) -> NormalizedString {
        let offsets = (0, self.parts.iter().last().map_or(0, |sub| sub.offsets.1));
        let normalized: NormalizedString = self.into_iter().map(|sub| sub.normalized).collect();
        assert_eq!(offsets, (0, normalized.len_original()));
        normalized
    }
}

impl From<NormalizedString> for PreTokenizedString {
    fn from(s: NormalizedString) -> Self {
        let offsets = (0, s.len_original());
        Self {
            parts: vec![SubString {
                normalized: s,
                offsets,
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

