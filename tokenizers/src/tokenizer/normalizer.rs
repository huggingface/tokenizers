use crate::pattern::Pattern;
use crate::{Offsets, Result};
use std::ops::{Bound, RangeBounds};
use unicode_normalization_alignments::UnicodeNormalization;

use serde::{Deserialize, Serialize};

/// Add or Substract a signed isize on a usize. Makes sure of avoiding
/// any substraction overflow, flooring at 0.
macro_rules! apply_signed {
    ($origin: expr, $signed: expr) => {
        if $signed.is_positive() {
            $origin += $signed as usize;
        } else {
            let (result, overflow) = $origin.overflowing_sub(-($signed) as usize);
            $origin = if overflow { 0 } else { result };
        }
    };
}

/// The possible offsets referential
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OffsetReferential {
    Original,
    Normalized,
}

/// Represents a Range usable by the NormalizedString to index its content.
/// A Range can use indices relative to either the `Original` or the `Normalized` string
#[derive(Debug, Clone, PartialEq)]
pub enum Range<T: RangeBounds<usize> + Clone> {
    Original(T),
    Normalized(T),
}

#[allow(clippy::len_without_is_empty)]
impl<T> Range<T>
where
    T: RangeBounds<usize> + Clone,
{
    /// Unwrap the underlying range
    pub fn unwrap(self) -> T {
        match self {
            Range::Original(r) => r,
            Range::Normalized(r) => r,
        }
    }

    /// Return the length of the current Range if not Unbounded
    pub fn len(&self) -> Option<usize> {
        let range = self.clone().unwrap();

        let end = match range.end_bound() {
            Bound::Unbounded => None,
            Bound::Included(i) => Some(*i + 1),
            Bound::Excluded(i) => Some(*i),
        }?;

        match range.start_bound() {
            Bound::Unbounded => Some(end),
            Bound::Included(i) => Some(end - (*i + 1)),
            Bound::Excluded(i) => Some(end - *i),
        }
    }

    /// Converts the current Range to a `std::ops::Range<usize>`. This requires the `max_len`
    /// of the represented string (in chars, not bytes) in order to cover the case where the
    /// original provided range was unbounded
    pub fn into_full_range(self, max_len: usize) -> std::ops::Range<usize> {
        let range = self.unwrap();

        let start = match range.start_bound() {
            Bound::Unbounded => 0,
            Bound::Included(i) => *i,
            Bound::Excluded(i) => *i + 1,
        };
        let end = match range.end_bound() {
            Bound::Unbounded => max_len,
            Bound::Included(i) => *i + 1,
            Bound::Excluded(i) => *i,
        };

        start..end
    }
}

/// Defines the expected behavior for the delimiter of a Split Pattern
/// When splitting on `'-'` for example, with input `the-final--countdown`:
///  - Removed => `[ "the", "final", "countdown" ]`
///  - Isolated => `[ "the", "-", "final", "-", "-", "countdown" ]`
///  - MergedWithPrevious => `[ "the-", "final-", "-", "countdown" ]`
///  - MergedWithNext => `[ "the", "-final", "-", "-countdown" ]`
///  - Contiguous => `[ "the", "-", "final", "--", "countdown" ]`
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SplitDelimiterBehavior {
    Removed,
    Isolated,
    MergedWithPrevious,
    MergedWithNext,
    Contiguous,
}

/// A `NormalizedString` takes care of processing an "original" string to modify
/// it and obtain a "normalized" string. It keeps both version of the string,
/// alignments information between both and provides an interface to retrieve
/// ranges of each string, using offsets from any of them.
///
/// It is possible to retrieve a part of the original string, by indexing it with
/// offsets from the normalized one, and the other way around too. It is also
/// possible to convert offsets from one referential to the other one easily.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct NormalizedString {
    /// The original version of the string, before any modification
    original: String,
    /// The normalized version of the string, after all modifications
    normalized: String,
    /// Mapping from normalized string to original one: (start, end) for each
    /// byte of the normalized string
    alignments: Vec<(usize, usize)>,
    /// If this NormalizedString is a slice of a bigger one, we keep the track
    /// of the missing part, so that we can still give offsets from this original
    /// string.
    original_shift: usize,
}

impl NormalizedString {
    #[cfg(test)]
    pub(crate) fn new(
        original: String,
        normalized: String,
        alignments: Vec<(usize, usize)>,
        original_shift: usize,
    ) -> Self {
        Self {
            original,
            normalized,
            alignments,
            original_shift,
        }
    }
    /// Return the normalized string
    pub fn get(&self) -> &str {
        &self.normalized
    }

    /// Return the original string
    pub fn get_original(&self) -> &str {
        &self.original
    }

    /// Return the original offsets
    pub fn offsets_original(&self) -> Offsets {
        (
            self.original_shift,
            self.original_shift + self.len_original(),
        )
    }

    /// Convert the given offsets range from one referential to the other one:
    /// `Original => Normalized` or `Normalized => Original`
    ///
    /// Returns `None` when targeting something that is outside range
    pub fn convert_offsets<T>(&self, range: Range<T>) -> Option<std::ops::Range<usize>>
    where
        T: RangeBounds<usize> + Clone,
    {
        let len_original = self.len_original();
        let len_normalized = self.len();

        let (target, original) = match range {
            Range::Original(_) => (range.into_full_range(len_original), true),
            Range::Normalized(_) => (range.into_full_range(len_normalized), false),
        };

        // If we target an empty range, let's return the same
        if target.start == target.end {
            return Some(target);
        }
        // If the target goes reverse, return None
        if target.start > target.end {
            return None;
        }

        // If we target 0..0 on an empty string, we want to expand to the entire equivalent
        if original && self.original.is_empty() && target == (0..0) {
            return Some(0..len_normalized);
        }
        if !original && self.normalized.is_empty() && target == (0..0) {
            return Some(0..len_original);
        }

        if original {
            let (mut start, mut end) = (None, None);
            self.alignments
                .iter()
                .enumerate()
                .take_while(|(_, alignment)| target.end >= alignment.1)
                .for_each(|(i, alignment)| {
                    if start.is_none() && target.start <= alignment.0 {
                        // For now, don't update if width == 0
                        if alignment.0 != alignment.1 {
                            start = Some(i);
                        }
                    }
                    if target.end >= alignment.1 {
                        end = Some(i + 1);
                    }
                });

            match (start, end) {
                // Targetting inexistant beginning
                (Some(s), None) => Some(s..s),
                // Targetting inexistant end
                (None, Some(e)) => Some(e..e),
                // Found the range
                (Some(s), Some(e)) => Some(s..e),
                _ => None,
            }
        } else {
            self.alignments.get(target).map(expand_alignments).flatten()
        }
    }

    /// Return a range of the normalized string
    pub fn get_range<T>(&self, range: Range<T>) -> Option<&str>
    where
        T: RangeBounds<usize> + Clone,
    {
        match range {
            Range::Original(_) => self.normalized.get(self.convert_offsets(range)?),
            Range::Normalized(_) => self.normalized.get(range.into_full_range(self.len())),
        }
    }

    /// Return a range of the original string
    pub fn get_range_original<T>(&self, range: Range<T>) -> Option<&str>
    where
        T: RangeBounds<usize> + Clone,
    {
        match range {
            Range::Original(_) => self
                .original
                .get(range.into_full_range(self.len_original())),
            Range::Normalized(_) => self.original.get(self.convert_offsets(range)?),
        }
    }

    /// Validate the given range, to make sure it is on char boundaries
    fn validate_range<T: RangeBounds<usize> + Clone>(
        &self,
        range: Range<T>,
    ) -> Option<Range<std::ops::Range<usize>>> {
        match range {
            Range::Original(_) => {
                let r = range.into_full_range(self.original.len());
                if !(self.original.is_char_boundary(r.start)
                    && self.original.is_char_boundary(r.end))
                {
                    None
                } else {
                    Some(Range::Original(r))
                }
            }
            Range::Normalized(_) => {
                let r = range.into_full_range(self.normalized.len());
                if !(self.normalized.is_char_boundary(r.start)
                    && self.normalized.is_char_boundary(r.end))
                {
                    None
                } else {
                    Some(Range::Normalized(r))
                }
            }
        }
    }

    /// Return a slice of the current NormalizedString
    /// If the range is not on char boundaries, return None
    pub fn slice<T>(&self, range: Range<T>) -> Option<NormalizedString>
    where
        T: RangeBounds<usize> + Clone,
    {
        let full_range = self.validate_range(range)?;
        let (normalized_range, original_range) = match full_range {
            Range::Original(_) => (
                self.convert_offsets(full_range.clone())?,
                full_range.clone().unwrap(),
            ),
            Range::Normalized(_) => (
                full_range.clone().unwrap(),
                self.convert_offsets(full_range.clone())?,
            ),
        };

        let n_shift = original_range.start;

        Some(Self {
            original: self
                .get_range_original(full_range.clone())
                .unwrap_or_default()
                .into(),
            normalized: self.get_range(full_range).unwrap_or_default().into(),
            alignments: self
                .alignments
                .get(normalized_range)?
                .to_vec()
                .iter()
                .map(|(start, end)| (start - n_shift, end - n_shift))
                .collect(),
            original_shift: self.original_shift + original_range.start,
        })
    }

    /// Applies transformations to the current normalized version of the string,
    /// while updating the alignments.
    /// This method expect an Iterator yielding each char of the new normalized string
    /// with a `change` isize equals to:
    ///   - `1` if this is a new char
    ///   - `-N` if the char is right before N removed chars
    ///   - `0` if the char is replacing the existing one
    /// Since it is possible that the normalized string doesn't include some of the characters at
    /// the beginning of the original one, we need an `initial_offset` which represents the number
    /// of removed chars at the very beginning.
    pub fn transform_range<T, I>(&mut self, range: Range<T>, dest: I, initial_offset: usize)
    where
        T: RangeBounds<usize> + Clone,
        I: IntoIterator<Item = (char, isize)>,
    {
        let n_range = match range {
            Range::Normalized(_) => range.into_full_range(self.len()),
            Range::Original(_) => match self.convert_offsets(range) {
                Some(range) => range,
                None => return,
            },
        };
        trace!(
            "===== transform_range call with {:?} (initial_offset: {}) =====",
            n_range,
            initial_offset
        );

        // Retrieve the original characters that are being replaced. This let us
        // compute the change in byte sizes along the way.
        let mut replaced_normalized = self.normalized[n_range.clone()]
            .chars()
            .collect::<Vec<_>>()
            .into_iter();
        let initial_removed: usize = (&mut replaced_normalized)
            .take(initial_offset)
            .map(|c| c.len_utf8())
            .sum();

        let mut offset = (initial_removed + n_range.start) as isize;
        let mut alignments = Vec::with_capacity(n_range.len());
        trace!("=> Applying transformations");
        let normalized = dest
            .into_iter()
            .map(|(c, changes)| {
                trace!(
                    "### {:?} with size {}: {} with offset {} ###",
                    c,
                    c.len_utf8(),
                    match changes {
                        0 => "Replacing".into(),
                        ch if ch > 0 => "Adding".into(),
                        ch if ch < 0 => format!("Replacing + removing {} following chars", ch),
                        _ => "Undefined".into(),
                    },
                    offset
                );

                let idx = offset as usize;
                let align = if changes.is_positive() {
                    if idx < 1 {
                        (0, 0)
                    } else {
                        // This is a newly inserted character, so it shares the same alignment
                        // than the previous one
                        self.alignments[idx - 1]
                    }
                } else {
                    self.alignments[idx]
                };

                // If we are replacing a character, find it and compute the change in size
                let replaced_char = if !changes.is_positive() {
                    replaced_normalized.next()
                } else {
                    None
                };
                let replaced_char_size = replaced_char.map_or(0, |c| c.len_utf8());
                let replaced_char_size_change = c.len_utf8() as isize - replaced_char_size as isize;
                if let Some(ref replaced_char) = replaced_char {
                    trace!(
                        "Replacing char {:?} - with a change in size: {}",
                        replaced_char,
                        replaced_char_size_change
                    );
                }

                // If we are removing some characters, find them too
                let total_bytes_to_remove = if changes.is_negative() {
                    (&mut replaced_normalized)
                        .take(-changes as usize)
                        .map(|c| c.len_utf8())
                        .sum()
                } else {
                    0
                };
                trace!("Total bytes to remove: {}", total_bytes_to_remove);

                // Keep track of the changes for next offsets
                offset += replaced_char_size as isize;
                offset += total_bytes_to_remove as isize;
                trace!("New offset: {}", offset);

                trace!("New normalized alignment: {}x {:?}", c.len_utf8(), align);
                alignments.extend((0..c.len_utf8()).map(|_| align));

                // Then we keep only the char for string reconstruction
                c
            })
            .collect::<String>();

        self.alignments.splice(n_range.clone(), alignments);
        unsafe {
            self.normalized
                .as_mut_vec()
                .splice(n_range, normalized.bytes());
        }
    }

    /// Applies transformations to the current normalized version of the string,
    /// while updating the alignments.
    /// This method expect an Iterator yielding each char of the new normalized string
    /// with a `change` isize equals to:
    ///   - `1` if this is a new char
    ///   - `-N` if the char is right before N removed chars
    ///   - `0` if the char is replacing the existing one
    /// Since it is possible that the normalized string doesn't include some of the characters at
    /// the beginning of the original one, we need an `initial_offset` which represents the number
    /// of removed chars at the very beginning.
    pub fn transform<I>(&mut self, dest: I, initial_offset: usize)
    where
        I: IntoIterator<Item = (char, isize)>,
    {
        self.transform_range(Range::Original(..), dest, initial_offset)
    }

    /// Applies NFD normalization
    pub fn nfd(&mut self) -> &mut Self {
        self.transform(self.get().to_owned().nfd(), 0);
        self
    }

    /// Applies NFKD normalization
    pub fn nfkd(&mut self) -> &mut Self {
        self.transform(self.get().to_owned().nfkd(), 0);
        self
    }

    /// Applies NFC normalization
    pub fn nfc(&mut self) -> &mut Self {
        self.transform(self.get().to_owned().nfc(), 0);
        self
    }

    /// Applies NFKC normalization
    pub fn nfkc(&mut self) -> &mut Self {
        self.transform(self.get().to_owned().nfkc(), 0);
        self
    }

    /// Applies filtering over our characters
    pub fn filter<F: Fn(char) -> bool>(&mut self, keep: F) -> &mut Self {
        let mut removed: isize = 0;
        let mut removed_start: usize = 0;

        let mut transforms = Vec::with_capacity(self.normalized.len());
        let mut last_c = None;
        for c in self.normalized.chars() {
            if keep(c) {
                match last_c {
                    Some(lc) => {
                        transforms.push((lc, -removed));
                    }
                    None => {
                        removed_start = removed as usize;
                    }
                }
                last_c = Some(c);
                removed = 0;
            } else {
                removed += 1;
            }
        }
        if let Some(lc) = last_c {
            transforms.push((lc, -removed));
        }
        self.transform(transforms, removed_start);
        self
    }

    /// Prepend the given string to ourself
    pub fn prepend(&mut self, s: &str) -> &mut Self {
        if let Some(next) = self.normalized.chars().next() {
            let transformations = s
                .chars()
                .enumerate()
                .map(|(i, c)| (c, if i == 0 { 0 } else { 1 }))
                .chain(std::iter::once((next, 1)));

            self.transform_range(Range::Normalized(0..next.len_utf8()), transformations, 0);
        }
        self
    }

    /// Append the given string to ourself
    pub fn append(&mut self, s: &str) -> &mut Self {
        if let Some((b, prev)) = self.normalized.char_indices().last() {
            let transformations = std::iter::once((prev, 0)).chain(s.chars().map(|c| (c, 1)));
            self.transform_range(Range::Normalized(b..), transformations, 0);
        }
        self
    }

    /// Map our characters
    pub fn map<F: Fn(char) -> char>(&mut self, map: F) -> &mut Self {
        let transformations = self
            .normalized
            .chars()
            .map(|c| (map(c), 0))
            .collect::<Vec<_>>();
        self.transform(transformations, 0);
        self
    }

    /// Calls the given function for each characters
    pub fn for_each<F: FnMut(char)>(&self, foreach: F) -> &Self {
        self.normalized.chars().for_each(foreach);
        self
    }

    /// Lowercase
    pub fn lowercase(&mut self) -> &mut Self {
        let mut new_chars: Vec<(char, isize)> = vec![];
        self.for_each(|c| {
            c.to_lowercase().enumerate().for_each(|(index, c)| {
                new_chars.push((c, if index > 0 { 1 } else { 0 }));
            })
        });
        self.transform(new_chars.into_iter(), 0);
        self
    }

    /// Uppercase
    pub fn uppercase(&mut self) -> &mut Self {
        let mut new_chars: Vec<(char, isize)> = vec![];
        self.for_each(|c| {
            c.to_uppercase().enumerate().for_each(|(index, c)| {
                new_chars.push((c, if index > 0 { 1 } else { 0 }));
            })
        });
        self.transform(new_chars.into_iter(), 0);
        self
    }

    /// Replace anything that matches the pattern with the given content.
    pub fn replace<P: Pattern>(&mut self, pattern: P, content: &str) -> Result<()> {
        let mut offset: isize = 0;
        pattern
            .find_matches(&self.normalized)?
            .into_iter()
            .for_each(|((start, end), is_match)| {
                if is_match {
                    let mut range = start..end;
                    apply_signed!(range.start, offset);
                    apply_signed!(range.end, offset);

                    let mut new_len = 0;
                    let removed_chars = self.normalized[range.clone()].chars().count();
                    self.transform_range(
                        Range::Normalized(range),
                        content.chars().map(|c| {
                            new_len += c.len_utf8();
                            (c, 1)
                        }),
                        removed_chars,
                    );

                    let old_len = end - start;
                    offset += new_len as isize - old_len as isize;
                }
            });
        Ok(())
    }

    /// Clear the normalized part of the string
    pub fn clear(&mut self) -> usize {
        let len = self.len();
        self.transform(std::iter::empty(), len);
        len
    }

    /// Split the current string in many subparts. Specify what to do with the
    /// delimiter.
    ///
    /// ## Splitting Behavior for the delimiter
    ///
    /// The behavior can be one of the followings:
    /// When splitting on `'-'` for example, with input `the-final--countdown`:
    ///  - Removed => `[ "the", "", "final", "", "", "countdown" ]`
    ///  - Isolated => `[ "the", "-", "final", "-", "-", "countdown" ]`
    ///  - MergedWithPrevious => `[ "the-", "final-", "-", "countdown" ]`
    ///  - MergedWithNext => `[ "the", "-final", "-", "-countdown" ]`
    pub fn split<P: Pattern>(
        &self,
        pattern: P,
        behavior: SplitDelimiterBehavior,
    ) -> Result<Vec<NormalizedString>> {
        let matches = pattern.find_matches(&self.normalized)?;

        // Process the matches according to the selected behavior: Vec<(Offsets, should_remove)>
        use SplitDelimiterBehavior::*;
        let splits = match behavior {
            Isolated => matches
                .into_iter()
                .map(|(offsets, _)| (offsets, false))
                .collect(),
            Removed => matches,
            Contiguous => {
                let mut previous_match = false;
                matches
                    .into_iter()
                    .fold(vec![], |mut acc, (offsets, is_match)| {
                        if is_match == previous_match {
                            if let Some(((_, end), _)) = acc.last_mut() {
                                *end = offsets.1;
                            } else {
                                acc.push((offsets, false));
                            }
                        } else {
                            acc.push((offsets, false));
                        }
                        previous_match = is_match;
                        acc
                    })
            }
            MergedWithPrevious => {
                let mut previous_match = false;
                matches
                    .into_iter()
                    .fold(vec![], |mut acc, (offsets, is_match)| {
                        if is_match && !previous_match {
                            if let Some(((_, end), _)) = acc.last_mut() {
                                *end = offsets.1;
                            } else {
                                acc.push((offsets, false));
                            }
                        } else {
                            acc.push((offsets, false));
                        }
                        previous_match = is_match;
                        acc
                    })
            }
            MergedWithNext => {
                let mut previous_match = false;
                let mut matches =
                    matches
                        .into_iter()
                        .rev()
                        .fold(vec![], |mut acc, (offsets, is_match)| {
                            if is_match && !previous_match {
                                if let Some(((start, _), _)) = acc.last_mut() {
                                    *start = offsets.0;
                                } else {
                                    acc.push((offsets, false));
                                }
                            } else {
                                acc.push((offsets, false));
                            }
                            previous_match = is_match;
                            acc
                        });
                matches.reverse();
                matches
            }
        };

        // Then we split according to the computed splits
        Ok(splits
            .into_iter()
            .filter_map(|(offsets, remove)| {
                if !remove {
                    Some(
                        self.slice(Range::Normalized(offsets.0..offsets.1))
                            .expect("NormalizedString bad split"),
                    )
                } else {
                    None
                }
            })
            .collect())
    }

    /// Remove any leading space(s) of the normalized string
    pub fn lstrip(&mut self) -> &mut Self {
        self.lrstrip(true, false)
    }

    /// Remove any trailing space(s) of the normalized string
    pub fn rstrip(&mut self) -> &mut Self {
        self.lrstrip(false, true)
    }

    /// Remove any leading and trailing space(s) of the normalized string
    pub fn strip(&mut self) -> &mut Self {
        self.lrstrip(true, true)
    }

    fn lrstrip(&mut self, left: bool, right: bool) -> &mut Self {
        let leading_spaces = if left {
            self.get().chars().take_while(|c| c.is_whitespace()).count()
        } else {
            0
        };
        let trailing_spaces = if right {
            self.get()
                .chars()
                .rev()
                .take_while(|c| c.is_whitespace())
                .count()
        } else {
            0
        };

        if leading_spaces > 0 || trailing_spaces > 0 {
            let transformation = self
                .normalized
                .chars()
                .enumerate()
                .filter_map(|(i, c)| {
                    if i < leading_spaces || i >= self.len() - trailing_spaces {
                        None
                    } else if i == self.len() - trailing_spaces - 1 {
                        Some((c, -(trailing_spaces as isize)))
                    } else {
                        Some((c, 0))
                    }
                })
                .collect::<Vec<_>>();
            self.transform(transformation, leading_spaces);
        }
        self
    }

    /// Returns the length of the normalized string (counting chars not bytes)
    pub fn len(&self) -> usize {
        self.normalized.len()
    }

    /// Returns the length of the original string (counting chars not bytes)
    pub fn len_original(&self) -> usize {
        self.original.len()
    }

    /// Whether empty
    pub fn is_empty(&self) -> bool {
        self.normalized.is_empty()
    }

    /// Recalculate original alignments
    #[allow(dead_code)]
    pub(crate) fn alignments_original(&self) -> Vec<(usize, usize)> {
        // Start, end are in alignments
        // offset, length are in alignments_original
        let mut alignments_original = Vec::with_capacity(self.original.len());

        // Eventual gap before first group
        let start = self.alignments[0].0;
        if start != 0 {
            alignments_original.extend(vec![(0, 0); start]);
        }

        let mut last = (&self.alignments[0].0, &self.alignments[0].1);
        let mut offset = 0;
        let mut length = 0;
        for (start, end) in &self.alignments {
            if last == (start, end) {
                // This is the same group
                length += 1;
            } else {
                // This is a new group
                if start < last.1 {
                    panic!("We can't have overlapping ranges.");
                }

                // Add the old group
                alignments_original.extend(vec![(offset, offset + length); last.1 - last.0]);
                offset += length;
                length = 1;

                // Eventual gap between the 2 groups
                alignments_original.extend(vec![(offset, offset); start - last.1]);
            }

            last = (start, end);
        }
        // Add the last group
        alignments_original.extend(vec![(offset, offset + length); last.1 - last.0]);

        // Add eventual last gap
        offset += length;
        alignments_original.extend(vec![
            (offset, offset);
            self.original.len() - alignments_original.len()
        ]);

        // assert_eq!(alignments_original.len(), self.original.len());
        alignments_original
    }
}

/// Returns the range covered by a slice of alignments
fn expand_alignments(alignments: &[(usize, usize)]) -> Option<std::ops::Range<usize>> {
    if alignments.is_empty() {
        None
    } else {
        let start = alignments[0].0;
        let end = alignments[alignments.len() - 1].1;
        Some(start..end)
    }
}

/// Returns a range of the given string slice, by indexing chars instead of bytes
pub fn get_range_of<T: RangeBounds<usize>>(s: &str, range: T) -> Option<&str> {
    let len = s.chars().count();
    let start = match range.start_bound() {
        Bound::Unbounded => 0,
        Bound::Included(i) => *i,
        Bound::Excluded(i) => *i + 1,
    };
    let end = match range.end_bound() {
        Bound::Unbounded => len,
        Bound::Included(i) => *i + 1,
        Bound::Excluded(i) => *i,
    };

    if start == 0 && end == 0 {
        Some(&s[0..0])
    } else if start >= len || end > len || start >= end {
        None
    } else {
        let start_b = s
            .char_indices()
            .map(|(i, _)| i)
            .nth(start as usize)
            .unwrap_or(0);
        let end_b = s
            .char_indices()
            .map(|(i, _)| i)
            .nth(end as usize)
            .unwrap_or_else(|| s.len());
        Some(&s[start_b..end_b])
    }
}

/// Convert the given range from bytes to char
pub fn bytes_to_char(s: &str, range: std::ops::Range<usize>) -> Option<std::ops::Range<usize>> {
    let (mut start, mut end) = if range == (0..0) {
        (Some(0), Some(0))
    } else {
        (None, None)
    };

    s.char_indices()
        .enumerate()
        .take_while(|(_, (b, _))| *b <= range.end)
        .filter(|(_, (b, _))| *b >= range.start)
        .for_each(|(i, (b, c))| {
            if b == range.start {
                start = Some(i);
            }
            if b == range.end {
                end = Some(i);
            }
            if b + c.len_utf8() == range.end {
                end = Some(i + 1);
            }
        });

    Some(start?..end?)
}

/// Convert the given range from char to bytes
pub fn char_to_bytes(s: &str, range: std::ops::Range<usize>) -> Option<std::ops::Range<usize>> {
    let (mut start, mut end) = if range == (0..0) {
        (Some(0), Some(0))
    } else {
        (None, None)
    };

    if range.start == range.end {
        s.char_indices()
            .skip(range.start)
            .take(1)
            .for_each(|(b, _)| {
                start = Some(b);
                end = Some(b);
            });
    } else {
        s.char_indices()
            .skip(range.start)
            .take(range.end - range.start)
            .for_each(|(b, c)| {
                if start.is_none() {
                    start = Some(b);
                }
                end = Some(b + c.len_utf8());
            });
    }

    Some(start?..end?)
}

impl From<String> for NormalizedString {
    fn from(s: String) -> Self {
        let alignments = s
            .char_indices()
            .flat_map(|(b, c)| {
                let len = c.len_utf8();
                (0..len).map(move |_| (b, b + len))
            })
            .collect::<Vec<_>>();
        Self {
            original: s.clone(),
            normalized: s,
            alignments,
            original_shift: 0,
        }
    }
}

impl From<&str> for NormalizedString {
    fn from(s: &str) -> Self {
        Self::from(s.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use regex::Regex;
    use unicode_categories::UnicodeCategories;

    #[test]
    fn nfd_adds_new_chars() {
        let mut n = NormalizedString::from("élégant");
        n.nfd();
        assert_eq!(
            &n.alignments,
            &[
                (0, 2),
                (0, 2),
                (0, 2),
                (2, 3),
                (3, 5),
                (3, 5),
                (3, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9)
            ]
        );
        assert_eq!(
            n.alignments_original(),
            vec![
                (0, 3),
                (0, 3),
                (3, 4),
                (4, 7),
                (4, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11)
            ]
        );
    }

    #[test]
    fn remove_chars_added_by_nfd() {
        let mut n = NormalizedString::from("élégant");
        n.nfd().filter(|c| !c.is_mark_nonspacing());

        assert_eq!(n.get(), "elegant");

        assert_eq!(
            &n.alignments,
            &[(0, 2), (2, 3), (3, 5), (5, 6), (6, 7), (7, 8), (8, 9)]
        );
        assert_eq!(
            n.alignments_original(),
            vec![
                (0, 1),
                (0, 1),
                (1, 2),
                (2, 3),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7)
            ]
        );
    }

    #[test]
    fn remove_chars() {
        let mut n = NormalizedString::from("élégant");
        n.filter(|c| c != 'n');
        assert_eq!(n.get(), "élégat");
        assert_eq!(
            &n.alignments,
            &[
                (0, 2),
                (0, 2),
                (2, 3),
                (3, 5),
                (3, 5),
                (5, 6),
                (6, 7),
                // Skipped range
                (8, 9)
            ]
        );
        assert_eq!(
            n.alignments_original(),
            vec![
                (0, 2),
                (0, 2),
                (2, 3),
                (3, 5),
                (3, 5),
                (5, 6),
                (6, 7),
                (7, 7), // Eaten n
                (7, 8)
            ]
        );
    }

    #[test]
    fn mixed_addition_and_removal() {
        let mut n = NormalizedString::from("élégant");
        n.nfd().filter(|c| !c.is_mark_nonspacing() && c != 'n');
        assert_eq!(n.get(), "elegat");
        assert_eq!(
            &n.alignments,
            &[(0, 2), (2, 3), (3, 5), (5, 6), (6, 7), (8, 9)]
        );
        assert_eq!(
            n.alignments_original(),
            vec![
                (0, 1),
                (0, 1),
                (1, 2),
                (2, 3),
                (2, 3),
                (3, 4), // g
                (4, 5), // a
                (5, 5), // Eaten n
                (5, 6)
            ]
        );
    }

    #[test]
    fn range_conversion() {
        let mut n = NormalizedString::from("    __Hello__   ");
        n.filter(|c| !c.is_whitespace()).lowercase();
        let hello_n = n.convert_offsets(Range::Original(6..11));
        assert_eq!(hello_n, Some(2..7));
        assert_eq!(
            n.get_range(Range::Normalized(hello_n.clone().unwrap())),
            Some("hello")
        );
        assert_eq!(
            n.get_range_original(Range::Normalized(hello_n.unwrap())),
            Some("Hello")
        );
        assert_eq!(n.get_range(Range::Original(6..11)), Some("hello"));
        assert_eq!(n.get_range_original(Range::Original(6..11)), Some("Hello"));

        // Make sure we get None only in specific cases
        assert_eq!(n.convert_offsets(Range::Original(0..0)), Some(0..0));
        assert_eq!(n.convert_offsets(Range::Original(3..3)), Some(3..3));
        assert_eq!(n.convert_offsets(Range::Original(15..)), Some(9..9));
        assert_eq!(n.convert_offsets(Range::Original(16..)), Some(16..16));
        assert_eq!(n.convert_offsets(Range::Original(17..)), None);
        assert_eq!(n.convert_offsets(Range::Normalized(0..0)), Some(0..0));
        assert_eq!(n.convert_offsets(Range::Normalized(3..3)), Some(3..3));
        assert_eq!(n.convert_offsets(Range::Normalized(9..)), Some(9..9));
        assert_eq!(n.convert_offsets(Range::Normalized(10..)), None);
    }

    #[test]
    fn original_range() {
        let mut n = NormalizedString::from("Hello_______ World!");
        n.filter(|c| c != '_').lowercase();
        let world_n = n.get_range(Range::Normalized(6..11)).unwrap();
        let world_o = n.get_range_original(Range::Normalized(6..11)).unwrap();
        assert_eq!(world_n, "world");
        assert_eq!(world_o, "World");
        let original_range = Range::Original(n.convert_offsets(Range::Normalized(6..11)).unwrap());
        assert_eq!(n.get_range(original_range.clone()).unwrap(), "world");
        assert_eq!(
            n.get_range_original(original_range.clone()).unwrap(),
            "World"
        );
        assert_eq!(original_range.into_full_range(n.len_original()), 13..18);
    }

    #[test]
    fn added_around_edges() {
        let mut n = NormalizedString::from("Hello");
        n.transform(
            vec![
                (' ', 1),
                ('H', 0),
                ('e', 0),
                ('l', 0),
                ('l', 0),
                ('o', 0),
                (' ', 1),
            ]
            .into_iter(),
            0,
        );

        assert_eq!(&n.normalized, " Hello ");
        assert_eq!(
            n.get_range_original(Range::Normalized(1..n.normalized.len() - 1)),
            Some("Hello")
        );
    }

    #[test]
    fn added_characters_alignment() {
        let mut n = NormalizedString::from("野口 No");
        n.transform(
            n.get().to_owned().chars().flat_map(|c| {
                if (c as usize) > 0x4E00 {
                    vec![(' ', 0), (c, 1), (' ', 1)]
                } else {
                    vec![(c, 0)]
                }
            }),
            0,
        );

        assert_eq!(
            n,
            NormalizedString {
                original: "野口 No".into(),
                normalized: " 野  口  No".into(),
                alignments: vec![
                    (0, 3),
                    (0, 3),
                    (0, 3),
                    (0, 3),
                    (0, 3),
                    (3, 6),
                    (3, 6),
                    (3, 6),
                    (3, 6),
                    (3, 6),
                    (6, 7),
                    (7, 8),
                    (8, 9)
                ],
                original_shift: 0
            }
        );
        assert_eq!(
            n.alignments_original(),
            vec![
                (0, 5),
                (0, 5),
                (0, 5),
                (5, 10),
                (5, 10),
                (5, 10),
                (10, 11),
                (11, 12),
                (12, 13)
            ]
        );
    }

    #[test]
    fn remove_at_beginning() {
        let mut n = NormalizedString::from("     Hello");
        n.filter(|c| !c.is_whitespace());
        assert_eq!(
            n.get_range_original(Range::Normalized(1.."Hello".len())),
            Some("ello")
        );
        assert_eq!(
            n.get_range_original(Range::Normalized(0..n.normalized.len())),
            Some("Hello")
        );
    }

    #[test]
    fn remove_at_end() {
        let mut n = NormalizedString::from("Hello    ");
        n.filter(|c| !c.is_whitespace());
        assert_eq!(n.get_range_original(Range::Normalized(0..4)), Some("Hell"));
        assert_eq!(
            n.get_range_original(Range::Normalized(0..n.normalized.len())),
            Some("Hello")
        );
    }

    #[test]
    fn removed_around_both_edges() {
        let mut n = NormalizedString::from("  Hello  ");
        n.filter(|c| !c.is_whitespace());
        assert_eq!(&n.normalized, "Hello");

        assert_eq!(
            n.get_range_original(Range::Normalized(0.."Hello".len())),
            Some("Hello")
        );
        assert_eq!(
            n.get_range_original(Range::Normalized(1.."Hell".len())),
            Some("ell")
        );
    }

    #[test]
    fn lstrip() {
        let mut n = NormalizedString::from("  This is an example  ");
        n.lstrip();
        assert_eq!(&n.normalized, "This is an example  ");
        assert_eq!(
            n.get_range_original(Range::Normalized(0..n.normalized.len())),
            Some("This is an example  ")
        );
    }

    #[test]
    fn rstrip() {
        let mut n = NormalizedString::from("  This is an example  ");
        n.rstrip();
        assert_eq!(&n.normalized, "  This is an example");
        assert_eq!(
            n.get_range_original(Range::Normalized(0..n.normalized.len())),
            Some("  This is an example")
        );
    }

    #[test]
    fn strip() {
        let mut n = NormalizedString::from("  This is an example  ");
        n.strip();
        assert_eq!(&n.normalized, "This is an example");
        assert_eq!(
            n.get_range_original(Range::Normalized(0..n.normalized.len())),
            Some("This is an example")
        );
    }

    #[test]
    fn prepend() {
        let mut n = NormalizedString::from("there");
        n.prepend("Hey ");
        assert_eq!(&n.normalized, "Hey there");
        assert_eq!(
            n.alignments,
            vec![
                (0, 1),
                (0, 1),
                (0, 1),
                (0, 1),
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5)
            ]
        );
        assert_eq!(n.convert_offsets(Range::Normalized(0..4)), Some(0..1));
    }

    #[test]
    fn append() {
        let mut n = NormalizedString::from("Hey");
        n.append(" there");
        assert_eq!(&n.normalized, "Hey there");
        assert_eq!(
            n.alignments,
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3),
                (2, 3)
            ]
        );
        assert_eq!(
            n.convert_offsets(Range::Normalized(3.." there".len())),
            Some(2..3)
        );
    }

    #[test]
    fn get_range() {
        let s = String::from("Hello my name is John 👋");
        assert_eq!(get_range_of(&s, ..), Some(&s[..]));
        assert_eq!(get_range_of(&s, 17..), Some("John 👋"));
    }

    #[test]
    fn slice() {
        let mut s = NormalizedString::from("𝔾𝕠𝕠𝕕 𝕞𝕠𝕣𝕟𝕚𝕟𝕘");
        s.nfkc();

        let original_slice = s.slice(Range::Original(0..4)).unwrap();
        assert_eq!(original_slice.get(), "G");
        assert_eq!(original_slice.get_original(), "𝔾");

        let normalized_slice = s.slice(Range::Normalized(0..4)).unwrap();
        assert_eq!(normalized_slice.get(), "Good");
        assert_eq!(normalized_slice.get_original(), "𝔾𝕠𝕠𝕕");

        // Make sure the sliced NormalizedString is still aligned as expected
        let mut s = NormalizedString::from("   Good Morning!   ");
        s.strip();

        // If we keep the whole slice
        let slice = s.slice(Range::Original(..)).unwrap();
        assert_eq!(
            slice.get_range_original(Range::Normalized(0..4)),
            Some("Good")
        );
        let slice = s.slice(Range::Normalized(..)).unwrap();
        assert_eq!(
            slice.get_range_original(Range::Normalized(0..4)),
            Some("Good")
        );

        // If we keep after the modified piece
        let slice = s.slice(Range::Original(4..15)).unwrap();
        assert_eq!(
            slice.get_range_original(Range::Normalized(0..3)),
            Some("ood")
        );

        // If we keep only the modified piece
        let slice = s.slice(Range::Original(3..16)).unwrap();
        assert_eq!(
            slice.get_range_original(Range::Normalized(0..4)),
            Some("Good")
        );
    }

    #[test]
    fn replace() {
        // Simple
        let mut s = NormalizedString::from(" Hello   friend ");
        s.replace(' ', "_").unwrap();
        assert_eq!(s.get(), "_Hello___friend_");
        let mut s = NormalizedString::from("aaaab");
        s.replace('a', "b").unwrap();
        assert_eq!(s.get(), "bbbbb");

        // Overlapping
        let mut s = NormalizedString::from("aaaab");
        s.replace("aaa", "b").unwrap();
        assert_eq!(s.get(), "bab");

        // Regex
        let mut s = NormalizedString::from(" Hello   friend ");
        let re = Regex::new(r"\s+").unwrap();
        s.replace(&re, "_").unwrap();
        assert_eq!(s.get(), "_Hello_friend_");
    }

    #[test]
    fn split() {
        use SplitDelimiterBehavior::*;
        let s = NormalizedString::from("The-final--countdown");

        let test = |behavior: SplitDelimiterBehavior, result: Vec<&str>| {
            let splits = s.split('-', behavior).unwrap();
            assert_eq!(splits.iter().map(|n| n.get()).collect::<Vec<_>>(), result);
        };

        test(Removed, vec!["The", "final", "countdown"]);
        test(Isolated, vec!["The", "-", "final", "-", "-", "countdown"]);
        test(MergedWithPrevious, vec!["The-", "final-", "-", "countdown"]);
        test(MergedWithNext, vec!["The", "-final", "-", "-countdown"]);
        test(Contiguous, vec!["The", "-", "final", "--", "countdown"]);
    }

    #[test]
    fn transform_range_single_bytes() {
        let s = NormalizedString::from("Hello friend");

        // Removing at the beginning
        let mut current = s.clone();
        current.transform_range(Range::Original(0..4), vec![('Y', 0)], 3);
        assert_eq!(
            current,
            NormalizedString {
                original: "Hello friend".into(),
                normalized: "Yo friend".into(),
                alignments: vec![
                    (3, 4),
                    (4, 5),
                    (5, 6),
                    (6, 7),
                    (7, 8),
                    (8, 9),
                    (9, 10),
                    (10, 11),
                    (11, 12)
                ],
                original_shift: 0,
            }
        );

        assert_eq!(
            current.alignments_original(),
            vec![
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9)
            ]
        );

        // Removing in the middle
        let mut current = s.clone();
        current.transform_range(
            Range::Original(3..10),
            vec![('_', 0), ('F', 0), ('R', -2)],
            2,
        );
        assert_eq!(
            current,
            NormalizedString {
                original: "Hello friend".into(),
                normalized: "Hel_FRnd".into(),
                alignments: vec![
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (5, 6),
                    (6, 7),
                    (7, 8),
                    (10, 11),
                    (11, 12)
                ],
                original_shift: 0,
            }
        );

        assert_eq!(
            current.alignments_original(),
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 3),
                (3, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 6),
                (6, 6),
                (6, 7),
                (7, 8)
            ]
        );

        // Removing at the end
        let mut current = s.clone();
        current.transform_range(Range::Original(5..), vec![('_', 0), ('F', -5)], 0);
        assert_eq!(
            current,
            NormalizedString {
                original: "Hello friend".into(),
                normalized: "Hello_F".into(),
                alignments: vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)],
                original_shift: 0,
            }
        );
        assert_eq!(
            current.alignments_original(),
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 7),
                (7, 7),
                (7, 7),
                (7, 7),
                (7, 7)
            ]
        );

        // Adding at the beginning
        let mut current = s.clone();
        current.transform_range(Range::Original(0..1), vec![('H', 1), ('H', 0)], 0);
        assert_eq!(
            current,
            NormalizedString {
                original: "Hello friend".into(),
                normalized: "HHello friend".into(),
                alignments: vec![
                    (0, 0),
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (4, 5),
                    (5, 6),
                    (6, 7),
                    (7, 8),
                    (8, 9),
                    (9, 10),
                    (10, 11),
                    (11, 12)
                ],
                original_shift: 0,
            }
        );
        assert_eq!(
            current.alignments_original(),
            vec![
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13)
            ]
        );
        // Equivalent to the previous one
        let mut current = s.clone();
        current.transform_range(Range::Original(0..0), vec![('H', 1)], 0);
        assert_eq!(
            current,
            NormalizedString {
                original: "Hello friend".into(),
                normalized: "HHello friend".into(),
                alignments: vec![
                    (0, 0),
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (4, 5),
                    (5, 6),
                    (6, 7),
                    (7, 8),
                    (8, 9),
                    (9, 10),
                    (10, 11),
                    (11, 12)
                ],
                original_shift: 0,
            }
        );
        assert_eq!(
            current.alignments_original(),
            vec![
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13)
            ]
        );
        // Adding as part of the first character
        let mut current = s.clone();
        current.transform_range(Range::Original(0..1), vec![('H', 0), ('H', 1)], 0);
        assert_eq!(
            current,
            NormalizedString {
                original: "Hello friend".into(),
                normalized: "HHello friend".into(),
                alignments: vec![
                    (0, 1),
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (4, 5),
                    (5, 6),
                    (6, 7),
                    (7, 8),
                    (8, 9),
                    (9, 10),
                    (10, 11),
                    (11, 12)
                ],
                original_shift: 0,
            }
        );

        assert_eq!(
            current.alignments_original(),
            vec![
                (0, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13)
            ]
        );

        // Adding in the middle
        let mut current = s.clone();
        current.transform_range(
            Range::Original(5..6),
            vec![('_', 0), ('m', 1), ('y', 1), ('_', 1)],
            0,
        );
        assert_eq!(
            current,
            NormalizedString {
                original: "Hello friend".into(),
                normalized: "Hello_my_friend".into(),
                alignments: vec![
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (4, 5),
                    (5, 6),
                    (5, 6),
                    (5, 6),
                    (5, 6),
                    (6, 7),
                    (7, 8),
                    (8, 9),
                    (9, 10),
                    (10, 11),
                    (11, 12)
                ],
                original_shift: 0,
            }
        );
        assert_eq!(
            current.alignments_original(),
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 15)
            ]
        );

        // Adding at the end
        let mut current = s;
        current.transform_range(Range::Original(11..), vec![('d', 0), ('_', 1), ('!', 1)], 0);
        assert_eq!(
            current,
            NormalizedString {
                original: "Hello friend".into(),
                normalized: "Hello friend_!".into(),
                alignments: vec![
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (4, 5),
                    (5, 6),
                    (6, 7),
                    (7, 8),
                    (8, 9),
                    (9, 10),
                    (10, 11),
                    (11, 12),
                    (11, 12),
                    (11, 12)
                ],
                original_shift: 0,
            }
        );
        assert_eq!(
            current.alignments_original(),
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 14)
            ]
        );
    }

    #[test]
    fn transform_range_multiple_bytes() {
        let s = NormalizedString::from("𝔾𝕠𝕠𝕕");

        // Removing at the beginning
        let mut current = s.clone();
        current.transform_range(Range::Original(0..8), vec![('G', -1)], 0);
        assert_eq!(
            current,
            NormalizedString {
                original: "𝔾𝕠𝕠𝕕".into(),
                normalized: "G𝕠𝕕".into(),
                alignments: vec![
                    (0, 4),
                    (8, 12),
                    (8, 12),
                    (8, 12),
                    (8, 12),
                    (12, 16),
                    (12, 16),
                    (12, 16),
                    (12, 16)
                ],
                original_shift: 0,
            }
        );
        assert_eq!(
            current.alignments_original(),
            vec![
                (0, 1),
                (0, 1),
                (0, 1),
                (0, 1),
                (1, 1),
                (1, 1),
                (1, 1),
                (1, 1),
                (1, 5),
                (1, 5),
                (1, 5),
                (1, 5),
                (5, 9),
                (5, 9),
                (5, 9),
                (5, 9)
            ]
        );
        assert_eq!(current.get_range(Range::Original(0..8)).unwrap(), "G");
        assert_eq!(current.get_range(Range::Original(0..4)).unwrap(), "G");
        assert_eq!(
            current.get_range_original(Range::Original(0..4)).unwrap(),
            "𝔾"
        );
        assert_eq!(
            current.get_range_original(Range::Original(0..8)).unwrap(),
            "𝔾𝕠"
        );

        // Removing in the middle
        let mut current = s.clone();
        current.transform_range(Range::Original(4..12), vec![('o', -1)], 0);
        assert_eq!(
            current,
            NormalizedString {
                original: "𝔾𝕠𝕠𝕕".into(),
                normalized: "𝔾o𝕕".into(),
                alignments: vec![
                    (0, 4),
                    (0, 4),
                    (0, 4),
                    (0, 4),
                    (4, 8),
                    (12, 16),
                    (12, 16),
                    (12, 16),
                    (12, 16)
                ],
                original_shift: 0,
            }
        );
        assert_eq!(
            current.alignments_original(),
            vec![
                (0, 4),
                (0, 4),
                (0, 4),
                (0, 4),
                (4, 5),
                (4, 5),
                (4, 5),
                (4, 5),
                (5, 5),
                (5, 5),
                (5, 5),
                (5, 5),
                (5, 9),
                (5, 9),
                (5, 9),
                (5, 9)
            ]
        );

        // Removing at the end
        let mut current = s.clone();
        current.transform_range(Range::Original(12..), vec![('d', 0), ('!', 1)], 0);
        assert_eq!(
            current,
            NormalizedString {
                original: "𝔾𝕠𝕠𝕕".into(),
                normalized: "𝔾𝕠𝕠d!".into(),
                alignments: vec![
                    (0, 4),
                    (0, 4),
                    (0, 4),
                    (0, 4),
                    (4, 8),
                    (4, 8),
                    (4, 8),
                    (4, 8),
                    (8, 12),
                    (8, 12),
                    (8, 12),
                    (8, 12),
                    (12, 16),
                    (12, 16)
                ],
                original_shift: 0,
            }
        );

        // Adding at the beginning
        let mut current = s.clone();
        current.transform_range(Range::Original(0..4), vec![('_', 1), ('𝔾', 0)], 0);
        assert_eq!(
            current,
            NormalizedString {
                original: "𝔾𝕠𝕠𝕕".into(),
                normalized: "_𝔾𝕠𝕠𝕕".into(),
                alignments: vec![
                    (0, 0),
                    (0, 4),
                    (0, 4),
                    (0, 4),
                    (0, 4),
                    (4, 8),
                    (4, 8),
                    (4, 8),
                    (4, 8),
                    (8, 12),
                    (8, 12),
                    (8, 12),
                    (8, 12),
                    (12, 16),
                    (12, 16),
                    (12, 16),
                    (12, 16)
                ],
                original_shift: 0,
            }
        );
        assert_eq!(
            current.alignments_original(),
            vec![
                (1, 5),
                (1, 5),
                (1, 5),
                (1, 5),
                (5, 9),
                (5, 9),
                (5, 9),
                (5, 9),
                (9, 13),
                (9, 13),
                (9, 13),
                (9, 13),
                (13, 17),
                (13, 17),
                (13, 17),
                (13, 17)
            ]
        );

        assert_eq!(current.get_range(Range::Original(0..8)).unwrap(), "𝔾𝕠");
        assert_eq!(current.get_range(Range::Original(0..4)).unwrap(), "𝔾");
        assert_eq!(
            current.get_range_original(Range::Original(0..4)).unwrap(),
            "𝔾"
        );
        assert_eq!(
            current.get_range_original(Range::Original(0..8)).unwrap(),
            "𝔾𝕠"
        );
        // Equivalent to the previous one
        let mut current = s.clone();
        current.transform_range(Range::Original(0..0), vec![('_', 1)], 0);
        assert_eq!(
            current,
            NormalizedString {
                original: "𝔾𝕠𝕠𝕕".into(),
                normalized: "_𝔾𝕠𝕠𝕕".into(),
                alignments: vec![
                    (0, 0),
                    (0, 4),
                    (0, 4),
                    (0, 4),
                    (0, 4),
                    (4, 8),
                    (4, 8),
                    (4, 8),
                    (4, 8),
                    (8, 12),
                    (8, 12),
                    (8, 12),
                    (8, 12),
                    (12, 16),
                    (12, 16),
                    (12, 16),
                    (12, 16)
                ],
                original_shift: 0,
            }
        );
        assert_eq!(
            current.alignments_original(),
            vec![
                (1, 5),
                (1, 5),
                (1, 5),
                (1, 5),
                (5, 9),
                (5, 9),
                (5, 9),
                (5, 9),
                (9, 13),
                (9, 13),
                (9, 13),
                (9, 13),
                (13, 17),
                (13, 17),
                (13, 17),
                (13, 17)
            ]
        );

        assert_eq!(current.get_range(Range::Original(0..8)).unwrap(), "𝔾𝕠");
        assert_eq!(current.get_range(Range::Original(0..4)).unwrap(), "𝔾");
        assert_eq!(
            current.get_range_original(Range::Original(0..4)).unwrap(),
            "𝔾"
        );
        assert_eq!(
            current.get_range_original(Range::Original(0..8)).unwrap(),
            "𝔾𝕠"
        );
        // Adding as part of the first character
        let mut current = s.clone();
        current.transform_range(Range::Original(0..4), vec![('𝔾', 0), ('o', 1)], 0);
        assert_eq!(
            current,
            NormalizedString {
                original: "𝔾𝕠𝕠𝕕".into(),
                normalized: "𝔾o𝕠𝕠𝕕".into(),
                alignments: vec![
                    (0, 4),
                    (0, 4),
                    (0, 4),
                    (0, 4),
                    (0, 4),
                    (4, 8),
                    (4, 8),
                    (4, 8),
                    (4, 8),
                    (8, 12),
                    (8, 12),
                    (8, 12),
                    (8, 12),
                    (12, 16),
                    (12, 16),
                    (12, 16),
                    (12, 16)
                ],
                original_shift: 0,
            }
        );
        assert_eq!(
            current.alignments_original(),
            vec![
                (0, 5),
                (0, 5),
                (0, 5),
                (0, 5),
                (5, 9),
                (5, 9),
                (5, 9),
                (5, 9),
                (9, 13),
                (9, 13),
                (9, 13),
                (9, 13),
                (13, 17),
                (13, 17),
                (13, 17),
                (13, 17)
            ]
        );
        assert_eq!(current.get_range(Range::Original(0..8)).unwrap(), "𝔾o𝕠");
        assert_eq!(current.get_range(Range::Original(0..4)).unwrap(), "𝔾o");
        assert_eq!(
            current.get_range_original(Range::Original(0..4)).unwrap(),
            "𝔾"
        );
        assert_eq!(
            current.get_range_original(Range::Original(0..8)).unwrap(),
            "𝔾𝕠"
        );

        // Adding in the middle
        let mut current = s.clone();
        current.transform_range(
            Range::Original(4..8),
            vec![('𝕠', 0), ('o', 1), ('o', 1), ('o', 1)],
            0,
        );
        assert_eq!(
            current,
            NormalizedString {
                original: "𝔾𝕠𝕠𝕕".into(),
                normalized: "𝔾𝕠ooo𝕠𝕕".into(),
                alignments: vec![
                    (0, 4),
                    (0, 4),
                    (0, 4),
                    (0, 4),
                    (4, 8),
                    (4, 8),
                    (4, 8),
                    (4, 8),
                    (4, 8),
                    (4, 8),
                    (4, 8),
                    (8, 12),
                    (8, 12),
                    (8, 12),
                    (8, 12),
                    (12, 16),
                    (12, 16),
                    (12, 16),
                    (12, 16)
                ],
                original_shift: 0,
            }
        );
        assert_eq!(
            current.alignments_original(),
            vec![
                (0, 4),
                (0, 4),
                (0, 4),
                (0, 4),
                (4, 11),
                (4, 11),
                (4, 11),
                (4, 11),
                (11, 15),
                (11, 15),
                (11, 15),
                (11, 15),
                (15, 19),
                (15, 19),
                (15, 19),
                (15, 19)
            ]
        );

        // Adding at the end
        let mut current = s;
        current.transform_range(Range::Original(16..), vec![('!', 1)], 0);
        assert_eq!(
            current,
            NormalizedString {
                original: "𝔾𝕠𝕠𝕕".into(),
                normalized: "𝔾𝕠𝕠𝕕!".into(),
                alignments: vec![
                    (0, 4),
                    (0, 4),
                    (0, 4),
                    (0, 4),
                    (4, 8),
                    (4, 8),
                    (4, 8),
                    (4, 8),
                    (8, 12),
                    (8, 12),
                    (8, 12),
                    (8, 12),
                    (12, 16),
                    (12, 16),
                    (12, 16),
                    (12, 16),
                    (12, 16)
                ],
                original_shift: 0,
            }
        );
        assert_eq!(
            current.alignments_original(),
            vec![
                (0, 4),
                (0, 4),
                (0, 4),
                (0, 4),
                (4, 8),
                (4, 8),
                (4, 8),
                (4, 8),
                (8, 12),
                (8, 12),
                (8, 12),
                (8, 12),
                (12, 17),
                (12, 17),
                (12, 17),
                (12, 17)
            ]
        );
    }

    #[test]
    fn transform_check() {
        let mut s = NormalizedString::from("abc…");
        s.nfkd();
        let transforms = vec![('a', -2), ('.', 0), ('.', 0), ('.', 0)];
        s.transform(transforms, 0);
        s.lowercase();
        assert_eq!(s.get(), "a...");
    }
}
