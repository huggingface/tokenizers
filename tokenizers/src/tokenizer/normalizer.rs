#![allow(clippy::reversed_empty_ranges)]

use crate::pattern::Pattern;
use crate::{Offsets, Result};
use std::ops::{Bound, RangeBounds};
use unicode_normalization_alignments::UnicodeNormalization;

/// The possible offsets referential
pub enum OffsetReferential {
    Original,
    Normalized,
}

/// Represents a Range usable by the NormalizedString to index its content.
/// A Range can use indices relative to either the `Original` or the `Normalized` string
#[derive(Debug, Clone)]
pub enum Range<T: RangeBounds<usize> + Clone> {
    Original(T),
    Normalized(T),
}

impl<T> Range<T>
where
    T: RangeBounds<usize> + Clone,
{
    /// Unwrap the underlying range
    fn unwrap(self) -> T {
        match self {
            Range::Original(r) => r,
            Range::Normalized(r) => r,
        }
    }

    /// Converts the current Range to a `std::ops::Range<usize>`. This requires the `max_len`
    /// of the represented string (in chars, not bytes) in order to cover the case where the
    /// original provided range was unbounded
    fn into_full_range(self, max_len: usize) -> std::ops::Range<usize> {
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
pub enum SplitDelimiterBehavior {
    Removed,
    Isolated,
    MergedWithPrevious,
    MergedWithNext,
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
    /// character of the normalized string
    alignments: Vec<(usize, usize)>,
}

impl NormalizedString {
    /// Return the normalized string
    pub fn get(&self) -> &str {
        &self.normalized
    }

    /// Return the original string
    pub fn get_original(&self) -> &str {
        &self.original
    }

    /// Return the offsets of the normalized part
    pub fn offsets(&self) -> Offsets {
        (0, self.len())
    }

    /// Return the offsets of the original part
    pub fn offsets_original(&self) -> Offsets {
        (0, self.len_original())
    }

    /// Convert the given offsets range from one referential to the other one:
    /// `Original => Normalized` or `Normalized => Original`
    pub fn convert_offsets<T>(&self, range: Range<T>) -> Option<std::ops::Range<usize>>
    where
        T: RangeBounds<usize> + Clone,
    {
        match range {
            Range::Original(_) => {
                let (mut start, mut end) = (None, None);
                let target = range.into_full_range(self.len_original());

                // If we target before the start of the normalized string
                if let Some((start, _)) = self.alignments.first() {
                    if target.end <= *start {
                        return Some(0..0);
                    }
                }
                // If we target after the end of the normalized string
                if let Some((_, end)) = self.alignments.last() {
                    if target.start >= *end {
                        let len = self.len();
                        return Some(len..len);
                    }
                }

                // Otherwise lets find the range
                self.alignments
                    .iter()
                    .enumerate()
                    .take_while(|(_, alignment)| target.end >= alignment.1)
                    .for_each(|(i, alignment)| {
                        if alignment.0 >= target.start && start.is_none() {
                            // Here we want to keep the first char in the normalized string
                            // that is on or *after* the target start.
                            start = Some(i);
                        }
                        if alignment.1 <= target.end {
                            end = Some(i + 1);
                        }
                    });

                // If we didn't find the start, let's use the end of the normalized string
                let start = start.unwrap_or_else(|| self.len());
                // The end must be greater or equal to start, and might be None otherwise
                let end = end.filter(|e| *e >= start);

                Some(start..end?)
            }
            Range::Normalized(_) => {
                // If we target 0..0 on an empty normalized string, we want to return the
                // entire original one
                let range = range.into_full_range(self.len());

                if self.alignments.is_empty() && range == (0..0) {
                    Some(0..self.len_original())
                } else {
                    self.alignments
                        .get(range)
                        .map(|alignments| {
                            if alignments.is_empty() {
                                None
                            } else {
                                let start = alignments[0].0;
                                let end = alignments[alignments.len() - 1].1;
                                Some(start..end)
                            }
                        })
                        .flatten()
                }
            }
        }
    }

    /// Return a range of the normalized string (indexing on char, not bytes)
    pub fn get_range<T>(&self, range: Range<T>) -> Option<&str>
    where
        T: RangeBounds<usize> + Clone,
    {
        match range {
            Range::Original(_) => self
                .convert_offsets(range)
                .map(|r| get_range_of(&self.normalized, r))
                .flatten(),
            Range::Normalized(r) => get_range_of(&self.normalized, r),
        }
    }

    /// Return a range of the original string (indexing on char, not bytes)
    pub fn get_range_original<T>(&self, range: Range<T>) -> Option<&str>
    where
        T: RangeBounds<usize> + Clone,
    {
        match range {
            Range::Original(r) => get_range_of(&self.original, r),
            Range::Normalized(_) => self
                .convert_offsets(range)
                .map(|r| get_range_of(&self.original, r))
                .flatten(),
        }
    }

    /// Return a new NormalizedString that contains only the specified range,
    /// indexing on bytes. Any range that splits a UTF-8 char will return None.
    ///
    /// If we want a slice of the `NormalizedString` based on a `Range::Normalized``,
    /// the original part of the `NormalizedString` will contain any "additional"
    /// content on the right, and also on the left. The left will be included
    /// only if we are retrieving the very beginning of the string, since there
    /// is no previous part. The right is always included, up to what's covered
    /// by the next part of the normalized string.  This is important to be able
    /// to build a new `NormalizedString` from multiple contiguous slices
    pub fn slice_bytes<T>(&self, range: Range<T>) -> Option<NormalizedString>
    where
        T: RangeBounds<usize> + Clone,
    {
        let (r, s) = match range {
            Range::Original(_) => (
                range.clone().into_full_range(self.original.len()),
                &self.original,
            ),
            Range::Normalized(_) => (
                range.clone().into_full_range(self.normalized.len()),
                &self.normalized,
            ),
        };

        let (mut start, mut end) = if r == (0..0) {
            (Some(0), Some(0))
        } else {
            (None, None)
        };
        s.char_indices()
            .enumerate()
            .take_while(|(_, (b, _))| *b < r.end)
            .filter(|(_, (b, _))| *b >= r.start)
            .for_each(|(i, (b, c))| {
                if b == r.start {
                    start = Some(i);
                }
                if b + c.len_utf8() == r.end {
                    end = Some(i + 1);
                }
            });

        match range {
            Range::Original(_) => self.slice(Range::Original(start?..end?)),
            Range::Normalized(_) => self.slice(Range::Normalized(start?..end?)),
        }
    }

    /// Return a new NormalizedString that contains only the specified range,
    /// indexing on char
    ///
    /// If we want a slice of the `NormalizedString` based on a `Range::Normalized``,
    /// the original part of the `NormalizedString` will contain any "additional"
    /// content on the right, and also on the left. The left will be included
    /// only if we are retrieving the very beginning of the string, since there
    /// is no previous part. The right is always included, up to what's covered
    /// by the next part of the normalized string.  This is important to be able
    /// to build a new `NormalizedString` from multiple contiguous slices
    pub fn slice<T>(&self, range: Range<T>) -> Option<NormalizedString>
    where
        T: RangeBounds<usize> + Clone,
    {
        let len_original = self.len_original();
        let len_normalized = self.len();

        // Find out the part of the normalized string we should keep
        let r_normalized = match range {
            Range::Original(_) => self.convert_offsets(range.clone())?,
            Range::Normalized(_) => range.clone().into_full_range(len_normalized),
        };

        let r_original = match range {
            Range::Original(_) => range.into_full_range(len_original),
            Range::Normalized(_) => {
                let end_range = self.convert_offsets(Range::Normalized(r_normalized.end..));
                let mut range = self.convert_offsets(range)?;

                // If we take the very beginning of the normalized string, we should take
                // all the beginning of the original too
                if r_normalized.start == 0 && range.start != 0 {
                    range.start = 0;
                }
                // If there is a void between the `end` char we target and the next one, we
                // want to include everything in-between from the original string
                match end_range {
                    Some(r) if r.start > range.end => range.end = r.start,
                    _ => {}
                }
                // If we target the end of the normalized but the original is longer
                if r_normalized.end == self.alignments.len() && len_original > range.end {
                    range.end = len_original;
                }

                range
            }
        };

        // We need to shift the alignments according to the part of the original string that we
        // keep
        let alignment_shift = r_original.start;

        Some(Self {
            original: get_range_of(&self.original, r_original)
                .unwrap_or_default()
                .into(),
            normalized: get_range_of(&self.normalized, r_normalized.clone())
                .unwrap_or_default()
                .into(),
            alignments: self
                .alignments
                .get(r_normalized)?
                .to_vec()
                .iter()
                .map(|(start, end)| (start - alignment_shift, end - alignment_shift))
                .collect(),
        })
    }

    /// Applies transformations to the current normalized version, updating the current
    /// alignments with the new ones.
    /// This method expect an Iterator yielding each char of the new normalized string
    /// with a `change` isize equals to:
    ///   - `1` if this is a new char
    ///   - `-N` if the char is right before N removed chars
    ///   - `0` if this char represents the old one (even if changed)
    /// Since it is possible that the normalized string doesn't include some of the characters at
    /// the beginning of the original one, we need an `initial_offset` which represents the number
    /// of removed chars at the very beginning.
    ///
    /// `change` should never be more than `1`. If multiple chars are added, each of
    /// them has a `change` of `1`, but more doesn't make any sense.
    /// We treat any value above `1` as `1`.
    pub fn transform<I: Iterator<Item = (char, isize)>>(&mut self, dest: I, initial_offset: usize) {
        let mut offset = -(initial_offset as isize);
        let (normalized, alignments): (String, Vec<_>) = dest
            .enumerate()
            .map(|(index, (c, changes))| {
                // A positive offset means we added characters. So we need to remove this offset
                // from the current index to find out the previous id
                let idx = (index as isize - offset) as usize;
                offset += changes;
                let align = if changes.is_positive() {
                    if idx < 1 {
                        (0, 0)
                    } else {
                        // This is a newly inserted character, so it has a length of 0 at the
                        // position of the last character.
                        let end_previous = self.alignments[idx - 1].1;
                        (end_previous, end_previous)
                    }
                } else {
                    self.alignments[idx]
                };
                // Then we keep only the char for string reconstruction
                (c, align)
            })
            .unzip();
        self.alignments = alignments;
        self.normalized = normalized;
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
        let mut removed = 0;
        let filtered = self
            .normalized
            .chars()
            .rev()
            .map(|c| {
                if keep(c) {
                    if removed > 0 {
                        let res = (c, -(removed as isize));
                        removed = 0;
                        Some(res)
                    } else {
                        Some((c, 0))
                    }
                } else {
                    removed += 1;
                    None
                }
            })
            .collect::<Vec<_>>();
        self.transform(filtered.into_iter().rev().filter_map(|o| o), removed);
        self
    }

    /// Prepend the given string to ourself
    pub fn prepend(&mut self, s: &str) -> &mut Self {
        self.normalized.insert_str(0, s);
        self.alignments.splice(0..0, s.chars().map(|_| (0, 0)));
        self
    }

    /// Append the given string to ourself
    pub fn append(&mut self, s: &str) -> &mut Self {
        self.normalized.push_str(s);
        let last_offset = self.alignments.last().map_or((0, 0), |o| (o.1, o.1));
        self.alignments.extend(s.chars().map(|_| last_offset));
        self
    }

    /// Map our characters
    pub fn map<F: Fn(char) -> char>(&mut self, map: F) -> &mut Self {
        self.normalized = self.normalized.chars().map(map).collect::<String>();
        self
    }

    /// Calls the given function for each characters
    pub fn for_each<F: FnMut(char)>(&mut self, foreach: F) -> &mut Self {
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
        let matches = pattern.find_matches(&self.normalized)?;

        let (normalized, alignments): (String, Vec<Offsets>) = matches
            .into_iter()
            .flat_map(|((start, end), is_match)| {
                let len = end - start;
                if is_match {
                    let original_offsets = self
                        .convert_offsets(Range::Normalized(start..end))
                        .expect("Bad offsets when replacing");

                    // Here, since we don't know the exact alignment, each character in
                    // the new normalized part will align to the whole replaced one.
                    itertools::Either::Left(content.chars().zip(std::iter::repeat((
                        original_offsets.start,
                        original_offsets.end,
                    ))))
                } else {
                    // No need to replace anything, just zip the relevant parts
                    itertools::Either::Right(
                        self.normalized
                            .chars()
                            .skip(start)
                            .take(len)
                            .zip(self.alignments.iter().skip(start).take(len).copied()),
                    )
                }
            })
            .unzip();

        self.normalized = normalized;
        self.alignments = alignments;

        Ok(())
    }

    /// Clear the normalized part of the string
    pub fn clear(&mut self) {
        self.normalized = "".into();
        self.alignments = vec![];
    }

    /// Split the current string in many subparts. Specify what to do with the
    /// delimiter.
    ///
    /// This method will always ensure that the entire `self` is covered in the
    /// produced subparts. This means that the delimiter parts will also be included,
    /// and will appear empty if we don't want to include them (their `original`
    /// part will still be present). It should always be possible to merge all the
    /// subparts back to the original `NormalizedString`
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
        self,
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
            .map(|(offsets, remove)| {
                let mut slice = self
                    .slice(Range::Normalized(offsets.0..offsets.1))
                    .expect("NormalizedString bad split");
                if remove {
                    slice.clear();
                }
                slice
            })
            .collect())
    }

    /// Split off ourselves, returning a new Self that contains the range [at, len).
    /// self will then contain the range [0, at).
    /// The provided `at` indexes on `char` not bytes.
    pub fn split_off(&mut self, at: usize) -> Self {
        if at > self.len() {
            return NormalizedString::from("");
        }

        // Split normalized
        let byte_index = self.normalized.chars().enumerate().fold(0, |acc, (i, c)| {
            if i < at {
                acc + c.len_utf8()
            } else {
                acc
            }
        });
        let normalized = self.normalized.split_off(byte_index);
        let alignments = self.alignments.split_off(at);

        // Split original
        let original_at = self.alignments.last().map(|(_, end)| *end).unwrap_or(0);
        let original_byte_index = self.original.chars().enumerate().fold(0, |acc, (i, c)| {
            if i < original_at {
                acc + c.len_utf8()
            } else {
                acc
            }
        });
        let original = self.original.split_off(original_byte_index);

        NormalizedString {
            original,
            normalized,
            alignments,
        }
    }

    /// Merge with the given NormalizedString by appending it to self
    pub fn merge_with(&mut self, other: &NormalizedString) {
        let shift_len = self.len_original();
        self.original.push_str(&other.original);
        self.normalized.push_str(&other.normalized);
        self.alignments.extend(
            other
                .alignments
                .iter()
                .map(|(start, end)| (start + shift_len, end + shift_len)),
        );
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
            self.transform(transformation.into_iter(), leading_spaces);
        }
        self
    }

    /// Returns the length of the normalized string (counting chars not bytes)
    pub fn len(&self) -> usize {
        self.normalized.chars().count()
    }

    /// Returns the length of the original string (counting chars not bytes)
    pub fn len_original(&self) -> usize {
        self.original.chars().count()
    }

    /// Whether empty
    pub fn is_empty(&self) -> bool {
        self.normalized.len() == 0
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

    if start >= len || end > len || start >= end {
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

impl From<String> for NormalizedString {
    fn from(s: String) -> Self {
        let len = s.chars().count();
        Self {
            original: s.clone(),
            normalized: s,
            alignments: (0..len).map(|v| (v, v + 1)).collect(),
        }
    }
}

impl From<&str> for NormalizedString {
    fn from(s: &str) -> Self {
        Self::from(s.to_owned())
    }
}

impl std::iter::FromIterator<NormalizedString> for NormalizedString {
    fn from_iter<I: IntoIterator<Item = NormalizedString>>(iter: I) -> NormalizedString {
        let mut normalized: NormalizedString = "".into();
        for sub in iter {
            normalized.merge_with(&sub)
        }
        normalized
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::reversed_empty_ranges)]
    use super::*;
    use regex::Regex;
    use unicode_categories::UnicodeCategories;

    #[test]
    fn new_chars() {
        let mut n = NormalizedString::from("élégant");
        n.nfd();
        assert_eq!(
            &n.alignments,
            &[
                (0, 1),
                (1, 1),
                (1, 2),
                (2, 3),
                (3, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7)
            ]
        );
    }

    #[test]
    fn unchanged() {
        let mut n = NormalizedString::from("élégant");
        n.nfd().filter(|c| !c.is_mark_nonspacing());
        assert_eq!(
            &n.alignments,
            &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
        );
    }

    #[test]
    fn removed_chars() {
        let mut n = NormalizedString::from("élégant");
        n.filter(|c| c != 'n');
        assert_eq!(
            &n.alignments,
            &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (6, 7)]
        );
    }

    #[test]
    fn mixed_addition_and_removal() {
        let mut n = NormalizedString::from("élégant");
        n.nfd().filter(|c| !c.is_mark_nonspacing() && c != 'n');
        assert_eq!(
            &n.alignments,
            &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (6, 7)]
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
                    vec![(' ', 1), (c, 0), (' ', 1)]
                } else {
                    vec![(c, 0)]
                }
            }),
            0,
        );

        assert_eq!(
            n,
            NormalizedString {
                original: "野口 No".to_owned(),
                normalized: " 野  口  No".to_owned(),
                alignments: vec![
                    (0, 0),
                    (0, 1),
                    (1, 1),
                    (1, 1),
                    (1, 2),
                    (2, 2),
                    (2, 3),
                    (3, 4),
                    (4, 5),
                ]
            }
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
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5)
            ]
        );
        assert_eq!(n.convert_offsets(Range::Normalized(0..4)), Some(0..0));
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
                (3, 3),
                (3, 3),
                (3, 3),
                (3, 3),
                (3, 3),
                (3, 3)
            ]
        );
        assert_eq!(
            n.convert_offsets(Range::Normalized(3.." there".len())),
            Some(3..3)
        );
    }

    #[test]
    fn get_range() {
        let s = String::from("Hello my name is John 👋");
        assert_eq!(get_range_of(&s, ..), Some(&s[..]));
        assert_eq!(get_range_of(&s, 17..), Some("John 👋"));
    }

    #[test]
    fn merge() {
        // Merge unmodified
        let s = NormalizedString::from("A sentence that will be merged");
        let mut merged = NormalizedString::from("A sentence");
        let s2 = NormalizedString::from(" that will");
        let s3 = NormalizedString::from(" be merged");
        merged.merge_with(&s2);
        merged.merge_with(&s3);
        assert_eq!(s, merged);

        // Merge grown normalized
        let mut s = NormalizedString::from("A sentence that will be merged");
        s.prepend(" ");
        let mut merged = NormalizedString::from("A sentence");
        let s2 = NormalizedString::from(" that will");
        let s3 = NormalizedString::from(" be merged");
        merged.prepend(" ");
        merged.merge_with(&s2);
        merged.merge_with(&s3);
        assert_eq!(s, merged);

        // Merge shrinked normalized
        let mut s = NormalizedString::from("  A sentence that will be merged  ");
        s.strip();
        let mut merged = NormalizedString::from("  A sentence");
        merged.strip();
        let s2 = NormalizedString::from(" that will");
        let mut s3 = NormalizedString::from(" be merged  ");
        s3.rstrip();
        merged.merge_with(&s2);
        merged.merge_with(&s3);
        assert_eq!(s, merged);
    }

    #[test]
    fn slice() {
        let mut s = NormalizedString::from("𝔾𝕠𝕠𝕕 𝕞𝕠𝕣𝕟𝕚𝕟𝕘");
        s.nfkc();

        assert_eq!(
            s.slice(Range::Original(0..4)),
            Some(NormalizedString {
                original: "𝔾𝕠𝕠𝕕".to_string(),
                normalized: "Good".to_string(),
                alignments: vec![(0, 1), (1, 2), (2, 3), (3, 4)]
            })
        );
        assert_eq!(
            s.slice(Range::Normalized(0..4)),
            Some(NormalizedString {
                original: "𝔾𝕠𝕠𝕕".to_string(),
                normalized: "Good".to_string(),
                alignments: vec![(0, 1), (1, 2), (2, 3), (3, 4)]
            })
        );

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
    fn slice_bytes() {
        let mut s = NormalizedString::from("𝔾𝕠𝕠𝕕 𝕞𝕠𝕣𝕟𝕚𝕟𝕘");
        s.nfkc();

        assert_eq!(
            s.slice_bytes(Range::Original(0..16)),
            Some(NormalizedString {
                original: "𝔾𝕠𝕠𝕕".to_string(),
                normalized: "Good".to_string(),
                alignments: vec![(0, 1), (1, 2), (2, 3), (3, 4)]
            })
        );
        assert_eq!(
            s.slice_bytes(Range::Original(17..)),
            Some(NormalizedString {
                original: "𝕞𝕠𝕣𝕟𝕚𝕟𝕘".to_string(),
                normalized: "morning".to_string(),
                alignments: vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
            })
        );
        assert_eq!(
            s.slice_bytes(Range::Normalized(0..4)),
            Some(NormalizedString {
                original: "𝔾𝕠𝕠𝕕".to_string(),
                normalized: "Good".to_string(),
                alignments: vec![(0, 1), (1, 2), (2, 3), (3, 4)]
            })
        );
        assert_eq!(s.slice_bytes(Range::Original(0..10)), None);

        // Check that we get a `None` if we try to split chars
        for cut_at in 1..s.len() {
            let res = s.slice_bytes(Range::Original(..cut_at));
            // The chars in the original string all take 4 bytes.
            assert!(if cut_at % 4 == 0 {
                res.is_some()
            } else {
                res.is_none()
            });
        }
    }

    #[test]
    fn slice_coverage() {
        let mut s = NormalizedString::from(" Hello   friend ");
        s.filter(|c| !c.is_whitespace());
        assert_eq!(s.get(), "Hellofriend");

        // Multiple slices with Normalized range
        for cut_at in 1..s.len() {
            let mut slices = vec![];
            slices.push(s.slice(Range::Normalized(..cut_at)).unwrap());
            slices.push(s.slice(Range::Normalized(cut_at..)).unwrap());
            let rebuilt: NormalizedString = slices.into_iter().collect();
            assert_eq!(rebuilt, s);
        }

        // Multiple slices with Original range
        for cut_at in 1..s.len_original() {
            let mut slices = vec![];
            slices.push(s.slice(Range::Original(..cut_at)).unwrap());
            slices.push(s.slice(Range::Original(cut_at..)).unwrap());
            let rebuilt: NormalizedString = slices.into_iter().collect();
            assert_eq!(rebuilt, s);
        }
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
            let splits = s.clone().split('-', behavior).unwrap();
            assert_eq!(splits.iter().map(|n| n.get()).collect::<Vec<_>>(), result);
        };

        test(Removed, vec!["The", "", "final", "", "", "countdown"]);
        test(Isolated, vec!["The", "-", "final", "-", "-", "countdown"]);
        test(MergedWithPrevious, vec!["The-", "final-", "-", "countdown"]);
        test(MergedWithNext, vec!["The", "-final", "-", "-countdown"]);
    }
}
