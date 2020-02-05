use super::Result;
use std::cmp::Ordering;
use unicode_normalization_alignments::UnicodeNormalization;

/// Takes care of pre-processing strings.
pub trait Normalizer {
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()>;
}

/// A normalized string takes care of keeping both versions of a `String`, and
/// provides necessary alignments to retrieve ranges of both strings.
#[derive(Default, Debug, Clone)]
pub struct NormalizedString {
    original: String,
    normalized: String,
    /// Mapping from normalized string to original one
    /// (pos, changes) where pos is the position in the modified string, and changes an isize
    /// representing the number of insertions or deletions
    alignments: Vec<(usize, usize)>,
}

impl std::cmp::PartialEq for NormalizedString {
    fn eq(&self, other: &NormalizedString) -> bool {
        self.normalized == other.normalized
    }
}

impl NormalizedString {
    pub fn from(s: &str) -> Self {
        NormalizedString {
            original: s.to_owned(),
            normalized: s.to_owned(),
            alignments: (0..s.chars().count()).map(|v| (v, v + 1)).collect(),
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

    /// Return the range of the original string corresponding to the received range on the
    /// normalized string. Returns None if out of bounds
    pub fn get_original_offsets(
        &self,
        range: std::ops::Range<usize>,
    ) -> Option<std::ops::Range<usize>> {
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

    fn get_range_of(&self, s: &str, range: std::ops::Range<usize>) -> Option<String> {
        let len = s.chars().count();
        if range.start >= len || range.end > len {
            None
        } else {
            Some(
                s.chars()
                    .enumerate()
                    .skip(range.start)
                    .map(|(i, c)| {
                        if i >= range.start && i < range.end {
                            Some(c)
                        } else {
                            None
                        }
                    })
                    .fuse()
                    .filter(|c| c.is_some())
                    .map(|c| c.unwrap())
                    .collect::<String>(),
            )
        }
    }

    /// Return a range of the normalized string (indexing on char not bytes)
    pub fn get_range(&self, range: std::ops::Range<usize>) -> Option<String> {
        self.get_range_of(&self.normalized, range)
    }

    /// Return a range of the original string, using a range from the normalized string
    pub fn get_range_original(&self, range: std::ops::Range<usize>) -> Option<String> {
        self.get_original_offsets(range)
            .map(|range| self.get_range_of(&self.original, range))
            .flatten()
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
        let mut offset = 0;
        let mut remaining_offset = initial_offset;
        let (ch, alignments): (Vec<_>, Vec<_>) = dest
            .enumerate()
            .map(|(index, (c, changes))| {
                let changes = if remaining_offset != 0 {
                    let c = changes - remaining_offset as isize;
                    remaining_offset = 0;
                    c
                } else {
                    changes
                };

                let uof = if offset < 0 {
                    -offset as usize
                } else {
                    offset as usize
                };
                // A positive offset means we added characters. So we need to remove this offset
                // from the current index to find out the previous id
                let idx = if offset < 0 { index + uof } else { index - uof };
                let align = match changes.cmp(&0) {
                    // This is a newly inserted character, so we use the alignment from the
                    // previous one
                    Ordering::Greater => {
                        offset += 1;
                        if idx < 1 {
                            Some((0, 0))
                        } else {
                            self.alignments.get(idx - 1).copied()
                        }
                    }
                    // No changes required here
                    Ordering::Equal => self.alignments.get(idx).copied(),
                    // Some characters where removed, so we merge our range with the one from the
                    // removed characters as the new alignment
                    Ordering::Less => {
                        let uch = -changes as usize;
                        offset += changes;
                        self.alignments.get(idx..=idx + uch).map(|alignments| {
                            let min = alignments
                                .iter()
                                .map(|(start, end)| usize::min(*start, *end))
                                .min()
                                .unwrap();
                            let max = alignments
                                .iter()
                                .map(|(start, end)| usize::max(*start, *end))
                                .max()
                                .unwrap();
                            (min, max)
                        })
                    }
                };

                // Then we keep only the char for string reconstruction
                (
                    c,
                    align.expect("Bad alignement in NormalizedString::transform"),
                )
            })
            .unzip();
        self.alignments = alignments;
        self.normalized = ch.iter().collect::<String>();
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
    pub fn filter<F: Fn(&char) -> bool>(&mut self, filter: F) -> &mut Self {
        let mut removed: usize = 0;
        let mut filtered = self
            .normalized
            .chars()
            // We need to collect here to be able to reverse the iterator because Char is not ended
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|c| {
                let keep = filter(&c);
                if keep {
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
        // For some reason, if we use rev, and unwrap directly, some parts of the tuples we return
        // above get mixed up... So we collect first, then reverse in place
        filtered.reverse();
        self.transform(
            filtered.iter().filter(|o| o.is_some()).map(|o| o.unwrap()),
            removed,
        );
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

    /// Split off ourselves, returning a new Self that contains the range [at, len).
    /// self will then contain the range [0, at).
    /// The provided `at` indexes on `char` not bytes.
    pub fn split_off(&mut self, at: usize) -> Self {
        if at > self.len() {
            return NormalizedString::from("");
        }

        // Split normalized
        let byte_index = self
            .normalized
            .chars()
            .enumerate()
            .map(|(i, c)| if i < at { Some(c.len_utf8()) } else { None })
            .fuse()
            .filter(|c| c.is_some())
            .map(|c| c.unwrap())
            .sum::<usize>();
        let normalized = self.normalized.split_off(byte_index);
        let alignments = self.alignments.split_off(at);

        // Split original
        let original_at = self.alignments.last().map(|(_, end)| *end).unwrap_or(0);
        let original_byte_index = self
            .original
            .chars()
            .enumerate()
            .map(|(i, c)| {
                if i < original_at {
                    Some(c.len_utf8())
                } else {
                    None
                }
            })
            .fuse()
            .filter(|c| c.is_some())
            .map(|c| c.unwrap())
            .sum::<usize>();
        let original = self.original.split_off(original_byte_index);

        NormalizedString {
            original,
            normalized,
            alignments,
        }
    }

    /// Merge with the given NormalizedString by appending it to self
    pub fn merge_with(&mut self, other: &NormalizedString) {
        self.original.push_str(&other.original);
        let len = self.len();
        self.alignments.extend(
            other
                .alignments
                .iter()
                .map(|(start, end)| (start + len, end + len)),
        );
        self.normalized.push_str(&other.normalized);
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

#[cfg(test)]
mod tests {
    use super::*;
    use unicode_categories::UnicodeCategories;

    #[test]
    fn new_chars() {
        let mut n = NormalizedString::from("élégant");
        n.nfd();
        assert_eq!(
            &n.alignments,
            &[
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
        n.filter(|c| *c != 'n');
        assert_eq!(
            &n.alignments,
            &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 7)]
        );
    }

    #[test]
    fn mixed_addition_and_removal() {
        let mut n = NormalizedString::from("élégant");
        n.nfd().filter(|c| !c.is_mark_nonspacing() && *c != 'n');
        assert_eq!(
            &n.alignments,
            &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 7)]
        );
    }

    #[test]
    fn original_range() {
        let mut n = NormalizedString::from("Hello_______ World!");
        n.filter(|c| *c != '_').lowercase();
        let world_n = n.get_range(6..11).unwrap();
        let world_o = n.get_range_original(6..11).unwrap();
        assert_eq!(world_n, "world");
        assert_eq!(world_o, "World");
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
            n.get_range_original(0..n.normalized.len()),
            Some("Hello".into())
        );
    }

    #[test]
    fn remove_at_beginning() {
        let mut n = NormalizedString::from("     Hello");
        n.filter(|c| !c.is_whitespace());
        assert_eq!(n.get_range_original(1.."Hello".len()), Some("ello".into()));
        assert_eq!(
            n.get_range_original(0..n.normalized.len()),
            Some("     Hello".into())
        );
    }

    #[test]
    fn remove_at_end() {
        let mut n = NormalizedString::from("Hello    ");
        n.filter(|c| !c.is_whitespace());
        assert_eq!(n.get_range_original(0..4), Some("Hell".into()));
        assert_eq!(
            n.get_range_original(0..n.normalized.len()),
            Some("Hello    ".into())
        );
    }

    #[test]
    fn removed_around_both_edges() {
        let mut n = NormalizedString::from("  Hello  ");
        n.filter(|c| !c.is_whitespace());
        assert_eq!(&n.normalized, "Hello");

        assert_eq!(
            n.get_range_original(0.."Hello".len()),
            Some("  Hello  ".into())
        );
        assert_eq!(n.get_range_original(1.."Hell".len()), Some("ell".into()));
    }
}
