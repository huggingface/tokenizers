//! Cutting a SentencePiece sequence into `▁▁▁word` units before BPE.
//!
//! Some Metaspace-style models reach the model stage with no word boundaries at
//! all. llama-2 has no pre-tokenizer, and gemma-4's `Split` on `" "` never
//! matches because its normalizer already replaced every space with `▁`. The
//! model then merges a whole document as a single word: slow, and far too long
//! for any per-word cache to help.
//!
//! Cutting at the `▁`s is not automatically safe: gemma-4's vocab holds `>▁</`,
//! so a merge really can span two words there. But a merge result is always a
//! vocab piece, so only a piece holding a `▁` *after* a non-`▁` character can
//! span a cut. When no piece does, cutting gives the ids of merging the whole
//! sequence; gemma-4's single such piece is kept and checked at each cut.

/// UTF-8 of `▁` (U+2581), the SentencePiece space marker.
const MARK: &[u8] = "\u{2581}".as_bytes();

/// Past this many boundary-crossing pieces, checking them all at every cut
/// costs more than cutting saves, so we keep merging whole sequences.
const MAX_CROSSING: usize = 32;

/// A vocab piece holding a `▁` after a non-`▁` char, split around that `▁`: it
/// can span a cut, so a cut it sits across is skipped.
#[derive(Debug, PartialEq, Eq)]
struct Crossing {
    before: Box<[u8]>,
    after: Box<[u8]>,
}

/// Where a sequence may be cut into independently-mergeable units.
pub(crate) struct MarkSplit {
    /// Empty means every candidate cut is safe.
    crossing: Vec<Crossing>,
    /// Bitset of the bytes that can precede a crossing piece's `▁`. A cut
    /// preceded by any other byte needs no piece check.
    crossing_prev: [u64; 4],
}

impl MarkSplit {
    /// `None` when cutting cannot be proven exact: the vocab lacks `▁`, or its
    /// crossing pieces are too many or too tangled for the per-cut check.
    pub(crate) fn build(vocab: &[(Vec<u8>, u32)]) -> Option<Self> {
        if !vocab.iter().any(|(piece, _)| piece.as_slice() == MARK) {
            return None;
        }
        let mut crossing = Vec::new();
        let mut crossing_prev = [0u64; 4];
        for (piece, _) in vocab {
            let mut at = 0;
            while let Some(mark) = next_mark(piece, at) {
                at = mark + MARK.len();
                if !starts_unit(piece, mark) {
                    continue;
                }
                let (before, after) = (&piece[..mark], &piece[at..]);
                // Another `▁` on either side means the piece can span the *next*
                // cut too, which one before/after compare cannot rule out.
                if crossing.len() == MAX_CROSSING || has_mark(before) || has_mark(after) {
                    return None;
                }
                let prev = *before
                    .last()
                    .expect("a unit-starting mark has a byte before it");
                crossing_prev[(prev >> 6) as usize] |= 1 << (prev & 63);
                crossing.push(Crossing {
                    before: before.into(),
                    after: after.into(),
                });
            }
        }
        Some(Self {
            crossing,
            crossing_prev,
        })
    }

    /// The units of `sequence`, each starting at a `▁` that follows a non-`▁`
    /// char. Merging these one by one gives the ids of merging `sequence`
    /// whole.
    pub(crate) fn units<'a>(&'a self, sequence: &'a str) -> Units<'a> {
        Units {
            split: self,
            sequence,
            start: 0,
            marks: memchr::memchr_iter(MARK[0], sequence.as_bytes()),
        }
    }

    /// Does a crossing piece sit across the mark at `at`? Only then can a merge
    /// span the cut. `at` is the offset of a unit-starting mark, so it is > 0.
    fn crosses(&self, bytes: &[u8], at: usize) -> bool {
        let prev = bytes[at - 1];
        if self.crossing_prev[(prev >> 6) as usize] & (1 << (prev & 63)) == 0 {
            return false;
        }
        let mark_end = at + MARK.len();
        self.crossing.iter().any(|piece| {
            let (before, after) = (piece.before.as_ref(), piece.after.as_ref());
            at >= before.len()
                && &bytes[at - before.len()..at] == before
                && bytes.len() >= mark_end + after.len()
                && &bytes[mark_end..mark_end + after.len()] == after
        })
    }
}

/// Yields the units of one sequence. See [`MarkSplit::units`].
pub(crate) struct Units<'a> {
    split: &'a MarkSplit,
    sequence: &'a str,
    start: usize,
    marks: memchr::Memchr<'a>,
}

impl<'a> Iterator for Units<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        if self.start >= self.sequence.len() {
            return None;
        }
        let bytes = self.sequence.as_bytes();
        // `marks` yields every `▁` first byte, including the ones inside a mark
        // run; `starts_unit` keeps only the first of each run.
        for mark in self.marks.by_ref() {
            if !bytes[mark..].starts_with(MARK)
                || !starts_unit(bytes, mark)
                || self.split.crosses(bytes, mark)
            {
                continue;
            }
            let unit = &self.sequence[self.start..mark];
            self.start = mark;
            return Some(unit);
        }
        let unit = &self.sequence[self.start..];
        self.start = self.sequence.len();
        Some(unit)
    }
}

/// Does the mark at `at` begin a unit? Leading marks and the marks inside a
/// `▁▁▁` run belong to the unit they start, so only the first of a run does.
fn starts_unit(bytes: &[u8], at: usize) -> bool {
    at > 0 && !(at >= MARK.len() && &bytes[at - MARK.len()..at] == MARK)
}

fn next_mark(bytes: &[u8], from: usize) -> Option<usize> {
    let mut at = from;
    while let Some(off) = memchr::memchr(MARK[0], &bytes[at..]) {
        at += off;
        if bytes[at..].starts_with(MARK) {
            return Some(at);
        }
        at += 1;
    }
    None
}

fn has_mark(bytes: &[u8]) -> bool {
    next_mark(bytes, 0).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab(pieces: &[&str]) -> Vec<(Vec<u8>, u32)> {
        pieces
            .iter()
            .enumerate()
            .map(|(id, p)| (p.as_bytes().to_vec(), id as u32))
            .collect()
    }

    fn units(split: &MarkSplit, sequence: &str) -> Vec<String> {
        split.units(sequence).map(str::to_string).collect()
    }

    #[test]
    fn needs_the_mark_in_vocab() {
        assert!(MarkSplit::build(&vocab(&["a", "b"])).is_none());
        assert!(MarkSplit::build(&vocab(&["a", "▁"])).is_some());
    }

    #[test]
    fn cuts_at_marks_that_follow_text() {
        let split = MarkSplit::build(&vocab(&["▁", "▁hello", "▁world"])).unwrap();
        assert!(split.crossing.is_empty());
        assert_eq!(units(&split, "▁hello▁world"), ["▁hello", "▁world"]);
        assert_eq!(units(&split, "hello▁world"), ["hello", "▁world"]);
    }

    #[test]
    fn keeps_mark_runs_with_the_word_they_precede() {
        let split = MarkSplit::build(&vocab(&["▁"])).unwrap();
        assert_eq!(units(&split, "a▁▁▁b"), ["a", "▁▁▁b"]);
        assert_eq!(units(&split, "▁▁▁a"), ["▁▁▁a"]);
        assert_eq!(units(&split, "▁▁▁"), ["▁▁▁"]);
        assert_eq!(units(&split, "a▁"), ["a", "▁"]);
    }

    #[test]
    fn no_marks_leaves_one_unit() {
        let split = MarkSplit::build(&vocab(&["▁"])).unwrap();
        assert_eq!(units(&split, "hello"), ["hello"]);
        assert_eq!(units(&split, ""), Vec::<String>::new());
    }

    #[test]
    fn skips_the_cut_a_crossing_piece_spans() {
        // gemma-4's shape: ">▁</" holds a mark after ">".
        let split = MarkSplit::build(&vocab(&["▁", ">▁</", "▁a"])).unwrap();
        assert_eq!(split.crossing.len(), 1);
        assert_eq!(units(&split, "<b>▁</b>"), ["<b>▁</b>"]);
        // Same prev byte, but "</" does not follow: the cut stands.
        assert_eq!(units(&split, "<b>▁a"), ["<b>", "▁a"]);
        // Other cuts in the same sequence are unaffected.
        assert_eq!(units(&split, "x▁<b>▁</b>▁y"), ["x", "▁<b>▁</b>", "▁y"]);
    }

    #[test]
    fn trailing_mark_piece_crosses_too() {
        let split = MarkSplit::build(&vocab(&["▁", "p▁"])).unwrap();
        assert_eq!(
            split.crossing,
            [Crossing {
                before: b"p".to_vec().into(),
                after: b"".to_vec().into(),
            }]
        );
        assert_eq!(units(&split, "p▁a"), ["p▁a"]);
        assert_eq!(units(&split, "q▁a"), ["q", "▁a"]);
    }

    #[test]
    fn gives_up_on_two_marks_in_one_piece() {
        assert!(MarkSplit::build(&vocab(&["▁", "a▁b▁c"])).is_none());
    }

    #[test]
    fn gives_up_past_the_crossing_limit() {
        let mut pieces = vec!["▁".to_string()];
        pieces.extend((0..=MAX_CROSSING).map(|i| format!("{i}▁x")));
        let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
        assert!(MarkSplit::build(&vocab(&refs)).is_none());
    }
}
