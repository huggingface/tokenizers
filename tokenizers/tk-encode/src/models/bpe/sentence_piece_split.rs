//! Cutting a SentencePiece sequence into words before BPE merges it.
//!
//! SentencePiece vocabs (gemma-4, llama-2, …) write a space as `▁` (U+2581),
//! so `"tell me"` reaches the model step as `"tell▁me"`.
//!
//! Those tokenizers usually have no pre-tokenizer, so the model gets the whole
//! input as a single sequence instead of one sequence per word.
//! Merging that in one go costs more than merging the same text word by word,
//! and big sequences are less cache-friendly as smaller words.
//!
//! This module implements a split on every `▁`, as long as the vocabulary proves
//! no BPE merge can cross the split boundary (which would make the split invalid).
//!
//! A merge only ever produces a piece that is in the vocabulary,
//! so a merge can only reach across a cut if some piece of vocabulary
//! holds a `▁` that is not at its start.
//!
//! gemma-4 has exactly one such piece: `>▁</`

/// UTF-8 bytes of `▁` (U+2581), the SentencePiece space marker. An array rather
/// than a slice, so `first_chunk`/`last_chunk` can match it without a length
/// check of their own.
const DELIMITER: [u8; 3] = {
    let bytes = "▁".as_bytes();
    [bytes[0], bytes[1], bytes[2]]
};

/// Width of one padded half of a crossing piece. Whatever the halves really
/// measure, they are compared a full `u128` at a time.
const HALF_WIDTH: usize = size_of::<u128>();

/// One half of a [`CrossingPiece`], padded out to [`HALF_WIDTH`] bytes.
///
/// The halves are a byte or two long, but their length is only known once the
/// vocabulary is read, and comparing a run-time number of bytes means calling
/// `memcmp`. Padding to a fixed width turns that into a couple of register
/// operations. Worth more than it looks: the call inlined into [`Words::next`],
/// spilling registers around the `memchr` scan for every sequence — including
/// the vocabs with no crossing piece to compare in the first place. Dropping it
/// takes ~5% off the splitting loop.
#[derive(Debug, PartialEq, Eq)]
struct HalfPiece {
    bytes: u128,
    /// `0xff` over the bytes of the half, `0` over the padding.
    mask: u128,
}

impl HalfPiece {
    /// The half at the end of the window, where the bytes running up to a cut land.
    fn ending(half: &[u8]) -> Option<Self> {
        Self::padded(half, HALF_WIDTH.checked_sub(half.len())?)
    }

    /// The half at the start of the window, where the bytes following a cut land.
    fn starting(half: &[u8]) -> Option<Self> {
        Self::padded(half, 0)
    }

    /// `None` when the half is wider than the window.
    fn padded(half: &[u8], at: usize) -> Option<Self> {
        let mut bytes = [0u8; HALF_WIDTH];
        let mut mask = [0u8; HALF_WIDTH];
        bytes.get_mut(at..at + half.len())?.copy_from_slice(half);
        mask[at..at + half.len()].fill(0xff);
        Some(Self {
            bytes: u128::from_le_bytes(bytes),
            mask: u128::from_le_bytes(mask),
        })
    }

    fn matches(&self, window: u128) -> bool {
        (window ^ self.bytes) & self.mask == 0
    }
}

/// A [`CrossingPiece`] is a piece of vocabulary that overlaps with a possible split, for example
/// `>▁</` in the gemma4 vocabulary. Splitting the sequence inside a crossing piece could output
/// different tokens, so we don't do it.
///
/// `>▁</` becomes `CrossingPiece { before: ">", after: "</" }`.
#[derive(Debug, PartialEq, Eq)]
struct CrossingPiece {
    before: HalfPiece,
    after: HalfPiece,
}

impl CrossingPiece {
    fn new(before: &[u8], after: &[u8]) -> Option<Self> {
        Some(Self {
            before: HalfPiece::ending(before)?,
            after: HalfPiece::starting(after)?,
        })
    }
}

/// How many crossing pieces we put up with before giving up on splitting the sequence
///
/// A crossing piece is a piece of vocabulary that overlaps with a possible split, for example
/// `>▁</` in the gemma4 vocabulary. Splitting the sequence inside a crossing piece could output
/// different tokens, so we don't do it.
const MAX_CROSSING_PIECES: usize = 32;

pub(crate) struct SentencePieceSplitter {
    /// Empty means no merge can reach across a cut, so every cut is kept.
    crossing: Vec<CrossingPiece>,
    /// Last bytes of the `before` halves, as a 256-bit set. A cut with any
    /// other byte in front of it matches no crossing piece, which is how most
    /// cuts get away without a single comparison.
    crossing_prev: [u64; 4],
}

impl SentencePieceSplitter {
    /// Collects the vocab pieces a merge could use to reach across a cut.
    ///
    /// `None` turns cutting off: either this is not a SentencePiece vocab (no
    /// `▁` piece in it), or its crossing pieces are too many, too long or too
    /// entangled to rule out cheaply.
    pub(crate) fn build(vocab: &[(Vec<u8>, u32)]) -> Option<Self> {
        if !vocab.iter().any(|(piece, _)| piece.as_slice() == DELIMITER) {
            return None;
        }
        let mut crossing = Vec::new();
        let mut crossing_prev = [0u64; 4];
        for (piece, _) in vocab {
            let mut at = 0;
            while let Some(mark) = next_mark(piece, at) {
                at = mark + DELIMITER.len();
                // A piece starting with `▁` sits right after a cut, not across
                // it, and a `▁` following another one is no cut at all.
                if !starts_word(piece, mark) {
                    continue;
                }
                let (before, after) = (&piece[..mark], &piece[at..]);
                // A piece with a second `▁` reaches across two cuts at once, and
                // checking one cut at a time no longer proves anything. None of
                // the vocabs we tested has one, so drop cutting instead.
                if crossing.len() == MAX_CROSSING_PIECES || has_mark(before) || has_mark(after) {
                    return None;
                }
                let prev = *before
                    .last()
                    .expect("a word-starting mark has a byte before it");
                crossing_prev[(prev >> 6) as usize] |= 1 << (prev & 63);
                crossing.push(CrossingPiece::new(before, after)?);
            }
        }
        Some(Self {
            crossing,
            crossing_prev,
        })
    }

    /// The words of `sequence`: the text up to the first safe cut, then one per
    /// cut. Merging them one after another gives the ids of merging `sequence`
    /// whole.
    pub(crate) fn split<'a>(&'a self, sequence: &'a str) -> Words<'a> {
        Words {
            split: self,
            sequence,
            start: 0,
            marks: memchr::memchr_iter(DELIMITER[0], sequence.as_bytes()),
        }
    }

    /// Could a crossing piece cover the `▁` at `at`, so that a merge reaches
    /// over a cut placed there? A match only means such a merge is possible, not
    /// that BPE performs it — either way we leave the text in one piece. `at`
    /// points at a word-starting `▁`, so there is a byte in front of it.
    fn crosses(&self, bytes: &[u8], at: usize) -> bool {
        let prev = bytes[at - 1];
        if self.crossing_prev[(prev >> 6) as usize] & (1 << (prev & 63)) == 0 {
            return false;
        }
        // Both windows are zero-padded, so a half can only match beyond the ends
        // of `bytes` if the half itself holds a NUL byte — and one match too many
        // only ever leaves the text uncut.
        let before = window_ending(&bytes[..at]);
        let after = window_starting(&bytes[at + DELIMITER.len()..]);
        self.crossing
            .iter()
            .any(|piece| piece.before.matches(before) && piece.after.matches(after))
    }
}

/// The last [`HALF_WIDTH`] bytes of `bytes`, padded on the left when there are
/// fewer, packed the way [`HalfPiece::ending`] packs a half.
fn window_ending(bytes: &[u8]) -> u128 {
    match bytes.last_chunk() {
        Some(window) => u128::from_le_bytes(*window),
        None => {
            let mut window = [0u8; HALF_WIDTH];
            window[HALF_WIDTH - bytes.len()..].copy_from_slice(bytes);
            u128::from_le_bytes(window)
        }
    }
}

/// The first [`HALF_WIDTH`] bytes of `bytes`, padded on the right when there are
/// fewer, packed the way [`HalfPiece::starting`] packs a half.
fn window_starting(bytes: &[u8]) -> u128 {
    match bytes.first_chunk() {
        Some(window) => u128::from_le_bytes(*window),
        None => {
            let mut window = [0u8; HALF_WIDTH];
            window[..bytes.len()].copy_from_slice(bytes);
            u128::from_le_bytes(window)
        }
    }
}

/// The words of one sequence, walked cut by cut. Built by
/// [`SentencePieceSplit::split`].
pub(crate) struct Words<'a> {
    split: &'a SentencePieceSplitter,
    sequence: &'a str,
    start: usize,
    marks: memchr::Memchr<'a>,
}

impl<'a> Iterator for Words<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        if self.start >= self.sequence.len() {
            return None;
        }
        let bytes = self.sequence.as_bytes();
        for mark in self.marks.by_ref() {
            // `marks` only matched `▁`'s first byte, which other characters
            // share, hence the full compare; the two other checks keep the
            // marks a cut is allowed at.
            if !is_mark(bytes, mark) || !starts_word(bytes, mark) || self.split.crosses(bytes, mark)
            {
                continue;
            }
            let word = &self.sequence[self.start..mark];
            self.start = mark;
            return Some(word);
        }
        let word = &self.sequence[self.start..];
        self.start = self.sequence.len();
        Some(word)
    }
}

/// Does a `▁` start at `at`?
fn is_mark(bytes: &[u8], at: usize) -> bool {
    bytes[at..].first_chunk() == Some(&DELIMITER)
}

/// Would a word start at the `▁` sitting at `at`? Not if it opens the sequence,
/// and not if another `▁` comes right before it: a run of marks stays whole with
/// the word that follows it (`a▁▁▁b` → `a`, `▁▁▁b`), because vocabs hold pieces
/// made of several marks (`▁▁`, `▁▁▁`, …) and a cut inside a run would lose
/// those merges.
fn starts_word(bytes: &[u8], at: usize) -> bool {
    at > 0 && bytes[..at].last_chunk() != Some(&DELIMITER)
}

/// Offset of the first `▁` at or after `from`. `memchr` matches its first byte,
/// which other characters share too, so every hit is compared in full.
fn next_mark(bytes: &[u8], from: usize) -> Option<usize> {
    let mut at = from;
    while let Some(off) = memchr::memchr(DELIMITER[0], &bytes[at..]) {
        at += off;
        if is_mark(bytes, at) {
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

    fn words(split: &SentencePieceSplitter, sequence: &str) -> Vec<String> {
        split.split(sequence).map(str::to_string).collect()
    }

    #[test]
    fn needs_the_mark_in_vocab() {
        assert!(SentencePieceSplitter::build(&vocab(&["a", "b"])).is_none());
        assert!(SentencePieceSplitter::build(&vocab(&["a", "▁"])).is_some());
    }

    #[test]
    fn cuts_at_marks_that_follow_text() {
        let split = SentencePieceSplitter::build(&vocab(&["▁", "▁hello", "▁world"])).unwrap();
        assert!(split.crossing.is_empty());
        assert_eq!(words(&split, "▁hello▁world"), ["▁hello", "▁world"]);
        assert_eq!(words(&split, "hello▁world"), ["hello", "▁world"]);
    }

    #[test]
    fn keeps_mark_runs_with_the_word_they_precede() {
        let split = SentencePieceSplitter::build(&vocab(&["▁"])).unwrap();
        assert_eq!(words(&split, "a▁▁▁b"), ["a", "▁▁▁b"]);
        assert_eq!(words(&split, "▁▁▁a"), ["▁▁▁a"]);
        assert_eq!(words(&split, "▁▁▁"), ["▁▁▁"]);
        assert_eq!(words(&split, "a▁"), ["a", "▁"]);
    }

    #[test]
    fn no_marks_leaves_one_word() {
        let split = SentencePieceSplitter::build(&vocab(&["▁"])).unwrap();
        assert_eq!(words(&split, "hello"), ["hello"]);
        assert_eq!(words(&split, ""), Vec::<String>::new());
    }

    #[test]
    fn skips_the_cut_a_crossing_piece_spans() {
        // gemma-4's real case: ">▁</" holds a `▁` in the middle, so BPE can
        // merge across the space between an HTML tag and its closing tag.
        let split = SentencePieceSplitter::build(&vocab(&["▁", ">▁</", "▁a"])).unwrap();
        assert_eq!(split.crossing.len(), 1);
        assert_eq!(words(&split, "<b>▁</b>"), ["<b>▁</b>"]);
        // ">" in front of the `▁` again, but "</" does not follow, so the piece
        // cannot form here and the cut stands.
        assert_eq!(words(&split, "<b>▁a"), ["<b>", "▁a"]);
        // Only the cut the piece covers is dropped, not the other ones.
        assert_eq!(words(&split, "x▁<b>▁</b>▁y"), ["x", "▁<b>▁</b>", "▁y"]);
    }

    #[test]
    fn trailing_mark_piece_crosses_too() {
        let split = SentencePieceSplitter::build(&vocab(&["▁", "p▁"])).unwrap();
        assert_eq!(split.crossing, [CrossingPiece::new(b"p", b"").unwrap()]);
        assert_eq!(words(&split, "p▁a"), ["p▁a"]);
        assert_eq!(words(&split, "q▁a"), ["q", "▁a"]);
    }

    #[test]
    fn gives_up_on_two_marks_in_one_piece() {
        assert!(SentencePieceSplitter::build(&vocab(&["▁", "a▁b▁c"])).is_none());
    }

    #[test]
    fn gives_up_past_the_crossing_limit() {
        let mut pieces = vec!["▁".to_string()];
        pieces.extend((0..=MAX_CROSSING_PIECES).map(|i| format!("{i}▁x")));
        let refs: Vec<&str> = pieces.iter().map(String::as_str).collect();
        assert!(SentencePieceSplitter::build(&vocab(&refs)).is_none());
    }

    #[test]
    fn matches_a_half_that_fills_the_window() {
        let before = "a".repeat(HALF_WIDTH);
        let split =
            SentencePieceSplitter::build(&vocab(&["▁", &format!("{before}▁x"), "▁x"])).unwrap();
        assert_eq!(
            words(&split, &format!("{before}▁x")),
            [format!("{before}▁x")]
        );
        // One byte short of the half: the window pads with a zero the half does
        // not hold, so the piece cannot form and the cut stands.
        let short = &before[1..];
        assert_eq!(words(&split, &format!("{short}▁x")), [short, "▁x"]);
    }

    #[test]
    fn gives_up_on_a_half_wider_than_the_window() {
        let before = "a".repeat(HALF_WIDTH + 1);
        assert!(
            SentencePieceSplitter::build(&vocab(&["▁", &format!("{before}▁x")])).is_none(),
            "before half is too wide for the fixed compare"
        );
        let after = "z".repeat(HALF_WIDTH + 1);
        assert!(
            SentencePieceSplitter::build(&vocab(&["▁", &format!("a▁{after}")])).is_none(),
            "after half is too wide for the fixed compare"
        );
    }
}
