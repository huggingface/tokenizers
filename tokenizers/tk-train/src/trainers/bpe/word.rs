//! The words a BPE trainer merges into, all in one buffer.
//!
//! Every word used to own a `Vec<Symbol>`. For a 588 MB corpus that is one allocation per unique
//! word -- millions of them -- and, worse, millions of separate heap blocks that the merge loop then
//! chases a pointer into. The loop visits a word once per merge that touches it, so that pointer
//! chase is the access pattern of the whole training run.
//!
//! Here the symbols live in a single `Vec`, one contiguous run per word. A run is allocated once,
//! at its full width, and only ever shrinks: merging replaces two symbols with one, so `live[i]`
//! falls and `start[i]` never moves. That is what keeps the layout static enough to hand out
//! disjoint `&mut` slices, which is what the parallel merge path needs.
//!
//! Because the runs are laid out in word order, a contiguous range of word indices is a contiguous
//! range of symbols -- so `split_at_mut` carves the arena up for workers with no `unsafe`.

use tk_encode::models::bpe::Pair;

/// One symbol of a word: which vocabulary entry it is, and how many of the original characters it
/// stands for.
///
/// 8 bytes, and both halves are read. It used to carry `prev`/`next` as well, an intrusive linked
/// list that only the dropout-aware `merge_all` walked; with that gone the two fields were written
/// on every append and every merge and read by nobody, at 4x the width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Symbol {
    pub c: u32,
    pub len: u32,
}

/// Every word's symbols, in one buffer.
///
/// The fields are `pub(super)` so the merge loop can peel disjoint slices off them directly; the
/// split has to stay where the chunking logic already lives.
#[derive(Default)]
pub struct WordArena {
    /// All runs, back to back in word order.
    pub(super) symbols: Vec<Symbol>,
    /// `start[i]` is where word `i`'s run begins. `n + 1` entries: the last is `symbols.len()`, so
    /// `start[i]..start[i + 1]` is word `i`'s run at full width.
    pub(super) start: Vec<u32>,
    /// `live[i]` is how much of word `i`'s run is still in use. Only ever shrinks.
    pub(super) live: Vec<u32>,
}

impl WordArena {
    /// `words` runs totalling at most `symbols` entries, so neither `Vec` grows while filling.
    pub fn with_capacity(words: usize, symbols: usize) -> Self {
        let mut start = Vec::with_capacity(words + 1);
        start.push(0);
        Self {
            symbols: Vec::with_capacity(symbols),
            start,
            live: Vec::with_capacity(words),
        }
    }

    pub fn len(&self) -> usize {
        self.live.len()
    }

    /// Appends one symbol to the word currently being built.
    ///
    /// Call between [`Self::open_word`] and [`Self::close_word`].
    #[inline]
    pub fn push_symbol(&mut self, c: u32, len: u32) {
        self.symbols.push(Symbol { c, len });
    }

    /// Starts a new word. Its run begins where the previous one ended.
    #[inline]
    pub fn open_word(&mut self) {}

    /// Ends the word being built, recording its run.
    #[inline]
    pub fn close_word(&mut self) {
        let begin = *self.start.last().expect("start always holds a sentinel");
        let end = self.symbols.len() as u32;
        self.start.push(end);
        self.live.push(end - begin);
    }

    /// Word `i`'s live symbols.
    #[inline]
    pub fn symbols(&self, i: usize) -> &[Symbol] {
        let begin = self.start[i] as usize;
        &self.symbols[begin..begin + self.live[i] as usize]
    }

    /// Word `i`'s live symbol ids.
    #[inline]
    pub fn chars(&self, i: usize) -> impl ExactSizeIterator<Item = u32> + '_ {
        self.symbols(i).iter().map(|s| s.c)
    }

    /// Merges every occurrence of `(c1, c2)` in word `i`, appending the pair deltas to `changes`.
    #[inline]
    pub fn merge(
        &mut self,
        i: usize,
        c1: u32,
        c2: u32,
        replacement: u32,
        max_length: usize,
        changes: &mut Vec<(Pair, i32)>,
    ) {
        let begin = self.start[i] as usize;
        let live = &mut self.live[i];
        let run = &mut self.symbols[begin..begin + *live as usize];
        merge_run(run, live, c1, c2, replacement, max_length, changes);
    }
}

/// Merges every occurrence of `(c1, c2)` in one word's run, rewriting it in place.
///
/// One forward pass, reading at `read` and writing at `write`. It used to `insert` the merged symbol
/// and then `remove` the two it replaced -- three `Vec` memmoves of the whole tail per occurrence,
/// to turn two symbols into one. Because `write` never runs ahead of `read`, everything at or after
/// `read` is still untouched when it is read, so nothing needs shifting.
///
/// The deltas match what the shifting version reported: the left neighbour is the last symbol
/// *written* (possibly one merged earlier in this same pass, exactly as `symbols[i - 1]` was after
/// the removals), and the right neighbour is the first symbol past the pair. They are appended
/// rather than returned in a fresh `Vec`: a training run visits hundreds of thousands of words and
/// reuses one buffer for all of them.
#[inline]
pub fn merge_run(
    run: &mut [Symbol],
    live: &mut u32,
    c1: u32,
    c2: u32,
    replacement: u32,
    max_length: usize,
    changes: &mut Vec<(Pair, i32)>,
) {
    let n = run.len();
    let mut write = 0;
    let mut read = 0;
    while read < n {
        if run[read].c == c1 && read + 1 < n && run[read + 1].c == c2 {
            let merged_len = run[read].len + run[read + 1].len;

            // If there are other characters before the pair
            if write > 0 {
                let left = run[write - 1];
                changes.push(((left.c, c1), -1));
                if ((left.len + merged_len) as usize) < max_length {
                    changes.push(((left.c, replacement), 1));
                }
            }

            // If there are other characters after the pair
            if read + 2 < n {
                let right = run[read + 2];
                changes.push(((c2, right.c), -1));
                if ((right.len + merged_len) as usize) < max_length {
                    changes.push(((replacement, right.c), 1));
                }
            }

            run[write] = Symbol {
                c: replacement,
                len: merged_len,
            };
            write += 1;
            // Both consumed. The merged symbol is not offered to a second merge in this pass, which
            // is what advancing past it did before.
            read += 2;
        } else {
            run[write] = run[read];
            write += 1;
            read += 1;
        }
    }
    *live = write as u32;
}

/// Yields each item with whether it is the first and whether it is the last.
pub trait WithFirstLastIterator: Iterator + Sized {
    fn with_first_and_last(self) -> FirstLastIterator<Self>;
}

impl<I: Iterator> WithFirstLastIterator for I {
    fn with_first_and_last(self) -> FirstLastIterator<Self> {
        FirstLastIterator {
            first: true,
            iter: self.peekable(),
        }
    }
}

pub struct FirstLastIterator<I: Iterator> {
    first: bool,
    iter: std::iter::Peekable<I>,
}

impl<I: Iterator> Iterator for FirstLastIterator<I> {
    /// (is_first, is_last, item)
    type Item = (bool, bool, I::Item);

    fn next(&mut self) -> Option<Self::Item> {
        let first = self.first;
        self.first = false;
        self.iter
            .next()
            .map(|e| (first, self.iter.peek().is_none(), e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a one-word arena from symbol ids, each standing for one character.
    fn one_word(chars: &[u32]) -> WordArena {
        let mut arena = WordArena::with_capacity(1, chars.len());
        arena.open_word();
        for &c in chars {
            arena.push_symbol(c, 1);
        }
        arena.close_word();
        arena
    }

    #[test]
    fn test_merge() {
        // Let's say we have the word 'hello' and know the following merges:
        //   h = 0, e = 1, l = 2, o = 3, ll = 4
        let mut arena = one_word(&[0, 1, 2, 2, 3]);

        // We're going to perform a merge on the pair ('l', 'l') ~= (2, 2). Let's
        // say that 'll' has the ID of 4 in the updated word-to-id vocab.
        let mut changes = Vec::new();
        arena.merge(0, 2, 2, 4, usize::MAX, &mut changes);

        // So the word should now look like this:
        assert_eq!(
            arena.chars(0).collect::<Vec<_>>(),
            &[
                0u32, // 'h'
                1u32, // 'e'
                4u32, // 'll'
                3u32, // 'o'
            ]
        );

        // The return value `changes` will be used to update the pair counts during
        // training. This merge affects the counts for the pairs
        // ('e', 'l') ~= (1, 2),
        // ('e', 'll') ~= (1, 4),
        // ('l', 'o') ~= (2, 3), and
        // ('ll', 'o') ~= (4, 3).
        // So the changes should reflect that:
        assert_eq!(
            changes,
            &[
                ((1u32, 2u32), -1i32), // count for ('e', 'l') should be decreased by 1.
                ((1u32, 4u32), 1i32),  // count for ('e', 'll') should be increased by 1.
                ((2u32, 3u32), -1i32), // count for ('l', 'o') should be decreased by 1.
                ((4u32, 3u32), 1i32),  // count for ('ll', 'o') should be increased by 1.
            ]
        );
    }

    #[test]
    fn test_merge_max_length() {
        // Let's say we have the word 'hello' and know the following merges:
        //   h = 0, e = 1, l = 2, o = 3, ll = 4
        let mut arena = one_word(&[0, 1, 2, 2, 3]);

        // We're going to perform a merge on the pair ('l', 'l') ~= (2, 2). Let's
        // say that 'll' has the ID of 4 in the updated word-to-id vocab.
        let mut changes = Vec::new();
        arena.merge(0, 2, 2, 4, 2, &mut changes);

        assert_eq!(
            arena.chars(0).collect::<Vec<_>>(),
            &[
                0u32, // 'h'
                1u32, // 'e'
                4u32, // 'll'
                3u32, // 'o'
            ]
        );

        // `max_length` of 2 blocks the pairs that would grow past it, so only the
        // decrements survive.
        assert_eq!(
            changes,
            &[
                ((1u32, 2u32), -1i32), // count for ('e', 'l') should be decreased by 1.
                ((2u32, 3u32), -1i32), // count for ('l', 'o') should be decreased by 1.
            ]
        );
    }

    /// A run only ever shrinks, so `start` stays put and the arena keeps its layout.
    #[test]
    fn runs_shrink_in_place() {
        let mut arena = WordArena::with_capacity(2, 8);
        arena.open_word();
        for &c in &[2u32, 2, 2, 2] {
            arena.push_symbol(c, 1);
        }
        arena.close_word();
        arena.open_word();
        for &c in &[7u32, 8] {
            arena.push_symbol(c, 1);
        }
        arena.close_word();

        let second_start = arena.start[1];
        let mut changes = Vec::new();
        arena.merge(0, 2, 2, 9, usize::MAX, &mut changes);

        // "2222" -> "99", one pass, both occurrences.
        assert_eq!(arena.chars(0).collect::<Vec<_>>(), &[9u32, 9u32]);
        // The neighbour is untouched and did not move.
        assert_eq!(arena.start[1], second_start);
        assert_eq!(arena.chars(1).collect::<Vec<_>>(), &[7u32, 8u32]);
    }
}
