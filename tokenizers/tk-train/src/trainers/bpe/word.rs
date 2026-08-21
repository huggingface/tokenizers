use std::{iter, mem};
use tk_encode::models::bpe::Pair;

/// Provides access to the `FirstLastIterator` to any Iterator
pub trait WithFirstLastIterator: Iterator + Sized {
    fn with_first_and_last(self) -> FirstLastIterator<Self>;
}

impl<I> WithFirstLastIterator for I
where
    I: Iterator,
{
    fn with_first_and_last(self) -> FirstLastIterator<Self> {
        FirstLastIterator {
            first: true,
            iter: self.peekable(),
        }
    }
}

/// Provides information about whether an item is the first and/or the last of the iterator
pub struct FirstLastIterator<I>
where
    I: Iterator,
{
    first: bool,
    iter: iter::Peekable<I>,
}

impl<I> Iterator for FirstLastIterator<I>
where
    I: Iterator,
{
    /// (is_first, is_last, item)
    type Item = (bool, bool, I::Item);

    fn next(&mut self) -> Option<Self::Item> {
        let first = mem::replace(&mut self.first, false);
        self.iter
            .next()
            .map(|e| (first, self.iter.peek().is_none(), e))
    }
}

/// 8 bytes, and every one of them is read.
///
/// It used to carry `prev`/`next` as well, an intrusive doubly-linked list that only the
/// dropout-aware `merge_all` ever walked. With that gone the two fields were written on every `add`
/// and every `merge` and read by nobody, at 4x the width: the merge loop sweeps these symbols once
/// per affected word per merge, so their size is the loop's memory traffic.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Symbol {
    c: u32,
    /// How many of the original characters this symbol stands for, for `max_token_length`.
    len: u32,
}

#[derive(Clone, Default)]
pub struct Word {
    symbols: Vec<Symbol>,
}

impl std::fmt::Debug for Word {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("Word")
            .field(
                "chars",
                &self
                    .symbols
                    .iter()
                    .map(|s| s.c.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
            )
            .field("symbols", &self.symbols)
            .finish()
    }
}

impl Word {
    // `new` and `get_chars` are used by the `parity-aware-bpe` trainer and by the tests below, so a
    // default build sees no caller for them.
    #[allow(dead_code)]
    pub fn new() -> Self {
        Word { symbols: vec![] }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            symbols: Vec::with_capacity(capacity),
        }
    }

    pub fn add(&mut self, c: u32, byte_len: usize) {
        self.symbols.push(Symbol {
            c,
            len: byte_len as u32,
        });
    }

    // this is a training only function, should potentially be feature gated.
    ///
    /// Rewrites the symbols in one forward pass, reading at `i` and writing at `write`.
    ///
    /// It used to `insert` the merged symbol then `remove` the two it replaced -- three `Vec`
    /// memmoves of the whole tail per occurrence, to turn two symbols into one. Since `write` never
    /// runs ahead of `i`, everything at or after `i` is still untouched when it is read, so the
    /// compaction needs no shifting at all.
    ///
    /// The reported changes are identical to the shifting version's: the left neighbour is the last
    /// symbol *written* (which may itself be a symbol merged earlier in this same pass, exactly as
    /// `symbols[i - 1]` was after the removals), and the right neighbour is the first symbol past
    /// the pair.
    ///
    /// Changes are *appended* to `changes` rather than returned in a fresh `Vec`: the caller visits
    /// hundreds of thousands of words per training run and reuses one buffer for all of them.
    pub fn merge(
        &mut self,
        c1: u32,
        c2: u32,
        replacement: u32,
        max_length: usize,
        changes: &mut Vec<(Pair, i32)>,
    ) {
        let mut write = 0;
        let mut i = 0;
        while i < self.symbols.len() {
            if self.symbols[i].c == c1
                && i + 1 < self.symbols.len()
                && self.symbols[i + 1].c == c2
            {
                let merged_len = self.symbols[i].len + self.symbols[i + 1].len;

                // If there are other characters before the pair
                if write > 0 {
                    let left = self.symbols[write - 1];
                    changes.push(((left.c, c1), -1));
                    if ((left.len + merged_len) as usize) < max_length {
                        changes.push(((left.c, replacement), 1));
                    }
                }

                // If there are other characters after the pair
                if i + 2 < self.symbols.len() {
                    let right = self.symbols[i + 2];
                    changes.push(((c2, right.c), -1));
                    if ((right.len + merged_len) as usize) < max_length {
                        changes.push(((replacement, right.c), 1));
                    }
                }

                self.symbols[write] = Symbol {
                    c: replacement,
                    len: merged_len,
                };
                write += 1;
                // Both consumed. The merged symbol is not offered to a second merge in this pass,
                // which is what advancing past it did before.
                i += 2;
            } else {
                self.symbols[write] = self.symbols[i];
                write += 1;
                i += 1;
            }
        }
        self.symbols.truncate(write);
    }

    #[allow(dead_code)]
    pub fn get_chars(&self) -> Vec<u32> {
        self.get_chars_iter().collect()
    }

    pub fn get_chars_iter(&self) -> impl ExactSizeIterator<Item = u32> + '_ {
        self.symbols.iter().map(|s| s.c)
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge() {
        // Let's say we have the word 'hello' and a word-to-id vocab that looks
        // like this: {'h': 0, 'e': 1, 'l': 2, 'o': 3}.
        let mut word = Word::new();
        word.add(0, 1); // 'h'
        word.add(1, 1); // 'e'
        word.add(2, 1); // 'l'
        word.add(2, 1); // 'l'
        word.add(3, 1); // 'o'

        // We're going to perform a merge on the pair ('l', 'l') ~= (2, 2). Let's
        // say that 'll' has the ID of 4 in the updated word-to-id vocab.
        let mut changes = Vec::new();
        word.merge(2, 2, 4, usize::MAX, &mut changes);

        // So the word should now look like this:
        assert_eq!(
            word.get_chars(),
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
        // Let's say we have the word 'hello' and a word-to-id vocab that looks
        // like this: {'h': 0, 'e': 1, 'l': 2, 'o': 3}.
        let mut word = Word::new();
        word.add(0, 1); // 'h'
        word.add(1, 1); // 'e'
        word.add(2, 1); // 'l'
        word.add(2, 1); // 'l'
        word.add(3, 1); // 'o'

        // We're going to perform a merge on the pair ('l', 'l') ~= (2, 2). Let's
        // say that 'll' has the ID of 4 in the updated word-to-id vocab.
        let mut changes = Vec::new();
        word.merge(2, 2, 4, 2, &mut changes);
        assert_eq!(
            word.get_chars(),
            &[
                0u32, // 'h'
                1u32, // 'e'
                4u32, // 'll'
                3u32, // 'o'
            ]
        );

        assert_eq!(
            changes,
            &[
                ((1u32, 2u32), -1i32), // count for ('e', 'l') should be decreased by 1.
                // ((1u32, 4u32), 1i32),  Missing since this would be larger than 2
                ((2u32, 3u32), -1i32), // count for ('l', 'o') should be decreased by 1.
                                       // ((4u32, 3u32), 1i32), Missing since this would be larger than 2
            ]
        );
    }
}
