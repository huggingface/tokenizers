use super::Pair;
use rand::{thread_rng, Rng};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

#[derive(Debug, Eq)]
struct Merge {
    pos: usize,
    rank: u32,
    new_id: u32,
}

impl PartialEq for Merge {
    fn eq(&self, other: &Self) -> bool {
        self.rank == other.rank && self.pos == other.pos
    }
}

impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // By manually implementing this, we make the containing BinaryHeap a
        // min-heap ordered first on the rank, and the pos otherwise
        if self.rank != other.rank {
            Some(other.rank.cmp(&self.rank))
        } else {
            Some(other.pos.cmp(&self.pos))
        }
    }
}

impl Ord for Merge {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Debug, Clone, Copy)]
struct Symbol {
    c: u32,
    prev: isize,
    next: isize,
    len: usize,
}
impl Symbol {
    /// Merges the current Symbol with the other one.
    /// In order to update prev/next, we consider Self to be the Symbol on the left,
    /// and other to be the next one on the right.
    pub fn merge_with(&mut self, other: &Self, new_c: u32) {
        self.c = new_c;
        self.len += other.len;
        self.next = other.next;
    }
}

#[derive(Clone, Default)]
pub(super) struct Word {
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
    pub(super) fn new() -> Self {
        Word { symbols: vec![] }
    }

    pub(super) fn with_capacity(capacity: usize) -> Self {
        Word {
            symbols: Vec::with_capacity(capacity),
        }
    }

    pub(super) fn add(&mut self, c: u32, byte_len: usize) {
        let (prev, next) = {
            let len = self.symbols.len() as isize;
            if let Some(last) = self.symbols.last_mut() {
                // Update `next` on the previous one
                last.next = len;
                (len - 1, -1)
            } else {
                (-1, -1)
            }
        };
        self.symbols.push(Symbol {
            c,
            prev,
            next,
            len: byte_len,
        });
    }

    pub(super) fn merge(&mut self, c1: u32, c2: u32, replacement: u32) -> Vec<(Pair, i32)> {
        let mut changes: Vec<(Pair, i32)> = vec![];
        let mut i = 0;
        loop {
            if i >= self.symbols.len() {
                break;
            }

            // Found a pair
            if self.symbols[i].c == c1 && i + 1 < self.symbols.len() && self.symbols[i + 1].c == c2
            {
                let first = self.symbols[i];
                let second = self.symbols[i + 1];

                // If there are other characters before the pair
                if i > 0 {
                    changes.push(((self.symbols[i - 1].c, first.c), -1));
                    changes.push(((self.symbols[i - 1].c, replacement), 1));
                }

                // Remove in place
                let new_s = Symbol {
                    c: replacement,
                    prev: first.prev,
                    next: second.next,
                    len: first.len + second.len,
                };
                self.symbols.insert(i, new_s); // Insert replacement before first char of pair
                self.symbols.remove(i + 1); // Remove first char of pair
                self.symbols.remove(i + 1); // And then the second

                // If there are other characters after the pair
                if i < self.symbols.len() - 1 {
                    changes.push(((second.c, self.symbols[i + 1].c), -1));
                    changes.push(((replacement, self.symbols[i + 1].c), 1));
                }
            }

            i += 1;
        }

        changes
    }

    pub(super) fn merge_all(&mut self, merges: &HashMap<Pair, (u32, u32)>, dropout: Option<f32>) {
        let mut queue = BinaryHeap::with_capacity(self.symbols.len());
        let mut skip = Vec::with_capacity(queue.len());

        queue.extend(
            self.symbols
                .windows(2)
                .enumerate()
                .filter_map(|(index, window)| {
                    let pair = (window[0].c, window[1].c);
                    merges.get(&pair).map(|m| Merge {
                        pos: index,
                        rank: m.0,
                        new_id: m.1,
                    })
                }),
        );

        while let Some(top) = queue.pop() {
            if dropout
                .map(|d| thread_rng().gen::<f32>() < d)
                .unwrap_or(false)
            {
                skip.push(top);
            } else {
                // Re-insert the skipped elements
                queue.extend(skip.drain(..));

                if self.symbols[top.pos].len == 0 {
                    continue;
                }
                // Do nothing if we are the last symbol
                if self.symbols[top.pos].next == -1 {
                    continue;
                }

                let next_pos = self.symbols[top.pos].next as usize;
                let right = self.symbols[next_pos];

                // Make sure we are not processing an expired queue entry
                let target_new_pair = (self.symbols[top.pos].c, right.c);
                if !merges
                    .get(&target_new_pair)
                    .map_or(false, |(_, new_id)| *new_id == top.new_id)
                {
                    continue;
                }

                // Otherwise, let's merge
                self.symbols[top.pos].merge_with(&right, top.new_id);
                // Tag the right part as removed
                self.symbols[next_pos].len = 0;

                // Update `prev` on the new `next` to the current pos
                if right.next > -1 && (right.next as usize) < self.symbols.len() {
                    self.symbols[right.next as usize].prev = top.pos as isize;
                }

                // Insert the new pair formed with the previous symbol
                let current = &self.symbols[top.pos];
                if current.prev >= 0 {
                    let prev = current.prev as usize;
                    let prev_symbol = self.symbols[prev];
                    let new_pair = (prev_symbol.c, current.c);
                    if let Some((rank, new_id)) = merges.get(&new_pair) {
                        queue.push(Merge {
                            pos: current.prev as usize,
                            rank: *rank,
                            new_id: *new_id,
                        });
                    }
                }

                // Insert the new pair formed with the next symbol
                let next = current.next as usize;
                if next < self.symbols.len() {
                    let next_symbol = self.symbols[next];
                    let new_pair = (current.c, next_symbol.c);
                    if let Some((rank, new_id)) = merges.get(&new_pair) {
                        queue.push(Merge {
                            pos: top.pos,
                            rank: *rank,
                            new_id: *new_id,
                        });
                    }
                }
            }
        }

        // Filter out the removed symbols
        self.symbols.retain(|s| s.len != 0);
    }

    pub(super) fn get_chars(&self) -> Vec<u32> {
        self.symbols.iter().map(|s| s.c).collect()
    }

    pub(super) fn get_chars_iter<'a>(&'a self) -> impl Iterator<Item = u32> + 'a {
        self.symbols.iter().map(|s| s.c)
    }

    pub(super) fn get_offsets_iter<'a>(&'a self) -> impl Iterator<Item = (usize, usize)> + 'a {
        let mut pos = 0;
        self.symbols.iter().map(move |symbol| {
            let new_pos = pos + symbol.len;
            let offset = (pos, new_pos);
            pos = new_pos;
            offset
        })
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
        let changes = word.merge(2, 2, 4);

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
}
