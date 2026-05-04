use super::Ngram;
use ahash::AHashMap;
use dary_heap::QuaternaryHeap;
use rand::{rng, Rng};
use std::cmp::Ordering;
use std::usize;

#[derive(Debug, Eq)]
struct Merge {
    pos: usize,
    rank: u32,
    new_id: u32,
    length: u32,
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
        Some(self.cmp(other))
    }
}

impl Ord for Merge {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.rank != other.rank {
            other.rank.cmp(&self.rank)
        } else {
            other.pos.cmp(&self.pos)
        }
    }
}

// Implement merge symbol with vec of symbols
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

    /*
    /// Merges the current Symbol with the vector of other symbols.
    /// Self is the left most symbol, all other symbols should be in order
    /// adjacent to self from left to right
    pub fn merge_with_vec(&mut self, other: Vec<&Self>, new_c: u32) {
        self.c = new_c;
        self.next = other[other.len()-1].next; // last mut
        for sym in other.iter() {
            self.len += sym.len;
        }
    }
    */
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
        Self {
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

    /// Original Merge Function
    pub(super) fn _merge_old(
        &mut self,
        c: Vec<u32>,
        replacement: u32,
        max_length: usize,
        max_ngram_length: usize,
    ) -> Vec<(Ngram, i32)> {
        //let mut changes: Vec<(Ngram, i32)> = vec![];
        let mut changes_ngrams: Vec<Ngram> = vec![];
        let mut changes_vals: Vec<i32> = vec![];
        let mut i = 0;
        loop {
            if i >= self.symbols.len() {
                break;
            }

            // Check for matching ngram
            let mut matching = (i + c.len() - 1) < self.symbols.len(); // Ngram fits in word starting at i
            let mut length: usize = 0;

            if matching {
                for j in 0..c.len() {
                    matching &= c[j] == self.symbols[i + j].c;
                    length += self.symbols[i + j].len;
                }
            }

            // Found matching Ngram
            if matching {
                // Remove in place
                let new_s = Symbol {
                    c: replacement,
                    prev: self.symbols[i].prev,
                    next: self.symbols[i + c.len() - 1].next,
                    len: length,
                };

                // Added all necessary changes for ngram manipulation. To be tested
                // Remove all Ngrams containing a part of the symbol in the word
                // Possibly optimize to reuse vecs for ngrams
                for end_index in i..self.symbols.len() {
                    let end = if i + c.len() < end_index {
                        i + c.len()
                    } else {
                        end_index
                    };
                    for start_index in 0..end {
                        let ngram_length = self.symbols[start_index..end_index + 1]
                            .iter()
                            .fold(0, |acc, sym| acc + sym.len);
                        if ngram_length <= max_length
                            && end_index + 1 - start_index <= max_ngram_length
                        {
                            if start_index != i || end_index - start_index + 1 != c.len() {
                                // reuse ngrams
                                let ngram = Ngram {
                                    ids: self.symbols[start_index..end_index + 1]
                                        .iter()
                                        .map(|elem| elem.c)
                                        .collect(),
                                };
                                let pos = changes_ngrams.iter().position(|n| *n == ngram);
                                if pos.is_some() {
                                    changes_vals[pos.unwrap()] += -1
                                } else {
                                    changes_ngrams.push(ngram);
                                    changes_vals.push(-1);
                                }
                                /*changes.push(
                                    (Ngram {
                                        ids:self.symbols[start_index..end_index+1].iter().map(|elem| elem.c).collect()
                                    }, -1)
                                );*/
                            }
                        }
                    }
                }

                // Changed to remove whole Ngram
                self.symbols.insert(i, new_s); // Insert replacement before first char of Ngram
                for _ in 0..c.len() {
                    self.symbols.remove(i + 1); // Remove all symbols of the new token from word
                }

                // Add back all Ngrams containing a the whole symbol as a token
                for end_index in i..self.symbols.len() {
                    let end = if i + 1 < end_index { i + 1 } else { end_index };
                    for start_index in 0..end {
                        let ngram_length = self.symbols[start_index..end_index + 1]
                            .iter()
                            .fold(0, |acc, sym| acc + sym.len);
                        if ngram_length <= max_length
                            && end_index + 1 - start_index <= max_ngram_length
                        {
                            // reuse ngrams
                            let ngram = Ngram {
                                ids: self.symbols[start_index..end_index + 1]
                                    .iter()
                                    .map(|elem| elem.c)
                                    .collect(),
                            };
                            let pos = changes_ngrams.iter().position(|n| *n == ngram);
                            if pos.is_some() {
                                changes_vals[pos.unwrap()] += 1
                            } else {
                                changes_ngrams.push(ngram);
                                changes_vals.push(1);
                            }
                            /*changes.push(
                                (Ngram {
                                    ids:self.symbols[start_index..end_index+1].iter().map(|elem| elem.c).collect()
                                }, 1)
                            );*/
                        }
                    }
                }
            }

            i += 1;
        }
        changes_ngrams
            .into_iter()
            .zip(changes_vals.into_iter())
            .filter(|(_, val)| *val != 0)
            .collect()
        //changes
    }

    /// Merges an Ngram in a word.
    /// maybe make c input to Ngram type
    /// subsequent merges should work, but check
    pub(super) fn merge(
        &mut self,
        c: Vec<u32>,
        replacement: u32,
        max_length: usize,
        max_ngram_length: usize,
    ) -> Vec<(Ngram, i32)> {
        //let mut changes: Vec<(Ngram, i32)> = vec![];
        let mut changes_ngrams: Vec<Ngram> = vec![];
        let mut changes_vals: Vec<i32> = vec![];
        let mut i = 0;
        let merge = Ngram { ids: c.clone() };

        // Count all N-grams currently in word
        let max_ngram_len_tmp = if max_ngram_length < self.symbols.len() {
            max_ngram_length
        } else {
            self.symbols.len()
        };
        for ngram_len in 2..max_ngram_len_tmp + 1 {
            let mut last_ngram = Ngram { ids: vec![] };
            let mut same_ngrams = 0;
            for window in self.symbols.windows(ngram_len) {
                // TODO: continue if exceeding max ngram length, expose function in word
                // Check if there are any characters with len > 1 at this point..

                // Skip if the ngram results in a token too large
                let ngram_length = window.iter().fold(0, |acc, sym| acc + sym.len);
                if ngram_length > max_length {
                    continue;
                }

                // Create new ngram
                let ngram = Ngram {
                    ids: window.iter().map(|elem| elem.c).collect(),
                };

                // Skip if it is to be merged ngram
                if ngram == merge {
                    continue;
                }

                // Skip Ngram if already counted in an overlapping Ngram that is exactely the same
                if ngram == last_ngram && same_ngrams + 1 < ngram_len {
                    same_ngrams += 1;
                    continue;
                }

                //  Update Skipping counter and current ngram
                last_ngram = ngram.clone();
                same_ngrams = 0;

                // Update Ngram Counts
                let pos = changes_ngrams.iter().position(|n| *n == ngram);
                if pos.is_some() {
                    changes_vals[pos.unwrap()] += -1
                } else {
                    changes_ngrams.push(ngram);
                    changes_vals.push(-1);
                }
            }
        }

        // Merge all possible merges in word
        loop {
            if i >= self.symbols.len() {
                break;
            }

            // Check for matching ngram
            let mut matching = (i + c.len() - 1) < self.symbols.len(); // Ngram fits in word starting at i
            let mut length: usize = 0;

            if matching {
                for j in 0..c.len() {
                    matching &= c[j] == self.symbols[i + j].c;
                    length += self.symbols[i + j].len;
                }
            }

            // Found matching Ngram
            if matching {
                // Remove in place
                let new_s = Symbol {
                    c: replacement,
                    prev: self.symbols[i].prev,
                    next: self.symbols[i + c.len() - 1].next,
                    len: length,
                };

                // Changed to remove whole Ngram
                self.symbols.insert(i, new_s); // Insert replacement before first char of Ngram
                for _ in 0..c.len() {
                    self.symbols.remove(i + 1); // Remove all symbols of the new token from word
                }
            }
            i += 1;
        }

        // Count all N-grams newly generated after merges
        let max_ngram_len_tmp = if max_ngram_length < self.symbols.len() {
            max_ngram_length
        } else {
            self.symbols.len()
        };
        for ngram_len in 2..max_ngram_len_tmp + 1 {
            let mut last_ngram = Ngram { ids: vec![] };
            let mut same_ngrams = 0;
            for window in self.symbols.windows(ngram_len) {
                // TODO: continue if exceeding max ngram length, expose function in word
                // Check if there are any characters with len > 1 at this point..

                // Skip if the ngram results in a token too large
                let ngram_length = window.iter().fold(0, |acc, sym| acc + sym.len);
                if ngram_length > max_length {
                    continue;
                }

                // Create new ngram
                let ngram = Ngram {
                    ids: window.iter().map(|elem| elem.c).collect(),
                };

                // Skip Ngram if already counted in an overlapping Ngram that is exactely the same
                if ngram == last_ngram && same_ngrams + 1 < ngram_len {
                    same_ngrams += 1;
                    continue;
                }

                //  Update Skipping counter and current ngram
                last_ngram = ngram.clone();
                same_ngrams = 0;

                // Update Ngram Counts
                let pos = changes_ngrams.iter().position(|n| *n == ngram);
                if pos.is_some() {
                    changes_vals[pos.unwrap()] += 1
                } else {
                    changes_ngrams.push(ngram);
                    changes_vals.push(1);
                }
            }
        }
        changes_ngrams
            .into_iter()
            .zip(changes_vals.into_iter())
            .filter(|(_, val)| *val != 0)
            .collect()
        //changes
    }

    /// Merges all merges in the merge hashmap in a single word
    pub(super) fn merge_all(&mut self, merges: &AHashMap<Ngram, (u32, u32)>, dropout: Option<f32>) {
        //println!("begin merge_all");
        if self.symbols.len() < 2 {
            return;
        }

        let mut queue =
            QuaternaryHeap::with_capacity(self.symbols.len() * (self.symbols.len() - 1) / 2);
        let mut skip = Vec::with_capacity(queue.len());

        // extend queue with all ngram sizes
        for i in 2..self.symbols.len() + 1 {
            queue.extend(
                self.symbols
                    .windows(i)
                    .enumerate()
                    .filter_map(|(index, window)| {
                        let ngram = Ngram {
                            ids: window.iter().map(|elem| elem.c).collect(),
                        };
                        merges.get(&ngram).map(|m| Merge {
                            pos: index,
                            rank: m.0,
                            new_id: m.1,
                            length: i as u32,
                        })
                    }),
            );
        }

        'queue: while let Some(top) = queue.pop() {
            if top.length < 2 {
                continue;
            }

            if dropout.map(|d| rng().random::<f32>() < d).unwrap_or(false) {
                skip.push(top);
            } else {
                // Re-insert the skipped elements
                queue.extend(skip.drain(..));

                // Do nothing if current symbol has already been removed
                if self.symbols[top.pos].len == 0 {
                    continue;
                }
                // Do nothing if we are the last symbol
                if self.symbols[top.pos].next == -1 {
                    continue;
                }

                /*
                println!("top.pos: {}", top.pos);
                println!("top.new_id: {}", top.new_id);
                println!("top.length: {}", top.length);
                println!("{:?}", self);
                println!("new_ids with len: {}", top.length);
                print!("[");*/

                let mut new_ids: Vec<u32> = Vec::with_capacity(top.length as usize);
                let mut curr = self.symbols[top.pos];
                for _ in 0..top.length - 1 {
                    // Do nothing if we are the last symbol
                    if curr.next == -1 {
                        continue 'queue;
                    }
                    /*if curr.c != top.ngram[_]:
                    continue 'queue;*/
                    new_ids.push(curr.c);
                    //print!("{}, ", curr.c);
                    curr = self.symbols[curr.next as usize];
                }
                new_ids.push(curr.c);
                //println!("{}]", curr.c);
                /*
                for id in &new_ids {
                    print!("{}", id);
                }
                println!();*/

                // Make sure we are not processing an expired queue entry
                // TODO: Necessary??
                let target_new_ngram = Ngram { ids: new_ids };
                if merges
                    .get(&target_new_ngram)
                    .is_none_or(|(_, new_id)| *new_id != top.new_id)
                {
                    continue;
                }

                // Otherwise, merge full ngram
                let mut curr_pos = top.pos;
                for _ in 0..top.length - 1 {
                    // go to the next symbol
                    curr_pos = self.symbols[curr_pos].next as usize;
                    let next_symbol = self.symbols[curr_pos];
                    // Merge next symbol on to first symbol
                    self.symbols[top.pos].merge_with(&next_symbol, top.new_id);
                    // Tag the next symbol as removed
                    self.symbols[curr_pos].len = 0;
                }

                // Update `prev` on the new `next` to the current pos
                // access could be done using top.pos -> next, as after merges, it points to required next symbol
                if self.symbols[curr_pos].next > -1
                    && (self.symbols[curr_pos].next as usize) < self.symbols.len()
                {
                    let next_symbol = self.symbols[curr_pos];
                    self.symbols[next_symbol.next as usize].prev = top.pos as isize;
                }

                let mut first_symbol_index = 0;
                while first_symbol_index < self.symbols.len()
                    && self.symbols[first_symbol_index].len == 0
                {
                    first_symbol_index += 1
                }
                // Insert Ngrams formed from new symbol
                while first_symbol_index <= top.pos {
                    let mut last_symbol_index = top.pos;
                    loop {
                        // construct a vector of all the ids up to top.pos excluding it
                        let mut ids: Vec<u32> = Vec::new();
                        let mut id_to_insert = first_symbol_index;
                        while id_to_insert <= last_symbol_index {
                            ids.push(self.symbols[id_to_insert].c);
                            id_to_insert = self.symbols[id_to_insert].next as usize;
                        }

                        let length = ids.len();
                        let new_ngram = Ngram { ids: ids };

                        if length >= 2 {
                            if let Some((rank, new_id)) = merges.get(&new_ngram) {
                                queue.push(Merge {
                                    pos: first_symbol_index,
                                    rank: *rank,
                                    new_id: *new_id,
                                    length: length as u32,
                                });
                            }
                        }

                        // break the loop if the last element is reached
                        if self.symbols[last_symbol_index].next == -1 {
                            break;
                        }
                        last_symbol_index = self.symbols[last_symbol_index].next as usize;
                    }
                    first_symbol_index = self.symbols[first_symbol_index].next as usize;
                }
            }
        }

        // Filter out the removed symbols
        self.symbols.retain(|s| s.len != 0);
    }

    pub(super) fn get_chars(&self) -> Vec<u32> {
        self.symbols.iter().map(|s| s.c).collect()
    }

    /*
    pub(super) fn get_char_len(&self) -> Vec<usize> {
        self.symbols.iter().map(|s| s.len).collect()
    }
    */

    pub(super) fn get_chars_iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.symbols.iter().map(|s| s.c)
    }

    pub(super) fn get_offsets_iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
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
        let mut changes = word.merge(vec![2, 2], 4, usize::MAX, usize::MAX);

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
        let mut changes_control = [
            (
                Ngram {
                    ids: vec![0u32, 1u32, 2u32],
                },
                -1i32,
            ), // count for ('h', 'e', 'l') should be decreased by 1.
            (
                Ngram {
                    ids: vec![1u32, 2u32],
                },
                -1i32,
            ), // count for ('e', 'l') should be decreased by 1.
            (
                Ngram {
                    ids: vec![0u32, 1u32, 2u32, 2u32],
                },
                -1i32,
            ), // count for ('h', 'e', 'l', 'l') should be decreased by 1.
            (
                Ngram {
                    ids: vec![1u32, 2u32, 2u32],
                },
                -1i32,
            ), // count for ('e', 'l', 'l') should be decreased by 1.
            (
                Ngram {
                    ids: vec![0u32, 1u32, 2u32, 2u32, 3u32],
                },
                -1i32,
            ), // count for ('h', 'e', 'l', 'l', 'o') should be decreased by 1.
            (
                Ngram {
                    ids: vec![1u32, 2u32, 2u32, 3u32],
                },
                -1i32,
            ), // count for ('e', 'l', 'l', 'o') should be decreased by 1.
            (
                Ngram {
                    ids: vec![2u32, 2u32, 3u32],
                },
                -1i32,
            ), // count for ('l', 'l', 'o') should be decreased by 1.
            (
                Ngram {
                    ids: vec![2u32, 3u32],
                },
                -1i32,
            ), // count for ('l', 'o') should be decreased by 1.
            (
                Ngram {
                    ids: vec![0u32, 1u32, 4u32],
                },
                1i32,
            ), // count for ('h', 'e', 'll') should be increased by 1.
            (
                Ngram {
                    ids: vec![1u32, 4u32],
                },
                1i32,
            ), // count for ('e', 'll') should be increased by 1.
            (
                Ngram {
                    ids: vec![0u32, 1u32, 4u32, 3u32],
                },
                1i32,
            ), // count for ('h', 'e', 'll', 'o') should be increased by 1.
            (
                Ngram {
                    ids: vec![1u32, 4u32, 3u32],
                },
                1i32,
            ), // count for ('e', 'll','o') should be increased by 1.
            (
                Ngram {
                    ids: vec![4u32, 3u32],
                },
                1i32,
            ), // count for ('ll','o') should be increased by 1.
        ];
        changes.sort();
        changes_control.sort();

        assert_eq!(changes, changes_control);
    }

    #[test]
    fn test_merge_2() {
        // Let's say we have the word 'hello' and a word-to-id vocab that looks
        // like this: {'h': 0, 'e': 1, 'l': 2, 'o': 3, '!': 4}.
        let mut word = Word::new();
        word.add(0, 1); // 'h'
        word.add(1, 1); // 'e'
        word.add(2, 1); // 'l'
        word.add(2, 1); // 'l'
        word.add(3, 1); // 'o'
        word.add(4, 1); // '!'

        // We're going to perform a merge on the ngram ('e', 'l', 'l') ~= [1, 2, 2)] Let's
        // say that 'ell' has the ID of 5 in the updated word-to-id vocab.
        let mut changes = word.merge(vec![1, 2, 2], 5, usize::MAX, usize::MAX);

        // So the word should now look like this:
        assert_eq!(
            word.get_chars(),
            &[
                0u32, // 'h'
                5u32, // 'ell'
                3u32, // 'o'
                4u32  // '!'
            ]
        );

        // The return value `changes` will be used to update the pair counts during
        let mut changes_control = [
            (
                Ngram {
                    ids: vec![0u32, 1u32],
                },
                -1i32,
            ), // count for ('h', 'e') should be decreased by 1.
            (
                Ngram {
                    ids: vec![0u32, 1u32, 2u32],
                },
                -1i32,
            ), // count for ('h', 'e', 'l') should be decreased by 1.
            (
                Ngram {
                    ids: vec![1u32, 2u32],
                },
                -1i32,
            ), // count for ('e', 'l') should be decreased by 1.
            (
                Ngram {
                    ids: vec![0u32, 1u32, 2u32, 2u32],
                },
                -1i32,
            ), // count for ('h', 'e', 'l', 'l') should be decreased by 1.
            (
                Ngram {
                    ids: vec![2u32, 2u32],
                },
                -1i32,
            ), // count for ('l', 'l') should be decreased by 1.
            (
                Ngram {
                    ids: vec![0u32, 1u32, 2u32, 2u32, 3u32],
                },
                -1i32,
            ), // count for ('h', 'e', 'l', 'l', 'o') should be decreased by 1.
            (
                Ngram {
                    ids: vec![1u32, 2u32, 2u32, 3u32],
                },
                -1i32,
            ), // count for ('e', 'l', 'l', 'o') should be decreased by 1.
            (
                Ngram {
                    ids: vec![2u32, 2u32, 3u32],
                },
                -1i32,
            ), // count for ('l', 'l', 'o') should be decreased by 1.
            (
                Ngram {
                    ids: vec![2u32, 3u32],
                },
                -1i32,
            ), // count for ('l', 'o') should be decreased by 1.
            (
                Ngram {
                    ids: vec![0u32, 1u32, 2u32, 2u32, 3u32, 4u32],
                },
                -1i32,
            ), // count for ('h', 'e', 'l', 'l', 'o', '!') should be decreased by 1.
            (
                Ngram {
                    ids: vec![1u32, 2u32, 2u32, 3u32, 4u32],
                },
                -1i32,
            ), // count for ('e', 'l', 'l', 'o', '!') should be decreased by 1.
            (
                Ngram {
                    ids: vec![2u32, 2u32, 3u32, 4u32],
                },
                -1i32,
            ), // count for ('l', 'l', 'o', '!') should be decreased by 1.
            (
                Ngram {
                    ids: vec![2u32, 3u32, 4u32],
                },
                -1i32,
            ), // count for ('l', 'o', '!') should be decreased by 1.
            (
                Ngram {
                    ids: vec![0u32, 5u32],
                },
                1i32,
            ), // count for ('h', 'ell') should be increased by 1.
            (
                Ngram {
                    ids: vec![0u32, 5u32, 3u32],
                },
                1i32,
            ), // count for ('h', 'ell', 'o') should be increased by 1.
            (
                Ngram {
                    ids: vec![5u32, 3u32],
                },
                1i32,
            ), // count for ('ell','o') should be increased by 1.
            (
                Ngram {
                    ids: vec![0u32, 5u32, 3u32, 4u32],
                },
                1i32,
            ), // count for ('h', 'ell', 'o', '!') should be increased by 1.
            (
                Ngram {
                    ids: vec![5u32, 3u32, 4u32],
                },
                1i32,
            ), // count for ('ell','o', '!') should be increased by 1.
        ];
        changes.sort();
        changes_control.sort();

        assert_eq!(changes, changes_control);
    }

    #[test]
    fn test_merge_multiple() {
        // Let's say we have the word 'hello' and a word-to-id vocab that looks
        // like this: {'h': 0, 'e': 1, 'l': 2, 'o': 3, '!': 4}.
        let mut word = Word::new();
        word.add(0, 1); // 'h'
        word.add(2, 1); // 'l'
        word.add(2, 1); // 'l'
        word.add(2, 1); // 'l'
        word.add(2, 1); // 'l'

        // We're going to perform a merge on the ngram ('e', 'l', 'l') ~= [1, 2, 2)] Let's
        // say that 'ell' has the ID of 5 in the updated word-to-id vocab.
        let mut changes = word.merge(vec![2, 2], 5, usize::MAX, usize::MAX);

        // So the word should now look like this:
        assert_eq!(
            word.get_chars(),
            &[
                0u32, // 'h'
                5u32, // 'll'
                5u32, // 'll'
            ]
        );

        // The return value `changes` will be used to update the pair counts during
        let mut changes_control = [
            (
                Ngram {
                    ids: vec![0u32, 2u32],
                },
                -1i32,
            ), // count for ('h', 'l') should be decreased by 1.
            (
                Ngram {
                    ids: vec![0u32, 2u32, 2u32],
                },
                -1i32,
            ), // count for ('h', 'l', 'l') should be decreased by 1.
            (
                Ngram {
                    ids: vec![0u32, 2u32, 2u32, 2u32],
                },
                -1i32,
            ), // count for ('h', 'l', 'l', 'l') should be decreased by 1.
            (
                Ngram {
                    ids: vec![2u32, 2u32, 2u32],
                },
                -1i32,
            ), // count for ('l', 'l', 'l') should be decreased by 1.
            (
                Ngram {
                    ids: vec![0u32, 2u32, 2u32, 2u32, 2u32],
                },
                -1i32,
            ), // count for ('h', 'l', 'l', 'l', 'l') should be decreased by 1.
            (
                Ngram {
                    ids: vec![2u32, 2u32, 2u32, 2u32],
                },
                -1i32,
            ), // count for ('l', 'l', 'l', 'l') should be decreased by 1.
            (
                Ngram {
                    ids: vec![0u32, 5u32],
                },
                1i32,
            ), // count for ('h', 'll') should be increased by 1.
            (
                Ngram {
                    ids: vec![0u32, 5u32, 5u32],
                },
                1i32,
            ), // count for ('h', 'll', 'll') should be increased by 1.
            (
                Ngram {
                    ids: vec![5u32, 5u32],
                },
                1i32,
            ), // count for ('ll','ll') should be increased by 1.
        ];
        changes.sort();
        changes_control.sort();

        assert_eq!(changes, changes_control);
    }

    #[test]
    fn test_merge_skip_ngrams() {
        // Let's say we have the word 'hello' and a word-to-id vocab that looks
        // like this: {'h': 0, 'e': 1, 'l': 2, 'o': 3, '!': 4}.
        let mut word = Word::new();
        word.add(0, 1); // 'h'
        word.add(2, 1); // 'l'
        word.add(2, 1); // 'l'
        word.add(2, 1); // 'l'
        word.add(2, 1); // 'l'
        word.add(0, 1); // 'h'

        // We're going to perform a merge on the ngram ('e', 'l', 'l') ~= [1, 2, 2)] Let's
        // say that 'ell' has the ID of 5 in the updated word-to-id vocab.
        let mut changes = word.merge(vec![2, 2], 5, usize::MAX, usize::MAX);

        // So the word should now look like this:
        assert_eq!(
            word.get_chars(),
            &[
                0u32, // 'h'
                5u32, // 'll'
                5u32, // 'll'
                0u32, // 'h'
            ]
        );

        // The return value `changes` will be used to update the pair counts during
        let mut changes_control = [
            (
                Ngram {
                    ids: vec![0u32, 2u32],
                },
                -1i32,
            ), // count for ('h', 'l') should be decreased by 1.
            (
                Ngram {
                    ids: vec![0u32, 2u32, 2u32],
                },
                -1i32,
            ), // count for ('h', 'l', 'l') should be decreased by 1.
            (
                Ngram {
                    ids: vec![0u32, 2u32, 2u32, 2u32],
                },
                -1i32,
            ), // count for ('h', 'l', 'l', 'l') should be decreased by 1.
            (
                Ngram {
                    ids: vec![2u32, 2u32, 2u32],
                },
                -1i32,
            ), // count for ('l', 'l', 'l') should be decreased by 1.
            (
                Ngram {
                    ids: vec![2u32, 2u32, 2u32, 0u32],
                },
                -1i32,
            ), // count for ('l', 'l', 'l', 'h') should be decreased by 1.
            (
                Ngram {
                    ids: vec![2u32, 2u32, 0u32],
                },
                -1i32,
            ), // count for ('l', 'l', 'h') should be decreased by 1.
            (
                Ngram {
                    ids: vec![2u32, 0u32],
                },
                -1i32,
            ), // count for ('l', 'h') should be decreased by 1.
            (
                Ngram {
                    ids: vec![0u32, 2u32, 2u32, 2u32, 2u32],
                },
                -1i32,
            ), // count for ('h', 'l', 'l', 'l', 'l') should be decreased by 1.
            (
                Ngram {
                    ids: vec![0u32, 2u32, 2u32, 2u32, 2u32, 0u32],
                },
                -1i32,
            ), // count for ('h', 'l', 'l', 'l', 'l', 'h') should be decreased by 1.
            (
                Ngram {
                    ids: vec![2u32, 2u32, 2u32, 2u32],
                },
                -1i32,
            ), // count for ('l', 'l', 'l', 'l') should be decreased by 1.
            (
                Ngram {
                    ids: vec![2u32, 2u32, 2u32, 2u32, 0u32],
                },
                -1i32,
            ), // count for ('l', 'l', 'l', 'l', 'h') should be decreased by 1.
            (
                Ngram {
                    ids: vec![0u32, 5u32],
                },
                1i32,
            ), // count for ('h', 'll') should be increased by 1.
            (
                Ngram {
                    ids: vec![0u32, 5u32, 5u32],
                },
                1i32,
            ), // count for ('h', 'll', 'll') should be increased by 1.
            (
                Ngram {
                    ids: vec![0u32, 5u32, 5u32, 0u32],
                },
                1i32,
            ), // count for ('h', 'll', 'll', 'h') should be increased by 1.
            (
                Ngram {
                    ids: vec![5u32, 5u32],
                },
                1i32,
            ), // count for ('ll','ll') should be increased by 1.
            (
                Ngram {
                    ids: vec![5u32, 5u32, 0u32],
                },
                1i32,
            ), // count for ('ll','ll', 'h') should be increased by 1.
            (
                Ngram {
                    ids: vec![5u32, 0u32],
                },
                1i32,
            ), // count for ('ll','h') should be increased by 1.
        ];
        changes.sort();
        changes_control.sort();

        assert_eq!(changes, changes_control);
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
        let mut changes = word.merge(vec![2, 2], 4, 3, usize::MAX);

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
        let mut changes_control = [
            (
                Ngram {
                    ids: vec![0u32, 1u32, 2u32],
                },
                -1i32,
            ), // count for ('h', 'e', 'l') should be decreased by 1.
            (
                Ngram {
                    ids: vec![1u32, 2u32],
                },
                -1i32,
            ), // count for ('e', 'l') should be decreased by 1.
            //(Ngram {ids:vec![0u32, 1u32, 2u32, 2u32]}, -1i32),    // count for ('h', 'e', 'l', 'l') should be decreased by 1.
            (
                Ngram {
                    ids: vec![1u32, 2u32, 2u32],
                },
                -1i32,
            ), // count for ('e', 'l', 'l') should be decreased by 1.
            //(Ngram {ids:vec![0u32, 1u32, 2u32, 2u32, 3u32]}, -1i32),    // count for ('h', 'e', 'l', 'l', 'o') should be decreased by 1.
            //(Ngram {ids:vec![1u32, 2u32, 2u32, 3u32]}, -1i32),    // count for ('e', 'l', 'l', 'o') should be decreased by 1.
            (
                Ngram {
                    ids: vec![2u32, 2u32, 3u32],
                },
                -1i32,
            ), // count for ('l', 'l', 'o') should be decreased by 1.
            (
                Ngram {
                    ids: vec![2u32, 3u32],
                },
                -1i32,
            ), // count for ('l', 'o') should be decreased by 1.
            //(Ngram {ids:vec![0u32, 1u32, 4u32]}, 1i32),  // count for ('h', 'e', 'll') should be increased by 1.
            (
                Ngram {
                    ids: vec![1u32, 4u32],
                },
                1i32,
            ), // count for ('e', 'll') should be increased by 1.
            //(Ngram {ids:vec![0u32, 1u32, 4u32, 3u32]}, 1i32),  // count for ('h', 'e', 'll', 'o') should be increased by 1.
            //(Ngram {ids:vec![1u32, 4u32, 3u32]}, 1i32),  // count for ('e', 'll','o') should be increased by 1.
            (
                Ngram {
                    ids: vec![4u32, 3u32],
                },
                1i32,
            ), // count for ('ll','o') should be increased by 1.
        ];
        changes.sort();
        changes_control.sort();

        assert_eq!(changes, changes_control);
    }

    #[test]
    fn test_merge_max_ngram_length() {
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
        let mut changes = word.merge(vec![2, 2], 4, usize::MAX, 3);

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
        let mut changes_control = [
            (
                Ngram {
                    ids: vec![0u32, 1u32, 2u32],
                },
                -1i32,
            ), // count for ('h', 'e', 'l') should be decreased by 1.
            (
                Ngram {
                    ids: vec![1u32, 2u32],
                },
                -1i32,
            ), // count for ('e', 'l') should be decreased by 1.
            //(Ngram {ids:vec![0u32, 1u32, 2u32, 2u32]}, -1i32),    // count for ('h', 'e', 'l', 'l') should be decreased by 1.
            (
                Ngram {
                    ids: vec![1u32, 2u32, 2u32],
                },
                -1i32,
            ), // count for ('e', 'l', 'l') should be decreased by 1.
            //(Ngram {ids:vec![0u32, 1u32, 2u32, 2u32, 3u32]}, -1i32),    // count for ('h', 'e', 'l', 'l', 'o') should be decreased by 1.
            //(Ngram {ids:vec![1u32, 2u32, 2u32, 3u32]}, -1i32),    // count for ('e', 'l', 'l', 'o') should be decreased by 1.
            (
                Ngram {
                    ids: vec![2u32, 2u32, 3u32],
                },
                -1i32,
            ), // count for ('l', 'l', 'o') should be decreased by 1.
            (
                Ngram {
                    ids: vec![2u32, 3u32],
                },
                -1i32,
            ), // count for ('l', 'o') should be decreased by 1.
            (
                Ngram {
                    ids: vec![0u32, 1u32, 4u32],
                },
                1i32,
            ), // count for ('h', 'e', 'll') should be increased by 1.
            (
                Ngram {
                    ids: vec![1u32, 4u32],
                },
                1i32,
            ), // count for ('e', 'll') should be increased by 1.
            //(Ngram {ids:vec![0u32, 1u32, 4u32, 3u32]}, 1i32),  // count for ('h', 'e', 'll', 'o') should be increased by 1.
            (
                Ngram {
                    ids: vec![1u32, 4u32, 3u32],
                },
                1i32,
            ), // count for ('e', 'll','o') should be increased by 1.
            (
                Ngram {
                    ids: vec![4u32, 3u32],
                },
                1i32,
            ), // count for ('ll','o') should be increased by 1.
        ];
        changes.sort();
        changes_control.sort();

        assert_eq!(changes, changes_control);
    }

    #[test]
    fn test_merge_all() {
        let mut word = Word::new();
        word.add(0, 1); // 'h'
        word.add(1, 1); // 'e'
        word.add(2, 1); // 'l'
        word.add(2, 1); // 'l'
        word.add(3, 1); // 'o'
        word.add(4, 1); // '!'

        let mut merges: AHashMap<Ngram, (u32, u32)> = AHashMap::new();
        merges.insert(Ngram { ids: vec![2, 2] }, (0, 5)); // merge 'l', 'l' -> 'll' (rank: 0, id: 5)
                                                          //merges.insert(Ngram{ids:vec![5, 3, 4]}, (1, 6));    // merge 'll', 'o', '!' -> 'llo!' (rank: 1, id: 6)

        word.merge_all(&merges, None);

        /*assert_eq!(
            word.get_chars(),
            &[
                0u32, // 'h'
                1u32, // 'e'
                6u32, // 'llo!'
            ]
        );*/
        assert_eq!(
            word.get_chars(),
            &[
                0u32, // 'h'
                1u32, // 'e'
                5u32, // 'll'
                3u32, // 'o'
                4u32, // '!'
            ]
        );
    }

    #[test]
    fn test_merge_all_2() {
        let mut word = Word::new();
        word.add(0, 1); // 'h'
        word.add(1, 1); // 'e'
        word.add(2, 1); // 'l'
        word.add(2, 1); // 'l'
        word.add(3, 1); // 'o'
        word.add(4, 1); // '!'

        let mut merges: AHashMap<Ngram, (u32, u32)> = AHashMap::new();
        merges.insert(Ngram { ids: vec![2, 2] }, (0, 5)); // merge 'l', 'l' -> 'll' (rank: 0, id: 5)
        merges.insert(Ngram { ids: vec![5, 3, 4] }, (1, 6)); // merge 'll', 'o', '!' -> 'llo!' (rank: 1, id: 6)

        word.merge_all(&merges, None);

        assert_eq!(
            word.get_chars(),
            &[
                0u32, // 'h'
                1u32, // 'e'
                6u32, // 'llo!'
            ]
        );
    }

    #[test]
    fn test_merge_all_3() {
        let mut word = Word::new();
        word.add(0, 1); // 'h'
        word.add(1, 1); // 'e'
        word.add(2, 1); // 'l'
        word.add(2, 1); // 'l'
        word.add(3, 1); // 'o'
        word.add(4, 1); // '!'

        let mut merges: AHashMap<Ngram, (u32, u32)> = AHashMap::new();
        merges.insert(Ngram { ids: vec![2, 2] }, (0, 5)); // merge 'l', 'l' -> 'll' (rank: 0, id: 5)
        merges.insert(Ngram { ids: vec![1, 2] }, (1, 6)); // merge 'e''l' -> 'el!' (rank: 1, id: 6)
        merges.insert(Ngram { ids: vec![0, 1] }, (2, 7)); // merge 'll', 'o', '!' -> 'llo!' (rank: 1, id: 6)
        merges.insert(Ngram { ids: vec![1, 5] }, (3, 8)); // merge 'e', 'll' -> 'llo!' (rank: 1, id: 6)

        word.merge_all(&merges, None);

        assert_eq!(
            word.get_chars(),
            &[
                7u32, // 'he'
                5u32, // 'll'
                3u32, // 'o'
                4u32, // '!'
            ]
        );
    }

    #[test]
    fn test_merge_all_4() {
        let mut word = Word::new();
        word.add(0, 1); // 'o'
        word.add(1, 1); // 'f'

        let mut merges: AHashMap<Ngram, (u32, u32)> = AHashMap::new();
        merges.insert(Ngram { ids: vec![0, 1] }, (0, 2)); // merge 'l', 'l' -> 'll' (rank: 0, id: 5)

        word.merge_all(&merges, None);

        assert_eq!(
            word.get_chars(),
            &[
                2u32, // 'of'
            ]
        );
    }

    #[test]
    fn test_merge_all_ignores_unary_merges() {
        let mut word = Word::new();
        word.add(0, 1);
        word.add(1, 1);

        let mut merges: AHashMap<Ngram, (u32, u32)> = AHashMap::new();
        merges.insert(Ngram { ids: vec![0, 1] }, (0, 2));
        merges.insert(Ngram { ids: vec![2] }, (1, 2));

        word.merge_all(&merges, None);

        assert_eq!(word.get_chars(), &[2u32]);
    }

    #[test]
    fn test_merge_all_empty_word() {
        let mut word = Word::new();
        let merges: AHashMap<Ngram, (u32, u32)> = AHashMap::new();

        word.merge_all(&merges, None);

        assert_eq!(word.get_chars(), Vec::<u32>::new());
    }

    #[test]
    fn test_merge_all_dropout() {}

    /*
    #[test]
    fn test_merge_with_vec() {
        let mut sym = Symbol{
            c: 0,
            prev: -1,
            next: 1,
            len: 1,
        };

        let sym1 = Symbol{
            c: 1,
            prev: 0,
            next: 3,
            len: 2,
        };

        let sym2 = Symbol{
            c: 3,
            prev: 1,
            next: 4,
            len: 1,
        };

        let sym3 = Symbol{
            c: 4,
            prev: 3,
            next: 8,
            len: 1,
        };

        let symbols = vec![&sym1, &sym2, &sym3];

        sym.merge_with_vec(symbols, 10);

        assert_eq!(sym.c,10);
        assert_eq!(sym.prev,-1);
        assert_eq!(sym.next,8);
        assert_eq!(sym.len,5);
    }
    */
}
