//! Small-input reference retaining upstream queue and birth-cohort semantics.
//! Observations expose merge choices without changing its queue or word updates.
use super::*;
use dary_heap::OctonaryHeap;
use std::cmp::Ordering;

#[derive(Debug, Eq)]
struct Merge {
    pair: Pair,
    count: u64,
    pos: AHashSet<usize>,
}

impl PartialEq for Merge {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}

impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Merge {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.count != other.count {
            self.count.cmp(&other.count)
        } else {
            // Here we want ascending order
            other.pair.cmp(&self.pair)
        }
    }
}

impl BpeTrainer {
    /// Add the provided special tokens to the initial vocabulary
    pub(super) fn add_special_tokens(
        &self,
        w2id: &mut AHashMap<CompactString, u32>,
        id2w: &mut Vec<CompactString>,
    ) {
        for token in &self.special_tokens {
            // get hash of content
            if !w2id.contains_key(&CompactString::from(&token.content)) {
                id2w.push(CompactString::from(&token.content));
                w2id.insert(CompactString::from(&token.content), (id2w.len() - 1) as u32);
            }
        }
    }

    pub(super) fn tokenize_words(
        &self,
        wc: &AHashMap<CompactString, u64>,
        w2id: &mut AHashMap<CompactString, u32>,
        id2w: &mut Vec<CompactString>,
    ) -> (Vec<Word>, Vec<u64>) {
        let mut words: Vec<Word> = Vec::with_capacity(wc.len());
        let mut counts: Vec<u64> = Vec::with_capacity(wc.len());

        for (word, count) in wc {
            let mut current_word = Word::new();
            counts.push(*count);

            for (is_first, is_last, c) in word.chars().with_first_and_last() {
                let mut s = c.to_string();
                if w2id.contains_key(&CompactString::from(&s)) {
                    // Found the initial char in the authorized alphabet

                    // Add the `continuing_subword_prefix` if relevant
                    if !is_first && let Some(prefix) = &self.continuing_subword_prefix {
                        s.insert_str(0, prefix);
                    }
                    // Add the `end_of_word_suffix` if relevant
                    if is_last && let Some(suffix) = &self.end_of_word_suffix {
                        s.push_str(suffix);
                    }

                    // Insert the new formed string if necessary
                    if !w2id.contains_key(&CompactString::from(&s)) {
                        id2w.push(CompactString::from(&s));
                        w2id.insert(CompactString::from(&s), (id2w.len() - 1) as u32);
                    }
                    current_word.add(w2id[&CompactString::from(&s)], 1); // We do not care about the len here
                }
            }
            words.push(current_word);
        }

        (words, counts)
    }

    fn count_pairs(
        &self,
        words: &[Word],
        counts: &[u64],
    ) -> (AHashMap<Pair, i32>, AHashMap<Pair, AHashSet<usize>>) {
        words
            .maybe_par_iter()
            .enumerate()
            .map(|(i, word)| {
                let mut pair_counts = AHashMap::new();
                let mut where_to_update: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();

                for window in word.get_chars().windows(2) {
                    let cur_pair: Pair = (window[0], window[1]);

                    // Initialize pair_counts and where_to_update for this pair if we just saw it
                    // Then update counts
                    *pair_counts.entry(cur_pair).or_default() += counts[i] as i32;
                    where_to_update.entry(cur_pair).or_default().insert(i);
                }

                (pair_counts, where_to_update)
            })
            .reduce(
                || (AHashMap::new(), AHashMap::new()),
                |(mut pair_counts, mut where_to_update), (pc, wtu)| {
                    for (k, v) in pc {
                        *pair_counts.entry(k).or_default() += v;
                    }
                    for (k, v) in wtu {
                        where_to_update.entry(k).or_default().extend(v);
                    }
                    (pair_counts, where_to_update)
                },
            )
    }

    pub(super) fn do_train_observed(
        &self,
        word_counts: &AHashMap<CompactString, u64>,
        mut observe: impl FnMut(Pair, u64, u32),
    ) -> Result<(Vocab, Merges, Vec<AddedToken>)> {
        let mut word_to_id: AHashMap<CompactString, u32> = AHashMap::with_capacity(self.vocab_size);
        let mut id_to_word: Vec<CompactString> = Vec::with_capacity(self.vocab_size);
        let max_token_length: usize = self.max_token_length.unwrap_or(usize::MAX);

        //
        // 1. Add all special tokens to the vocabulary
        //
        self.add_special_tokens(&mut word_to_id, &mut id_to_word);

        //
        // 2. Compute the initial alphabet
        //
        self.compute_alphabet(word_counts, &mut word_to_id, &mut id_to_word);

        //
        // 3. Tokenize words
        //

        let (mut words, counts) =
            self.tokenize_words(word_counts, &mut word_to_id, &mut id_to_word);

        //
        // 4. Count pairs in words
        //

        let (mut pair_counts, mut where_to_update) = self.count_pairs(&words, &counts);
        // Insert them in the queue
        let mut queue = OctonaryHeap::with_capacity(pair_counts.len());
        where_to_update.drain().for_each(|(pair, pos)| {
            let count = pair_counts[&pair];
            if count > 0 {
                queue.push(Merge {
                    pair,
                    count: count as u64,
                    pos,
                });
            }
        });

        //
        // 5. Do merges
        //

        let mut merges: Vec<(Pair, u32)> = vec![];
        loop {
            // Stop as soon as we have a big enough vocabulary
            if word_to_id.len() >= self.vocab_size {
                break;
            }

            let Some(mut top) = queue.pop() else {
                break;
            };

            if top.count != pair_counts[&top.pair] as u64 {
                top.count = pair_counts[&top.pair] as u64;
                queue.push(top);
                continue;
            }

            if top.count < 1 || self.min_frequency > top.count {
                break;
            }

            let part_a = &id_to_word[top.pair.0 as usize];
            let mut part_b = id_to_word[top.pair.1 as usize].as_str();

            // Build new token
            if let Some(prefix) = &self.continuing_subword_prefix
                && let Some(rest) = part_b.strip_prefix(prefix)
            {
                part_b = rest;
            }

            // Insert new token if it does not already exist
            let new_token = format!("{part_a}{part_b}");
            let new_token_id = word_to_id
                .get(&CompactString::from(&new_token))
                .copied()
                .unwrap_or(id_to_word.len() as u32);
            if !word_to_id.contains_key(&CompactString::from(&new_token)) {
                id_to_word.push(CompactString::from(&new_token));
                word_to_id.insert(CompactString::from(&new_token), new_token_id);
            }
            merges.push((top.pair, new_token_id));
            observe(top.pair, top.count, new_token_id);

            // Reference updates are sequential; each distinct word is merged once.
            let changes = top
                .pos
                .iter()
                .flat_map(|&index| {
                    words[index]
                        .merge(top.pair.0, top.pair.1, new_token_id, max_token_length)
                        .into_iter()
                        .map(|change| (change, index))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            // Introduce new formed pairs
            for ((pair, change), iw) in changes {
                let count = change * counts[iw] as i32;
                *pair_counts.entry(pair).or_default() += count;
                if change > 0 {
                    where_to_update.entry(pair).or_default().insert(iw);
                }
            }
            where_to_update.drain().for_each(|(pair, pos)| {
                let count = pair_counts[&pair];
                if count > 0 {
                    queue.push(Merge {
                        pair,
                        count: count as u64,
                        pos,
                    });
                }
            });
        }

        // The vocabulary, keyed by the token string rather than by `word_to_id`'s hash: we have to
        // look the string up in `id_to_word` either way.
        let vocab: Vocab = word_to_id
            .into_iter()
            .map(|(_key, val)| (id_to_word[val as usize].to_string(), val))
            .collect();

        // `merges` holds id pairs, highest priority first; the on-disk form is the two token
        // strings, which is also what `from_vocab_and_merges` re-derives its ranks from. Order is
        // the rank, so it has to be preserved.
        let merges: Merges = merges
            .into_iter()
            .map(|(pair, _new_token_id)| {
                (
                    id_to_word[pair.0 as usize].to_string(),
                    id_to_word[pair.1 as usize].to_string(),
                )
            })
            .collect();

        Ok((vocab, merges, self.special_tokens.clone()))
    }
}
