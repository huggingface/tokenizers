use crate::decoders::byte_level::CHAR_BYTES;
use crate::models::bpe::Pair;
use crate::pre_tokenizers::byte_level::ByteLevel;
use crate::pre_tokenizers::byte_level::BYTES_CHAR;
use crate::tokenizer::{Decoder, Result};
use ahash::AHashMap;
use aneubeck_daachorse::DoubleArrayAhoCorasick;
use aneubeck_daachorse::DoubleArrayAhoCorasickBuilder;
use fnv::{FnvHashMap, FnvHasher};
use itertools::Itertools;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::hash::{Hash, Hasher};
use std::ops::Range;

use super::MergeMap;
use super::Merges;
use super::Vocab;
use super::VocabR;

/// Small helper to manage a bit field which supports predecessor and successor queries with a simple scan implementation.
/// This is sufficient for our use case, since two one bits will be at most 128 bits apart.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BitField {
    bitfield: Vec<u64>,
}

impl BitField {
    /// All bits are initialized to 1.
    pub(crate) fn new(bits: usize) -> Self {
        Self {
            bitfield: vec![u64::MAX; (bits + 63) / 64],
        }
    }

    pub(crate) fn is_set(&self, bit: usize) -> bool {
        let (word, bit) = (bit / 64, bit % 64);
        self.bitfield[word] & (1 << bit) != 0
    }

    pub(crate) fn clear(&mut self, bit: usize) {
        let (word, bit) = (bit / 64, bit % 64);
        self.bitfield[word] &= !(1 << bit);
    }

    pub(crate) fn successor(&self, bit: usize) -> usize {
        let (mut word_idx, bit_idx) = (bit / 64, bit % 64);
        let word = self.bitfield[word_idx] >> bit_idx;
        if word != 0 {
            word.trailing_zeros() as usize + bit
        } else {
            loop {
                word_idx += 1;
                let word = self.bitfield[word_idx];
                if word != 0 {
                    break word.trailing_zeros() as usize + word_idx * 64;
                }
            }
        }
    }

    pub(crate) fn predecessor(&self, bit: usize) -> usize {
        let (mut word_idx, bit_idx) = (bit / 64, bit % 64);
        let word = self.bitfield[word_idx] << (63 - bit_idx);
        if word != 0 {
            bit - word.leading_zeros() as usize
        } else {
            loop {
                word_idx -= 1;
                let word = self.bitfield[word_idx];
                if word != 0 {
                    break word_idx * 64 + 63 - word.leading_zeros() as usize;
                }
            }
        }
    }
}

/// This can be thought of as a lazy variation of the dynamic programming approach.
/// It only computes those states which have to be visited in order to compute the tokenization
/// for a given input text.
/// It keeps track of visited states in a bitfield and only remembers the tokenization
/// of the currently processed dynamic programming state.
///
/// The biggest downside of this approach is that the search for the longest leftmost match (the firt token?)
/// has to be reset at every (backtracking) step which is still a net win in practice compared to other approaches.
#[derive(Clone, PartialEq)]
pub struct BacktrackState<'a> {
    pub(crate) text: &'a [u8],
    pub(crate) tokens: Vec<u32>,        // len of the tezt / 3
    pub(crate) next_token: Option<u32>, // bpe.next_match(text) wich is longest_searcher.leftmost_find_iter(text)'s first match value
    pub(crate) pos: usize,              // current pos in the text?
    pub(crate) bitfield: BitField, // keeps track of token boundaries? keeps track of all the valid tokenization positions and making the runtime linear in the input length.
}

impl<'a> BacktrackState<'a> {
    pub(crate) fn new(text: &'a [u8], next_token: Option<u32>) -> Self {
        Self::with_capacity(text, next_token, text.len() / 3)
    }

    pub(crate) fn with_capacity(text: &'a [u8], next_token: Option<u32>, cap: usize) -> Self {
        Self {
            text,
            tokens: Vec::with_capacity(cap),
            next_token,
            pos: 0,
            bitfield: BitField::new(text.len() + 1),
        }
    }
    pub(crate) fn count(&self) -> usize {
        self.tokens.len()
    }

    pub(crate) fn pos(&self) -> usize {
        self.pos
    }

    pub(crate) fn last_token(&self) -> Option<u32> {
        self.tokens.last().copied()
    }

    pub(crate) fn into_tokens(self) -> Vec<u32> {
        self.tokens
    }
}

#[derive(PartialEq, Clone)]
pub struct Backtrack {
    /// All the decoded tokens concatenated into? used to build the aho corasick searchers
    all_tokens: Vec<u8>,
    /// Start index of each token in all_tokens.
    /// The end is simply the next entry in this vector.
    token_starts: Vec<u32>,
    /// Mapping from hash of token to token id.
    bytes_hash_to_token: FnvHashMap<u32, u32>,
    /// The two tokens from which the token got merged.
    /// If the token is an original one, than the two tokens point back to itself.
    split_table: Vec<(u32, u32)>,
    /// Mapping from a pair of tokens to a merged token if such a merged token exists.
    pair_lookup: FnvHashMap<(u32, u32), u32>,
    /// An aho corasick automaton to find the next longest token in a byte sequence.
    // #[serde(
    //     serialize_with = "serialize_daac",
    //     deserialize_with = "deserialize_daac"
    // )]
    longest_searcher: DoubleArrayAhoCorasick<u32>,
    /// An aho corasick automaton to find ALL tokens in a byte sequence.
    // #[serde(
    //     serialize_with = "serialize_daac",
    //     deserialize_with = "deserialize_daac"
    // )]
    pub(crate) overlapping_searcher: DoubleArrayAhoCorasick<u32>,
    /// An aho corasick automaton to find ALL tokens in a byte sequence which is being processed in reverse order.
    // #[serde(
    //     serialize_with = "serialize_daac",
    //     deserialize_with = "deserialize_daac"
    // )]
    pub(crate) overlapping_searcher_rev: DoubleArrayAhoCorasick<u32>,
    /// Mapping from a token to the next longest prefix token.
    /// This is in principle information represented by the AhoCorasick automaton.
    /// But we don't have efficient access to it and therefore store it here again.
    /// If there is none, then the value is set to u32::MAX.
    next_prefix_match: Vec<u32>,
    /// Hash factor used to prevent hash collisions.
    hash_factor: u64,
    vocab: Vocab,
    vocab_r: VocabR,
    unk_token: Option<String>,
    merges: MergeMap,
}

fn hash_bytes(bytes: &[u8], factor: u64) -> u32 {
    let mut hasher = FnvHasher::default();
    bytes.hash(&mut hasher);
    // Note: we save 1/3 of space for the hashmap by only using the most significant bits of the hash.
    // To make them unique for the given tokens, we have to add unfortunately another multiplication.
    ((hasher.finish().wrapping_mul(factor)) >> 32) as u32
}

// #[cfg(feature = "rand")]
pub fn find_hash_factor_for_dictionary(tokens: impl IntoIterator<Item = Vec<u8>>) -> u64 {
    use std::collections::HashSet;

    use rand::Rng;

    let all_tokens: Vec<Vec<u8>> = tokens.into_iter().collect();
    let mut rnd = rand::rng();
    loop {
        let factor: u64 = rnd.random();
        let mut seen = HashSet::new();
        if all_tokens
            .iter()
            .all(|token| seen.insert(hash_bytes(token, factor)))
        {
            return factor;
        }
    }
}

impl Backtrack {
    pub(crate) fn new(vocab: Vocab, merge_map: MergeMap) -> Self {
        // let vocab_vec: Vec<_> = vocab
        //     .into_iter()
        //     .sorted_unstable_by(|a, b| a.1.cmp(&b.1))
        //     .map(|(k, _v)| k.chars().map(|b| CHAR_BYTES[&b] as u8).collect::<Vec<_>>())
        //     .collect();
        let mut merges: Vec<_> = merge_map.values().collect();
        merges.sort();
        let merge_vocab: Vec<u32> = merges
            .into_iter()
            .map(|(_rank, token_id)| *token_id)
            .collect();

        let vocab_r: AHashMap<_, _> = vocab.iter().map(|(k, v)| (v, k)).collect();
        let mut tokens: Vec<_> = vocab
            .clone()
            .into_iter()
            .flat_map(|(k, token_id)| {
                if merge_vocab.contains(&token_id) {
                    Some((token_id, k))
                } else {
                    None
                }
            })
            .collect();
        tokens.sort();
        let mut tokens: Vec<_> = tokens.into_iter().map(|(_token_id, k)| k).collect();

        let merge_vocab: Vec<String> = merge_vocab
            .into_iter()
            .map(|token_id| vocab_r[&token_id].clone())
            .collect();
        tokens.extend(merge_vocab);
        let vocab_vec: Vec<_> = tokens.into_iter().map(|k| k.as_bytes().to_vec()).collect();

        let hash_factor = find_hash_factor_for_dictionary(vocab_vec.clone());
        let mut all_tokens = Vec::new();
        let mut all_tokens_rev = Vec::new();
        let mut token_starts = vec![0]; // The begin byte index of each token in all_tokens.
        let mut bytes_hash_to_token = FnvHashMap::default();
        let tokens = vocab_vec;
        for (i, token) in tokens.into_iter().enumerate() {
            info!(
                "token byte: {:?}, {i}",
                ByteLevel::default()
                    .decode_chain(unsafe { vec![String::from_utf8_unchecked(token.clone())] })
                    .unwrap()
            );
            bytes_hash_to_token.insert(hash_bytes(&token, hash_factor), i as u32);
            all_tokens_rev.extend(token.iter().copied().rev());
            all_tokens.extend(token);
            token_starts.push(all_tokens.len() as u32);
        }
        assert_eq!(
            bytes_hash_to_token.len() + 1,
            token_starts.len(),
            "Some tokens are not unique under the hash function!"
        ); // TODO maybe this check is needed?
        let longest_searcher = DoubleArrayAhoCorasickBuilder::new()
            .match_kind(aneubeck_daachorse::MatchKind::LeftmostLongest)
            .build(token_iter(&all_tokens, &token_starts))
            .expect("failed to build AhoCorasick");

        let overlapping_searcher =
            DoubleArrayAhoCorasick::<u32>::new(token_iter(&all_tokens, &token_starts)).expect("");
        let overlapping_searcher_rev =
            DoubleArrayAhoCorasick::<u32>::new(token_iter(&all_tokens_rev, &token_starts))
                .expect("");

        let next_prefix_match: Vec<_> = token_iter(&all_tokens, &token_starts)
            .map(|token| {
                next_match(&longest_searcher, &token[0..token.len() - 1]).unwrap_or(u32::MAX)
            })
            .collect();

        let vocab: AHashMap<String, u32> = token_iter(&all_tokens, &token_starts)
            .enumerate()
            .map(|(id, bytes)| {
                (
                    bytes.iter().map(|b| BYTES_CHAR[b]).collect::<String>(),
                    id as u32,
                )
            })
            .collect();

        let vocab_r: AHashMap<u32, String> = token_iter(&all_tokens, &token_starts)
            .enumerate()
            .map(|(id, bytes)| {
                (
                    id as u32,
                    bytes.iter().map(|b| BYTES_CHAR[b]).collect::<String>(),
                )
            })
            .collect();

        let mut split_table = vec![];
        let mut pair_lookup = FnvHashMap::default();
        let mut merge_map = AHashMap::new();

        // // First option, use the input merge table.
        // if let Some(ref merges) = merges {
        //     for (index, pair) in merges.into_iter().enumerate() {
        //         let token1 = &pair.0.clone();
        //         let token2 = &pair.1.clone();
        //         // TODO something is weird here
        //         if token1.len() ==1{
        //             split_table.push((vocab[token1], vocab[token1]));
        //         }
        //         if token2.len() == 1 {
        //             split_table.push((vocab[token2], vocab[token2]));
        //         }
        //         let id1 = vocab[token1];
        //         let id2 = vocab[token2];
        //         let new_token = format!("{}{}", token1, &token2);
        //         let new_id = vocab
        //             .get(&new_token)
        //             .ok_or(Error::MergeTokenOutOfVocabulary(new_token));
        //         if let Ok(id) = new_id {
        //             pair_lookup.insert((id1, id2), *id);
        //             split_table.push((id1, id2));
        //             merge_map.insert(Pair::from((id1, id2)), (index as u32, *id));
        //         } else {
        //             println!("Token not added?");
        //         }

        //         // TODO wrong
        //     }
        //     split_table.push((merges.len() as u32, merges.len() as u32));
        // }
        // Second option, reverse engineer the merge/split table from the vocabulary.
        {
            for (id, token) in token_iter(&all_tokens, &token_starts).enumerate() {
                let mut id1 = next_prefix_match[id];
                while id1 != u32::MAX {
                    let rest = &token[token_range(&token_starts, id1).len()..];
                    if let Some(id2) = find_token_by_bytes(
                        &all_tokens,
                        &token_starts,
                        &bytes_hash_to_token,
                        rest,
                        hash_factor,
                    ) {
                        if id1 < id as u32
                            && id2 < id as u32
                            && is_valid_token_pair(&pair_lookup, &split_table, id1, id2)
                        {
                            pair_lookup.insert((id1, id2), id as u32);
                            split_table.push((id1, id2));
                            merge_map.insert(Pair::from((id1, id2)), (id as u32, id as u32));
                            break;
                        }
                    }
                    id1 = next_prefix_match[id1 as usize];
                }
                if id1 == u32::MAX {
                    split_table.push((id as u32, id as u32));
                }
            }
        };
        let bpe = Self {
            all_tokens,
            token_starts,
            bytes_hash_to_token,
            overlapping_searcher,
            overlapping_searcher_rev,
            longest_searcher,
            next_prefix_match,
            pair_lookup,
            split_table,
            hash_factor,
            unk_token: None,
            vocab,
            vocab_r,
            merges: merge_map,
        };
        // A health checkup
        for token_id in 0..bpe.num_tokens() as u32 {
            let bytes = bpe.token_bytes(token_id);
            let strs = bytes.iter().map(|b| char::from(*b)).collect::<Vec<_>>();
            // println!("Encoding {bytes:?} into bitfield");
            let tokens = bpe.encode_via_bitfield(bytes);
            assert_eq!(
                tokens,
                vec![token_id],
                "token {token_id} with bytes {bytes:?} (tokens {strs:?} encodes to {tokens:?} instead of to itself"
            );
        }
        bpe
    }

    fn bitfield_into_tokens(&self, bytes: &[u8], bitfield: BitField, count: usize) -> Vec<u32> {
        let mut encoded = Vec::with_capacity(count);
        let mut start = 0;
        while start < bytes.len() {
            let end = bitfield.successor(start + 1);
            // println!("bitfield's successor {:?}", &bytes[start..end]);
            let token = self
                .find_token_by_bytes(&bytes[start..end])
                .expect(&format!(
                    "Could not convert bytes to tokens for bytes: [{:?}]",
                    bytes.into_iter().map(|b| BYTES_CHAR[b]).join("")
                ));
            encoded.push(token);
            start = end;
        }
        encoded
    }

    fn encode_into_bitfield(&self, bytes: &[u8]) -> (BitField, usize) {
        // Reserve for every byte a bit in the bitfield.
        let mut bitfield = BitField::new(bytes.len() + 1);
        let mut heap = BinaryHeap::with_capacity(bytes.len() * 2);
        heap.extend((0..bytes.len().saturating_sub(1)).filter_map(|i| {
            self.find_token_by_bytes(&bytes[i..i + 2])
                .map(|e| Reverse((e, i as u32)))
        }));
        let mut count = bytes.len();
        while let Some(Reverse((token, start))) = heap.pop() {
            let start = start as usize;
            if !bitfield.is_set(start) {
                continue;
            }
            let mid = bitfield.successor(start + 1);
            if mid >= bytes.len() {
                continue;
            }
            let end = bitfield.successor(mid + 1);
            if self.token_len(token) != end - start {
                continue;
            }
            bitfield.clear(mid);
            count -= 1;
            if end < bytes.len() {
                let new_end = bitfield.successor(end + 1);
                if let Some(e) = self.find_token_by_bytes(&bytes[start..new_end]) {
                    heap.push(Reverse((e, start as u32)));
                }
            }
            if start > 0 {
                let new_start = bitfield.predecessor(start - 1);
                if let Some(e) = self.find_token_by_bytes(&bytes[new_start..end]) {
                    heap.push(Reverse((e, new_start as u32)));
                }
            }
        }
        (bitfield, count)
    }

    pub fn encode_via_bitfield(&self, text: &[u8]) -> Vec<u32> {
        let (bitfield, count) = self.encode_into_bitfield(text);
        self.bitfield_into_tokens(text, bitfield, count)
    }

    /// Return the number of tokens in this BPE dictionary.
    pub fn num_tokens(&self) -> usize {
        self.token_starts.len() - 1
    }

    /// Converts a token id into its corresponding token bytes.
    /// Panics if the token_id is not within the valid 0..num_tokens() range!
    pub fn token_bytes(&self, token_id: u32) -> &[u8] {
        token_bytes(&self.all_tokens, &self.token_starts, token_id)
    }

    pub(crate) fn is_valid_token_pair(&self, token1: u32, token2: u32) -> bool {
        is_valid_token_pair(&self.pair_lookup, &self.split_table, token1, token2)
    }

    /// Returns the length of the decoded byte slice of a token.
    pub fn token_len(&self, token_id: u32) -> usize {
        token_range(&self.token_starts, token_id).len()
    }

    /// Returns the first longest match in the provided text.
    pub(crate) fn next_match(&self, text: &[u8]) -> Option<u32> {
        next_match(&self.longest_searcher, text)
    }

    /// Returns the next token which shares the longest prefix with the specified token.
    pub(crate) fn next_prefix(&self, token_id: u32) -> Option<u32> {
        let prefix = self.next_prefix_match[token_id as usize];
        if prefix == u32::MAX {
            None
        } else {
            Some(prefix)
        }
    }

    fn find_token_by_bytes(&self, bytes: &[u8]) -> Option<u32> {
        find_token_by_bytes(
            &self.all_tokens,
            &self.token_starts,
            &self.bytes_hash_to_token,
            bytes,
            self.hash_factor,
        )
    }

    /// Decode a sequence of tokens back to its original byte sequence.
    /// Note: we don't return here a str, since not every token sequence corresponds to a valid
    /// utf8 sequence.
    pub fn decode_tokens(&self, tokens: &[u32]) -> Vec<u8> {
        let mut text = vec![];
        for token in tokens {
            text.extend(self.token_bytes(*token));
        }
        text
    }

    /// Computes for every prefix of the input text a corresponding last token.
    pub(crate) fn encode_all_prefixes(&self, text: &[u8]) -> Vec<u32> {
        let mut last_token = Vec::with_capacity(text.len());
        let mut state = self.overlapping_searcher.start_state();
        for (pos, c) in text.iter().enumerate() {
            let (s, iter) = self.overlapping_searcher.consume(state, pos + 1, *c);
            state = s;
            for m in iter {
                let new_token = m.value();
                let new_range = m.start()..m.end();
                assert_eq!(new_range.end, last_token.len() + 1);
                if new_range.start == 0 {
                    last_token.push(new_token);
                    break;
                } else {
                    let prev_token = unsafe { *last_token.get_unchecked(new_range.start - 1) };
                    if self.is_valid_token_pair(prev_token, new_token) {
                        last_token.push(new_token);
                        break;
                    }
                    // println!("Finished encoding prefix")
                }
            }
        }
        last_token
    }

    /// Counts the number tokens produced when encoding the text.
    pub fn count(&mut self, text: &[u8]) -> usize {
        let mut enc = BacktrackState::new(text, None);
        while self.step(&mut enc).is_some() {}
        enc.count()
    }

    pub fn encode_via_table(&self, text: &[u8]) -> Vec<u32> {
        let last_token = self.encode_all_prefixes(text);
        let mut encoded = Vec::with_capacity(text.len() / 3);
        let mut pos = text.len();
        while pos > 0 {
            let token = last_token[pos - 1];
            encoded.push(token);
            pos -= self.token_len(token);
        }
        encoded.reverse();
        encoded
    }

    pub fn encode_via_backtracking(&self, text: &[u8]) -> Vec<u32> {
        let next_token = self.next_match(text);
        let mut enc = BacktrackState::new(text, next_token);
        while self.step(&mut enc).is_some() {}
        enc.into_tokens()
    }

    pub fn get_vocab(&self) -> Vocab {
        self.vocab.clone()
    }

    pub fn get_unk_token(&self) -> &Option<String> {
        &self.unk_token
    }

    pub fn step(&self, backtrack_state: &mut BacktrackState) -> Option<u32> {
        let mut token = backtrack_state.next_token?;
        let last = backtrack_state.tokens.last().copied();
        loop {
            // println!("in step, token: {last:?}, {token}");
            let token_len = self.token_len(token);
            let end_pos = backtrack_state.pos + token_len;
            if backtrack_state.bitfield.is_set(end_pos)
                && last
                    .map(|last_token| self.is_valid_token_pair(last_token, token))
                    .unwrap_or(true)
            {
                backtrack_state.tokens.push(token);
                backtrack_state.pos = end_pos;
                // In principle, we could in some cases reuse the leftmost longest match iterator.
                // Especially when it has to look ahead, this could save scanning the input multiple times.
                // But on average this seems to be slower due to the overhead of storing the iterator as part of the struct.
                backtrack_state.next_token = self.next_match(&backtrack_state.text[end_pos..]);
                break;
            } else if let Some(shorter) = self.next_prefix(token) {
                token = shorter;
            } else {
                // Clearing the bitfield when we pop tokens saves a little bit of work...
                backtrack_state.bitfield.clear(backtrack_state.pos);
                backtrack_state.tokens.pop();
                backtrack_state.pos -= last.map(|t| self.token_len(t)).unwrap_or(0);
                backtrack_state.next_token = last;
                break;
            }
        }
        // println!("finished step, token: {last:?}, {token}");

        backtrack_state.next_token
    }
}

// A helper function to iterate over the tokens in a byte sequence
fn token_iter<'a>(all_tokens: &'a [u8], token_starts: &'a [u32]) -> impl Iterator<Item = &'a [u8]> {
    token_starts
        .iter()
        .tuple_windows()
        .map(move |(start, end)| &all_tokens[*start as usize..*end as usize])
}

fn next_match(longest_searcher: &DoubleArrayAhoCorasick<u32>, text: &[u8]) -> Option<u32> {
    longest_searcher
        .leftmost_find_iter(text)
        .map(|m| m.value())
        .next()
}

fn is_valid_token_pair(
    pair_lookup: &FnvHashMap<(u32, u32), u32>,
    split_table: &[(u32, u32)],
    mut token1: u32,
    mut token2: u32,
) -> bool {
    // Keep track of the maximum token which can still be chosen across the split point.
    let mut limit = u32::MAX;
    // println!("checking if {token1}, {token2} is a valid token_pair");
    loop {
        // Check whether BPE would choose a different token pair across the split point.
        // this is super super important
        if let Some(combined) = pair_lookup.get(&(token1, token2)) {
            if *combined < limit {
                // println!("Done1");
                return false;
            }
        }
        // Reverse the merge operation from BPE.

        // println!("{:?}", split_table);
        if token1 > token2 {
            limit = token1;
            token1 = unsafe { split_table.get_unchecked(token1 as usize).1 };
            if token1 == limit {
                limit = token2 + 1;
                token2 = unsafe { split_table.get_unchecked(token2 as usize).0 };
                if token2 + 1 == limit {
                    // println!("Done2");
                    return true;
                }
            }
        } else {
            limit = token2 + 1;
            token2 = unsafe { split_table.get_unchecked(token2 as usize).0 };
            if token2 + 1 == limit {
                limit = token1;
                token1 = unsafe { split_table.get_unchecked(token1 as usize).1 };
                if token1 == limit {
                    // println!("Done3");
                    return true;
                }
            }
        }
    }
}

fn token_range(token_starts: &[u32], token_id: u32) -> Range<usize> {
    unsafe {
        *token_starts.get_unchecked(token_id as usize) as usize
            ..*token_starts.get_unchecked(token_id as usize + 1) as usize
    }
}

fn token_bytes<'a>(all_tokens: &'a [u8], token_starts: &[u32], token_id: u32) -> &'a [u8] {
    &all_tokens[token_range(token_starts, token_id)]
}

fn find_token_by_bytes(
    all_tokens: &[u8],
    token_starts: &[u32],
    bytes_hash_to_token: &FnvHashMap<u32, u32>,
    bytes: &[u8],
    hash_factor: u64,
) -> Option<u32> {
    let hash = hash_bytes(bytes, hash_factor);
    let token = *bytes_hash_to_token.get(&hash)?;
    if token_bytes(all_tokens, token_starts, token) == bytes {
        Some(token)
    } else {
        None
    }
}

/// Converts the merges strings (for example from `merges.txt` file) with the format
/// "{pair_a} {pair_b}" into the format expected by the BacktrackingBpe struct
pub(crate) fn convert_merges_to_hashmap<I: Iterator<Item = String>>(
    iter: I,
    _vocab: &Vocab,
) -> Result<Merges> {
    let mut merges = vec![];

    let lines = iter.filter(|l| !l.starts_with("#version"));
    for (rank, line) in lines.enumerate() {
        let parts = line.split(' ').collect::<Vec<_>>();
        if parts.len() != 2 {
            return Err(super::Error::BadMerges(rank + 1).into());
        }

        merges.push((parts[0].to_string(), parts[1].to_string()));
    }

    Ok(merges)
}
