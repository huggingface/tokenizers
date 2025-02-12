use super::bitfield::BitField;
use super::{super::bpe::trainer::BpeTrainer, super::bpe::Error, super::OrderedVocabIter};
use crate::models::Bpe;
use crate::{decoders, pre_tokenizers, Decoder};
use crate::models::bpe::{MergeMap, Pair, BPE};
use crate::tokenizer::{Model, Result, Token};
use crate::utils::iter::ResultShunt;
use aneubeck_daachorse::{DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder};
use fnv::{FnvHashMap, FnvHasher};
use itertools::Itertools;
use serde::de::Visitor;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::{
    collections::HashMap,
    fs::File,
    io::prelude::*,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};
pub type Vocab = HashMap<String, u32>;
type VocabR = HashMap<u32, String>;
pub type Merges = Vec<(String, String)>;

use super::backtracking_state::BacktrackState;

struct Config {
    files: Option<(String, String)>,
    vocab: Vocab,
    merges: Merges,
    dropout: Option<f32>,
    unk_token: Option<String>,
    fuse_unk: bool,
    byte_fallback: bool,
}

pub struct BacktrackingBpeBuilder {
    config: Config,
}

impl Default for BacktrackingBpeBuilder {
    fn default() -> Self {
        Self {
            config: Config {
                files: None,
                vocab: HashMap::new(),
                merges: vec![],
                dropout: None,
                unk_token: None,
                fuse_unk: false,
                byte_fallback: false,
            },
        }
    }
}

/// A [Byte Pair Encoding](https://www.aclweb.org/anthology/P16-1162/) model.
#[derive(PartialEq, Clone)]
pub struct BacktrackingBpe {
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
    pub vocab: Vocab,
    pub vocab_r: VocabR,
    unk_token: Option<String>,
    pub merges: MergeMap,
}

use std::fmt;

// Manually implement the Debug trait to exclude the `cache` field
impl fmt::Debug for BacktrackingBpe {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("BacktrackingBpe")
            .field("vocab", &self.vocab)
            .field("vocab_r", &self.vocab_r)
            .field("split_table", &self.split_table)
            .field("token_starts", &self.token_starts)
            .field("pair_lookup", &self.pair_lookup)
            // Skipping `cache` field here, it won't be included in debug output
            .finish()
    }
}

impl BacktrackingBpeBuilder {
    /// Constructs a new `BacktrackingBpeBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the input files.
    #[must_use]
    pub fn files(mut self, vocab: String, merges: String) -> Self {
        self.config.files = Some((vocab, merges));
        self
    }

    /// Set the vocab (token -> ID) and merges mappings.
    #[must_use]
    pub fn vocab_and_merges(mut self, vocab: Vocab, merges: Merges) -> Self {
        self.config.vocab = vocab;
        self.config.merges = merges;
        self
    }

    /// Use [dropout](https://arxiv.org/abs/1910.13267) with the model.
    #[must_use]
    pub fn dropout(mut self, dropout: f32) -> Self {
        self.config.dropout = Some(dropout);
        self
    }

    /// Set the `UNK` token for the vocab.
    #[must_use]
    pub fn unk_token(mut self, unk_token: String) -> Self {
        self.config.unk_token = Some(unk_token);
        self
    }

    /// Set the `fuse_unk` option.
    #[must_use]
    pub fn fuse_unk(mut self, fuse_unk: bool) -> Self {
        self.config.fuse_unk = fuse_unk;
        self
    }

    /// Set the `byte_fallback` option.
    #[must_use]
    pub fn byte_fallback(mut self, byte_fallback: bool) -> Self {
        self.config.byte_fallback = byte_fallback;
        self
    }

    /// Returns a `BacktrackingBpe` model that uses the `BacktrackingBpeBuilder`'s configuration.
    pub fn build(mut self) -> Result<BacktrackingBpe> {
        // Validate dropout.
        if let Some(p) = self.config.dropout {
            if !(0.0..=1.0).contains(&p) {
                return Err(Error::InvalidDropout.into());
            }
        }

        // Read files if necessary
        if let Some((vocab, merges)) = self.config.files {
            let (v, m) = BPE::read_file(&vocab, &merges)?;
            self.config.vocab = v;
            self.config.merges = m;
        }

        let backtraching_bpe = BacktrackingBpe::from_dictionary(
            self.config.vocab.into_iter().sorted_unstable_by(|a,b| a.1.cmp(&b.1)).map(|(k, v)| k.into_bytes()),
            Some(self.config.merges),
            None,
        );
        Ok(backtraching_bpe)
    }
}

impl Default for BacktrackingBpe {
    fn default() -> Self {
        Self::builder().build().unwrap()
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
    loop {
        // Check whether BPE would choose a different token pair across the split point.
        if let Some(combined) = pair_lookup.get(&(token1, token2)) {
            if *combined < limit {
                return false;
            }
        }
        // Reverse the merge operation from BPE.
        println!("{token1}, {token2}");
        println!("{:?}", split_table);
        if token1 > token2 {
            limit = token1;
            token1 = unsafe { split_table.get_unchecked(token1 as usize).1 };
            if token1 == limit {
                limit = token2 + 1;
                token2 = unsafe { split_table.get_unchecked(token2 as usize).0 };
                if token2 + 1 == limit {
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

fn hash_bytes(bytes: &[u8], factor: u64) -> u32 {
    let mut hasher = FnvHasher::default();
    bytes.hash(&mut hasher);
    // Note: we save 1/3 of space for the hashmap by only using the most significant bits of the hash.
    // To make them unique for the given tokens, we have to add unfortunately another multiplication.
    ((hasher.finish().wrapping_mul(factor)) >> 32) as u32
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
            return Err(Error::BadMerges(rank + 1).into());
        }

        merges.push((parts[0].to_string(), parts[1].to_string()));
    }

    Ok(merges)
}

impl BacktrackingBpe {
    /// Initialize a `BacktrackingBpeBuilder`.
    pub fn builder() -> BacktrackingBpeBuilder {
        BacktrackingBpeBuilder::new()
    }

    /// Create a new BacktrackingBpe model with the given vocab and merges.
    pub fn new(vocab: Vocab, merges: Merges) -> Self {
        Self::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .unwrap()
    }

    fn bitfield_into_tokens(&self, bytes: &[u8], bitfield: BitField, count: usize) -> Vec<u32> {
        let mut encoded = Vec::with_capacity(count);
        let mut start = 0;
        while start < bytes.len() {
            let end = bitfield.successor(start + 1);
            println!("bitfield's successor {:?}", &bytes[start..end]);
            let token = self.find_token_by_bytes(&bytes[start..end]).expect(&format!("Could not convert bytes to tokens for bytes: [{:?}]", bytes.into_iter().map(|b| char::from(*b)).join("|")));
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

    /// Construct a BytePairEncoding instance from an iterator that enumerates all tokens.
    /// A suitable hash factor may be necessary to prevent hash collisions, which can be
    /// found using [`find_hash_factor_for_dictionary`].
    ///
    /// The recommended approach is to store the serialized value and reuse that,
    /// to prevent repeating the cost of computing the hash factor and encoding.
    pub fn from_dictionary(
        tokens: impl IntoIterator<Item = Vec<u8>>,
        merges: Option<Merges>,
        hash_factor: Option<u64>,
    ) -> Self {
        let hash_factor = hash_factor
            .inspect(|f| assert_ne!(*f, 0, "hash factor must be larger than zero"))
            .unwrap_or(1);
        let mut all_tokens = Vec::new();
        let mut all_tokens_rev = Vec::new();
        let mut token_starts = vec![0]; // The begin byte index of each token in all_tokens.
        let mut bytes_hash_to_token = FnvHashMap::default();
        let mut merge_map: HashMap<Pair, (u32, u32)> = HashMap::new();
        for (i, token) in tokens.into_iter().enumerate() {
            use pre_tokenizers::byte_level::ByteLevel;
            info!("token byte: {:?}, {i}", ByteLevel::default().decode_chain(unsafe {
                vec![String::from_utf8_unchecked(token.clone())]
            }).unwrap());
            bytes_hash_to_token.insert(hash_bytes(&token, hash_factor), i as u32);
            all_tokens_rev.extend(token.iter().copied().rev());
            all_tokens.extend(token);
            token_starts.push(all_tokens.len() as u32);
        }
        // assert_eq!(bytes_hash_to_token.len() + 1, token_starts.len()); # TODO maybe this check is needed?
        let longest_searcher = DoubleArrayAhoCorasickBuilder::new()
            .match_kind(aneubeck_daachorse::MatchKind::LeftmostLongest)
            .build(token_iter(&all_tokens, &token_starts))
            .expect("failed to build AhoCorasick");

        let overlapping_searcher =
            DoubleArrayAhoCorasick::<u32>::new(token_iter(&all_tokens, &token_starts)).expect("");
        let overlapping_searcher_rev =
            DoubleArrayAhoCorasick::<u32>::new(token_iter(&all_tokens, &token_starts)).expect("");

        let next_prefix_match: Vec<_> = token_iter(&all_tokens, &token_starts)
            .map(|token| {
                next_match(&longest_searcher, &token[0..token.len() - 1]).unwrap_or(u32::MAX)
            })
            .collect();

        let vocab: HashMap<String, u32> = token_iter(&all_tokens, &token_starts)
            .enumerate()
            .map(|(id, item)| {
                (
                    unsafe { String::from_utf8_unchecked(Vec::from(item)) },
                    id as u32,
                )
            })
            .collect();

        let vocab_r: HashMap<u32, String> = token_iter(&all_tokens, &token_starts)
            .enumerate()
            .map(|(id, item)| {
                (id as u32, unsafe {
                    String::from_utf8_unchecked(Vec::from(item))
                })
            })
            .collect();

        let mut split_table = vec![];
        let mut pair_lookup = FnvHashMap::default();

        // First option, use the input merge table.
        if let Some( ref merges) = merges{
            for (index, pair) in merges.into_iter().enumerate() {
                let token1 = &pair.0.clone();
                let token2 = &pair.1.clone();
                let id1 = vocab[token1];
                let id2 = vocab[token2];
                let new_token = format!("{}{}", token1, &token2);
                let new_id = vocab
                    .get(&new_token)
                    .ok_or(Error::MergeTokenOutOfVocabulary(new_token));
                if let Ok(id) = new_id {
                    println!("adding to the split table: ({token1}, {token2}), ({id1}, {id2}), {id}");
                    pair_lookup.insert((id1, id2), *id);
                    split_table.push((id1, id2));
                    merge_map.insert(Pair::from((id1, id2)), (index as u32, *id ));
                }else{
                    println!("Token not added?");
                }

                // TODO wrong
            }
            split_table.push((merges.len() as u32, merges.len() as u32));
        }  

        // Second option, reverse engineer the merge/split table from the vocabulary.
        else
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
        for token_id in 0..bpe.num_tokens() as u32 {
            let bytes = bpe.token_bytes(token_id);
            let strs = bytes.iter().map(|b| char::from(*b)).collect::<Vec<_>>();
            println!("Encoding {bytes:?} into bitfield");
            let tokens = bpe.encode_via_bitfield(bytes);
            assert_eq!(
                tokens,
                vec![token_id],
                "token {token_id} with bytes {bytes:?} (tokens {strs:?} encodes to {tokens:?} instead of to itself"
            );
        }
        println!("{:#?}", bpe);
        bpe
    }

    /// Initialize a BacktrackingBpeBuilder model from vocab and merges files
    pub fn from_file(vocab: &str, merges: &str) -> BacktrackingBpeBuilder {
        Self::builder().files(vocab.to_owned(), merges.to_owned())
    }

    /// Read the given files to extract the vocab and merges
    pub fn read_file(vocab: &str, merges: &str) -> Result<(Vocab, Merges)> {
        // Read vocab.json
        let vocab_file = File::open(vocab)?;
        let mut vocab_file = BufReader::new(vocab_file);

        let mut buffer = String::new();
        vocab_file.read_to_string(&mut buffer)?;
        let json: Value = serde_json::from_str(&buffer)?;
        let mut vocab = HashMap::new();
        match json {
            Value::Object(m) => {
                for (token, id) in m {
                    if let Value::Number(id) = id {
                        let id = id.as_u64().ok_or(Error::BadVocabulary)? as u32;
                        vocab.insert(token, id);
                    }
                }
            }
            _ => return Err(Box::new(Error::BadVocabulary)),
        };

        // Read merges file
        let merge_file = File::open(merges)?;
        let merge_file = BufReader::new(merge_file);
        let merges = ResultShunt::process(merge_file.lines(), |iter| {
            convert_merges_to_hashmap(iter, &vocab)
        })??; // TODO correctly process to fill the split and pair lookup

        Ok((vocab, merges))
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
            let token_len = self.token_len(token);
            let end_pos = backtrack_state.pos + token_len;
            println!("in step, token: {last:?}, {token}");
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
        backtrack_state.next_token
    }

    fn word_to_tokens<'a, 'b: 'a>(
        &'a self,
        word: &'b Vec<u32>,
    ) -> impl Iterator<Item = Token> + 'a {
        word.into_iter()
            .map(move |id| Token::new(*id, self.vocab_r[&id].clone(), (0usize, 0usize)))
        // TODO offsets should be easy to integrate as well!
    }
}
impl Model for BacktrackingBpe {
    type Trainer = BpeTrainer;

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.clone()
    }

    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        if sequence.is_empty() {
            return Ok(vec![]);
        }
        let byte_text = sequence.as_bytes();
        let word = self.encode_via_backtracking(byte_text);
        Ok(self.word_to_tokens(&word).collect())
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        Some(self.vocab_r[&id].clone())
    }

    fn save(&self, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>> {
        let vocab_file_name = match name {
            Some(name) => format!("{name}-vocab.json"),
            None => "vocab.json".to_string(),
        };

        // Write vocab.json
        let vocab_path: PathBuf = [folder, Path::new(vocab_file_name.as_str())]
            .iter()
            .collect();
        let mut vocab_file = File::create(&vocab_path)?;
        let order_vocab_iter = OrderedVocabIter::new(&self.vocab_r);
        let serialized = serde_json::to_string(&order_vocab_iter)?;
        vocab_file.write_all(serialized.as_bytes())?;
        Ok(vec![vocab_path])
        // Ok(vec![vocab_path, merges_path])
    }

    fn get_trainer(&self) -> BpeTrainer {
        BpeTrainer::default()
    }
}

impl Bpe for BacktrackingBpe {
    fn with_vocab(&mut self, vocab: HashMap<String, u32>) ->  &mut Self {
        self.vocab = vocab;
        self
    }

    fn with_vocab_r(&mut self, vocab_r: HashMap<u32, String>) -> &mut Self {
        self.vocab_r = vocab_r;
        self    
    }

    fn with_merges(&mut self, merge_map:HashMap<(u32, u32), (u32, u32)>) -> &mut Self {
        self.merges = merge_map;
        self
    }

}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn my_example() {
        let tokens = [
            "a", "b", "c", // 1 character each
            "aac", "ac", "cc", "cca", "aacc", "aaccca", "acca", "acc", "aa", "aaa",
            "aaaa", // 2 characters each
        ];
        let mut bpe =
            BacktrackingBpe::from_dictionary(tokens.map(|t| t.as_bytes().to_vec()), None, None);
        // bpe.encode_via_backtracking(b"baacca");
        let tokens = bpe.tokenize("aaaacc").unwrap();
        println!("{:?}", bpe.tokenize("aaaacc"));
        assert_eq!(
            tokens,
            vec![
                Token {
                    id: 12,
                    value: String::from("aaa"),
                    offsets: (0, 0)
                },
                Token {
                    id: 10,
                    value: String::from("acc"),
                    offsets: (0, 0)
                }
            ]
        );
        println!("{:?}", bpe.tokenize("baaaaccca"));
        let tokens = bpe.tokenize("baaaaccca").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token {
                    id: 1,
                    value: String::from("b"),
                    offsets: (0, 0)
                },
                Token {
                    id: 12,
                    value: String::from("aaa"),
                    offsets: (0, 0)
                },
                Token {
                    id: 4,
                    value: String::from("ac"),
                    offsets: (0, 0)
                },
                Token {
                    id: 6,
                    value: String::from("cca"),
                    offsets: (0, 0)
                }
            ]
        );
        bpe.encode_via_backtracking(b"baaaaccca");
        let tokens = [
            "a", "b", "c", // 1 character each
            "acca", "cc", "ac", "aac", "cca",
        ];
        let mut bpe =
            BacktrackingBpe::from_dictionary(tokens.map(|t| t.as_bytes().to_vec()), None, None);
        bpe.encode_via_backtracking(b"baacca");
    }
}
