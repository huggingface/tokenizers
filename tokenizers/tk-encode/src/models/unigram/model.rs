use super::{
    lattice::Lattice,
    trie::{Trie, TrieBuilder},
};
use crate::utils::word_cache::{Lookup, WordCache};
use crate::{
    pipeline::{self, Token},
    tokenizer::Result,
};
use crate::{utils::cache::DEFAULT_CACHE_CAPACITY, vocab::bucket_vocab_store::BucketVocabStore};
use std::collections::HashMap;

pub(crate) type Vocab = Vec<(String, f64)>;

/// A `Unigram` model to encode sentences.
#[derive(Clone)]
pub struct Unigram {
    token_to_ids: BucketVocabStore,
    pub(crate) vocab: Vocab,
    // Compact, cache-friendly copy of `vocab`'s scores, indexed by vocabulary id.
    // `vocab` stores `(String, f64)` pairs, so reading a score through it strides
    // 16+ bytes and chases a heap allocation; this array is 8 bytes per entry.
    scores: Vec<f64>,
    trie: Trie<u8>,
    pub min_score: f64,
    pub(super) unk_id: Option<usize>,
    pub bos_id: usize,
    pub eos_id: usize,

    fuse_unk: bool,
    is_optimized: bool,
    byte_fallback: bool,

    pub(crate) alpha: Option<f64>,
    pub(crate) nbest_size: Option<usize>,
}
impl PartialEq for Unigram {
    fn eq(&self, other: &Self) -> bool {
        self.unk_id == other.unk_id
            && self.vocab == other.vocab
            && self.alpha == other.alpha
            && self.nbest_size == other.nbest_size
    }
}

impl std::fmt::Debug for Unigram {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("Unigram")
            .field("vocab", &self.vocab.len())
            .field("unk_id", &self.unk_id)
            .field("byte_fallback", &self.byte_fallback)
            .finish()
    }
}

static K_UNK_PENALTY: f64 = 10.0;

/// One row of the optimized Viterbi table.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct BestPathNode {
    /// The vocab id. (maybe UNK)
    id: usize,
    /// The total score of the best path ending at this node.
    best_path_score: f64,
    /// The starting position (in utf-8) of this node. The entire best
    /// path can be constructed by backtracking along this link.
    starts_at: Option<usize>,
}

/// One piece of the best path, and the id the path gave it.
///
/// A piece is always a span of the input, so neither consumer has to build or own it.
enum Piece<'s> {
    /// A node the vocabulary has an entry for, with that entry's id.
    Token { text: &'s str, id: u32 },
    /// A span with no id of its own: one unknown character, or a run of them fused.
    Unknown(&'s str),
}

impl<'s> Piece<'s> {
    fn text(&self) -> &'s str {
        match self {
            Piece::Token { text, .. } | Piece::Unknown(text) => text,
        }
    }
}

#[inline]
fn utf8_char_width(lead: u8) -> usize {
    if lead < 0x80 {
        1
    } else if lead < 0xE0 {
        2
    } else if lead < 0xF0 {
        3
    } else {
        4
    }
}

#[derive(thiserror::Error, Debug)]
pub(crate) enum UnigramError {
    #[error("The vocabulary is empty but at least <unk> is needed")]
    EmptyVocabulary,
    #[error("The `unk_id` is larger than vocabulary size")]
    UnkIdNotInVocabulary,
    #[error("Encountered an unknown token but `unk_id` is missing")]
    MissingUnkId,
}

impl Default for Unigram {
    fn default() -> Self {
        let vocab = vec![("<unk>".to_string(), 0.0)];
        Self::from(vocab, Some(0), false).unwrap()
    }
}

impl Unigram {
    /// Create a `Unigram` model from a given vocabulary.
    /// Vocabulary are the various tokens and their associated score which is a sort of a logprob of
    /// their frequency, which will enable tokenization and sampling.
    /// unk_id, is the index within the vocabulary.
    /// For now `Unigram` *requires* at least `unk` because we might find a never seen char.
    /// Further versions might allow that part to be hidden.
    pub fn from(
        vocab: Vec<(String, f64)>,
        unk_id: Option<usize>,
        byte_fallback: bool,
    ) -> Result<Self> {
        let n = vocab.len();
        let mut pairs: Vec<(Vec<u8>, u32)> = Vec::with_capacity(n);
        let mut builder = TrieBuilder::default();

        if let Some(unk_id) = unk_id {
            if vocab.is_empty() {
                return Err(Box::new(UnigramError::EmptyVocabulary));
            }
            if unk_id >= vocab.len() {
                return Err(Box::new(UnigramError::UnkIdNotInVocabulary));
            }
        }
        let bos_id = n + 1;
        let eos_id = n + 2;

        let mut min_score = f64::INFINITY;
        for (id, (token, score)) in vocab.iter().enumerate() {
            pairs.push((token.as_bytes().to_vec(), id as u32));
            builder.push_with_value(token.as_bytes(), id as u32);
            if score < &min_score {
                min_score = *score;
            }
        }
        let token_to_ids = BucketVocabStore::build(pairs)?;
        let trie = builder.build();
        let scores: Vec<f64> = vocab.iter().map(|(_, score)| *score).collect();
        let fuse_unk = true;
        let is_optimized = true;

        Ok(Self {
            vocab,
            scores,
            token_to_ids,
            trie,
            min_score,
            bos_id,
            eos_id,
            unk_id,
            fuse_unk,
            is_optimized,
            byte_fallback,
            alpha: None,
            nbest_size: None,
        })
    }

    #[cfg(test)]
    pub(super) fn set_fuse_unk(&mut self, fuse_unk: bool) {
        self.fuse_unk = fuse_unk;
    }

    #[cfg(test)]
    pub(super) fn set_optimized(&mut self, is_optimized: bool) {
        self.is_optimized = is_optimized;
    }
    pub fn byte_fallback(&self) -> bool {
        self.byte_fallback
    }
    pub fn len(&self) -> usize {
        self.vocab.len()
    }
    pub fn is_empty(&self) -> bool {
        self.vocab.is_empty()
    }

    pub fn populate_nodes(&self, lattice: &mut Lattice) {
        let sentence_len = lattice.sentence.len();
        let unk_score = self.min_score - K_UNK_PENALTY;
        lattice.nodes.reserve(sentence_len * 2 + 4);
        let mut begin_pos = 0;
        while begin_pos < sentence_len {
            let mblen = utf8_char_width(lattice.sentence.as_bytes()[begin_pos]);
            let mut has_single_node = false;

            for bytes in self
                .trie
                .common_prefix_search_values(lattice.sentence[begin_pos..].bytes())
            {
                let (n, id) = bytes;
                let score: f64 = self.scores[id as usize];
                lattice.insert(begin_pos, n, score, id as usize);
                if !has_single_node && n == mblen {
                    has_single_node = true;
                }
            }

            if !has_single_node && let Some(unk_id) = self.unk_id {
                lattice.insert(begin_pos, mblen, unk_score, unk_id);
            }

            begin_pos += mblen;
        }
    }

    /// Whether [`Unigram::alpha`] asks for a tokenization drawn at random from the
    /// lattice instead of its best path. Such a result is one draw out of many, so no
    /// cache may hold it.
    fn samples(&self) -> bool {
        matches!(self.alpha, Some(alpha) if alpha != 0.0)
    }

    /// The pieces `sentence` encodes to: the lattice's best path, or a draw from it when
    /// [`Unigram::samples`].
    pub(super) fn encode(&self, sentence: &str) -> Result<Vec<String>> {
        if self.is_optimized && !self.samples() {
            self.encode_optimized(sentence)
        } else {
            self.encode_unoptimized(sentence)
        }
    }

    /// The pieces `sentence` encodes to, built into a fresh `Vec`.
    ///
    /// This is the model-level entry point, which has to hand back owned pieces. The pipeline
    /// calls [`Self::encode_ids_optimized`] instead, which skips the strings entirely.
    fn encode_optimized(&self, sentence: &str) -> Result<Vec<String>> {
        let mut best_path = Vec::new();
        let mut path = Vec::new();
        let mut results = Vec::new();
        self.fill_best_path(sentence, &mut best_path)?;
        backtrack(&best_path, &mut path);
        self.for_each_piece(sentence, &path, |piece| {
            results.push(piece.text().to_string());
            Ok(())
        })?;
        Ok(results)
    }

    /// Fill `best_path[i]` with the highest-scoring path that ends at byte `i`, and the byte it
    /// started at, so the best path can be read back by following `starts_at` from the end.
    fn fill_best_path(&self, sentence: &str, best_path: &mut Vec<BestPathNode>) -> Result<()> {
        // https://github.com/google/sentencepiece/blob/d48247191a6d50e469ed1a4a36e877befffd1851/src/unigram_model.cc#L600
        let size = sentence.len();
        let unk_score = self.min_score - K_UNK_PENALTY;

        // `starts_at` is what marks a row taken, so clearing is all a reused table needs.
        best_path.clear();
        best_path.resize(size + 1, BestPathNode::default());
        let mut starts_at = 0;
        while starts_at < size {
            let best_path_score_till_here = best_path[starts_at].best_path_score;
            let mut has_single_node = false;
            let mblen = utf8_char_width(sentence.as_bytes()[starts_at]);
            for (length, id) in self
                .trie
                .common_prefix_search_values(sentence[starts_at..].bytes())
            {
                let key_pos = starts_at + length;
                let target_node = &mut best_path[key_pos];
                let score = self.scores[id as usize];
                let candidate_best_path_score = score + best_path_score_till_here;
                if target_node.starts_at.is_none()
                    || candidate_best_path_score > target_node.best_path_score
                {
                    target_node.best_path_score = candidate_best_path_score;
                    target_node.starts_at = Some(starts_at);
                    target_node.id = id as usize;
                }
                if !has_single_node && length == mblen {
                    has_single_node = true;
                }
            }
            if !has_single_node {
                let target_node = &mut best_path[starts_at + mblen];
                let candidate_best_path_score = unk_score + best_path_score_till_here;
                if target_node.starts_at.is_none()
                    || candidate_best_path_score > target_node.best_path_score
                {
                    target_node.best_path_score = candidate_best_path_score;
                    target_node.starts_at = Some(starts_at);
                    // A model declared without an unknown id still encodes text its
                    // vocabulary covers, so this is the one place that error belongs.
                    target_node.id = self.unk_id.ok_or(UnigramError::MissingUnkId)?;
                }
            }
            starts_at += mblen;
        }
        Ok(())
    }

    /// Walk the best path and hand the caller each piece, one span of `sentence` at a time,
    /// with runs of unknown tokens fused into a single piece when [`Unigram::fuse_unk`] says so.
    ///
    /// The path is walked forwards here, which is the reverse of the order it was backtracked
    /// in, so a fused run ends up as one `first..last` span of the input rather than a
    /// concatenation of its parts.
    fn for_each_piece<'s>(
        &self,
        sentence: &'s str,
        path: &[(usize, usize, usize)],
        mut emit: impl FnMut(Piece<'s>) -> Result<()>,
    ) -> Result<()> {
        let mut fused: Option<(usize, usize)> = None;
        for &(start, end, id) in path.iter().rev() {
            if self.fuse_unk && Some(id) == self.unk_id {
                fused = Some(match fused {
                    Some((first, _)) => (first, end),
                    None => (start, end),
                });
                continue;
            }
            if let Some((first, last)) = fused.take() {
                emit(Piece::Unknown(&sentence[first..last]))?;
            }
            emit(match Some(id) == self.unk_id {
                true => Piece::Unknown(&sentence[start..end]),
                false => Piece::Token {
                    text: &sentence[start..end],
                    id: id as u32,
                },
            })?;
        }
        if let Some((first, last)) = fused {
            emit(Piece::Unknown(&sentence[first..last]))?;
        }
        Ok(())
    }

    /// Optimized Viterbi that resolves each piece to its id directly.
    ///
    /// `encode` builds a `String` per piece only for the pipeline to look it straight back up.
    /// This keeps the same lattice, tie-breaking and unknown/byte-fallback rules, and fills
    /// caller-owned buffers instead.
    fn encode_ids_optimized(
        &self,
        sentence: &str,
        best_path: &mut Vec<BestPathNode>,
        path: &mut Vec<(usize, usize, usize)>,
        ids: &mut Vec<u32>,
    ) -> Result<()> {
        self.fill_best_path(sentence, best_path)?;
        backtrack(best_path, path);
        ids.clear();
        self.for_each_piece(sentence, path, |piece| match piece {
            // The path already carries the id, so a known piece is never looked up again.
            Piece::Token { id, .. } => {
                ids.push(id);
                Ok(())
            }
            Piece::Unknown(text) => self.push_unresolved(text, ids),
        })
    }

    /// A piece the best path has no id for: an unknown character, or a run of them fused into
    /// one span. The fused span may still name a real token, so it is looked up whole first,
    /// exactly as the piece path does with its `String`.
    fn push_unresolved(&self, text: &str, ids: &mut Vec<u32>) -> Result<()> {
        match self.token_to_ids.token_to_id(text) {
            Some(id) => {
                ids.push(id);
                Ok(())
            }
            None => self.push_unknown(text, ids),
        }
    }

    /// The pipeline's fallback for a piece the vocabulary has no id for: byte tokens when the
    /// model has them and every byte is covered, otherwise the unknown id.
    fn push_unknown(&self, text: &str, ids: &mut Vec<u32>) -> Result<()> {
        if self.byte_fallback {
            const HEX: &[u8; 16] = b"0123456789ABCDEF";
            let start = ids.len();
            let mut covered = true;
            for byte in text.bytes() {
                let mut token = *b"<0x00>";
                token[3] = HEX[(byte >> 4) as usize];
                token[4] = HEX[(byte & 0x0f) as usize];
                // Built entirely from ASCII above.
                let token = std::str::from_utf8(&token).unwrap();
                match self.token_to_ids.token_to_id(token) {
                    Some(id) => ids.push(id),
                    None => {
                        covered = false;
                        break;
                    }
                }
            }
            if covered {
                return Ok(());
            }
            ids.truncate(start);
        }
        ids.push(self.unk_id.ok_or(UnigramError::MissingUnkId)? as u32);
        Ok(())
    }

    fn encode_unoptimized(&self, sentence: &str) -> Result<Vec<String>> {
        let mut lattice = Lattice::from(sentence, self.bos_id, self.eos_id);
        self.populate_nodes(&mut lattice);
        let path = match (self.nbest_size, self.alpha) {
            (Some(n), Some(alpha)) if n > 0 => lattice.sample_nbest(n, alpha),
            (_, Some(alpha)) => lattice.sample(alpha),
            _ => lattice.viterbi(),
        };
        if self.fuse_unk {
            let mut results = vec![];
            let mut token = String::new();
            for node in path.iter() {
                let item = lattice.piece(node);
                if node.id == self.unk_id.ok_or(UnigramError::MissingUnkId)? {
                    token.push_str(&item);
                } else {
                    if !token.is_empty() {
                        results.push(token);
                        token = String::new();
                    }
                    results.push(item);
                }
            }
            if !token.is_empty() {
                results.push(token);
            }
            Ok(results)
        } else {
            let results: Vec<String> = path.iter().map(|node| lattice.piece(node)).collect();
            Ok(results)
        }
    }

    /// Iterate of vocabulary of the model as a pair of `(token, score)`.
    pub fn iter(&self) -> UnigramIterator<'_> {
        UnigramIterator { model: self, i: 0 }
    }

    /// The id of the unknown-token entry, if this model declares one.
    ///
    /// This and [`Self::vocab`] exist for `tk-convert`'s serde mirror: the fields themselves are
    /// crate-private, and a `unigram.json` is written from exactly these three values plus
    /// [`Self::byte_fallback`].
    pub fn unk_id(&self) -> Option<usize> {
        self.unk_id
    }

    /// The `(token, score)` table, in id order. [`Self::iter`] is the same data, one entry at a
    /// time.
    pub fn vocab(&self) -> &[(String, f64)] {
        &self.vocab
    }
}

/// Iterator to iterate of vocabulary of the model, and their relative score.
pub struct UnigramIterator<'a> {
    model: &'a Unigram,
    i: usize,
}

impl<'a> Iterator for UnigramIterator<'a> {
    type Item = &'a (String, f64);

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.i;
        if i < self.model.len() {
            let r = Some(&self.model.vocab[i]);
            self.i += 1;
            r
        } else {
            None
        }
    }
}

/// The model methods the pipeline and the readers call. These used to be the legacy
/// `Model` trait; that trait had no implementor left that needed polymorphism, so they are
/// plain inherent methods now and every call site is unchanged.
impl Unigram {
    pub(crate) fn get_vocab(&self) -> HashMap<String, u32> {
        self.token_to_ids.content().into_iter().collect()
    }

    pub fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_ids.token_to_id(token)
    }

    pub(crate) fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab.get(id as usize).map(|item| item.0.clone())
    }
}

/// Read the best path back from `best_path` into `path`, one `(start, end, id)` per node, in
/// the order the path was backtracked (last node first).
fn backtrack(best_path: &[BestPathNode], path: &mut Vec<(usize, usize, usize)>) {
    path.clear();
    let mut ends_at = best_path.len() - 1;
    while ends_at > 0 {
        let node = &best_path[ends_at];
        let starts_at = node.starts_at.unwrap();
        path.push((starts_at, ends_at, node.id));
        ends_at = starts_at;
    }
}

pub struct UnigramScratch {
    /// Outlives the encode call that fills it, or it would never see a word twice.
    pub(crate) word_cache: WordCache,
    /// Reused Viterbi table; sized by the longest sequence this scratch has seen.
    pub(super) best_path: Vec<BestPathNode>,
    /// Reused best-path spans, `(start, end, id)`, filled from the end backwards.
    pub(super) path: Vec<(usize, usize, usize)>,
    /// Reused id buffer for the optimized path.
    pub(super) ids: Vec<u32>,
}

impl UnigramScratch {
    fn new() -> Self {
        Self {
            word_cache: WordCache::new(DEFAULT_CACHE_CAPACITY),
            best_path: Vec::new(),
            path: Vec::new(),
            ids: Vec::new(),
        }
    }
}

impl pipeline::Tokenize for Unigram {
    type Scratch = UnigramScratch;

    fn init_scratch(&self) -> Self::Scratch {
        UnigramScratch::new()
    }

    /// The word cache holds ids, not pieces. A hit skips the lattice and the vocabulary lookup;
    /// on a miss the optimized path backtracks to ids in the scratch buffers, so neither path
    /// builds a `String` per piece.
    fn tokenize(
        &self,
        sequence: &str,
        scratch: &mut Self::Scratch,
        output: &mut Vec<pipeline::Token>,
    ) -> Result<()> {
        if sequence.is_empty() {
            return Ok(());
        }
        let mut placement = None;
        if !self.samples() {
            match scratch.word_cache.lookup(sequence.as_bytes()) {
                Lookup::Hit(ids) => {
                    output.extend(ids.iter().copied().map(Token::from));
                    return Ok(());
                }
                Lookup::Miss(at) => placement = Some(at),
            }
        }

        let start = output.len();
        if self.is_optimized && !self.samples() {
            // Backtrack straight to ids: the piece strings `encode` builds exist only for
            // this lookup, and this loop already knows each piece's id.
            self.encode_ids_optimized(
                sequence,
                &mut scratch.best_path,
                &mut scratch.path,
                &mut scratch.ids,
            )?;
            output.extend(scratch.ids.iter().copied().map(Token::from));
        } else {
            for string in self.encode(sequence)? {
                match self.token_to_ids.token_to_id(&string) {
                    Some(id) => {
                        output.push(Token::from(id));
                    }
                    None => {
                        scratch.ids.clear();
                        self.push_unknown(&string, &mut scratch.ids)?;
                        output.extend(scratch.ids.iter().copied().map(Token::from));
                    }
                };
            }
        }

        // Sampling skipped the lookup, so `placement` is `None` and this stores
        // nothing, which is what `samples` says has to happen.
        if let Some(at) = placement {
            scratch
                .word_cache
                .insert(at, output[start..].iter().map(|token| token.id()));
        }
        Ok(())
    }
}
