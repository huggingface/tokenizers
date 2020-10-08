use super::{
    lattice::Lattice,
    trainer::UnigramTrainer,
    trie::{Trie, TrieBuilder},
};
use crate::tokenizer::{Model, Result, Token};
use crate::utils::cache::Cache;

use std::collections::HashMap;
use std::convert::TryInto;
use std::fs::read_to_string;
use std::path::{Path, PathBuf};

type TokenMap = HashMap<String, u32>;
type Vocab = Vec<(String, f64)>;

/// A `Unigram` model to encode sentences.
pub struct Unigram {
    token_to_ids: TokenMap,
    pub(crate) vocab: Vocab,
    cache: Cache<String, Vec<String>>,
    trie: Trie<u8>,
    pub min_score: f64,
    pub(super) unk_id: Option<usize>,
    pub(super) bos_id: usize,
    pub(super) eos_id: usize,

    fuse_unk: bool,
    is_optimized: bool,
}
impl PartialEq for Unigram {
    fn eq(&self, other: &Self) -> bool {
        self.unk_id == other.unk_id && self.vocab == other.vocab
    }
}

impl Clone for Unigram {
    // `Clone` can't be derive because it's not implemented for `Cache`.
    // To keep things simple when we clone, the new Unigram will start with a fresh cache.
    fn clone(&self) -> Self {
        let fresh_cache = self.cache.fresh();
        Self {
            vocab: self.vocab.clone(),
            cache: fresh_cache,
            token_to_ids: self.token_to_ids.clone(),
            trie: self.trie.clone(),
            min_score: self.min_score,
            unk_id: self.unk_id,
            bos_id: self.bos_id,
            eos_id: self.eos_id,
            fuse_unk: self.fuse_unk,
            is_optimized: self.is_optimized,
        }
    }
}

impl std::fmt::Debug for Unigram {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("Unigram")
            .field("vocab", &self.vocab.len())
            .field("unk_id", &self.unk_id)
            .finish()
    }
}

static K_UNK_PENALTY: f64 = 10.0;

#[derive(Debug)]
pub enum UnigramError {
    EmptyVocabulary,
    UnkIdNotInVocabulary,
    MissingUnkId,
}

impl std::fmt::Display for UnigramError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            UnigramError::EmptyVocabulary => {
                write!(f, "The vocabulary is empty but at least <unk> is needed")
            }
            UnigramError::UnkIdNotInVocabulary => {
                write!(f, "The `unk_id` is larger than vocabulary size")
            }
            UnigramError::MissingUnkId => {
                write!(f, "Encountered an unknown token but `unk_id` is missing")
            }
        }
    }
}

impl std::error::Error for UnigramError {}

impl Default for Unigram {
    fn default() -> Self {
        let vocab = vec![("<unk>".to_string(), 0.0)];
        Self::from(vocab, Some(0)).unwrap()
    }
}

impl Unigram {
    /// Create a `Unigram` model from a given vocabulary.
    /// Vocabulary are the various tokens and their associated score which is a sort of a logprob of
    /// their frequency, which will enable tokenization and sampling.
    /// unk_id, is the index within the vocabulary.
    /// For now `Unigram` *requires* at least `unk` because we might find a never seen char.
    /// Further versions might allow that part to be hidden.
    pub fn from(vocab: Vec<(String, f64)>, unk_id: Option<usize>) -> Result<Self> {
        let n = vocab.len();
        let mut token_to_ids: TokenMap = HashMap::new();
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
            token_to_ids.insert(token.to_string(), id as u32);
            let bytes: Vec<u8> = token.bytes().collect();
            builder.push(&bytes);
            if score < &min_score {
                min_score = *score;
            }
        }
        let trie = builder.build();
        let fuse_unk = true;
        let is_optimized = true;

        Ok(Unigram {
            vocab,
            token_to_ids,
            trie,
            min_score,
            bos_id,
            eos_id,
            unk_id,
            fuse_unk,
            cache: Cache::default(),
            is_optimized,
        })
    }

    #[cfg(test)]
    pub(super) fn set_fuse_unk(&mut self, fuse_unk: bool) {
        self.fuse_unk = fuse_unk;
        self.cache = self.cache.fresh();
    }

    #[cfg(test)]
    pub(super) fn set_optimized(&mut self, is_optimized: bool) {
        self.is_optimized = is_optimized;
    }

    pub(super) fn len(&self) -> usize {
        self.vocab.len()
    }

    pub(super) fn populate_nodes(&self, lattice: &mut Lattice) {
        let unk_score = self.min_score - K_UNK_PENALTY;

        let len = lattice.len();

        let mut begin_pos = 0;
        while begin_pos < len {
            let mblen = lattice.sentence[begin_pos..]
                .chars()
                .next()
                .unwrap()
                .len_utf8();

            let mut has_single_node = false;

            for bytes in self
                .trie
                .common_prefix_search(lattice.sentence.bytes().skip(begin_pos))
            {
                let n = bytes.len();
                let tok = String::from_utf8(bytes).unwrap();
                let id = *self.token_to_ids.get(&tok).unwrap();

                let item = &self.vocab[id as usize];
                assert_eq!(item.0, tok);
                let score: f64 = item.1;
                lattice.insert(begin_pos, n, score, id.try_into().unwrap());
                if !has_single_node && n == mblen {
                    has_single_node = true;
                }
            }

            if !has_single_node {
                if let Some(unk_id) = self.unk_id {
                    lattice.insert(begin_pos, mblen, unk_score, unk_id);
                }
            }
            begin_pos += mblen
        }
    }

    /// This functions take a String, and will encode it in a Vec of Strings,
    /// of the best tokenization available to the current model.
    /// ```
    /// use tokenizers::models::unigram::Unigram;
    ///
    /// let pieces = vec![
    ///     ("<unk>".to_string(), 0.0),
    ///     ("a".to_string(), 0.0),
    ///     ("b".to_string(), 0.0),
    ///     ("c".to_string(), 0.0),
    ///     ("d".to_string(), 0.0),
    ///     ("cd".to_string(), 1.0),
    ///     ("ab".to_string(), 2.0),
    ///     ("abc".to_string(), 5.0),
    ///     ("abcd".to_string(), 10.0),
    /// ];
    /// let model = Unigram::from(pieces, Some(0)).unwrap();
    /// let result = model.encode("abcdacdxx").unwrap();
    /// assert_eq!(result, vec!["abcd", "a", "cd", "xx"]);
    /// ```
    pub fn encode(&self, sentence: &str) -> Result<Vec<String>> {
        if sentence.is_empty() {
            return Ok(vec![]);
        }
        if let Some(result) = self.cache.get(sentence) {
            Ok(result.to_vec())
        } else {
            let result = if self.is_optimized {
                self.encode_optimized(sentence)?
            } else {
                self.encode_unoptimized(sentence)?
            };
            self.cache.set(sentence.to_owned(), result.clone());
            Ok(result)
        }
    }

    fn encode_optimized(&self, sentence: &str) -> Result<Vec<String>> {
        // https://github.com/google/sentencepiece/blob/d48247191a6d50e469ed1a4a36e877befffd1851/src/unigram_model.cc#L600
        #[derive(Debug, Clone)]
        struct BestPathNode {
            /// The vocab id. (maybe UNK)
            id: usize,
            /// The total score of the best path ending at this node.
            best_path_score: f64,
            /// The starting position (in utf-8) of this node. The entire best
            /// path can be constructed by backtracking along this link.
            starts_at: Option<usize>,
        };
        impl Default for BestPathNode {
            fn default() -> Self {
                Self {
                    id: 0,
                    best_path_score: 0.0,
                    starts_at: None,
                }
            }
        }
        let size = sentence.len();
        let unk_score = self.min_score - K_UNK_PENALTY;

        let mut best_path_ends_at = vec![BestPathNode::default(); size + 1];
        let mut starts_at = 0;
        while starts_at < size {
            let best_path_score_till_here = best_path_ends_at[starts_at].best_path_score;
            let mut has_single_node = false;
            let mblen = sentence[starts_at..].chars().next().unwrap().len_utf8();
            for tok_bytes in self
                .trie
                .common_prefix_search(sentence.bytes().skip(starts_at))
            {
                let key_pos = starts_at + tok_bytes.len();
                let token: String = String::from_utf8(tok_bytes).unwrap();
                let mut target_node = &mut best_path_ends_at[key_pos];
                let length = key_pos - starts_at;
                let id = self.token_to_ids.get(&token).unwrap();
                let score = self.vocab.get(*id as usize).unwrap().1;
                let candidate_best_path_score = score + best_path_score_till_here;
                if target_node.starts_at.is_none()
                    || candidate_best_path_score > target_node.best_path_score
                {
                    target_node.best_path_score = candidate_best_path_score;
                    target_node.starts_at = Some(starts_at);
                    target_node.id = *id as usize;
                }
                if !has_single_node && length == mblen {
                    has_single_node = true;
                }
            }
            if !has_single_node {
                let mut target_node = &mut best_path_ends_at[starts_at + mblen];
                let candidate_best_path_score = unk_score + best_path_score_till_here;
                if target_node.starts_at.is_none()
                    || candidate_best_path_score > target_node.best_path_score
                {
                    target_node.best_path_score = candidate_best_path_score;
                    target_node.starts_at = Some(starts_at);
                    target_node.id = self.unk_id.ok_or(UnigramError::MissingUnkId)?;
                }
            }
            starts_at += mblen
        }
        let mut ends_at = size;
        let mut results: Vec<String> = vec![];
        let mut token = vec![];
        while ends_at > 0 {
            let node = &best_path_ends_at[ends_at];
            let starts_at = node.starts_at.unwrap();
            if self.fuse_unk
                && self.unk_id.is_some()
                && node.id == self.unk_id.ok_or(UnigramError::MissingUnkId)?
            {
                token.push(
                    String::from_utf8(sentence[starts_at..ends_at].as_bytes().to_vec()).unwrap(),
                );
            } else {
                if !token.is_empty() {
                    token.reverse();
                    results.push(token.concat());
                    token = vec![];
                }
                results.push(
                    String::from_utf8(sentence[starts_at..ends_at].as_bytes().to_vec()).unwrap(),
                );
            }
            ends_at = starts_at;
        }
        if !token.is_empty() {
            token.reverse();
            results.push(token.concat());
        }
        results.reverse();
        Ok(results)
    }

    fn encode_unoptimized(&self, sentence: &str) -> Result<Vec<String>> {
        let mut lattice = Lattice::from(sentence, self.bos_id, self.eos_id);
        self.populate_nodes(&mut lattice);
        if self.fuse_unk {
            let mut results = vec![];
            let mut token = String::new();
            for node in lattice.viterbi().iter() {
                let item = lattice.piece(&node.borrow());
                if node.borrow().id == self.unk_id.ok_or(UnigramError::MissingUnkId)? {
                    token.push_str(&item);
                } else {
                    if !token.is_empty() {
                        results.push(token);
                        token = String::new();
                    }
                    results.push(item.to_string());
                }
            }
            if !token.is_empty() {
                results.push(token);
            }
            Ok(results)
        } else {
            Ok(lattice.tokens())
        }
    }

    /// Iterate of vocabulary of the model as a pair of `(token, score)`.
    pub fn iter(&self) -> UnigramIterator {
        UnigramIterator { model: self, i: 0 }
    }

    /// Loads a SentencePiece output model after being trained by tokenizers.
    /// After that you can use the model with tokenizers library.
    /// ```no_run
    /// use tokenizers::models::unigram::Unigram;
    /// use std::path::Path;
    ///
    /// let model = Unigram::load("mymodel-unigram.json").unwrap();
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Unigram> {
        let string = read_to_string(path)?;
        Ok(serde_json::from_str(&string)?)
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

impl Model for Unigram {
    type Trainer = UnigramTrainer;

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.token_to_ids.clone()
    }

    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn tokenize(&self, sentence: &str) -> Result<Vec<Token>> {
        let str_tokens = self.encode(sentence)?;
        let mut offset = 0;
        let mut tokens = Vec::with_capacity(str_tokens.len());
        for string in str_tokens {
            let id: u32 = match self.token_to_ids.get(&string) {
                Some(id) => *id,
                None => self.unk_id.ok_or(UnigramError::MissingUnkId)? as u32,
            };
            let len = string.len();
            let offsets = (offset, offset + len);
            offset += len;
            tokens.push(Token::new(id, string, offsets));
        }
        Ok(tokens)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_ids.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        match self.vocab.get(id as usize) {
            Some(item) => Some(item.0.clone()),
            None => None,
        }
    }

    fn save(&self, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>> {
        let name = match name {
            Some(name) => format!("{}-unigram.json", name),
            None => "unigram.json".to_string(),
        };
        let mut fullpath = PathBuf::new();
        fullpath.push(folder);
        fullpath.push(name);
        let string = serde_json::to_string_pretty(self)?;
        std::fs::write(&fullpath, string)?;
        Ok(vec![fullpath])
    }

    fn get_trainer(&self) -> Self::Trainer {
        UnigramTrainer::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_populate_nodes_unk() {
        let pieces = vec![("<unk>".to_string(), 0.0)];
        let model = Unigram::from(pieces, Some(0)).unwrap();

        let mut lattice = Lattice::from("abc", model.bos_id, model.eos_id);
        model.populate_nodes(&mut lattice);

        assert_eq!(lattice.begin_nodes[0].len(), 1);
        assert_eq!(lattice.begin_nodes[1].len(), 1);
        assert_eq!(lattice.begin_nodes[2].len(), 1);
        assert_eq!(lattice.begin_nodes[0][0].borrow().id, 0);
        assert_eq!(lattice.begin_nodes[1][0].borrow().id, 0);
        assert_eq!(lattice.begin_nodes[2][0].borrow().id, 0);
        assert_eq!(lattice.begin_nodes[0][0].borrow().node_id, 2);
        assert_eq!(lattice.begin_nodes[1][0].borrow().node_id, 3);
        assert_eq!(lattice.begin_nodes[2][0].borrow().node_id, 4);
    }

    #[test]
    fn test_populate_nodes() {
        let pieces = vec![
            ("<unk>".to_string(), 0.0),
            ("a".to_string(), 0.1),
            ("b".to_string(), 0.2),
            ("ab".to_string(), 0.3),
            ("bc".to_string(), 0.4),
        ];
        let model = Unigram::from(pieces, Some(0)).unwrap();

        let mut lattice = Lattice::from("abc", model.bos_id, model.eos_id);
        model.populate_nodes(&mut lattice);

        assert_eq!(lattice.begin_nodes[0].len(), 2); // a, ab
        assert_eq!(lattice.begin_nodes[1].len(), 2); // b, bc
        assert_eq!(lattice.begin_nodes[2].len(), 1); // c(unk)

        // Id is the vocabulary id from Unigram model
        // node_id is simply the rank of the given node in the lattice.
        assert_eq!(lattice.begin_nodes[0][0].borrow().id, 1);
        assert_eq!(lattice.begin_nodes[0][1].borrow().id, 3);
        assert_eq!(lattice.begin_nodes[1][0].borrow().id, 2);
        assert_eq!(lattice.begin_nodes[1][1].borrow().id, 4);
        assert_eq!(lattice.begin_nodes[2][0].borrow().id, 0);
        assert_eq!(lattice.begin_nodes[0][0].borrow().node_id, 2);
        assert_eq!(lattice.begin_nodes[0][1].borrow().node_id, 3);
        assert_eq!(lattice.begin_nodes[1][0].borrow().node_id, 4);
        assert_eq!(lattice.begin_nodes[1][1].borrow().node_id, 5);
        assert_eq!(lattice.begin_nodes[2][0].borrow().node_id, 6);
    }

    #[test]
    fn test_encode() {
        let sentencepieces = vec![
            ("<unk>".to_string(), 0.0),
            ("a".to_string(), 0.0),
            ("b".to_string(), 0.0),
            ("c".to_string(), 0.0),
            ("d".to_string(), 0.0),
            ("cd".to_string(), 1.0),
            ("ab".to_string(), 2.0),
            ("abc".to_string(), 5.0),
            ("abcd".to_string(), 10.0),
        ];

        let model = Unigram::from(sentencepieces, Some(0)).unwrap();
        let result = model.encode("abcd").unwrap();
        assert_eq!(result, vec!["abcd"]);
    }

    #[test]
    fn test_encode2() {
        let sentencepieces = vec![
            ("<unk>".to_string(), 0.0),
            ("ab".to_string(), 0.0),
            ("cd".to_string(), -0.1),
            ("abc".to_string(), -0.2),
            ("a".to_string(), -0.3),
            ("b".to_string(), -0.4),
            ("c".to_string(), -0.5),
            ("ABC".to_string(), -0.5),
            ("abcdabcd".to_string(), 20.0), // User defined just max the scores.
            ("q".to_string(), 20.5),
            ("r".to_string(), 20.5),
            ("qr".to_string(), -0.5),
        ];

        let mut model = Unigram::from(sentencepieces, Some(0)).unwrap();

        for is_optimized in &[true, false] {
            model.set_optimized(*is_optimized);
            println!("IsOptimized {:?}", is_optimized);
            assert_eq!(model.encode("abc").unwrap(), vec!["abc"]);
            assert_eq!(model.encode("AB").unwrap(), vec!["AB"]);

            model.set_fuse_unk(false);
            assert_eq!(model.encode("AB").unwrap(), vec!["A", "B"]);
            model.set_fuse_unk(true);
            assert_eq!(model.encode("AB").unwrap(), vec!["AB"]);

            assert_eq!(model.encode("abcd").unwrap(), vec!["ab", "cd"]);
            assert_eq!(model.encode("abcc").unwrap(), vec!["abc", "c"]);
            assert_eq!(
                model.encode("xabcabaabcdd").unwrap(),
                vec!["x", "abc", "ab", "a", "ab", "cd", "d"]
            );
            model.set_fuse_unk(false);
            assert_eq!(
                model.encode("xyz東京").unwrap(),
                vec!["x", "y", "z", "東", "京"]
            );
            model.set_fuse_unk(true);
            assert_eq!(model.encode("xyz東京").unwrap(), vec!["xyz東京"]);

            // User encoded in original version
            assert_eq!(model.encode("ABC").unwrap(), vec!["ABC"]);
            assert_eq!(model.encode("abABCcd").unwrap(), vec!["ab", "ABC", "cd"]);
            assert_eq!(
                model.encode("ababcdabcdcd").unwrap(),
                vec!["ab", "abcdabcd", "cd"]
            );
            assert_eq!(model.encode("abqrcd").unwrap(), vec!["ab", "q", "r", "cd"]);
        }
    }
}
