use crate::models::unigram::lattice::Lattice;
use crate::models::unigram::trie::{Trie, TrieBuilder};
use crate::tokenizer::{Model, Result, Token};

use std::collections::HashMap;
use std::convert::TryInto;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

type TokenMap = HashMap<String, u32>;
type Vocab = Vec<(String, f64)>;

/// A `Unigram` model to encode sentences.
#[derive(Clone)]
pub struct Unigram {
    token_to_ids: TokenMap,
    pub(crate) vocab: Vocab,
    trie: Trie<char>,
    pub min_score: f64,
    pub(super) unk_id: usize,
    pub(super) bos_id: usize,
    pub(super) eos_id: usize,

    fuse_unk: bool,
}
impl PartialEq for Unigram {
    fn eq(&self, other: &Self) -> bool {
        self.unk_id == other.unk_id && self.vocab == other.vocab
    }
}

impl std::fmt::Debug for Unigram {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("BPE")
            .field("vocab", &self.vocab.len())
            .finish()
    }
}

static K_UNK_PENALTY: f64 = 10.0;

#[derive(Debug)]
pub enum UnigramError {
    EmptyVocabulary,
    UnkIdNotInVocabulary,
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
        }
    }
}

impl std::error::Error for UnigramError {}

impl Default for Unigram {
    fn default() -> Self {
        let vocab = vec![("<unk>".to_string(), 0.0)];
        Self::from(vocab, 0).unwrap()
    }
}

impl Unigram {
    /// Create a `Unigram` model from a given vocabulary.
    /// Vocabulary are the various tokens and their associated score which is a sort of a logprob of
    /// their frequency, which will enable tokenization and sampling.
    /// unk_id, is the index within the vocabulary.
    /// For now `Unigram` *requires* at least `unk` because we might find a never seen char.
    /// Further versions might allow that part to be hidden.
    pub fn from(vocab: Vec<(String, f64)>, unk_id: usize) -> Result<Self> {
        let n = vocab.len();
        let mut token_to_ids: TokenMap = HashMap::new();
        let mut builder = TrieBuilder::default();

        if vocab.is_empty() {
            return Err(Box::new(UnigramError::EmptyVocabulary));
        }
        if unk_id >= vocab.len() {
            return Err(Box::new(UnigramError::UnkIdNotInVocabulary));
        }

        let bos_id = n + 1;
        let eos_id = n + 2;

        let mut min_score = f64::INFINITY;
        for (id, (token, score)) in vocab.iter().enumerate() {
            token_to_ids.insert(token.to_string(), id as u32);
            let chars: Vec<char> = token.chars().collect();
            builder.push(&chars);
            if score < &min_score {
                min_score = *score;
            }
        }
        let trie = builder.build();
        let fuse_unk = true;

        Ok(Unigram {
            vocab,
            token_to_ids,
            trie,
            min_score,
            bos_id,
            eos_id,
            unk_id,
            fuse_unk,
        })
    }

    #[cfg(test)]
    pub(super) fn set_fuse_unk(&mut self, fuse_unk: bool) {
        self.fuse_unk = fuse_unk;
    }

    pub(super) fn len(&self) -> usize {
        self.vocab.len()
    }

    pub(super) fn populate_nodes(&self, lattice: &mut Lattice) {
        let unk_score = self.min_score - K_UNK_PENALTY;

        let len = lattice.len();

        for begin_pos in 0..len {
            let trie_results: Vec<String> = self
                .trie
                .common_prefix_search(&lattice.chars[begin_pos..])
                .iter()
                .map(|chars| chars.iter().collect())
                .collect();

            let mut has_single_node = false;

            for tok in trie_results {
                let n = tok.chars().count();

                let id = *self.token_to_ids.get(&tok).unwrap();

                let item = &self.vocab[id as usize];
                assert_eq!(item.0, tok);
                let score: f64 = item.1;
                lattice.insert(begin_pos, n, score, id.try_into().unwrap());
                if !has_single_node && n == 1 {
                    has_single_node = true;
                }
            }

            if !has_single_node {
                lattice.insert(begin_pos, 1, unk_score, self.unk_id);
            }
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
    /// let model = Unigram::from(pieces, 0).unwrap();
    /// let result = model.encode("abcdacdxx");
    /// assert_eq!(result, vec!["abcd", "a", "cd", "xx"]);
    /// ```
    pub fn encode(&self, sentence: &str) -> Vec<String> {
        // TODO optimized version
        // https://github.com/google/sentencepiece/blob/d48247191a6d50e469ed1a4a36e877befffd1851/src/unigram_model.cc#L600
        let mut lattice = Lattice::from(sentence, self.unk_id, self.bos_id, self.eos_id);
        self.populate_nodes(&mut lattice);
        if self.fuse_unk {
            let mut results = vec![];
            let mut token = String::new();
            for node in lattice.viterbi().iter() {
                let item = lattice.piece(&node.borrow());
                if node.borrow().id == self.unk_id {
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
            results
        } else {
            lattice.tokens()
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
    /// let model = Unigram::load(Path::new("mymodel-unigram.json")).unwrap();
    /// ```
    pub fn load(path: &Path) -> Result<Unigram> {
        let file = File::open(path).unwrap();
        let reader = BufReader::new(file);
        let u = serde_json::from_reader(reader)?;
        Ok(u)
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
    fn get_vocab(&self) -> &HashMap<String, u32> {
        &self.token_to_ids
    }

    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn tokenize(&self, sentence: &str) -> Result<Vec<Token>> {
        let tokens = self.encode(sentence);
        let mut offset = 0;
        Ok(tokens
            .iter()
            .map(|string| {
                let id = self.token_to_ids.get(string).unwrap_or(&0);
                let len = string.len();
                let offsets = (offset, offset + len);
                offset += len;
                Token::new(*id, string.to_string(), offsets)
            })
            .collect())
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_ids.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<&str> {
        match self.vocab.get(id as usize) {
            Some(item) => Some(&item.0),
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_populate_nodes_unk() {
        let pieces = vec![("<unk>".to_string(), 0.0)];
        let model = Unigram::from(pieces, 0).unwrap();

        let mut lattice = Lattice::from("abc", 0, model.bos_id, model.eos_id);
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
        let model = Unigram::from(pieces, 0).unwrap();

        let mut lattice = Lattice::from("abc", 0, model.bos_id, model.eos_id);
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

        let model = Unigram::from(sentencepieces, 0).unwrap();
        let result = model.encode("abcd");
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

        let mut model = Unigram::from(sentencepieces, 0).unwrap();
        assert_eq!(model.encode("abc"), vec!["abc"]);
        assert_eq!(model.encode("AB"), vec!["AB"]);

        model.set_fuse_unk(false);
        assert_eq!(model.encode("AB"), vec!["A", "B"]);
        model.set_fuse_unk(true);

        assert_eq!(model.encode("abcd"), vec!["ab", "cd"]);
        assert_eq!(model.encode("abcc"), vec!["abc", "c"]);
        assert_eq!(
            model.encode("xabcabaabcdd"),
            vec!["x", "abc", "ab", "a", "ab", "cd", "d"]
        );
        model.set_fuse_unk(false);
        assert_eq!(model.encode("xyz東京"), vec!["x", "y", "z", "東", "京"]);
        model.set_fuse_unk(true);

        // User encoded in original version
        assert_eq!(model.encode("ABC"), vec!["ABC"]);
        assert_eq!(model.encode("abABCcd"), vec!["ab", "ABC", "cd"]);
        assert_eq!(model.encode("ababcdabcdcd"), vec!["ab", "abcdabcd", "cd"]);
        assert_eq!(model.encode("abqrcd"), vec!["ab", "q", "r", "cd"]);
    }
}
