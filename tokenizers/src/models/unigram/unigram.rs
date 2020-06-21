use crate::models::unigram::lattice::{Lattice, Node};
use crate::pre_tokenizers::whitespace::Whitespace;
use crate::tokenizer::{
    AddedToken, Model, NormalizedString, Offsets, PreTokenizer, Result, Token, Trainer,
};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::path::{Path, PathBuf};
use trie_rs::{Trie, TrieBuilder};

type Vocab = HashMap<String, u32>;
type Ids = Vec<String>;

#[derive(Serialize, Deserialize)]
pub struct Unigram<'a> {
    vocab: Vocab,
    ids: Ids,
    #[serde(skip_serializing, skip_deserializing, default = "empty_trie")]
    trie: Trie<&'a str>,
    min_score: f64,
}
impl std::fmt::Debug for Unigram {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("BPE")
            .field("vocab", &self.vocab.len())
            .finish()
    }
}

fn empty_trie<'a>() -> Trie<&'a str> {
    TrieBuilder::new().build()
}

static K_UNK_PENALTY: f64 = 10.0;
static UNK_ID: usize = 2;

impl Unigram {
    pub fn new() -> Self {
        Unigram {
            vocab: HashMap::new(),
            ids: vec![],
            trie: empty_trie(),
            min_score: 0.0,
        }
    }
    pub fn populate_nodes<'a>(&'a mut self, lattice: &mut Lattice) {
        //TODO
        //  auto get_chars_length = [&lattice](int begin_pos, const char *end) {
        //   int pos = begin_pos;
        //   while (lattice->surface(pos) < end) ++pos;
        //   return pos - begin_pos;
        // };
        let unk_score = self.min_score - K_UNK_PENALTY;

        // const float unk_score = min_score() - kUnkPenalty;

        // const int len = lattice->size();
        let len = lattice.len();
        // const char *end = lattice->sentence() + lattice->utf8_size();

        // // +1 just in case.
        // std::vector<Darts::DoubleArray::result_pair_type> trie_results(
        //     trie_results_size_ + 1);

        for begin_pos in 0..len {
            let trie_results: Vec<Vec<&str>> = self
                .trie
                .common_prefix_search(&lattice.graphemes[begin_pos..]);

            let mut has_single_node = false;

            for result in trie_results {
                // TODO score comes from proto id.
                let score: f64 = *self.vocab.get(&result.concat()).unwrap() as f64;
                lattice.insert(best_pos, result.len(), score);
                if !has_single_node && node.length == 1 {
                    has_single_node = true;
                }
            }

            if !has_single_node {
                lattice.insert_with_id(begin_pos, 1, unk_score, UNK_ID);
            }
        }
        // for (int begin_pos = 0; begin_pos < len; ++begin_pos) {
        //   const char *begin = lattice->surface(begin_pos);

        //   // Finds all pieces which are prefix of surface(begin_pos).
        //   const size_t num_nodes = trie_->commonPrefixSearch(
        //       begin, trie_results.data(), trie_results.size(),
        //       static_cast<int>(end - begin));
        //   CHECK_LT(num_nodes, trie_results.size());

        //   bool has_single_node = false;

        //   // Inserts pieces to the lattice.
        //   for (size_t k = 0; k < num_nodes; ++k) {
        //     const int length =
        //         get_chars_length(begin_pos, begin + trie_results[k].length);
        //     const int id = trie_results[k].value;
        //     if (IsUnusedInlined(id)) continue;
        //     Lattice::Node *node = lattice->Insert(begin_pos, length);
        //     node->id = id;  // the value of Trie stores vocab_id.
        //     // User defined symbol receives extra bonus to always be selected.
        //     node->score = IsUserDefinedInlined(id) ? (length * max_score_ - 0.1)
        //                                            : GetScoreInlined(id);
        //     if (!has_single_node && node->length == 1) {
        //       has_single_node = true;
        //     }
        //   }

        //   if (!has_single_node) {
        //     Lattice::Node *node = lattice->Insert(begin_pos, 1);
        //     node->id = unk_id_;  // add UNK node.
        //     node->score = unk_score;
        //   }
        // }
    }
    pub fn encode(&self, sentence: &str) -> Vec<String> {
        // let pretokenizer = Whitespace;
        // let mut input = NormalizedString::from(sentence);
        // let encoded = pretokenizer.pre_tokenize(&mut input)?;
        // self.tokenize(encoded)
        // TODO optimized version
        let mut lattice = Lattice::from(sentence);
        self.populate_nodes(&mut lattice);
        lattice.tokens()
    }
}

#[typetag::serde]
impl Model for Unigram {
    fn get_vocab(&self) -> &HashMap<String, u32> {
        &self.vocab
    }

    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn tokenize(&self, sentence: Vec<(String, Offsets)>) -> Result<Vec<Token>> {
        Ok(vec![])
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        None
    }

    fn save(&self, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>> {
        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode() {
        // let mut proto = make_model_proto();
        // proto.add_piece("abcd", 10.0); // 3
        // proto.add_piece("abc", 5.0); // 4
        // proto.add_piece("ab", 2.0); // 5
        // proto.add_piece("cd", 1.0); // 6
        // proto.add_piece("a", 0.0); // 7
        // proto.add_piece("b", 0.0); // 8
        // proto.add_piece("c", 0.0); // 9
        // proto.add_piece("d", 0.0); // 10

        //TODO
        let model = Unigram::new();
        let result = model.encode("abcd");
        let token = Token::new(0u32, "u".into(), (0, 4), 0);
        assert_eq!(result, vec!["abcd"]);
    }
}
