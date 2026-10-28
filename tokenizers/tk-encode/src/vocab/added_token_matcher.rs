use std::ops::Range;

use aho_corasick::{AhoCorasick, MatchKind};

use crate::Result;

use super::bucket_vocab_store::BucketVocabStore;

#[derive(Clone, PartialEq, Debug, Default)]
pub struct AddedTokenFlags {
    pub special: bool,
    pub normalized: bool,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
}

/// Finds added tokens in text, and maps them to and from their ids.
///
/// Matching runs an [`AhoCorasick`] automaton over every token with
/// [`MatchKind::LeftmostLongest`]: the match that starts first wins, and of those starting at the
/// same byte, the longest.
#[derive(Clone, Default)]
pub struct AddedTokenMatcher {
    /// `None` without tokens. An automaton with no patterns has no prefilter and walks every byte.
    automaton: Option<AhoCorasick>,
    /// The token id of each automaton pattern, indexed by pattern id.
    pattern_ids: Box<[u32]>,
    vocab: BucketVocabStore,
}

impl AddedTokenMatcher {
    pub fn new(tokens: Vec<(Vec<u8>, u32)>) -> Result<Self> {
        if tokens.is_empty() {
            return Ok(Self::default());
        }
        let vocab = BucketVocabStore::build(tokens)?;
        // The vocab keeps one id per token, so a token listed twice matches as `token_to_id` says.
        let tokens = vocab.byte_content();
        let automaton = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(tokens.iter().map(|(bytes, _)| bytes))?;
        let pattern_ids = tokens.into_iter().map(|(_, id)| id).collect();
        Ok(Self {
            automaton: Some(automaton),
            pattern_ids,
            vocab,
        })
    }

    /// The leftmost-longest added token in `bytes`: its id and its byte range.
    pub fn find(&self, bytes: &[u8]) -> Option<(u32, Range<usize>)> {
        let found = self.automaton.as_ref()?.find(bytes)?;
        Some((self.pattern_ids[found.pattern()], found.range()))
    }

    pub fn len(&self) -> usize {
        self.vocab.len()
    }
    pub fn is_empty(&self) -> bool {
        self.vocab.is_empty()
    }
    pub fn get_vocab_bytes(&self) -> Vec<(Vec<u8>, u32)> {
        self.vocab.byte_content()
    }
    pub fn get_vocab(&self) -> Vec<(String, u32)> {
        self.vocab.content()
    }
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.token_to_id(token)
    }
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab.id_to_token(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matcher(tokens: &[(&str, u32)]) -> AddedTokenMatcher {
        let tokens = tokens.iter().map(|(t, id)| (t.as_bytes().to_vec(), *id));
        AddedTokenMatcher::new(tokens.collect()).unwrap()
    }

    #[test]
    fn finds_the_leftmost_token() {
        let m = matcher(&[("<|eos|>", 0), ("[CLS]", 1), ("|SLS]", 2)]);
        assert_eq!(m.find(b"kwown<s><|eos|>[CLS]"), Some((0, 8..15)));
        assert_eq!(m.find(b"x|SLS]<|eos|>"), Some((2, 1..6)));
        assert_eq!(m.find(b"no token here"), None);
    }

    #[test]
    fn prefers_the_longest_token_at_the_same_start() {
        let m = matcher(&[("<|im", 0), ("<|im_start|>", 1), ("<|im_end|>", 2)]);
        assert_eq!(m.find(b"a<|im_start|>"), Some((1, 1..13)));
        assert_eq!(m.find(b"a<|im_sta"), Some((0, 1..5)));
    }

    #[test]
    fn a_leftmost_short_token_beats_a_later_long_one() {
        let m = matcher(&[("ab", 0), ("bcdef", 1)]);
        assert_eq!(m.find(b"abcdef"), Some((0, 0..2)));
    }

    #[test]
    fn a_token_listed_twice_matches_with_the_id_token_to_id_gives() {
        let m = matcher(&[("<s>", 3), ("<s>", 7)]);
        let id = m.token_to_id("<s>").unwrap();
        assert_eq!(m.find(b"a<s>"), Some((id, 1..4)));
    }

    #[test]
    fn an_empty_matcher_finds_nothing() {
        let m = AddedTokenMatcher::default();
        assert!(m.is_empty());
        assert_eq!(m.find(b"<|eos|>"), None);
    }
}
