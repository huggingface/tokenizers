use super::{
    normalizer::Range, Model, NormalizedString, Normalizer, Offsets, PreTokenizedString, Token,
};
use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use regex::Regex;
use serde::{ser::SerializeSeq, Deserialize, Serialize, Serializer};
use std::collections::{HashMap, HashSet};

/// Represent a token added by the user on top of the existing Model vocabulary.
/// AddedToken can be configured to specify the behavior they should have in various situations
/// like:
///   - Whether they should only match single words
///   - Whether to include any whitespace on its left or right
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AddedToken {
    /// The content of the added token
    pub content: String,
    /// Whether this token must be a single word or can break words
    pub single_word: bool,
    /// Whether this token should strip whitespaces on its left
    pub lstrip: bool,
    /// Whether this token should strip whitespaces on its right
    pub rstrip: bool,
    /// Whether this token should be normalized
    pub normalized: bool,
    /// Whether this token is special
    pub special: bool,
}

impl AddedToken {
    /// Build this token from the given content, specifying if it is intented to be a
    /// special token. Special tokens are not normalized by default.
    pub fn from<S: Into<String>>(content: S, special: bool) -> Self {
        Self {
            content: content.into(),
            normalized: !special,
            special,
            ..Default::default()
        }
    }
    /// Specify whether this token should only match on whole single words, and never
    /// part of a word.
    #[must_use]
    pub fn single_word(mut self, single_word: bool) -> Self {
        self.single_word = single_word;
        self
    }
    /// Specify whether this token should include all the whitespaces on its left, in
    /// order to strip them out.
    #[must_use]
    pub fn lstrip(mut self, lstrip: bool) -> Self {
        self.lstrip = lstrip;
        self
    }
    /// Specify whether this token should include all the whitespaces on its right, in
    /// order to strip them out.
    #[must_use]
    pub fn rstrip(mut self, rstrip: bool) -> Self {
        self.rstrip = rstrip;
        self
    }
    /// Specify whether this token should be normalized and match against its normalized
    /// version in the input text.
    #[must_use]
    pub fn normalized(mut self, normalized: bool) -> Self {
        self.normalized = normalized;
        self
    }
    /// Specify whether this token is special, meaning if it should be skipped when decoding
    #[must_use]
    pub fn special(mut self, special: bool) -> Self {
        self.special = special;
        self
    }
}
impl Default for AddedToken {
    fn default() -> Self {
        Self {
            content: String::new(),
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: true,
            special: false,
        }
    }
}
// AddedTokens can be updated if value changed
impl std::hash::Hash for AddedToken {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.content.hash(state);
    }
}

type MatchingSet = (AhoCorasick, Vec<u32>);

lazy_static! {
    static ref STARTS_WITH_WORD: Regex = Regex::new(r"^\w").unwrap();
    static ref ENDS_WITH_WORD: Regex = Regex::new(r"\w$").unwrap();
    static ref RIGHTMOST_SPACE_AT_START: Regex = Regex::new(r"^\s*").unwrap();
    static ref LEFTMOST_SPACE_AT_END: Regex = Regex::new(r"\s*$").unwrap();
}

fn ends_with_word(sentence: &str) -> bool {
    ENDS_WITH_WORD.is_match(sentence)
}

fn starts_with_word(sentence: &str) -> bool {
    STARTS_WITH_WORD.is_match(sentence)
}

fn space_leftmost_at_end(sentence: &str) -> usize {
    if let Some(match_) = LEFTMOST_SPACE_AT_END.find(sentence) {
        match_.start()
    } else {
        sentence.len()
    }
}
fn space_rightmost_at_start(sentence: &str) -> usize {
    if let Some(match_) = RIGHTMOST_SPACE_AT_START.find(sentence) {
        match_.end()
    } else {
        0
    }
}
///
/// A vocabulary built on top of the Model
///
/// This provides a way to add new vocabulary to a Tokenizer that has already been trained,
/// in a previous process, maybe by someone else. This is especially interesting in the case
/// of fine-tunings, where we want to finetune a model while adding some new functionalities
/// using some new special tokens, or maybe add some tokens in the case of unknown tokens, etc.
///
/// One of the reasons we need to handle these tokens outside of the model is simply that
/// for many models, it is not possible to add new tokens after the training process. For example,
/// using BPE, the training process generates merges pairs along the vocabulary, and any token
/// in the vocabulary can be decomposed in other tokens, down to the original alphabet. If we
/// were to add new tokens after this training process, we couldn't make sure the merges pairs
/// exist as required.
///
#[derive(Clone, Debug)]
pub struct AddedVocabulary {
    /// Contains the mapping from String (token content) to ID. This map contains both special
    /// tokens and classic added tokens that were added to the this vocabulary.
    added_tokens_map: HashMap<String, u32>,
    /// Contains the mapping from ID to AddedToken for all the added tokens, both special
    /// and classic.
    added_tokens_map_r: HashMap<u32, AddedToken>,

    /// Contains only the classic AddedToken, in the specific order the user gave them.
    added_tokens: Vec<AddedToken>,
    /// Contains only the special AddedToken, in the specific order the user gave them.
    special_tokens: Vec<AddedToken>,

    /// A Set, containing all the special token for easy access while decoding. This let's
    /// us remove them easily with an O(1) complexity.
    special_tokens_set: HashSet<String>,

    /// A RegexSet containing all the non-normalized patterns used to split on AddedTokens
    split_trie: MatchingSet,
    /// A RegexSet containing all the normalized patterns used to split on AddedTokens
    split_normalized_trie: MatchingSet,

    /// Whether or not special tokens should be splitted when encoding. This is equivalent to ignoring them
    encode_special_tokens: bool,
}

impl AddedVocabulary {
    pub fn new() -> Self {
        let trie = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build::<_, &&[u8]>([])
            .expect("The trie should build correctly");
        let normalized_trie = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build::<_, &&[u8]>([])
            .expect("The normalized trie should build correctly");
        Self {
            added_tokens_map: HashMap::new(),
            added_tokens_map_r: HashMap::new(),
            added_tokens: vec![],
            special_tokens: vec![],
            special_tokens_set: HashSet::new(),
            split_trie: (trie, vec![]),
            split_normalized_trie: (normalized_trie, vec![]),
            encode_special_tokens: false,
        }
    }
    /// Size of the additional vocabulary
    #[allow(dead_code)] // Suppress the "method is never used" warning
    pub fn len(&self) -> usize {
        self.added_tokens_map.len()
    }

    /// Whether or not this vocabulary is empty
    pub fn is_empty(&self) -> bool {
        self.added_tokens_map.is_empty()
    }

    /// Get the additional vocabulary
    pub fn get_vocab(&self) -> &HashMap<String, u32> {
        &self.added_tokens_map
    }

    /// Get the additional vocabulary with the AddedTokens
    pub fn get_added_tokens_decoder(&self) -> &HashMap<u32, AddedToken> {
        &self.added_tokens_map_r
    }

    /// Get the id matching one of our token if it exists
    pub fn token_to_id(&self, token: &str, model: &impl Model) -> Option<u32> {
        self.added_tokens_map
            .get(token)
            .copied()
            .or_else(|| model.token_to_id(token))
    }

    /// Get the token matching the given id if it exists
    #[deprecated(
        since = "0.19.0",
        note = "please use `added_vocabulary.simple_id_to_token(id).or_else(|| model.id_to_token(id)` instead"
    )]
    pub fn id_to_token(&self, id: u32, model: &impl Model) -> Option<String> {
        self.added_tokens_map_r
            .get(&id)
            .map(|t| t.content.clone())
            .or_else(|| model.id_to_token(id))
    }

    pub fn simple_id_to_token(&self, id: u32) -> Option<String> {
        self.added_tokens_map_r.get(&id).map(|t| t.content.clone())
    }

    //
    pub fn set_encode_special_tokens(&mut self, value: bool) {
        self.encode_special_tokens = value;
    }

    pub fn get_encode_special_tokens(&self) -> bool {
        self.encode_special_tokens
    }

    /// Check if a token is a special token
    pub fn is_special_token(&self, token: &str) -> bool {
        self.special_tokens_set.contains(token)
    }

    /// Add some special tokens to the vocabulary
    pub fn add_special_tokens<N: Normalizer>(
        &mut self,
        tokens: &[AddedToken],
        model: &impl Model,
        normalizer: Option<&N>,
    ) -> usize {
        self.add_tokens(tokens, model, normalizer)
    }

    /// Add some tokens to the vocabulary
    pub fn add_tokens<N: Normalizer>(
        &mut self,
        tokens: &[AddedToken],
        model: &impl Model,
        normalizer: Option<&N>,
    ) -> usize {
        // Handle special tokens (if any)
        for token in tokens {
            if token.special
                && !token.content.is_empty()
                && !self.special_tokens_set.contains(&token.content)
            {
                self.special_tokens.push(token.to_owned());
                self.special_tokens_set.insert(token.content.clone());
            }
        }

        // Then we delegate to `add_tokens`, that will take care of refreshing added tokens too.
        let mut ignored = 0;
        for token in tokens {
            if token.content.is_empty() || self.added_tokens_map_r.values().any(|val| val == token)
            {
                ignored += 1;
                continue;
            }
            // If a token is already part of the vocabulary, we mark it as added
            let new_id = if let Some(new_id) = self.token_to_id(&token.content, model) {
                new_id
            } else {
                self.added_tokens_map.values().cloned().max().map_or(
                    model.get_vocab_size() as u32,
                    |max| {
                        if (max >= model.get_vocab_size() as u32) || model.get_vocab_size() == 0 {
                            max + 1
                        } else {
                            model.get_vocab_size() as u32
                        }
                    },
                )
            };
            // Make sure we modify the previous entry
            self.added_tokens_map
                .entry(token.content.clone())
                .and_modify(|old_id| *old_id = new_id)
                .or_insert_with(|| new_id);
            // Update the current revert operation
            self.added_tokens_map_r
                .entry(new_id)
                .and_modify(|t| *t = token.clone())
                .or_insert_with(|| token.clone());
            // Make sure to remove previous entry (if the token gets a new id)

            // Finally add the token to the classic set if special
            if !self.special_tokens_set.contains(&token.content) {
                self.added_tokens.push(token.clone());
            }
        }

        self.refresh_added_tokens(model, normalizer);

        // Return the number of added tokens
        tokens.len() - ignored
    }

    /// Reconstruct our internal RegexSet when new tokens are added to the vocabulary.
    ///
    /// We keep two different RegexSet, one that will take care of matching against the
    /// non-normalized string, and one matching against the normalized one.
    fn refresh_added_tokens<N: Normalizer>(&mut self, model: &impl Model, normalizer: Option<&N>) {
        type TupleTokenId<'a> = (&'a AddedToken, u32);
        let (normalized, non_normalized): (Vec<TupleTokenId>, Vec<TupleTokenId>) = self
            .special_tokens
            .iter()
            .chain(self.added_tokens.iter())
            .map(|token| {
                (
                    token,
                    self.token_to_id(&token.content, model)
                        .expect("Missing additional token"),
                )
            })
            .partition(|(token, _)| token.normalized);

        let (tokens, ids): (Vec<&AddedToken>, Vec<u32>) = non_normalized.into_iter().unzip();
        let trie = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build(tokens.iter().map(|token| &token.content))
            .expect("Failed to build tried when refreshing tokens");
        self.split_trie = (trie, ids);

        let (ntokens, nids): (Vec<&AddedToken>, Vec<u32>) = normalized.into_iter().unzip();
        let patterns: Vec<_> = ntokens
            .iter()
            .map(|token| {
                let mut content = NormalizedString::from(token.content.as_ref());
                if let Some(n) = normalizer {
                    n.normalize(&mut content).unwrap();
                }
                content
            })
            .collect();
        let normalized_trie = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build(patterns.iter().map(|content| content.get()))
            .expect("Failed to build tried when refreshing tokens (normalized)");
        self.split_normalized_trie = (normalized_trie, nids);
    }

    /// Find any AddedToken in the given sentence, using the provided MatchingSet.
    /// This method returns a list "splits", each of them being a pair of Offsets
    /// and an optional ID if it is an AddedToken.
    /// The list of splits cover the entire input string.
    fn find_matches(&self, sentence: &str, split_re: &MatchingSet) -> Vec<(Option<u32>, Offsets)> {
        if sentence.is_empty() {
            return vec![(None, (0, 0))];
        }

        let mut start_offset = 0;
        let mut splits = vec![];

        for mat in split_re.0.find_iter(sentence) {
            let mut start = mat.start();
            let mut stop = mat.end();
            let aho_id = mat.pattern();
            let id = split_re.1[aho_id];
            let added_token = &self.added_tokens_map_r.get(&id).unwrap();

            if self.encode_special_tokens && self.special_tokens_set.contains(&added_token.content)
            {
                continue;
            }

            if added_token.single_word {
                let start_space = start == 0 || !ends_with_word(&sentence[..start]);
                let stop_space = stop == sentence.len() || !starts_with_word(&sentence[stop..]);

                if !stop_space || !start_space {
                    // Discard not single word
                    continue;
                }
            }
            if added_token.lstrip {
                // This will be strictly inferior to start and in correct sentence offset
                let newstart = space_leftmost_at_end(&sentence[..start]);

                // The previous match could have already matched those spaces
                // Ignore them if it's already matched
                start = std::cmp::max(newstart, start_offset);
            }
            if added_token.rstrip {
                // This will starting a the stop+1 character, so we need
                // to add the previous stop value
                stop += space_rightmost_at_start(&sentence[stop..])
            }
            if start_offset < start {
                splits.push((None, (start_offset, start)));
            }
            splits.push((Some(id), (start, stop)));
            start_offset = stop;
        }

        let total_byte_len = sentence.len();
        if start_offset != total_byte_len {
            splits.push((None, (start_offset, total_byte_len)));
        }

        splits
    }

    /// Split the input sentence to extract anything we found from the `MatchingSet`, as well as
    /// the list of corresponding IDs
    /// The list of IDs have the exact same number of elements than the Iterator.
    fn split_with_indices(
        &self,
        sentence: NormalizedString,
        split_re: &MatchingSet,
    ) -> Vec<(NormalizedString, Option<Vec<Token>>)> {
        self.find_matches(sentence.get(), split_re)
            .into_iter()
            .map(|(id, byte_offsets)| {
                let slice = sentence
                    .slice(Range::Normalized(byte_offsets.0..byte_offsets.1))
                    .expect("AddedVocabulary bad split");
                if let Some(id) = id {
                    let value = slice.get().to_owned();
                    let len = value.len();
                    (slice, Some(vec![Token::new(id, value, (0, len))]))
                } else {
                    (slice, None)
                }
            })
            .collect()
    }

    /// Extract the additional vocabulary from the given sentence, normalizing it along the way.
    ///
    /// Some tokens should match against their normalized representation, as well as the
    /// non-normalized one. For example, when we expect to extract the token `yesterday` in the
    /// input sentence `I read a book Yesterday`, if the normalizer is supposed to lowercase
    /// everything, we expect a match.
    pub fn extract_and_normalize<N: Normalizer>(
        &self,
        normalizer: Option<&N>,
        sequence: &str,
    ) -> PreTokenizedString {
        let mut pretokenized: PreTokenizedString = sequence.into();

        // 1. We extract all the non-normalized tokens from the non-normalized string
        pretokenized
            .split(|_, sequence| Ok(self.split_with_indices(sequence, &self.split_trie)))
            .expect("AddedVocabulary bad split");

        // <s> normalized = False
        // "I read a book   <s>Hey" -> "I read a book", "   <s>", "Hey"

        // </s> normalized = True -> "▁</s>"
        // "I read a book</s>Hey" -> "I read a book</s>Hey"

        // Day normalized = True -> "Day"
        // "I read a book monday" -> "I read a book monday"

        // [DAY] normalized = False -> "Day"
        // "I read a [DAY] monday" -> "I read a " "[DAY]", "book monday"
        //                                         320055
        // 2. Then extract the normalized tokens from the normalized pieces of the string
        pretokenized
            .split(|_, mut sequence| {
                normalizer.map(|n| n.normalize(&mut sequence));
                Ok(self.split_with_indices(sequence, &self.split_normalized_trie))
            })
            .expect("AddedVocabulary bad split");

        // ["I read a book", "   <s>", "Hey"] -> ["▁I read a book", "▁   <s>", "▁Hey"]
        // ["▁I read a book", "▁   <s>", "▁Hey"] -> [.., "▁   ", "<s>", "▁Hey"]

        // </s> normalized = True -> "▁</s>"
        // "I read a book</s>Hey" -> ["▁I read a book", "<","/","s",">", "Hey"]

        // "I read a " "[DAY]", "book monday" -> "i read a " "[day]", "book monday"

        pretokenized
    }
}

impl Default for AddedVocabulary {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct AddedTokenWithId {
    /// The id assigned to this token
    pub id: u32,
    #[serde(flatten)]
    /// The target AddedToken
    pub token: AddedToken,
}

impl Serialize for AddedVocabulary {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut added_tokens = self
            .added_tokens_map_r
            .iter()
            .map(|(id, token)| AddedTokenWithId {
                id: *id,
                token: token.clone(),
            })
            .collect::<Vec<_>>();
        // We need to have these added tokens ordered by ascending ID
        added_tokens.sort_unstable_by_key(|o| o.id);

        let mut vocabulary = serializer.serialize_seq(Some(added_tokens.len()))?;
        for token in added_tokens {
            vocabulary.serialize_element(&token)?;
        }

        vocabulary.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalizers::byte_level::ByteLevel as ByteLevelNormalizer;
    use crate::normalizers::utils::Lowercase;
    use crate::normalizers::NormalizerWrapper;
    use crate::{OffsetReferential, OffsetType, Result, Token, Trainer};
    use std::path::{Path, PathBuf};

    #[derive(Serialize, Deserialize)]
    struct ModelMock {
        vocab: HashMap<String, u32>,
        vocab_r: HashMap<u32, String>,
    }
    impl ModelMock {
        pub fn new<I>(iter: I) -> Self
        where
            I: IntoIterator<Item = &'static (&'static str, u32)>,
        {
            let vocab: HashMap<String, u32> = iter
                .into_iter()
                .map(|&(tok, id)| (tok.to_string(), id))
                .collect();
            Self {
                vocab_r: vocab
                    .iter()
                    .map(|(tok, id)| (*id, tok.to_owned()))
                    .collect(),
                vocab,
            }
        }
    }

    fn simplify_output(result: &'_ PreTokenizedString) -> Vec<(&'_ str, Option<Vec<u32>>)> {
        result
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(s, _, tokens)| {
                (
                    s,
                    tokens
                        .as_ref()
                        .map(|t| t.iter().map(|t| t.id).collect::<Vec<_>>()),
                )
            })
            .collect::<Vec<_>>()
    }

    struct TrainerMock;
    impl Trainer for TrainerMock {
        type Model = ModelMock;
        fn should_show_progress(&self) -> bool {
            true
        }
        fn train(&self, _model: &mut ModelMock) -> Result<Vec<AddedToken>> {
            unimplemented!()
        }
        fn feed<I, S, F>(&mut self, _iterator: I, _process: F) -> Result<()>
        where
            I: Iterator<Item = S> + Send,
            S: AsRef<str> + Send,
            F: Fn(&str) -> Result<Vec<String>> + Sync,
        {
            unimplemented!()
        }
    }

    impl Model for ModelMock {
        type Trainer = TrainerMock;

        fn tokenize(&self, _sequence: &str) -> Result<Vec<Token>> {
            unimplemented!()
        }
        fn token_to_id(&self, token: &str) -> Option<u32> {
            self.vocab.get(token).copied()
        }
        fn id_to_token(&self, id: u32) -> Option<String> {
            self.vocab_r.get(&id).cloned()
        }
        fn get_vocab(&self) -> HashMap<String, u32> {
            self.vocab.clone()
        }
        fn get_vocab_size(&self) -> usize {
            self.vocab.len()
        }
        fn save(&self, _folder: &Path, _name: Option<&str>) -> Result<Vec<PathBuf>> {
            unimplemented!()
        }
        fn get_trainer(&self) -> Self::Trainer {
            TrainerMock
        }
    }

    #[test]
    fn can_add_tokens() {
        let model = ModelMock::new(&[("test", 0), ("tost", 1)]);
        let mut vocab = AddedVocabulary::new();
        let normalizer: Option<&NormalizerWrapper> = None;

        // Add tokens normally
        assert_eq!(
            vocab.add_tokens(
                &[AddedToken::from("added_token_1", false)],
                &model,
                normalizer
            ),
            1
        );

        let vocab_len: usize = vocab.len();
        assert_eq!(vocab_len, 1);

        // Does not add multiple time the same token
        assert_eq!(
            vocab.add_tokens(
                &[
                    AddedToken::from("added_token_2", false),
                    AddedToken::from("added_token_2", false)
                ],
                &model,
                normalizer
            ),
            1
        );
        assert_eq!(vocab.len(), 2);

        // Also adds tokens already covered by the model
        let added_token = AddedToken::from("test", false);
        assert_eq!(
            vocab.add_tokens(&[added_token.clone()], &model, normalizer),
            1
        );
        assert_eq!(vocab.len(), 3);

        assert_eq!(vocab.get_added_tokens_decoder()[&0], added_token);
    }

    #[test]
    fn can_add_special_tokens() {
        let model = ModelMock::new(&[("test", 0), ("tost", 1)]);
        let mut vocab = AddedVocabulary::new();
        let normalizer: Option<&NormalizerWrapper> = None;
        // Add tokens normally
        assert_eq!(
            vocab.add_special_tokens(
                &[AddedToken::from("added_token_1", true)],
                &model,
                normalizer
            ),
            1
        );
        assert_eq!(vocab.len(), 1);

        // Does not add multiple time the same token
        assert_eq!(
            vocab.add_special_tokens(
                &[
                    AddedToken::from("added_token_2", true),
                    AddedToken::from("added_token_2", true)
                ],
                &model,
                normalizer
            ),
            1
        );
        assert_eq!(vocab.len(), 2);

        // Can add tokens already covered by the model
        assert_eq!(
            vocab.add_special_tokens(&[AddedToken::from("test", true)], &model, normalizer),
            1
        );
        assert_eq!(vocab.len(), 3); // New token was added
        assert!(vocab.is_special_token("test"));
        assert_eq!(
            *vocab.get_added_tokens_decoder(),
            HashMap::from([
                (0, AddedToken::from("test", true)),
                (2, AddedToken::from("added_token_1", true)),
                (3, AddedToken::from("added_token_2", true)),
            ])
        );
        assert!(vocab.added_tokens_map.contains_key("test"));
        assert!(vocab.added_tokens_map_r.contains_key(&0));

        vocab.add_tokens(
            &[
                AddedToken::from("tost", true),
                AddedToken::from("another_two", false),
            ],
            &model,
            normalizer,
        );
        assert_eq!(vocab.len(), 5); // New token was added
        assert_eq!(vocab.get_vocab()["another_two"], 4); // New token was added, but the index is not the length of the vocab

        // Let's add an already added token again
        assert_eq!(
            vocab.add_special_tokens(&[AddedToken::from("another_two", true)], &model, normalizer),
            1
        );
        assert_eq!(vocab.len(), 5); // Token was already there
        assert_eq!(vocab.get_vocab()["another_two"], 4); // Token idx not changed

        // Just checking that we can set the content of the string in rust
        let mut token: AddedToken = AddedToken::from("Hey", false);
        token.content = "hey".to_string();
        assert_eq!(token.content, "hey"); // Token was already there

        token.special = true;
        assert!(token.special); // Token was already there
    }

    #[test]
    fn can_extract_added_tokens() {
        // Is able to extract both normal and special tokens
        let model = ModelMock::new(&[]);
        let mut vocab = AddedVocabulary::new();
        let normalizer: Option<&NormalizerWrapper> = None;

        vocab.add_tokens(
            &[
                AddedToken::from("my", false),
                AddedToken::from("name", false),
            ],
            &model,
            normalizer,
        );
        vocab.add_special_tokens(
            &[
                AddedToken::from("[CLS]", true),
                AddedToken::from("[SEP]", true),
            ],
            &model,
            normalizer,
        );

        let result = vocab.extract_and_normalize(normalizer, "[CLS] My name is Anthony [SEP]");
        assert_eq!(
            result
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, _, tokens)| (
                    s,
                    tokens
                        .as_ref()
                        .map(|t| t.iter().map(|t| t.id).collect::<Vec<_>>())
                ))
                .collect::<Vec<_>>(),
            vec![
                ("[CLS]", Some(vec![2])),
                (" My ", None),
                ("name", Some(vec![1])),
                (" is Anthony ", None),
                ("[SEP]", Some(vec![3]))
            ]
        );
    }

    #[test]
    fn options_use_cases() {
        // Is able to extract both normal and special tokens, with various options (lstrip, rstrip,
        // single_word, normalized)
        let model = ModelMock::new(&[]);
        let normalizer = Lowercase;
        let mut vocab = AddedVocabulary::new();

        vocab.add_tokens(
            &[
                AddedToken::from("my", false).lstrip(true).rstrip(true),
                AddedToken::from("name", false),
                AddedToken::from("ony", false).single_word(true),
            ],
            &model,
            Some(&normalizer),
        );
        vocab.add_special_tokens(
            &[
                AddedToken::from("[CLS]", true),
                AddedToken::from("[SEP]", true),
            ],
            &model,
            Some(&normalizer),
        );

        let result =
            vocab.extract_and_normalize(Some(&normalizer), "[CLS] My name is Anthony [SEP]");

        assert_eq!(
            simplify_output(&result),
            vec![
                ("[CLS]", Some(vec![3])),
                // This one includes both spaces because of the lstrip & rstrip
                // And it matches because normalized == true
                (" my ", Some(vec![0])),
                ("name", Some(vec![1])),
                // `ony` is not extracted here thanks to single_word
                (" is anthony ", None),
                ("[SEP]", Some(vec![4])),
            ]
        );
    }

    #[test]
    fn empty_matches() {
        let vocab = AddedVocabulary::new();
        let matches = vocab.find_matches("", &vocab.split_trie);
        assert_eq!(matches, vec![(None, (0, 0))]);
    }

    #[test]
    fn test_single_word_is_correct() {
        // Is able to extract both normal and special tokens, with various options (lstrip, rstrip,
        // single_word, normalized)
        let model = ModelMock::new(&[]);
        let mut vocab = AddedVocabulary::new();
        let normalizer = Lowercase;

        vocab.add_tokens(
            &[AddedToken::from("<mask>", false).single_word(true)],
            &model,
            Some(&normalizer),
        );
        // Left, in the middle, non single world left, non single word right, end of sentence valid
        let result = vocab.extract_and_normalize(
            Some(&normalizer),
            "<mask> My name <mask> A<mask> <mask>ony <mask>",
        );
        assert_eq!(
            simplify_output(&result),
            vec![
                ("<mask>", Some(vec![0])),
                (" my name ", None),
                ("<mask>", Some(vec![0])),
                (" a<mask> <mask>ony ", None),
                ("<mask>", Some(vec![0]))
            ]
        );
    }

    #[test]
    fn test_single_word_is_unicode_correct() {
        let model = ModelMock::new(&[]);
        let mut vocab = AddedVocabulary::new();
        let normalizer = Lowercase;

        assert_eq!(vocab.len(), 0);

        vocab.add_tokens(
            &[AddedToken::from("<mask>", false).single_word(true)],
            &model,
            Some(&normalizer),
        );
        let result = vocab.extract_and_normalize(Some(&normalizer), "<mask>, <mask>- ◌̰<mask>");
        assert_eq!(
            simplify_output(&result),
            vec![
                // Punctuation is not word
                ("<mask>", Some(vec![0])),
                (", ", None),
                // dash is not word
                ("<mask>", Some(vec![0])),
                // This is unicode combining mark character and is word: https://en.wikipedia.org/wiki/Combining_Diacritical_Marks
                ("- ◌̰<mask>", None),
            ]
        );
    }

    #[test]
    fn test_lstrip_unicode_space() {
        let model = ModelMock::new(&[]);
        let mut vocab = AddedVocabulary::new();
        let normalizer = Lowercase;

        vocab.add_tokens(
            &[AddedToken::from("<mask>", false)
                .lstrip(true)
                .rstrip(true)
                .single_word(true)],
            &model,
            Some(&normalizer),
        );
        let result = vocab
            .extract_and_normalize(Some(&normalizer), "Hi <mask> there\t<mask>\t<mask>\u{2000}");
        assert_eq!(
            simplify_output(&result),
            vec![
                ("hi", None),
                // Regular space
                (" <mask> ", Some(vec![0])),
                ("there", None),
                // \t is a spacing character
                ("\t<mask>\t", Some(vec![0])),
                // Non overlapping
                // \u{2000} is mongolian vowel separator: https://jkorpela.fi/chars/spaces.html
                ("<mask>\u{2000}", Some(vec![0])),
            ]
        );
    }

    #[test]
    fn test_encode_special_tokens() {
        let model = ModelMock::new(&[]);
        let mut vocab = AddedVocabulary::new();
        let normalizer = Lowercase;

        vocab.add_tokens(
            &[
                AddedToken::from("<mask>", true)
                    .lstrip(true)
                    .rstrip(true)
                    .single_word(true),
                AddedToken::from("ask>", false),
                AddedToken::from("<pad>", true),
            ],
            &model,
            Some(&normalizer),
        );
        vocab.set_encode_special_tokens(true);

        let result = vocab.extract_and_normalize(
            Some(&normalizer),
            "Hi <mask> there\t<mask>\t<mask>\u{2000} <pad> <mask><pad><pad>",
        );

        assert_eq!(
            simplify_output(&result),
            vec![
                ("hi <m", None),
                ("ask>", Some(vec![1])),
                (" there\t<m", None),
                ("ask>", Some(vec![1])),
                ("\t<m", None),
                ("ask>", Some(vec![1])),
                ("\u{2000} <pad> <m", None),
                ("ask>", Some(vec![1])),
                ("<pad><pad>", None)
            ]
        );

        vocab.set_encode_special_tokens(false);

        let result = vocab.extract_and_normalize(
            Some(&normalizer),
            "Hi <mask> there\t<mask>\t<mask>\u{2000} <pad> <mask><pad><pad>",
        );
        assert_eq!(
            simplify_output(&result),
            vec![
                ("hi", None),
                (" <mask> ", Some(vec![0])),
                ("there", None),
                ("\t<mask>\t", Some(vec![0])),
                ("<mask>\u{2000} ", Some(vec![0])),
                ("<pad>", Some(vec![2])),
                (" <mask>", Some(vec![0])),
                ("<pad>", Some(vec![2])),
                ("<pad>", Some(vec![2]))
            ]
        );
    }
    #[test]
    fn byte_level_normalizer() {
        // Is able to extract both normal and special tokens
        let model = ModelMock::new(&[]);
        let mut vocab = AddedVocabulary::new();
        let from = NormalizerWrapper::from(ByteLevelNormalizer::new());
        let normalizer: Option<&NormalizerWrapper> = Some(&from);

        vocab.add_tokens(
            &[AddedToken::from("my", false), AddedToken::from("今", false)],
            &model,
            normalizer,
        );
        let result = vocab.extract_and_normalize(normalizer, "my今");
        assert_eq!(
            result
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(s, _, tokens)| (
                    s,
                    tokens
                        .as_ref()
                        .map(|t| t.iter().map(|t| t.id).collect::<Vec<_>>())
                ))
                .collect::<Vec<_>>(),
            vec![("my", Some(vec![0])), ("ä»Ĭ", Some(vec![1])),]
        );
    }
}
