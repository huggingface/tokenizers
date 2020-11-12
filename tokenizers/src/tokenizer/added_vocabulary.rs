use super::{
    normalizer::Range, Model, NormalizedString, Normalizer, Offsets, PreTokenizedString, Token,
};
use serde::{ser::SerializeSeq, Deserialize, Serialize, Serializer};
use std::collections::{HashMap, HashSet};

/// Represent a token added by the user on top of the existing Model vocabulary.
/// AddedToken can be configured to specify the behavior they should have in various situations
/// like:
///   - Whether they should only match single words
///   - Whether to include any whitespace on its left or right
#[derive(Debug, Clone, Serialize, Deserialize)]
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
}

impl AddedToken {
    /// Build this token from the given content, specifying if it is intented to be a
    /// special token. Special tokens are not normalized by default.
    pub fn from<S: Into<String>>(content: S, special: bool) -> Self {
        AddedToken {
            content: content.into(),
            normalized: !special,
            ..Default::default()
        }
    }
    /// Specify whether this token should only match on whole single words, and never
    /// part of a word.
    pub fn single_word(mut self, single_word: bool) -> Self {
        self.single_word = single_word;
        self
    }
    /// Specify whether this token should include all the whitespaces on its left, in
    /// order to strip them out.
    pub fn lstrip(mut self, lstrip: bool) -> Self {
        self.lstrip = lstrip;
        self
    }
    /// Specify whether this token should include all the whitespaces on its right, in
    /// order to strip them out.
    pub fn rstrip(mut self, rstrip: bool) -> Self {
        self.rstrip = rstrip;
        self
    }
    /// Specify whether this token should be normalized and match against its normalized
    /// version in the input text.
    pub fn normalized(mut self, normalized: bool) -> Self {
        self.normalized = normalized;
        self
    }
    /// Retrive the pattern built for this token, according to all the specified parameters.
    pub fn get_pattern<N: Normalizer>(&self, normalizer: Option<&N>) -> String {
        let mut r = if self.single_word {
            let first_b = self
                .content
                .chars()
                .next()
                .map(|c| {
                    if regex_syntax::is_word_character(c) {
                        r"\b"
                    } else {
                        ""
                    }
                })
                .unwrap();
            let last_b = self
                .content
                .chars()
                .last()
                .map(|c| {
                    if regex_syntax::is_word_character(c) {
                        r"\b"
                    } else {
                        ""
                    }
                })
                .unwrap();

            // Normalize the content
            let mut content = NormalizedString::from(self.content.as_ref());
            normalizer.map(|n| n.normalize(&mut content));
            format!(r"{}{}{}", first_b, regex::escape(content.get()), last_b)
        } else {
            regex::escape(&self.content)
        };

        if self.lstrip && self.rstrip {
            r = format!(r"(\s)?{}(\s)?", r);
        } else if self.lstrip {
            r = format!(r"(\s)?{}", r);
        } else if self.rstrip {
            r = format!(r"{}(\s)?", r);
        }

        r
    }
}
impl Default for AddedToken {
    fn default() -> Self {
        AddedToken {
            content: String::new(),
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: true,
        }
    }
}
// We only want to hash on the content. AddedToken cannot be added multiple times with different
// options
impl std::hash::Hash for AddedToken {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.content.hash(state);
    }
}
impl std::cmp::PartialEq for AddedToken {
    fn eq(&self, other: &Self) -> bool {
        self.content == other.content
    }
}
impl std::cmp::Eq for AddedToken {}

type MatchingSet = (regex::RegexSet, Vec<u32>);

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
pub(super) struct AddedVocabulary {
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
    split_re: MatchingSet,
    /// A RegexSet containing all the normalized patterns used to split on AddedTokens
    split_normalized_re: MatchingSet,
}

impl AddedVocabulary {
    pub fn new() -> Self {
        Self {
            added_tokens_map: HashMap::new(),
            added_tokens_map_r: HashMap::new(),
            added_tokens: vec![],
            special_tokens: vec![],
            special_tokens_set: HashSet::new(),
            split_re: (regex::RegexSet::new::<_, &&str>(&[]).unwrap(), vec![]),
            split_normalized_re: (regex::RegexSet::new::<_, &&str>(&[]).unwrap(), vec![]),
        }
    }

    /// Size of the additional vocabulary
    pub fn len(&self) -> usize {
        self.added_tokens_map.len()
    }

    /// Get the additional vocabulary
    pub fn get_vocab(&self) -> &HashMap<String, u32> {
        &self.added_tokens_map
    }

    /// Get the id matching one of our token if it exists
    pub fn token_to_id(&self, token: &str, model: &impl Model) -> Option<u32> {
        self.added_tokens_map
            .get(token)
            .copied()
            .or_else(|| model.token_to_id(token))
    }

    /// Get the token matching the given id if it exists
    pub fn id_to_token(&self, id: u32, model: &impl Model) -> Option<String> {
        self.added_tokens_map_r
            .get(&id)
            .map(|t| t.content.clone())
            .or_else(|| model.id_to_token(id))
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
        for token in tokens {
            if !token.content.is_empty() && !self.special_tokens_set.contains(&token.content) {
                self.special_tokens.push(token.to_owned());
                self.special_tokens_set.insert(token.content.clone());
            }
        }
        // Then we delegate to `add_tokens`, that will take care of refreshing added tokens too.
        self.add_tokens(&tokens, model, normalizer)
    }

    /// Add some tokens to the vocabulary
    pub fn add_tokens<N: Normalizer>(
        &mut self,
        tokens: &[AddedToken],
        model: &impl Model,
        normalizer: Option<&N>,
    ) -> usize {
        let mut ignored = 0;
        for token in tokens {
            if token.content.is_empty() {
                ignored += 1;
                continue;
            }

            let id = if let Some(id) = self.token_to_id(&token.content, model) {
                ignored += 1;
                id
            } else {
                let new_id = (model.get_vocab_size() + self.added_tokens_map.len()) as u32;
                self.added_tokens_map.insert(token.content.clone(), new_id);

                if !self.special_tokens_set.contains(&token.content) {
                    self.added_tokens.push(token.clone());
                }

                new_id
            };

            // Update the current revert operation
            self.added_tokens_map_r
                .entry(id)
                .and_modify(|t| *t = token.clone())
                .or_insert_with(|| token.clone());
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
        self.split_re = (
            regex::RegexSet::new(tokens.iter().map(|t| t.get_pattern(normalizer))).unwrap(),
            ids,
        );

        let (tokens, ids): (Vec<&AddedToken>, Vec<u32>) = normalized.into_iter().unzip();
        self.split_normalized_re = (
            regex::RegexSet::new(tokens.iter().map(|t| t.get_pattern(normalizer))).unwrap(),
            ids,
        );
    }

    /// Find any AddedToken in the given sentence, using the provided MatchingSet.
    /// This method returns a list "splits", each of them being a pair of Offsets
    /// and an optional ID if it is an AddedToken.
    /// The list of splits cover the entire input string.
    fn find_matches<'a>(
        &self,
        sentence: &str,
        split_re: &'a MatchingSet,
    ) -> Vec<(Option<u32>, Offsets)> {
        if sentence.is_empty() {
            return vec![(None, (0, 0))];
        }

        let mut matches = split_re
            .0
            .matches(sentence)
            .into_iter()
            .flat_map(|idx| {
                regex::Regex::new(&split_re.0.patterns()[idx])
                    .unwrap()
                    .find_iter(sentence)
                    .map(|m| (idx, (m.start(), m.end())))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        // We sort all the matches by their start and then by their pattern id
        matches.sort_by(
            |(idxa, (sa, _)), (idxb, (sb, _))| {
                if sa != sb {
                    sa.cmp(sb)
                } else {
                    idxa.cmp(idxb)
                }
            },
        );

        // Select the matches (if some are overlapping) we want to keep
        let mut i = 0;
        let mut current_offset = 0;
        let mut splits = Vec::with_capacity(matches.len());
        while i < matches.len() {
            let (idx, (start, end)) = matches[i];

            // current match is before the currentt offset, let's skip it
            if start < current_offset {
                i += 1;
                continue;
            }

            // Find out if we have overlapping neighbors. If so, we keep the one with the lowest
            // idx, and apply it, then continue. All others will be skipped since `current_offset`
            // will have been increased
            if i + 1 < matches.len() {
                if let Some((idx, (s, e))) = matches[i..]
                    .iter()
                    .take_while(|(_, (s, e))| *s < end && start < *e)
                    .min() // Order on idx first
                    .copied()
                {
                    splits.push((idx, (s, e)));
                    current_offset = e;
                    i += 1;
                    continue;
                }
            }

            // We didn't find overlapping neighbors, apply ourself
            splits.push((idx, (start, end)));
            current_offset = end;
            i += 1;
        }

        // We also insert the splits that are inbetween the added tokens, to split the entire string
        let mut start_offset = 0;
        let mut splits = splits
            .into_iter()
            .flat_map(|(idx, (start, end))| {
                let mut splits = vec![];
                if start_offset < start {
                    splits.push((None, (start_offset, start)));
                }
                splits.push((Some(split_re.1[idx] as u32), (start, end)));
                start_offset = end;

                splits
            })
            .collect::<Vec<_>>();

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
            .split(|_, sequence| Ok(self.split_with_indices(sequence, &self.split_re)))
            .expect("AddedVocabulary bad split");

        // 2. Then extract the normalized tokens from the normalized pieces of the string
        pretokenized
            .split(|_, mut sequence| {
                normalizer.map(|n| n.normalize(&mut sequence));
                Ok(self.split_with_indices(sequence, &self.split_normalized_re))
            })
            .expect("AddedVocabulary bad split");

        pretokenized
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct AddedTokenWithId {
    /// The id assigned to this token
    pub id: u32,
    /// Whether this is a special token
    pub special: bool,

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
                special: self.special_tokens_set.contains(&token.content),
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
        assert_eq!(vocab.len(), 1);

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

        // Does not add tokens already covered by the model
        assert_eq!(
            vocab.add_tokens(&[AddedToken::from("test", false)], &model, normalizer),
            0
        );
        assert_eq!(vocab.len(), 2);
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
            0
        );
        assert_eq!(vocab.len(), 2); // Did not add a new token, since it exist in the original model
        assert_eq!(vocab.is_special_token("test"), true);
        assert_eq!(vocab.added_tokens_map.contains_key("test"), false);
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
        let matches = vocab.find_matches("", &vocab.split_re);
        assert_eq!(matches, vec![(None, (0, 0))]);
    }
}
