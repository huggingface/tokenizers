use super::{Model, NormalizedString, Normalizer, Range};
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
    pub fn from(content: String, special_token: bool) -> Self {
        AddedToken {
            content,
            normalized: !special_token,
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
    /// Specify whether this token should be normalized, and/or match against its normalized
    /// version in the input text.
    pub fn normalized(mut self, normalized: bool) -> Self {
        self.normalized = normalized;
        self
    }
    /// Retrive the pattern built for this token, according to all the specified parameters.
    pub fn get_pattern(&self, normalizer: Option<&dyn Normalizer>) -> String {
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
            let mut content = NormalizedString::from(&self.content);
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
            normalized: false,
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
pub(super) struct AddedVocabulary {
    /// The size of the original vocabulary. This is what we use to determine the new
    /// ids we need to generate
    original_vocab_size: usize,
    /// Contains the mapping from String to ID as the user intended it. This map
    /// contains both special tokens and classic added tokens.
    added_tokens_map: HashMap<String, u32>,
    /// Contains the mapping from ID to AddedToken for all the added tokens, both special
    /// and classic.
    added_tokens_map_r: HashMap<u32, AddedToken>,
    /// Contains only the classic AddedToken, in the specific order the user gave them.
    added_tokens: Vec<AddedToken>,
    /// Contains only the special AddedToken, in the specific order the user gave them.
    special_tokens: Vec<AddedToken>,
    /// A Set, containing all the special token for easy access while decoding. This let's
    /// use remove them easily with an O(1) complexity.
    special_tokens_set: HashSet<String>,
    /// A RegexSet containing all the non-normalized patterns used to split on AddedTokens
    split_re: MatchingSet,
    /// A RegexSet containing all the normalized patterns used to split on AddedTokens
    split_normalized_re: MatchingSet,
}

impl AddedVocabulary {
    pub fn new(original_vocab_size: usize) -> Self {
        Self {
            original_vocab_size,
            added_tokens_map: HashMap::new(),
            added_tokens_map_r: HashMap::new(),
            added_tokens: vec![],
            special_tokens: vec![],
            special_tokens_set: HashSet::new(),
            split_re: (regex::RegexSet::new::<_, &&str>(&[]).unwrap(), vec![]),
            split_normalized_re: (regex::RegexSet::new::<_, &&str>(&[]).unwrap(), vec![]),
        }
    }

    /// Sets the original vocabulary size. We need this value to return IDs that
    /// are shifted according to the original vocabulary.
    pub fn update_original_vocab_size(&mut self, size: usize) {
        self.original_vocab_size = size;
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
    pub fn token_to_id(&self, token: &str) -> Option<&u32> {
        self.added_tokens_map.get(token)
    }

    /// Get the token matching the given id if it exists
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.added_tokens_map_r.get(&id).map(|t| t.content.as_ref())
    }

    /// Check if a token is a special token
    pub fn is_special_token(&self, token: &str) -> bool {
        self.special_tokens_set.contains(token)
    }

    /// Add some special tokens to the vocabulary
    pub fn add_special_tokens(
        &mut self,
        tokens: &[AddedToken],
        model: &dyn Model,
        normalizer: Option<&dyn Normalizer>,
    ) -> usize {
        for token in tokens {
            if !self.special_tokens_set.contains(&token.content) {
                self.special_tokens.push(token.to_owned());
                self.special_tokens_set.insert(token.content.clone());
            }
        }
        let added = self.add_tokens(&tokens, model, normalizer);

        self.refresh_added_tokens(normalizer);

        added
    }

    /// Add some tokens to the vocabulary
    pub fn add_tokens(
        &mut self,
        tokens: &[AddedToken],
        model: &dyn Model,
        normalizer: Option<&dyn Normalizer>,
    ) -> usize {
        let mut ignored = 0;
        for token in tokens {
            if token.content.is_empty() {
                ignored += 1;
                continue;
            }

            let id = if let Some(id) = model.token_to_id(&token.content) {
                ignored += 1;
                id
            } else {
                let new_id = (self.original_vocab_size + self.added_tokens_map.len()) as u32;
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

        self.refresh_added_tokens(normalizer);

        // Return the number of added tokens
        tokens.len() - ignored
    }

    /// Reconstruct our internal RegexSet when new tokens are added to the vocabulary.
    ///
    /// We keep two different RegexSet, one that will take care of matching against the
    /// non-normalized string, and one matching against the normalized one.
    fn refresh_added_tokens(&mut self, normalizer: Option<&dyn Normalizer>) {
        type TupleTokenId<'a> = (&'a AddedToken, u32);
        let (normalized, non_normalized): (Vec<TupleTokenId>, Vec<TupleTokenId>) = self
            .special_tokens
            .iter()
            .chain(self.added_tokens.iter())
            // TODO: Fix this: special tokens that are part of the original vocabulary are
            // not part of the `self.added_tokens_map` and so it crashes.
            .map(|token| (token, self.added_tokens_map[&token.content]))
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

    /// TODO: Add doc string here
    fn extract(
        &self,
        sentence: NormalizedString,
        split_re: &MatchingSet,
    ) -> Vec<(NormalizedString, Option<u32>)> {
        let mut matches = split_re
            .0
            .matches(sentence.get())
            .into_iter()
            .flat_map(|idx| {
                regex::Regex::new(&split_re.0.patterns()[idx])
                    .unwrap()
                    .find_iter(sentence.get())
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
                splits.push((Some(idx), (start, end)));
                start_offset = end;

                splits
            })
            .collect::<Vec<_>>();
        if let Some((_, (_, end))) = splits.iter().last().copied() {
            if end < sentence.get().len() {
                splits.push((None, (end, sentence.get().len())));
            }
        }

        if splits.is_empty() {
            vec![(sentence, None)]
        } else {
            splits
                .into_iter()
                .map(|(idx, (start, end))| {
                    // TODO: Check this works
                    let normalized = sentence
                        .slice_bytes(Range::Normalized(start..end))
                        .expect("Error while extracting normalized Range");

                    // Find out the associated AddedToken, and its id
                    let id = idx.map(|idx| split_re.1[idx]);

                    (normalized, id)
                })
                .collect()
        }
    }

    /// Extract the additional vocabulary from the given sentence, normalizing it along the way.
    ///
    /// Some tokens should match against their normalized representation, as well as the
    /// non-normalized one. For example, when we expect to extract the token `yesterday` in the
    /// input sentence `I read a book Yesterday`, if the normalizer is supposed to lowercase
    /// everything, we expect a match.
    ///
    /// This method returns a `Vec` of `(NormalizedString, Option<u32>)`, where the optional `u32`
    /// contains the relevant ID if this is an additional token.
    pub fn extract_and_normalize(
        &self,
        normalizer: Option<&dyn Normalizer>,
        sentence: &str,
    ) -> Vec<(NormalizedString, Option<u32>)> {
        // 1. We extract all the non-normalized tokens from the non-normalized string
        let pieces = self.extract(NormalizedString::from(sentence), &self.split_re);

        // 2. Then extract the normalized tokens from the normalized pieces of the string
        pieces
            .into_iter()
            .flat_map(|(mut normalized, id)| {
                if id.is_some() {
                    // If the piece has an associated ID, we already extracted something,
                    // so we just return it
                    vec![(normalized, id)]
                } else {
                    // Otherwise, we need to normalized the string, and then proceed to extracting
                    normalizer.map(|n| n.normalize(&mut normalized));
                    self.extract(normalized, &self.split_normalized_re)
                }
            })
            .collect::<Vec<_>>()
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
        let mut vocabulary = serializer.serialize_seq(Some(self.added_tokens_map.len()))?;

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

        for token in added_tokens {
            vocabulary.serialize_element(&token)?;
        }

        vocabulary.end()
    }
}
