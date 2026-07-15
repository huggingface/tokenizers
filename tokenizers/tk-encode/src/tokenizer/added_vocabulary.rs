use super::{Model, NormalizedString, Normalizer, Result};
use ahash::{AHashMap, AHashSet};
use serde::{ser::SerializeSeq, Deserialize, Serialize, Serializer};

/// Represent a token added by the user on top of the existing Model vocabulary.
/// AddedToken can be configured to specify the behavior they should have in various situations
/// like:
///   - Whether they should only match single words
///   - Whether to include any whitespace on its left or right
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AddedToken {
    /// The content of the added token (original, as provided by the user)
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
    /// Build this token from the given content, specifying if it is intended to be a
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
#[derive(Clone)]
pub struct AddedVocabulary {
    /// Contains the mapping from String (token content) to ID. This map contains both special
    /// tokens and classic added tokens that were added to the this vocabulary.
    added_tokens_map: AHashMap<String, u32>,
    /// Contains the mapping from ID to AddedToken for all the added tokens, both special
    /// and classic.
    added_tokens_map_r: AHashMap<u32, AddedToken>,

    /// A Set, containing all the special token for easy access while decoding. This let's
    /// us remove them easily with an O(1) complexity.
    special_tokens_set: AHashSet<String>,

    /// Cache of the normalizer output for tokens that have `normalized = true`.
    /// Keyed by token ID. Not serialized — rebuilt by `add_tokens` and
    /// `refresh_normalized_tokens` whenever the normalizer changes.
    /// Kept separate from `AddedToken` so the token struct stays lean.
    normalized_cache: AHashMap<u32, String>,

    /// Whether or not special tokens should be splitted when encoding. This is equivalent to ignoring them
    encode_special_tokens: bool,
}

impl std::fmt::Debug for AddedVocabulary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AddedVocabulary")
            .field("added_tokens_map", &self.added_tokens_map)
            .field("added_tokens_map_r", &self.added_tokens_map_r)
            .field("special_tokens_set", &self.special_tokens_set)
            .field("encode_special_tokens", &self.encode_special_tokens)
            .finish_non_exhaustive()
    }
}

impl AddedVocabulary {
    pub fn new() -> Self {
        Self {
            added_tokens_map: AHashMap::new(),
            added_tokens_map_r: AHashMap::new(),
            special_tokens_set: AHashSet::new(),
            normalized_cache: AHashMap::new(),
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
    pub fn get_vocab(&self) -> &AHashMap<String, u32> {
        &self.added_tokens_map
    }

    /// Get the additional vocabulary with the AddedTokens
    pub fn get_added_tokens_decoder(&self) -> &AHashMap<u32, AddedToken> {
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

    /// Return the string form of an added token used during **decoding**.
    ///
    /// For tokens that were normalized on the way *in* (e.g. byte-level encoding),
    /// this returns the cached normalized form so that the configured `Decoder` can
    /// invert the transformation correctly. For all other tokens, the original
    /// `content` is returned.
    pub fn simple_id_to_token(&self, id: u32) -> Option<String> {
        self.added_tokens_map_r.get(&id).map(|t| {
            self.normalized_cache
                .get(&id)
                .cloned()
                .unwrap_or_else(|| t.content.clone())
        })
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
        tokens: impl IntoIterator<Item = AddedToken>,
        model: &impl Model,
        normalizer: Option<&N>,
    ) -> Result<usize> {
        self.add_tokens(tokens, model, normalizer)
    }

    /// Add some tokens to the vocabulary
    pub fn add_tokens<N: Normalizer>(
        &mut self,
        tokens: impl IntoIterator<Item = AddedToken>,
        model: &impl Model,
        normalizer: Option<&N>,
    ) -> Result<usize> {
        let mut ignored = 0;
        let mut total = 0;

        let mut next_id =
            self.added_tokens_map_r
                .keys()
                .max()
                .map_or(model.get_vocab_size() as u32, |max| {
                    if *max >= model.get_vocab_size() as u32 || model.get_vocab_size() == 0 {
                        max + 1
                    } else {
                        model.get_vocab_size() as u32
                    }
                });

        for token in tokens {
            total += 1;
            if token.content.is_empty() {
                ignored += 1;
                continue;
            }
            // Fast path: skip if this content is already in the map with identical properties.
            if let Some(id) = self.added_tokens_map.get(&token.content) {
                if self.added_tokens_map_r.get(id) == Some(&token) {
                    ignored += 1;
                    continue;
                }
            }

            let new_id = if let Some(new_id) = self.token_to_id(&token.content, model) {
                new_id
            } else {
                let id = next_id;
                next_id += 1;
                id
            };

            if token.normalized {
                if let Some(n) = normalizer {
                    let mut s = NormalizedString::from(token.content.as_ref());
                    n.normalize(&mut s)?;
                    let normed = s.get().to_string();
                    if normed != token.content {
                        self.normalized_cache.insert(new_id, normed);
                    }
                }
            }

            *self
                .added_tokens_map
                .entry(token.content.clone())
                .or_default() = new_id;

            let is_new_special = token.special
                && !token.content.is_empty()
                && !self.special_tokens_set.contains(&token.content);
            if is_new_special {
                self.special_tokens_set.insert(token.content.clone());
            }
            self.added_tokens_map_r.insert(new_id, token);
        }

        // Return the number of added tokens
        Ok(total - ignored)
    }

    /// Re-apply normalization to every added token that has `normalized = true`, then
    /// rebuild the matching tries.
    ///
    /// This is called automatically by [`TokenizerImpl::with_normalizer`] when the
    /// normalizer is replaced. For tokenizers with many added tokens the trie rebuild
    /// can be slow; prefer setting the normalizer *before* calling `add_tokens` when
    /// constructing a tokenizer programmatically. During deserialization this is never
    /// triggered because the normalizer is set via the builder before tokens are added.
    pub fn refresh_normalized_tokens<N: Normalizer>(
        &mut self,
        normalizer: Option<&N>,
    ) -> Result<()> {
        self.normalized_cache.clear();
        for (id, token) in &self.added_tokens_map_r {
            if token.normalized {
                if let Some(n) = normalizer {
                    let mut s = NormalizedString::from(token.content.as_ref());
                    n.normalize(&mut s)?;
                    let normed = s.get().to_string();
                    if normed != token.content {
                        self.normalized_cache.insert(*id, normed);
                    }
                }
            }
        }
        Ok(())
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
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
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

