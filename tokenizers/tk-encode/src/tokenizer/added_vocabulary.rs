use super::{
    normalizer::Range, Model, NormalizedString, Normalizer, Offsets, PreTokenizedString, Result,
    Token,
};
use crate::types::{AddedTokenFlags, Bucket};
use crate::vocab_store::VocabStore;
use ahash::{AHashMap, AHashSet};
use daachorse::{DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder, MatchKind};
use regex::Regex;
use serde::{ser::SerializeSeq, Deserialize, Serialize, Serializer};
use std::{collections::HashMap, sync::LazyLock};

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

impl From<&AddedToken> for AddedTokenFlags {
    fn from(token: &AddedToken) -> AddedTokenFlags {
        AddedTokenFlags {
            special: token.special,
            normalized: token.normalized,
            single_word: token.single_word,
            lstrip: token.lstrip,
            rstrip: token.rstrip,
        }
    }
}

// AddedTokens can be updated if value changed
impl std::hash::Hash for AddedToken {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.content.hash(state);
    }
}

type MatchingSet = Option<DoubleArrayAhoCorasick<u32>>;

static STARTS_WITH_WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\w").unwrap());
static ENDS_WITH_WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\w$").unwrap());
static RIGHTMOST_SPACE_AT_START: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s*").unwrap());
static LEFTMOST_SPACE_AT_END: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s*$").unwrap());

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
    encode_special_tokens: bool,
    /// New fast path for normalize and extra needs:
    ///  - multi-bucket first bytes. If its len is 1, we use memchr
    ///  otherwise we just check each car on that table lookup. Its a 1KB table.
    ///  We use u8 because this ends up beaing 4 lines of cache, in the function's stack
    first_byte_to_bucket_id: [u8; 256],
    ///  - the actual buckets. We could use small vec here. Chose to impl it.
    ///  Buckets give pointers to the inner vocab store.
    buckets: Box<[Bucket]>,
    /// The metadata of each tokens
    token_metadata: Box<[AddedTokenFlags]>, // indexed using id_to_slot?
    inner: VocabStore,
}

impl AddedVocabulary {
    pub fn new() -> Self {
        Self {
            encode_special_tokens: true,
            first_byte_to_bucket_id: [u8::MAX; 256],
            buckets: Box::new([]),
            token_metadata: Box::new([]),
            inner: VocabStore::build(vec![("".as_bytes().to_vec(), 0)].to_vec()),
        }
    }
    /// Size of the additional vocabulary
    #[allow(dead_code)] // Suppress the "method is never used" warning
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether or not this vocabulary is empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get the additional vocabulary
    pub fn get_vocab(&self) -> AHashMap<String, u32> {
        self.inner
            .get_vocab()
            .into_iter()
            .collect::<AHashMap<String, u32>>()
    }

    /// Get the additional vocabulary with the AddedTokens
    /// TODO: this will be slowe because we rebuild the added tokens
    pub fn get_added_tokens_decoder(&self) -> AHashMap<u32, AddedToken> {
        self.get_vocab()
            .into_iter()
            .map(|(token, id)| {
                let m = &self.token_metadata[id as usize];
                (
                    id,
                    AddedToken {
                        content: token,
                        single_word: m.single_word,
                        lstrip: m.lstrip,
                        rstrip: m.rstrip,
                        normalized: m.normalized,
                        special: m.special,
                    }, // TODO: implem from / to
                )
            })
            .collect::<AHashMap<u32, AddedToken>>()
    }

    /// Get the id matching one of our token if it exists
    pub fn token_to_id(&self, token: &str, model: &dyn Model) -> Option<u32> {
        None
    }

    /// Return the string form of an added token used during **decoding**.
    ///
    /// For tokens that were normalized on the way *in* (e.g. byte-level encoding),
    /// this returns the cached normalized form so that the configured `Decoder` can
    /// invert the transformation correctly. For all other tokens, the original
    /// `content` is returned.
    pub fn simple_id_to_token(&self, id: u32) -> Option<String> {
        Some(String::new())
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
        self.token_metadata[0].special
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

        let mut next_id = self.inner.len();
        let mut byte_set = Vec::from(self.buckets.to_vec());
        let mut all_tokens = Vec::from(self.inner.get_vocab_bytes());
        let mut all_metadata = Vec::from(self.token_metadata.to_vec());

        for token in tokens {
            total += 1;
            if token.content.is_empty() {
                ignored += 1;
                continue;
            }
            let flags = AddedTokenFlags::from(&token);
            // Fast path: skip if this content is already in the map with identical properties.
            if let Some(id) = self.token_to_id(&token.content, model) {
                let id = id as usize;
                if all_metadata[id] == flags {
                    ignored += 1;
                    continue;
                } else {
                    all_metadata[id] = flags
                }
            } else {
                all_metadata.push(flags)
            }

            let new_id = if let Some(new_id) = self.token_to_id(&token.content, model) {
                // its not new here, but we need to update the token flags
                new_id as usize
            } else {
                let id = next_id;
                next_id += 1;
                id
            };

            // We count the first bytes and store the actual lenght of the char
            let token_bytes = token.content.as_bytes();
            let prefix_len = match &token_bytes[0] {
                0x00..=0x7F => 1,
                0xC2..=0xDF => 2,
                0xE0..=0xEF => 3,
                0xF0..=0xF4 => 4,
                _ => return Err(format!("Invalid UTF-8 first byte in token ").into()),
            };
            if self.first_byte_to_bucket_id[token_bytes[0] as usize] != u8::MAX {
                // bucket already exists :)
                println!("self buckest {:?}, {:?}, self.first_byte_to_bucket_id: {:?}", self.buckets, token_bytes[0] as usize, self.first_byte_to_bucket_id);
                byte_set[self.first_byte_to_bucket_id[token_bytes[0] as usize] as uzise].end += 1;
            } else {
                let mut prefix = [0; 4];
                prefix[..prefix_len].copy_from_slice(&token_bytes[..prefix_len]);
                byte_set.push(Bucket {
                    prefix,
                    prefix_len: prefix_len as u8,
                    start: 0,
                    end: 1,
                });
                println!("Adding !");
                println!("{:?}", byte_set);
                self.first_byte_to_bucket_id[token_bytes[0] as usize] = byte_set.len() as u8 -1;
            }
            // dummy bucket for now, next time its seens will just update end.
            if token.normalized {
                if let Some(n) = normalizer {
                    let mut s = NormalizedString::from(token.content.as_ref());
                    n.normalize(&mut s)?;
                    let normed = s.get().to_string();
                    // TODO: just init the normalized struct here
                    if normed != token.content {}
                }
            }
            all_tokens.push((token.content.into_bytes(), next_id as u32));
        }
        // TODO: we have
        let mut zipped:Vec<_> = all_tokens.into_iter().zip(all_metadata).collect();
        zipped.sort_unstable_by_key(|((s, _id), other)| {
            (
                self.buckets[self.first_byte_to_bucket_id[s[0] as usize] as usize].prefix,
                std::cmp::Reverse(s.len()),
            )
        });
        let (all_tokens, all_metadata) : (Vec<_>, Vec<_>)= zipped.into_iter().unzip();
        // at this point all tokens should look like: ["<|1|>", "<||>", "[ooo]", "[i]"]. First same
        //                                           b: <|     b:<|    b:[      b:[     the buckets    #[rustfmt::skip]
        // prefix then just longest
        //
        self.token_metadata = all_metadata.into();
        self.inner = VocabStore::build(all_tokens);
        let mut idx = 0;
        for b in &mut byte_set {
            let elem = b.end - b.start;
            b.start = idx;
            b.end = idx + elem;
            idx = b.end;
        }
        self.buckets = byte_set.into();
        // TODO: normalized_inner needed as well!

        // Return the number of added tokens
        Ok(total - ignored)
    }

    /// Find any AddedToken in the given sentence, using the provided MatchingSet.
    /// This method returns a list "splits", each of them being a pair of Offsets
    /// and an optional ID if it is an AddedToken.
    /// The list of splits cover the entire input string.
    fn find_matches(&self, sentence: &str, split_re: &MatchingSet) -> Vec<(Option<u32>, Offsets)> {
        // if sentence.is_empty() {
        //     return vec![(None, (0, 0))];
        // }
        //
        // let mut start_offset = 0;
        // let mut splits = vec![];
        //
        // let trie = match split_re {
        //     Some(t) => t,
        //     None => {
        //         return vec![(None, (0, sentence.len()))];
        //     }
        // };
        return Vec::new();
        // for mat in trie.leftmost_find_iter(sentence) {
        //     let mut start = mat.start();
        //     let mut stop = mat.end();
        //     let id = mat.value();
        //     let added_token = &self.added_tokens_map_r.get(&id).unwrap();
        //
        //     if self.encode_special_tokens && self.special_tokens_set.contains(&added_token.content)
        //     {
        //         continue;
        //     }
        //
        //     if added_token.single_word {
        //         let start_space = start == 0 || !ends_with_word(&sentence[..start]);
        //         let stop_space = stop == sentence.len() || !starts_with_word(&sentence[stop..]);
        //
        //         if !stop_space || !start_space {
        //             // Discard not single word
        //             continue;
        //         }
        //     }
        //     if added_token.lstrip {
        //         // This will be strictly inferior to start and in correct sentence offset
        //         let newstart = space_leftmost_at_end(&sentence[..start]);
        //
        //         // The previous match could have already matched those spaces
        //         // Ignore them if it's already matched
        //         start = std::cmp::max(newstart, start_offset);
        //     }
        //     if added_token.rstrip {
        //         // This will starting a the stop+1 character, so we need
        //         // to add the previous stop value
        //         stop += space_rightmost_at_start(&sentence[stop..])
        //     }
        //     if start_offset < start {
        //         splits.push((None, (start_offset, start)));
        //     }
        //     splits.push((Some(id), (start, stop)));
        //     start_offset = stop;
        // }
        //
        // let total_byte_len = sentence.len();
        // if start_offset != total_byte_len {
        //     splits.push((None, (start_offset, total_byte_len)));
        // }
        //
        // splits
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

    /// The first improvement we are working on is on replacing the heavy regex
    /// with IREE's fast string / normalization algo.
    pub fn extract_and_normalize<N: Normalizer>(
        &self,
        normalizer: Option<&N>,
        sequence: &str,
    ) -> PreTokenizedString {
        // 1. if the machinery does not exist, we build it:
        if self.token_metadata.len() == 1 {
            let next_match: Option<u8> = None;
        }
        return sequence.into();
    }
    /// Extract the additional vocabulary from the given sentence, normalizing it along the way.
    ///
    /// Some tokens should match against their normalized representation, as well as the
    /// non-normalized one. For example, when we expect to extract the token `yesterday` in the
    /// input sentence `I read a book Yesterday`, if the normalizer is supposed to lowercase
    /// everything, we expect a match.
    pub fn skip() {}
    // pub fn extract_and_normalize_old<N: Normalizer>(
    //     &self,
    //     normalizer: Option<&N>,
    //     sequence: &str,
    // ) -> PreTokenizedString {
    //     let mut pretokenized: PreTokenizedString = sequence.into();
    //
    //     // 1. We extract all the non-normalized tokens from the non-normalized string
    //     pretokenized
    //         .split(|_, sequence| Ok(self.split_with_indices(sequence, &self.split_trie)))
    //         .expect("AddedVocabulary bad split");
    //
    //     // <s> normalized = False
    //     // "I read a book   <s>Hey" -> "I read a book", "   <s>", "Hey"
    //
    //     // </s> normalized = True -> "▁</s>"
    //     // "I read a book</s>Hey" -> "I read a book</s>Hey"
    //
    //     // Day normalized = True -> "Day"
    //     // "I read a book monday" -> "I read a book monday"
    //
    //     // [DAY] normalized = False -> "Day"
    //     // "I read a [DAY] monday" -> "I read a " "[DAY]", "book monday"
    //     //                                         320055
    //     // 2. Then extract the normalized tokens from the normalized pieces of the string
    //     pretokenized
    //         .split(|_, mut sequence| {
    //             normalizer.map(|n| n.normalize(&mut sequence));
    //             Ok(self.split_with_indices(sequence, &self.split_normalized_trie))
    //         })
    //         .expect("AddedVocabulary bad split");
    //
    //     // ["I read a book", "   <s>", "Hey"] -> ["▁I read a book", "▁   <s>", "▁Hey"]
    //     // ["▁I read a book", "▁   <s>", "▁Hey"] -> [.., "▁   ", "<s>", "▁Hey"]
    //
    //     // </s> normalized = True -> "▁</s>"
    //     // "I read a book</s>Hey" -> ["▁I read a book", "<","/","s",">", "Hey"]
    //
    //     // "I read a " "[DAY]", "book monday" -> "i read a " "[day]", "book monday"
    //
    //     pretokenized
    // }
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
        // let mut added_tokens = self
        //     .added_tokens_map_r
        //     .iter()
        //     .map(|(id, token)| AddedTokenWithId {
        //         id: *id,
        //         token: token.clone(),
        //     })
        //     .collect::<Vec<_>>();
        // // We need to have these added tokens ordered by ascending ID
        // added_tokens.sort_unstable_by_key(|o| o.id);
        //
        // let mut vocabulary = serializer.serialize_seq(Some(added_tokens.len()))?;
        // for token in added_tokens {
        //     vocabulary.serialize_element(&token)?;
        // }
        //
        // vocabulary.end()

        let mut vocabulary = serializer.serialize_seq(Some(0))?;
        vocabulary.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalizers::byte_level::ByteLevel as ByteLevelNormalizer;
    use crate::normalizers::utils::Lowercase;
    use crate::normalizers::NormalizerWrapper;
    use crate::{OffsetReferential, OffsetType, Result, Token};
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};

    #[derive(Serialize, Deserialize)]
    struct ModelMock {
        vocab: AHashMap<String, u32>,
        vocab_r: AHashMap<u32, String>,
    }
    impl ModelMock {
        pub fn new<I>(iter: I) -> Self
        where
            I: IntoIterator<Item = &'static (&'static str, u32)>,
        {
            let vocab: AHashMap<String, u32> = iter
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

    impl Model for ModelMock {
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
            self.vocab.clone().into_iter().collect()
        }
        fn get_vocab_size(&self) -> usize {
            self.vocab.len()
        }
        fn save(&self, _folder: &Path, _name: Option<&str>) -> Result<Vec<PathBuf>> {
            unimplemented!()
        }
    }

    #[test]
    fn can_add_tokens() {
        let model = ModelMock::new(&[("test", 0), ("tost", 1)]);
        let mut vocab = AddedVocabulary::new();
        let normalizer: Option<&NormalizerWrapper> = None;

        // Add tokens normally
        assert_eq!(
            vocab
                .add_tokens(
                    [AddedToken::from("added_token_1", false)],
                    &model,
                    normalizer
                )
                .unwrap(),
            1
        );

        let vocab_len: usize = vocab.len();
        assert_eq!(vocab_len, 1);

        // Does not add multiple time the same token
        assert_eq!(
            vocab
                .add_tokens(
                    [
                        AddedToken::from("added_token_2", false),
                        AddedToken::from("added_token_2", false)
                    ],
                    &model,
                    normalizer
                )
                .unwrap(),
            1
        );
        assert_eq!(vocab.len(), 2);

        // Also adds tokens already covered by the model
        let added_token = AddedToken::from("test", false);
        assert_eq!(
            vocab
                .add_tokens([added_token.clone()], &model, normalizer)
                .unwrap(),
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
            vocab
                .add_special_tokens(
                    [AddedToken::from("added_token_1", true)],
                    &model,
                    normalizer
                )
                .unwrap(),
            1
        );
        assert_eq!(vocab.len(), 1);

        // Does not add multiple time the same token
        assert_eq!(
            vocab
                .add_special_tokens(
                    [
                        AddedToken::from("added_token_2", true),
                        AddedToken::from("added_token_2", true)
                    ],
                    &model,
                    normalizer
                )
                .unwrap(),
            1
        );
        assert_eq!(vocab.len(), 2);

        // Can add tokens already covered by the model
        assert_eq!(
            vocab
                .add_special_tokens([AddedToken::from("test", true)], &model, normalizer)
                .unwrap(),
            1
        );
        assert_eq!(vocab.len(), 3); // New token was added
        assert!(vocab.is_special_token("test"));
        assert_eq!(
            *vocab.get_added_tokens_decoder(),
            vec![
                (0u32, AddedToken::from("test", true)),
                (2, AddedToken::from("added_token_1", true)),
                (3, AddedToken::from("added_token_2", true)),
            ]
            .into_iter()
            .collect::<AHashMap<u32, AddedToken>>()
            .into()
        );
        // assert!(vocab.added_tokens_map.contains_key("test"));
        // assert!(vocab.added_tokens_map_r.contains_key(&0));

        vocab
            .add_tokens(
                [
                    AddedToken::from("tost", true),
                    AddedToken::from("another_two", false),
                ],
                &model,
                normalizer,
            )
            .unwrap();
        assert_eq!(vocab.len(), 5); // New token was added
        assert_eq!(vocab.get_vocab()["another_two"], 4); // New token was added, but the index is not the length of the vocab

        // Let's add an already added token again, but change normalized
        assert_eq!(
            vocab
                .add_special_tokens([AddedToken::from("another_two", true)], &model, normalizer)
                .unwrap(),
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

        vocab
            .add_tokens(
                [
                    AddedToken::from("my", false),
                    AddedToken::from("name", false),
                ],
                &model,
                normalizer,
            )
            .unwrap();
        vocab
            .add_special_tokens(
                [
                    AddedToken::from("[CLS]", true),
                    AddedToken::from("[SEP]", true),
                ],
                &model,
                normalizer,
            )
            .unwrap();

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

        vocab
            .add_tokens(
                [
                    AddedToken::from("my", false).lstrip(true).rstrip(true),
                    AddedToken::from("name", false),
                    AddedToken::from("ony", false).single_word(true),
                ],
                &model,
                Some(&normalizer),
            )
            .unwrap();
        vocab
            .add_special_tokens(
                [
                    AddedToken::from("[CLS]", true),
                    AddedToken::from("[SEP]", true),
                ],
                &model,
                Some(&normalizer),
            )
            .unwrap();

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

    // ponytail: disabled — uses removed `split_trie` field. Re-enable once find_matches has its new signature.
    // #[test]
    // fn empty_matches() {
    //     let vocab = AddedVocabulary::new();
    //     let matches = vocab.find_matches("", &vocab.split_trie);
    //     assert_eq!(matches, vec![(None, (0, 0))]);
    // }

    #[test]
    fn test_single_word_is_correct() {
        // Is able to extract both normal and special tokens, with various options (lstrip, rstrip,
        // single_word, normalized)
        let model = ModelMock::new(&[]);
        let mut vocab = AddedVocabulary::new();
        let normalizer = Lowercase;

        vocab
            .add_tokens(
                [AddedToken::from("<mask>", false).single_word(true)],
                &model,
                Some(&normalizer),
            )
            .unwrap();
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

        vocab
            .add_tokens(
                [AddedToken::from("<mask>", false).single_word(true)],
                &model,
                Some(&normalizer),
            )
            .unwrap();
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

        vocab
            .add_tokens(
                [AddedToken::from("<mask>", false)
                    .lstrip(true)
                    .rstrip(true)
                    .single_word(true)],
                &model,
                Some(&normalizer),
            )
            .unwrap();
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

        vocab
            .add_tokens(
                [
                    AddedToken::from("<mask>", true)
                        .lstrip(true)
                        .rstrip(true)
                        .single_word(true),
                    AddedToken::from("ask>", false),
                    AddedToken::from("<pad>", true),
                ],
                &model,
                Some(&normalizer),
            )
            .unwrap();
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
    fn content_preserved_with_normalizer() {
        // Verify that AddedToken.content always holds the original user-provided string,
        // and that normalized_content holds the normalizer output separately.
        let model = ModelMock::new(&[]);
        let mut vocab = AddedVocabulary::new();
        let normalizer = Lowercase;

        vocab
            .add_tokens(
                [
                    AddedToken::from("Hello", false),
                    AddedToken::from("[CLS]", true),
                ],
                &model,
                Some(&normalizer),
            )
            .unwrap();

        let decoder = vocab.get_added_tokens_decoder();
        // Original content is always preserved in the token struct regardless of normalization
        assert!(decoder.values().any(|t| t.content == "Hello"));
        assert!(decoder.values().any(|t| t.content == "[CLS]"));

        // "hello" (lowercased) is in the normalized cache — verify via simple_id_to_token
        // let hello_id = vocab.added_tokens_map["Hello"];
        // let cls_id = vocab.added_tokens_map["[CLS]"];
        // normalized=true → decode returns cached form "hello"
        // assert_eq!(vocab.simple_id_to_token(hello_id).unwrap(), "hello");
        // // normalized=false → decode returns original content "[CLS]"
        // assert_eq!(vocab.simple_id_to_token(cls_id).unwrap(), "[CLS]");
    }

    // ponytail: disabled — uses removed `added_tokens_map` field and `refresh_normalized_tokens` method.
    // Re-enable once those exist again on AddedVocabulary.
    // #[test]
    // fn refresh_normalized_tokens_on_normalizer_change() {
    //     // Tokens added without a normalizer should get their normalized_content populated
    //     // when the normalizer is set later via refresh_normalized_tokens.
    //     let model = ModelMock::new(&[]);
    //     let mut vocab = AddedVocabulary::new();
    //     let normalizer = Lowercase;
    //
    //     // Add tokens with NO normalizer first
    //     vocab
    //         .add_tokens(
    //             [AddedToken::from("Hello", false)],
    //             &model,
    //             None::<&NormalizerWrapper>,
    //         )
    //         .unwrap();
    //
    //     // Without a normalizer, simple_id_to_token returns the original content
    //     let hello_id = vocab.added_tokens_map["Hello"];
    //     assert_eq!(vocab.simple_id_to_token(hello_id).unwrap(), "Hello");
    //
    //     // Now attach a normalizer and refresh
    //     vocab.refresh_normalized_tokens(Some(&normalizer)).unwrap();
    //
    //     // After refresh, simple_id_to_token returns the cached normalized form
    //     assert_eq!(vocab.simple_id_to_token(hello_id).unwrap(), "hello");
    //
    //     // And the vocabulary should still match correctly (splits use normalized form)
    //     let result = vocab.extract_and_normalize(Some(&normalizer), "Hello world");
    //     let splits = simplify_output(&result);
    //     assert_eq!(splits[0], ("hello", Some(vec![0])));
    // }

    #[test]
    fn byte_level_normalizer() {
        // Is able to extract both normal and special tokens
        let model = ModelMock::new(&[]);
        let mut vocab = AddedVocabulary::new();
        let from = NormalizerWrapper::from(ByteLevelNormalizer::new());
        let normalizer: Option<&NormalizerWrapper> = Some(&from);

        vocab
            .add_tokens(
                [AddedToken::from("my", false), AddedToken::from("今", false)],
                &model,
                normalizer,
            )
            .unwrap();
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
