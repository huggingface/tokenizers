use crate::normalizer::Range;
use crate::pipeline::PipelinePatternMatcher;
use crate::{Model, NormalizedString, Normalizer, PreTokenizedString, Result, Token};

use crate::buckets::{AddedTokenFlags, Buckets};
use crate::pre_tokenizers::whitespace::is_word_char;
use ahash::AHashMap;
use serde::{ser::SerializeSeq, Deserialize, Serialize, Serializer};
use std::fmt;
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

#[inline]
fn is_ws(cp: u32) -> bool {
    // check if its ascii -> we can answer fast if yes. ASCII rang is < 0x80
    if cp < 0x80 {
        cp == 0x20 || (0x09..=0x0d).contains(&cp)
    } else {
        char::from_u32(cp).is_some_and(|c| c.is_whitespace())
    }
}
fn is_single_word(bytes: &[u8], search: usize, match_start: usize, match_end: usize) -> bool {
    // FIXME: we use chr conversion for now, this can be inproved by using bitmap.
    // This is the equivalent of `\w`, so its letters, numbers and underscore
    let s = unsafe { std::str::from_utf8_unchecked(bytes) };
    let before_ok = s[search..match_start]
        .chars()
        .next_back()
        .is_none_or(|c| !is_word_char(c));
    if !before_ok {
        return false;
    };
    let after_ok = s[match_end..]
        .chars()
        .next()
        .is_none_or(|c| !is_word_char(c));
    before_ok && after_ok
}

fn skip_whitespace_backward(bytes: &[u8], match_start: usize) -> usize {
    let s = unsafe { std::str::from_utf8_unchecked(bytes) };
    s[..match_start].trim_end_matches(|c| is_ws(c as u32)).len()
}
fn skip_whitespace_forward(bytes: &[u8], match_start: usize) -> usize {
    let s = unsafe { std::str::from_utf8_unchecked(bytes) };
    s.len()
        - s[match_start..]
            .trim_start_matches(|c| is_ws(c as u32))
            .len()
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

// TODO: remove this once we have the final saving format
// its outside the scope of this PR and allows us to test
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
    encode_special_tokens: bool,
    /// New fast path for normalize and extra needs:
    ///  - multi-bucket first bytes. If its len is 1, we use memchr; otherwise we just check each
    ///    byte on that table lookup. Its a 1KB table.
    ///  - the actual buckets. We could use small vec here. Chose to impl it. Buckets give pointers
    ///    to the inner vocab store.
    ///  - the metadata of each token.
    token_metadata: Box<[AddedTokenFlags]>, // indexed using id_to_slot?
    normalized_vocab: Buckets,
    vocab: Buckets,
}
impl fmt::Debug for AddedVocabulary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TODO: make it clean.
        f.debug_struct("AddedVocabulary")
            // .field("buckets", &self.vocab)
            .field(
                "reconstructed_added_vocab",
                &self.get_added_tokens_decoder(),
            )
            .finish()
    }
}
impl AddedVocabulary {
    pub fn new() -> Self {
        Self {
            encode_special_tokens: true,
            token_metadata: Box::new([]),
            normalized_vocab: Buckets::new(),
            vocab: Buckets::new(),
        }
    }
    /// Size of the additional vocabulary
    #[allow(dead_code)] // Suppress the "method is never used" warning
    pub fn len(&self) -> usize {
        self.vocab.len() + self.normalized_vocab.len()
    }

    /// Whether or not this vocabulary is empty
    pub fn is_empty(&self) -> bool {
        self.vocab.is_empty() && self.normalized_vocab.is_empty()
    }

    /// Get the additional vocabulary (union of both matchers; normalized tokens appear by their
    /// normalized form, since the original content isn't retained after partitioning).
    pub fn get_vocab(&self) -> AHashMap<String, u32> {
        self.vocab
            .get_vocab()
            .into_iter()
            .chain(self.normalized_vocab.get_vocab())
            .collect::<AHashMap<String, u32>>()
    }

    /// Get the additional vocabulary with the AddedTokens
    /// TODO: this will be slow because we rebuild the added tokens
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
    pub fn token_to_id(&self, token: &str, _model: &dyn Model) -> Option<u32> {
        self.vocab
            .token_to_id(token)
            .or_else(|| self.normalized_vocab.token_to_id(token))
    }

    /// Return the string form of an added token used during **decoding**.
    ///
    /// For tokens that were normalized on the way *in* (e.g. byte-level encoding),
    /// this returns the cached normalized form so that the configured `Decoder` can
    /// invert the transformation correctly. For all other tokens, the original
    /// `content` is returned.
    pub fn simple_id_to_token(&self, _id: u32) -> Option<String> {
        self.vocab
            .id_to_token(_id)
            .or_else(|| self.normalized_vocab.id_to_token(_id))
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
        if let Some(tok) = self.vocab.token_to_id(token) {
            return self.token_metadata[tok as usize].special;
        }
        false
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

        // One matcher per `normalized` flag: non-normalized tokens live in `vocab` (matched on the
        // raw input), normalized ones in `normalized_vocab` (matched on normalized text). Single id
        // space; `metadata` is indexed by id and sized to (max id + 1), so `metadata.len()` is the
        // next free dense id (no scan/alloc). Added ids start above the model vocab; a token already
        // in the model reuses the model id.
        let mut metadata = self.token_metadata.to_vec();
        let model_size = model.get_vocab_size() as u32;
        let mut next_id = (metadata.len() as u32).max(model_size);

        let mut entries: AHashMap<u32, (Vec<u8>, bool)> = AHashMap::new();
        // Because we allow changing from normalized=true to false, we need to keep track of both
        for (form, id) in self.vocab.get_vocab_bytes() {
            entries.insert(id, (form, false));
        }
        for (form, id) in self.normalized_vocab.get_vocab_bytes() {
            entries.insert(id, (form, true));
        }
        let mut seen: AHashMap<String, u32> = AHashMap::new();

        for token in tokens {
            total += 1;
            if token.content.is_empty() {
                ignored += 1;
                continue;
            }
            let flags = AddedTokenFlags::from(&token);
            let is_norm = flags.normalized;
            let norm_form: String = match normalizer {
                Some(n) => {
                    // TODO: fix normalizer to remove allocations :)
                    let mut s = NormalizedString::from(token.content.as_str());
                    n.normalize(&mut s)?;
                    s.get().to_string()
                }
                None => token.content.clone(),
            };
            let form = if is_norm {
                norm_form.clone().into_bytes()
            } else {
                token.content.clone().into_bytes()
            };
            // token could be in model, in added vocab, in added vocab normalized
            let existing = seen
                .get(&token.content)
                .copied()
                .or_else(|| model.token_to_id(&token.content))
                .or_else(|| self.vocab.token_to_id(&token.content))
                .or_else(|| self.normalized_vocab.token_to_id(&norm_form));
            // Already present with the exact same flags AND matcher -> nothing to do.
            if let Some(id) = existing {
                if metadata.get(id as usize) == Some(&flags)
                    && entries.get(&id).map(|(_, n)| *n) == Some(is_norm)
                {
                    ignored += 1;
                    continue;
                }
            }
            let id = existing.unwrap_or_else(|| {
                let i = next_id;
                next_id += 1;
                i
            });
            if id as usize >= metadata.len() {
                metadata.resize(id as usize + 1, AddedTokenFlags::default());
            }
            metadata[id as usize] = flags;
            entries.insert(id, (form, is_norm)); // if id existed, we overwrite it
            seen.insert(token.content.clone(), id);
        }

        // Partition the final set of tokens into two VocabStore and rebuild both from scratch.
        let (mut raw_tokens, mut norm_tokens) = (Vec::new(), Vec::new());
        for (id, (form, is_norm)) in entries {
            if is_norm {
                norm_tokens.push((form, id));
            } else {
                raw_tokens.push((form, id));
            }
        }
        self.token_metadata = metadata.into();
        self.vocab = Buckets::from_tokens(raw_tokens);
        self.normalized_vocab = Buckets::from_tokens(norm_tokens);
        Ok(total - ignored)
    }

    pub fn extract_and_normalize<N: Normalizer>(
        &self,
        _normalizer: Option<&N>,
        sequence: &str,
    ) -> PreTokenizedString {
        let bytes = sequence.as_bytes();
        let mut splits: Vec<(usize, usize, Option<u32>)> = Vec::new();
        let mut emit = 0;
        let mut search = 0;
        while search < bytes.len() {
            // Find the next candidate start and the bucket whose entries we should scan.
            match self.vocab.match_bytes(&bytes[search..]) {
                Some((id, match_start, match_len)) if match_len > 0 => {
                    // TODO: after matching non-normalized we have to match normalized
                    // match_bytes positions are relative to the slice we passed (&bytes[search..])
                    let mut match_start = search + match_start as usize;
                    let mut match_end = match_start + match_len as usize;
                    // single_word: reject unless the token is a standalone word (neither neighbour
                    // char is a word char). Checked on the raw byte bounds, before any strip.
                    if self.token_metadata[id as usize].single_word
                        && !is_single_word(bytes, search, match_start, match_end)
                    {
                        search = match_start + 1; // resume just past the rejected match
                        continue;
                    }
                    if self.token_metadata[id as usize].lstrip {
                        match_start =
                            skip_whitespace_backward(&bytes[..match_end], match_start).max(emit)
                    }
                    // FIXME: here we need to split again on normalized!
                    // I am not doing because it means allocating a new NormalizedString...
                    if match_start > emit {
                        splits.push((emit, match_start, None));
                    }
                    if self.token_metadata[id as usize].rstrip {
                        match_end = skip_whitespace_forward(bytes, match_end)
                    }
                    splits.push((match_start, match_end, Some(id)));
                    emit = match_end;
                    search = match_end;
                }
                // since match_bytes goes to the end, this means we reach the end.
                _ => break,
            }
        }
        if emit < bytes.len() {
            splits.push((emit, bytes.len(), None));
        }
        // FIXME: this will go away once we have the 0-allocation in the hot path :)
        let mut pre = PreTokenizedString::from(sequence);
        pre.split(|_, normalized| {
            Ok(splits
                .iter()
                .filter_map(|&(start, end, id)| {
                    let ns = normalized.slice(Range::Normalized(start..end))?;
                    let tokens = id
                        .map(|id| vec![Token::new(id, ns.get().to_string(), (0, ns.get().len()))]);
                    Some((ns, tokens))
                })
                .collect::<Vec<_>>())
        })
        .unwrap();
        pre
    }
}

impl PipelinePatternMatcher for AddedVocabulary {
    fn extract_next(
        &self,
        bytes: &[u8],
        search: usize,
        normalized: bool,
    ) -> Option<((usize, usize), u32)> {
        let vocab = if normalized {
            &self.vocab
        } else {
            &self.normalized_vocab
        };
        match vocab.match_bytes(&bytes[search..]) {
            Some((id, match_start, match_len)) if match_len > 0 => {
                let mut match_start = search + match_start as usize;
                let mut match_end = match_start + match_len as usize;
                // single_word: reject unless the token is a standalone word (neither neighbour
                // char is a word char). Checked on the raw byte bounds, before any strip.
                if self.token_metadata[id as usize].single_word
                    && !is_single_word(bytes, search, match_start, match_end)
                {
                    match_end += 1;
                }
                if self.token_metadata[id as usize].lstrip {
                    match_start =
                        skip_whitespace_backward(&bytes[..match_end], match_start).max(search)
                }
                if self.token_metadata[id as usize].rstrip {
                    match_end = skip_whitespace_forward(bytes, match_end)
                }
                Some(((match_start, match_end), id))
            }
            // since match_bytes goes to the end, this means we reach the end.
            _ => None,
        }
    }
}

impl Default for AddedVocabulary {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct AddedTokenWithId {
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
        // Serialize the logical added tokens (id + content + flags), ordered by id — NOT the derived
        // `Buckets`/`VocabStore`, which are rebuilt from this list on deserialize via `add_tokens`.
        let mut added_tokens: Vec<AddedTokenWithId> = self
            .get_added_tokens_decoder()
            .into_iter()
            .map(|(id, token)| AddedTokenWithId { id, token })
            .collect();
        added_tokens.sort_unstable_by_key(|o| o.id);

        let mut vocabulary = serializer.serialize_seq(Some(added_tokens.len()))?;
        for token in &added_tokens {
            vocabulary.serialize_element(token)?;
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
        println!("{:?}", vocab);
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
                    AddedToken::from("my", true).lstrip(true).rstrip(true),
                    AddedToken::from("name", true),
                    AddedToken::from("ony", true).single_word(true),
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
