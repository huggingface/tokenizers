use super::super::Result;
use super::buckets::{AddedTokenFlags, Buckets};
use crate::pipeline::Normalizer;
use crate::pipeline::PipelinePatternMatcher;
use crate::pre_tokenizers::whitespace::is_word_char;
use crate::utils::perfect_hash::{Mphf, mix};
use ahash::AHashMap;
use ptr_hash::PtrHashParams;
use std::collections::BTreeMap;
use std::fmt;
/// Represent a token added by the user on top of the existing Model vocabulary.
/// AddedToken can be configured to specify the behavior they should have in various situations
/// like:
///   - Whether they should only match single words
///   - Whether to include any whitespace on its left or right
#[derive(Debug, Clone, PartialEq, Eq)]
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
    ids: InternalIds,

    /// New fast path for normalize and extra needs:
    ///  - multi-bucket first bytes. If its len is 1, we use memchr; otherwise we just check each
    ///    byte on that table lookup. Its a 256-byte table.
    ///  - the actual buckets. We could use small vec here. Chose to impl it. Buckets give pointers
    ///    to the inner vocab store.
    ///  - the metadata of each token.
    token_metadata: Box<[AddedTokenFlags]>, // indexed by internal id

    normalized_vocab: Buckets,
    vocab: Buckets,
}

/// Renumbers the added tokens 0..n in vocab-id order, and maps between those internal ids and the
/// true vocab token ids.
#[derive(Clone)]
struct InternalIds {
    vocab_ids: Box<[u32]>,
    mphf: Mphf,
    /// `internal_by_slot[mphf.index(hash(vocab_id))]` is the internal id of `vocab_id`.
    internal_ids: Box<[u32]>,
}

impl InternalIds {
    fn new(sorted_vocab_ids: Box<[u32]>) -> Self {
        debug_assert!(
            sorted_vocab_ids.is_sorted(),
            "add_tokens collects the vocab ids from a BTreeMap, which yields them in order"
        );
        let keys: Vec<u64> = sorted_vocab_ids.iter().map(|&id| mix(id as u64)).collect();
        let mphf = Mphf::new(&keys, PtrHashParams::default_fast());
        // Unused slots hold 0 - `internal_id` rejects them by comparing the vocab id.
        let mut internal_ids = vec![0; mphf.max_index()].into_boxed_slice();
        for (internal_id, key) in keys.iter().enumerate() {
            internal_ids[mphf.index(key)] = internal_id as u32;
        }
        Self {
            vocab_ids: sorted_vocab_ids,
            mphf,
            internal_ids,
        }
    }

    fn vocab_id(&self, internal_id: u32) -> u32 {
        self.vocab_ids[internal_id as usize]
    }

    fn internal_id(&self, vocab_id: u32) -> Option<u32> {
        // An empty vocabulary has no slots.
        let internal_id = *self
            .internal_ids
            .get(self.mphf.index(&mix(vocab_id as u64)))?;
        (self.vocab_id(internal_id) == vocab_id).then_some(internal_id)
    }

    fn highest_vocab_id(&self) -> Option<u32> {
        self.vocab_ids.last().copied()
    }
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
            ids: InternalIds::new(Box::new([])),
            token_metadata: Box::new([]),
            normalized_vocab: Buckets::new(),
            vocab: Buckets::new(),
        }
    }

    /// Every added token with its internal id, raw tokens first.
    fn tokens(&self) -> impl Iterator<Item = (String, u32)> {
        self.vocab
            .tokens()
            .into_iter()
            .chain(self.normalized_vocab.tokens())
    }

    fn matcher(&self, normalized: bool) -> &Buckets {
        if normalized {
            &self.normalized_vocab
        } else {
            &self.vocab
        }
    }

    fn token_to_internal_id(&self, token: &str) -> Option<u32> {
        self.vocab
            .token_to_internal_id(token)
            .or_else(|| self.normalized_vocab.token_to_internal_id(token))
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
        self.tokens()
            .map(|(token, internal_id)| (token, self.ids.vocab_id(internal_id)))
            .collect::<AHashMap<String, u32>>()
    }

    /// Get the additional vocabulary with the AddedTokens
    /// TODO: this will be slow because we rebuild the added tokens
    pub fn get_added_tokens_decoder(&self) -> AHashMap<u32, AddedToken> {
        self.tokens()
            .map(|(token, internal_id)| {
                let m = &self.token_metadata[internal_id as usize];
                (
                    self.ids.vocab_id(internal_id),
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
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_internal_id(token)
            .map(|internal_id| self.ids.vocab_id(internal_id))
    }

    /// Return the string form of an added token used during **decoding**.
    ///
    /// For tokens that were normalized on the way *in* (e.g. byte-level encoding),
    /// this returns the cached normalized form so that the configured `Decoder` can
    /// invert the transformation correctly. For all other tokens, the original
    /// `content` is returned.
    pub fn simple_id_to_token(&self, id: u32) -> Option<String> {
        let internal_id = self.ids.internal_id(id)?;
        self.matcher(self.token_metadata[internal_id as usize].normalized)
            .internal_id_to_token(internal_id)
    }

    /// Check if a token is a special token
    pub fn is_special_token(&self, token: &str) -> bool {
        self.token_to_internal_id(token)
            .is_some_and(|internal_id| self.token_metadata[internal_id as usize].special)
    }

    /// Add some special tokens to the vocabulary
    pub fn add_special_tokens<N: Normalizer>(
        &mut self,
        tokens: impl IntoIterator<Item = AddedToken>,
        model_vocab_size: usize,
        model_token_to_id: impl Fn(&str) -> Option<u32>,
        normalizer: Option<&N>,
    ) -> Result<usize> {
        self.add_tokens(tokens, model_vocab_size, model_token_to_id, normalizer)
    }

    /// Add some tokens to the vocabulary
    pub fn add_tokens<N: Normalizer>(
        &mut self,
        tokens: impl IntoIterator<Item = AddedToken>,
        model_vocab_size: usize,
        model_token_to_id: impl Fn(&str) -> Option<u32>,
        normalizer: Option<&N>,
    ) -> Result<usize> {
        let mut ignored = 0;
        let mut total = 0;

        // Added ids start above the model vocab; a token already in the model reuses the model id.
        let model_size = model_vocab_size as u32;
        let mut next_id = self
            .ids
            .highest_vocab_id()
            .map_or(0, |id| id + 1)
            .max(model_size);

        let mut entries: BTreeMap<u32, (Vec<u8>, AddedTokenFlags)> = BTreeMap::new();
        let existing_tokens = self
            .vocab
            .token_bytes()
            .into_iter()
            .chain(self.normalized_vocab.token_bytes());
        for (form, internal_id) in existing_tokens {
            let flags = self.token_metadata[internal_id as usize].clone();
            entries.insert(self.ids.vocab_id(internal_id), (form, flags));
        }
        let mut seen: AHashMap<String, u32> = AHashMap::new();

        for token in tokens {
            total += 1;
            if token.content.is_empty() {
                ignored += 1;
                continue;
            }
            let flags = AddedTokenFlags::from(&token);
            let norm_form: String = match normalizer {
                // An added token is matched anywhere in the text, so its form is normalized as if
                // it never opened the sequence.
                Some(n) => n.normalize(&token.content, 1)?.into_owned(),
                None => token.content.clone(),
            };
            let form = if flags.normalized {
                norm_form.clone().into_bytes()
            } else {
                token.content.clone().into_bytes()
            };
            // token could be in model, in added vocab, in added vocab normalized
            let existing = seen
                .get(&token.content)
                .copied()
                .or_else(|| model_token_to_id(&token.content))
                .or_else(|| {
                    self.vocab
                        .token_to_internal_id(&token.content)
                        .or_else(|| self.normalized_vocab.token_to_internal_id(&norm_form))
                        .map(|internal_id| self.ids.vocab_id(internal_id))
                });
            if let Some(id) = existing
                && entries.get(&id).map(|(_, f)| f) == Some(&flags)
            {
                ignored += 1;
                continue;
            }
            let id = existing.unwrap_or_else(|| {
                let i = next_id;
                next_id += 1;
                i
            });
            entries.insert(id, (form, flags));
            seen.insert(token.content.clone(), id);
        }

        let mut vocab_ids = Vec::with_capacity(entries.len());
        let mut token_metadata = Vec::with_capacity(entries.len());
        let (mut raw_tokens, mut norm_tokens) = (Vec::new(), Vec::new());
        for (internal_id, (vocab_id, (form, flags))) in entries.into_iter().enumerate() {
            let internal_id = internal_id as u32;
            if flags.normalized {
                norm_tokens.push((form, internal_id));
            } else {
                raw_tokens.push((form, internal_id));
            }
            vocab_ids.push(vocab_id);
            token_metadata.push(flags);
        }
        let vocab = Buckets::from_tokens(raw_tokens)?;
        let normalized_vocab = Buckets::from_tokens(norm_tokens)?;
        self.ids = InternalIds::new(vocab_ids.into());
        self.token_metadata = token_metadata.into();
        self.vocab = vocab;
        self.normalized_vocab = normalized_vocab;
        Ok(total - ignored)
    }
}

impl PipelinePatternMatcher for AddedVocabulary {
    fn extract_next(
        &self,
        bytes: &[u8],
        search_offset: usize,
        normalized: bool,
        encode_special_tokens: bool,
    ) -> Option<((usize, usize), u32)> {
        let vocab = self.matcher(normalized);
        let mut search = search_offset;
        while search < bytes.len() {
            // since match_bytes goes to the end, no match means we reached the end.
            let (internal_id, start, len) = match vocab.match_bytes(&bytes[search..]) {
                Some((internal_id, start, len)) if len > 0 => (internal_id, start, len),
                _ => return None,
            };
            let mut match_start = search + start as usize;
            let mut match_end = match_start + len as usize;
            let metadata = &self.token_metadata[internal_id as usize];
            if encode_special_tokens && metadata.special {
                search = match_end;
                continue;
            }
            // single_word: reject unless the token is a standalone word (neither neighbour
            // char is a word char). Checked on the raw byte bounds, before any strip.
            if metadata.single_word && !is_single_word(bytes, search, match_start, match_end) {
                search = match_start + 1;
                continue;
            }
            if metadata.lstrip {
                match_start =
                    skip_whitespace_backward(&bytes[..match_end], match_start).max(search_offset)
            }
            if metadata.rstrip {
                match_end = skip_whitespace_forward(bytes, match_end)
            }
            return Some(((match_start, match_end), self.ids.vocab_id(internal_id)));
        }
        None
    }
}

impl Default for AddedVocabulary {
    fn default() -> Self {
        Self::new()
    }
}

// `AddedTokenWithId` and `impl Serialize for AddedVocabulary` are in `super::serialization`, next to
// this file -- an added vocabulary writes its *logical* token list and never the derived
// `Buckets`/`VocabStore`, so that impl is hand-written rather than a derive.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalizers::utils::Lowercase;
    use crate::pipeline::{Segment, SpecialSegmentIterator};

    struct ModelMock {
        vocab: AHashMap<String, u32>,
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
            Self { vocab }
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            self.vocab.get(token).copied()
        }
        fn get_vocab_size(&self) -> usize {
            self.vocab.len()
        }
    }

    fn sparse_vocabulary() -> AddedVocabulary {
        let model = ModelMock::new(&[("[CLS]", 5)]);
        let mut vocab = AddedVocabulary::new();
        vocab
            .add_tokens(
                [
                    AddedToken::from("<|end|>", true),
                    AddedToken::from("[CLS]", true),
                    AddedToken::from("hello", false),
                ],
                200_000,
                |t| model.token_to_id(t),
                None::<&Lowercase>,
            )
            .unwrap();
        vocab
    }

    #[test]
    fn empty_vocabulary_has_no_ids() {
        assert_eq!(AddedVocabulary::new().simple_id_to_token(0), None);
    }

    #[test]
    fn sparse_ids_round_trip() {
        let vocab = sparse_vocabulary();
        for (token, id) in [("<|end|>", 200_000), ("[CLS]", 5), ("hello", 200_001)] {
            assert_eq!(vocab.token_to_id(token), Some(id));
            assert_eq!(vocab.simple_id_to_token(id).as_deref(), Some(token));
        }
        assert_eq!(vocab.simple_id_to_token(6), None);
        assert_eq!(vocab.simple_id_to_token(200_002), None);
        assert!(vocab.is_special_token("[CLS]"));
        assert!(!vocab.is_special_token("hello"));

        let text = b"say hello<|end|>";
        assert_eq!(
            vocab.extract_next(text, 0, false, false),
            Some(((9, 16), 200_000))
        );
        assert_eq!(
            vocab.extract_next(text, 0, true, false),
            Some(((4, 9), 200_001))
        );
    }

    #[test]
    fn changing_normalized_keeps_the_vocab_id() {
        let mut vocab = sparse_vocabulary();
        let text = b"say hello<|end|>";
        let model = ModelMock::new(&[]);
        vocab
            .add_tokens(
                [AddedToken::from("<|end|>", false)],
                200_000,
                |t| model.token_to_id(t),
                None::<&Lowercase>,
            )
            .unwrap();
        assert_eq!(
            vocab.extract_next(text, 0, true, false),
            Some(((4, 9), 200_001))
        );
        assert_eq!(
            vocab.extract_next(b"<|end|>", 0, true, false),
            Some(((0, 7), 200_000))
        );
        assert_eq!(vocab.extract_next(text, 0, false, false), None);
    }

    #[test]
    fn can_add_tokens() {
        let model = ModelMock::new(&[("test", 0), ("tost", 1)]);
        let mut vocab = AddedVocabulary::new();
        let normalizer: Option<&Lowercase> = None;

        // Add tokens normally
        assert_eq!(
            vocab
                .add_tokens(
                    [AddedToken::from("added_token_1", false)],
                    model.get_vocab_size(),
                    |t| model.token_to_id(t),
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
                    model.get_vocab_size(),
                    |t| model.token_to_id(t),
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
                .add_tokens(
                    [added_token.clone()],
                    model.get_vocab_size(),
                    |t| model.token_to_id(t),
                    normalizer
                )
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
        let normalizer: Option<&Lowercase> = None;
        // Add tokens normally
        assert_eq!(
            vocab
                .add_special_tokens(
                    [AddedToken::from("added_token_1", true)],
                    model.get_vocab_size(),
                    |t| model.token_to_id(t),
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
                    model.get_vocab_size(),
                    |t| model.token_to_id(t),
                    normalizer
                )
                .unwrap(),
            1
        );
        assert_eq!(vocab.len(), 2);

        // Can add tokens already covered by the model
        assert_eq!(
            vocab
                .add_special_tokens(
                    [AddedToken::from("test", true)],
                    model.get_vocab_size(),
                    |t| model.token_to_id(t),
                    normalizer
                )
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
                model.get_vocab_size(),
                |t| model.token_to_id(t),
                normalizer,
            )
            .unwrap();
        assert_eq!(vocab.len(), 5); // New token was added
        assert_eq!(vocab.get_vocab()["another_two"], 4); // New token was added, but the index is not the length of the vocab

        // Let's add an already added token again, but change normalized
        assert_eq!(
            vocab
                .add_special_tokens(
                    [AddedToken::from("another_two", true)],
                    model.get_vocab_size(),
                    |t| model.token_to_id(t),
                    normalizer
                )
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
    fn normalized_tokens_are_stored_by_normalized_form() {
        // The Buckets rewrite doesn't retain the original content of normalized tokens:
        // they are stored (and decoded) by their normalized form.
        let model = ModelMock::new(&[]);
        let mut vocab = AddedVocabulary::new();
        let normalizer = Lowercase;

        vocab
            .add_tokens(
                [
                    AddedToken::from("Hello", false),
                    AddedToken::from("[CLS]", true),
                ],
                model.get_vocab_size(),
                |t| model.token_to_id(t),
                Some(&normalizer),
            )
            .unwrap();

        let decoder = vocab.get_added_tokens_decoder();
        // normalized=true → stored under the normalizer output
        assert!(decoder.values().any(|t| t.content == "hello"));
        // normalized=false → stored under the original content
        assert!(decoder.values().any(|t| t.content == "[CLS]"));

        assert_eq!(vocab.simple_id_to_token(0).unwrap(), "hello");
        assert_eq!(vocab.simple_id_to_token(1).unwrap(), "[CLS]");
    }

    #[test]
    fn is_special_token_true_for_a_normalized_special_token() {
        let model = ModelMock::new(&[]);
        let mut vocab = AddedVocabulary::new();
        let normalizer: Option<&Lowercase> = None;
        vocab
            .add_special_tokens(
                [AddedToken::from("<unk>", true).normalized(true)],
                model.get_vocab_size(),
                |t| model.token_to_id(t),
                normalizer,
            )
            .unwrap();

        assert!(vocab.is_special_token("<unk>"));
    }

    /// Drive the real `SpecialSegmentIterator` over an `AddedVocabulary`, mapping each
    /// segment to `(text, id)` so `extract_next`'s behaviour is easy to assert.
    fn segments(
        vocab: &AddedVocabulary,
        input: &str,
        normalized: bool,
        encode_special_tokens: bool,
    ) -> Vec<(Option<String>, Option<u32>)> {
        SpecialSegmentIterator::new(input, vocab, normalized, encode_special_tokens)
            .map(|segment| match segment {
                Segment::Text { text, .. } => (Some(text.to_string()), None),
                Segment::SpecialToken(id) => (None, Some(id)),
            })
            .collect()
    }

    #[test]
    fn segment_iterator_carves_raw_added_tokens() {
        let model = ModelMock::new(&[]);
        let mut vocab = AddedVocabulary::new();
        let normalizer: Option<&Lowercase> = None;
        vocab
            .add_special_tokens(
                [
                    AddedToken::from("[CLS]", true),
                    AddedToken::from("[SEP]", true),
                ],
                model.get_vocab_size(),
                |t| model.token_to_id(t),
                normalizer,
            )
            .unwrap();

        // Special tokens are declared on raw text, so a single raw pass carves them out.
        assert_eq!(
            segments(&vocab, "[CLS] hello [SEP]", false, false),
            vec![
                (None, Some(0)),
                (Some(" hello ".to_string()), None),
                (None, Some(1)),
            ]
        );
    }

    #[test]
    fn segment_iterator_respects_single_word() {
        let model = ModelMock::new(&[]);
        let mut vocab = AddedVocabulary::new();
        let normalizer: Option<&Lowercase> = None;
        vocab
            .add_tokens(
                [AddedToken::from("mask", false)
                    .normalized(false)
                    .single_word(true)],
                model.get_vocab_size(),
                |t| model.token_to_id(t),
                normalizer,
            )
            .unwrap();

        // Standalone `mask` matches; the `mask` inside `bitmask` does not.
        assert_eq!(
            segments(&vocab, "a mask bitmask", false, false),
            vec![
                (Some("a ".to_string()), None),
                (None, Some(0)),
                (Some(" bitmask".to_string()), None),
            ]
        );
    }

    #[test]
    fn segment_iterator_applies_lstrip_rstrip() {
        let model = ModelMock::new(&[]);
        let mut vocab = AddedVocabulary::new();
        let normalizer: Option<&Lowercase> = None;
        vocab
            .add_tokens(
                [AddedToken::from("<mask>", false)
                    .normalized(false)
                    .lstrip(true)
                    .rstrip(true)],
                model.get_vocab_size(),
                |t| model.token_to_id(t),
                normalizer,
            )
            .unwrap();

        // The spaces around `<mask>` are absorbed into the matched span, so the surrounding
        // text segments come back trimmed.
        assert_eq!(
            segments(&vocab, "hi <mask> there", false, false),
            vec![
                (Some("hi".to_string()), None),
                (None, Some(0)),
                (Some("there".to_string()), None),
            ]
        );
    }

    #[test]
    fn segment_iterator_honors_encode_special_tokens() {
        let model = ModelMock::new(&[]);
        let mut vocab = AddedVocabulary::new();
        let normalizer: Option<&Lowercase> = None;
        vocab
            .add_special_tokens(
                [AddedToken::from("[CLS]", true)],
                model.get_vocab_size(),
                |t| model.token_to_id(t),
                normalizer,
            )
            .unwrap();

        // Default: the special token is carved out.
        assert_eq!(
            segments(&vocab, "[CLS] hi", false, false),
            vec![(None, Some(0)), (Some(" hi".to_string()), None)]
        );

        // encode_special_tokens = true: the special token is left in the text for the model.
        assert_eq!(
            segments(&vocab, "[CLS] hi", false, true),
            vec![(Some("[CLS] hi".to_string()), None)]
        );
    }

    #[test]
    fn segment_iterator_selects_raw_vs_normalized_matcher() {
        let model = ModelMock::new(&[]);
        let mut vocab = AddedVocabulary::new();
        let normalizer = Lowercase;
        // Non-special token, normalized by default → stored in the normalized matcher.
        vocab
            .add_tokens(
                [AddedToken::from("mask", false)],
                model.get_vocab_size(),
                |t| model.token_to_id(t),
                Some(&normalizer),
            )
            .unwrap();

        // Raw pass: the token lives in the normalized matcher, so nothing is carved.
        assert_eq!(
            segments(&vocab, "a mask", false, false),
            vec![(Some("a mask".to_string()), None)]
        );
        // Normalized pass over already-normalized text: the token is matched.
        assert_eq!(
            segments(&vocab, "a mask", true, false),
            vec![(Some("a ".to_string()), None), (None, Some(0))]
        );
    }
}
