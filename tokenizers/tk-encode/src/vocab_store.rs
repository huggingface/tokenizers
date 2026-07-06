//! Legacy, map-backed token store: the pre-#2129 behavior, kept as a drop-in twin of the fast
//! [`BucketVocabStore`](crate::bucket_vocab_store::BucketVocabStore). Both expose the same public
//! API, so the models compile against either — the legacy `Tokenizer` path uses this store and the
//! `PipelineTokenizer` path uses the bucket version, which lets the two be compared apples to apples.

use ahash::AHashMap;

use crate::bucket_vocab_store::BucketVocabStore;

#[derive(Clone, PartialEq)]
pub enum VocabStoreWrapper {
    Legacy(LegacyVocabStore),
    Bucket(BucketVocabStore),
}

impl VocabStoreWrapper {
    pub fn new() -> Self {
        Self::Legacy(LegacyVocabStore::new())
    }

    pub fn build(tokens: Vec<(Vec<u8>, u32)>) -> Self {
        Self::Legacy(LegacyVocabStore::build(tokens))
    }
}

impl Default for VocabStoreWrapper {
    fn default() -> Self {
        Self::new()
    }
}

impl VocabStoreWrapper {
    pub fn into_bucket(self) -> Self {
        match self {
            Self::Legacy(legacy) => Self::Bucket(legacy.into()),
            bucket => bucket,
        }
    }
}

pub trait VocabStore {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn get_bytes(&self, q: &[u8]) -> Option<u32>;
    fn id_to_token_bytes(&self, id: u32) -> Option<&[u8]>;
    fn content(&self) -> Vec<(String, u32)>;
    fn get_vocab(&self) -> Vec<(String, u32)>;
    fn byte_content(&self) -> Vec<(Vec<u8>, u32)>;

    #[inline]
    fn token_to_id(&self, s: &str) -> Option<u32> {
        self.get_bytes(s.as_bytes())
    }

    #[inline]
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.id_to_token_bytes(id)
            .map(|b| String::from_utf8_lossy(b).into_owned())
    }
}

macro_rules! delegate {
    ($(fn $name:ident(&self $(, $arg:ident: $ty:ty)*) -> $ret:ty;)*) => {$(
        #[inline]
        fn $name(&self $(, $arg: $ty)*) -> $ret {
            match self {
                Self::Legacy(v) => v.$name($($arg),*),
                Self::Bucket(v) => v.$name($($arg),*),
            }
        }
    )*};
}

impl VocabStore for VocabStoreWrapper {
    delegate! {
        fn len(&self) -> usize;
        fn is_empty(&self) -> bool;
        fn get_bytes(&self, q: &[u8]) -> Option<u32>;
        fn id_to_token_bytes(&self, id: u32) -> Option<&[u8]>;
        fn content(&self) -> Vec<(String, u32)>;
        fn get_vocab(&self) -> Vec<(String, u32)>;
        fn byte_content(&self) -> Vec<(Vec<u8>, u32)>;
    }
}

#[derive(Clone, Debug, Default)]
pub struct LegacyVocabStore {
    by_bytes: AHashMap<Vec<u8>, u32>,
    by_id: AHashMap<u32, Vec<u8>>,
}

impl PartialEq for LegacyVocabStore {
    fn eq(&self, other: &Self) -> bool {
        self.by_bytes == other.by_bytes
    }
}

impl LegacyVocabStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn build(tokens: Vec<(Vec<u8>, u32)>) -> Self {
        let mut by_bytes = AHashMap::with_capacity(tokens.len());
        let mut by_id = AHashMap::with_capacity(tokens.len());
        for (bytes, id) in tokens {
            by_bytes.insert(bytes.clone(), id);
            by_id.insert(id, bytes);
        }
        Self { by_bytes, by_id }
    }
}

impl VocabStore for LegacyVocabStore {
    #[inline]
    fn get_bytes(&self, q: &[u8]) -> Option<u32> {
        self.by_bytes.get(q).copied()
    }

    #[inline]
    fn id_to_token_bytes(&self, id: u32) -> Option<&[u8]> {
        self.by_id.get(&id).map(|b| b.as_slice())
    }

    fn len(&self) -> usize {
        self.by_bytes.len()
    }

    fn is_empty(&self) -> bool {
        self.by_bytes.is_empty()
    }

    fn content(&self) -> Vec<(String, u32)> {
        self.by_id
            .iter()
            .map(|(id, b)| (String::from_utf8_lossy(b).into_owned(), *id))
            .collect()
    }

    fn get_vocab(&self) -> Vec<(String, u32)> {
        self.content()
    }

    fn byte_content(&self) -> Vec<(Vec<u8>, u32)> {
        self.by_id.iter().map(|(id, b)| (b.clone(), *id)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_and_lookup() {
        let vocab = LegacyVocabStore::build(vec![
            (b"a".to_vec(), 0),
            (b"bb".to_vec(), 5),
            (b"ccc".to_vec(), 100),
        ]);
        assert_eq!(vocab.token_to_id("a"), Some(0));
        assert_eq!(vocab.token_to_id("bb"), Some(5));
        assert_eq!(vocab.token_to_id("ccc"), Some(100));
        assert_eq!(vocab.token_to_id("zzz"), None);
        assert_eq!(vocab.id_to_token(100), Some("ccc".to_string()));
        assert_eq!(vocab.id_to_token(1), None);
        assert_eq!(vocab.len(), 3);
    }

    #[test]
    fn empty() {
        let vocab = LegacyVocabStore::new();
        assert!(vocab.is_empty());
        assert_eq!(vocab.token_to_id("x"), None);
        assert_eq!(vocab.id_to_token(0), None);
    }

    #[test]
    fn eq_is_content_based_order_independent() {
        let a = LegacyVocabStore::build(vec![(b"x".to_vec(), 0), (b"y".to_vec(), 9)]);
        let b: LegacyVocabStore =
            LegacyVocabStore::build(vec![(b"y".to_vec(), 9), (b"x".to_vec(), 0)]);
        let c = LegacyVocabStore::build(vec![(b"x".to_vec(), 0), (b"z".to_vec(), 9)]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn roundtrip_byte_content() {
        let toks: Vec<(Vec<u8>, u32)> = ["the", "\u{2581}hello", "\u{4eca}", "\n"]
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_bytes().to_vec(), i as u32))
            .collect();
        let vocab = LegacyVocabStore::build(toks.clone());
        let mut got = vocab.byte_content();
        got.sort();
        let mut want = toks;
        want.sort();
        assert_eq!(got, want);
    }
}
