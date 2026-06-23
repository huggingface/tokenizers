use super::{super::OrderedVocabIter, Error, Pair, Word};
use crate::tokenizer::{Model, Result, Token};
use crate::utils::byte_level::BYTES_CHAR_LOOKUP;
use crate::utils::cache::{DEFAULT_CACHE_CAPACITY, MAX_LENGTH};
use crate::utils::iter::ResultShunt;
use ahash::AHashMap;
use serde_json::Value;
use std::borrow::Cow;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

use std::collections::HashMap;
use std::str::from_utf8_unchecked;
use std::{
    fs::File,
    io::prelude::*,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

pub type Vocab = AHashMap<String, u32>;
pub type VocabR = AHashMap<u32, String>;
pub type MergeMap = AHashMap<Pair, (u32, u32)>;

/// Process-wide monotonic counter used to assign a unique generation id
/// to every `BpeCache`, so per-instance thread-local caches never collide.
static NEXT_CACHE_ID: AtomicU64 = AtomicU64::new(0);

/// Per-BPE cache descriptor.
///
/// BPE no longer keeps a shared `RwLock<AHashMap>` cache: the encode hot
/// path reads and writes only the thread-local `BPE_LOCAL_CACHE` below,
/// keyed by `(BpeCache::id, sequence)`.  This struct only carries the
/// per-instance generation id and capacity so existing `clear_cache()`
/// and `resize_cache()` APIs keep their meaning: `clear()` bumps the id,
/// invalidating every thread's entries for this BPE in one shot.
#[derive(Debug)]
pub(crate) struct BpeCache {
    id: AtomicU64,
    pub capacity: usize,
}

// Matches the previous `Cache` impl: we never compare caches by value.
impl PartialEq for BpeCache {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl BpeCache {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            id: AtomicU64::new(NEXT_CACHE_ID.fetch_add(1, Ordering::Relaxed)),
            capacity,
        }
    }

    /// Return a fresh `BpeCache` with the same capacity but a new id,
    /// used by `impl Clone for BPE`.
    pub(crate) fn fresh(&self) -> Self {
        Self::new(self.capacity)
    }

    /// Current generation id.  Bumped on `clear()`.
    pub(crate) fn id(&self) -> u64 {
        self.id.load(Ordering::Relaxed)
    }

    /// Invalidate every thread's thread-local entries for this BPE by
    /// advancing the generation id; the next lookup re-computes.
    pub(crate) fn clear(&self) {
        self.id.store(
            NEXT_CACHE_ID.fetch_add(1, Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }

    pub(crate) fn resize(&mut self, capacity: usize) {
        self.capacity = capacity;
    }
}

thread_local! {
    /// Per-thread BPE tokenization cache.  This is the only BPE cache
    /// on the hot path: there is no shared global map, so lookups and
    /// inserts need no atomic synchronization at all.  The outer map is
    /// keyed by `BpeCache::id` so multiple `BPE` instances sharing the
    /// same rayon worker thread never see each other's entries.
    static BPE_LOCAL_CACHE: RefCell<AHashMap<u64, AHashMap<String, Word>>> =
        RefCell::new(AHashMap::new());

    /// Byte-space sibling of `BPE_LOCAL_CACHE`
    /// Keyed by raw bytes instead of the byte-level-mapped string. Used by the byte-level bypass path.
    /// Storing the vocab in byte space at load time allow merging the 2 statics.
    static BPE_LOCAL_CACHE_BYTES: RefCell<AHashMap<u64, AHashMap<Vec<u8>, Word>>> =
        RefCell::new(AHashMap::new());
}
pub type Merges = Vec<(String, String)>;

struct Config {
    files: Option<(String, String)>,
    vocab: Vocab,
    merges: Merges,
    cache_capacity: usize,
    dropout: Option<f32>,
    unk_token: Option<String>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    fuse_unk: bool,
    byte_fallback: bool,
    ignore_merges: bool,
    byte_level_bypass: Option<ByteLevelBypass>,
}

/// A `BpeBuilder` can be used to create a `BPE` model with a custom configuration.
pub struct BpeBuilder {
    config: Config,
}

impl Default for BpeBuilder {
    fn default() -> Self {
        Self {
            config: Config {
                files: None,
                vocab: AHashMap::new(),
                merges: vec![],
                cache_capacity: DEFAULT_CACHE_CAPACITY,
                dropout: None,
                unk_token: None,
                continuing_subword_prefix: None,
                end_of_word_suffix: None,
                fuse_unk: false,
                byte_fallback: false,
                ignore_merges: false,
                byte_level_bypass: None,
            },
        }
    }
}

impl BpeBuilder {
    /// Constructs a new `BpeBuilder`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the input files.
    #[must_use]
    pub fn files(mut self, vocab: String, merges: String) -> Self {
        self.config.files = Some((vocab, merges));
        self
    }

    /// Set the vocab (token -> ID) and merges mappings.
    #[must_use]
    pub fn vocab_and_merges<V: Into<AHashMap<String, u32>>>(
        mut self,
        vocab: V,
        merges: Merges,
    ) -> Self {
        self.config.vocab = vocab.into();
        self.config.merges = merges;
        self
    }

    /// Set the cache's capacity. Set to 0 if you want to disable caching.
    #[must_use]
    pub fn cache_capacity(mut self, capacity: usize) -> Self {
        self.config.cache_capacity = capacity;
        self
    }

    /// Use [dropout](https://arxiv.org/abs/1910.13267) with the model.
    #[must_use]
    pub fn dropout(mut self, dropout: f32) -> Self {
        self.config.dropout = Some(dropout);
        self
    }

    /// Set the `UNK` token for the vocab.
    #[must_use]
    pub fn unk_token(mut self, unk_token: String) -> Self {
        self.config.unk_token = Some(unk_token);
        self
    }

    /// Set the `continuing_subword_prefix` option.
    #[must_use]
    pub fn continuing_subword_prefix(mut self, prefix: String) -> Self {
        self.config.continuing_subword_prefix = Some(prefix);
        self
    }

    /// Set the `end_of_word_suffix` option.
    #[must_use]
    pub fn end_of_word_suffix(mut self, prefix: String) -> Self {
        self.config.end_of_word_suffix = Some(prefix);
        self
    }

    /// Set the `fuse_unk` option.
    #[must_use]
    pub fn fuse_unk(mut self, fuse_unk: bool) -> Self {
        self.config.fuse_unk = fuse_unk;
        self
    }

    /// Set the `byte_fallback` option.
    #[must_use]
    pub fn byte_fallback(mut self, byte_fallback: bool) -> Self {
        self.config.byte_fallback = byte_fallback;
        self
    }
    /// Set the `ignore_merges` option.
    #[must_use]
    pub fn ignore_merges(mut self, ignore_merges: bool) -> Self {
        self.config.ignore_merges = ignore_merges;
        self
    }

    /// Decide whether this config is eligible for the byte-level projection bypass.
    ///
    /// When eligible, encoding skips the projection into byte-level "vocabulary space"
    /// (see `BYTES_CHAR_LOOKUP`) and feeds raw bytes' token-id straight into Word
    /// bypassing both the byte→char string mapping AND the prefix/suffix/unk handling that
    /// `merge_word` performs. Only correct when *every* config option is compatible.
    ///
    /// Returns `None` when not eligible: encoding then defaults to the slow path, which is
    /// always correct.
    fn byte_level_bypass(&self) -> Option<ByteLevelBypass> {
        // Destructured exhaustively on purpose: adding a Config field is a compile error
        // until it is classified here.
        let Config {
            // Unsupported by the fast path — gate on these.
            continuing_subword_prefix,
            end_of_word_suffix,

            // If the vocab maps all 256 single bytes, these have no effect.
            unk_token: _,
            fuse_unk: _,
            byte_fallback: _,

            // Irrelevant to eligibility.
            dropout: _,
            ignore_merges: _,
            files: _,
            vocab,
            merges: _,
            cache_capacity: _,
            byte_level_bypass: _,
        } = &self.config;
        compute_byte_level_bypass(
            vocab,
            continuing_subword_prefix.as_deref(),
            end_of_word_suffix.as_deref(),
        )
    }

    /// Returns a `BPE` model that uses the `BpeBuilder`'s configuration.
    pub fn build(mut self) -> Result<BPE> {
        // Validate dropout.
        if let Some(p) = self.config.dropout {
            if !(0.0..=1.0).contains(&p) {
                return Err(Error::InvalidDropout.into());
            }
        }

        // Read files if necessary
        if let Some((vocab, merges)) = self.config.files.take() {
            let (v, m) = BPE::read_file(&vocab, &merges)?;
            self.config.vocab = v;
            self.config.merges = m;
        }
        self.config.byte_level_bypass = self.byte_level_bypass();

        let mut max_len = 0;
        let vocab_r = self
            .config
            .vocab
            .iter()
            .map(|(key, val)| {
                if max_len < key.len() {
                    max_len = key.len();
                }
                (*val, key.to_owned())
            })
            .collect();
        let cache = match self.config.cache_capacity {
            0 => None,
            capacity => Some(BpeCache::new(capacity)),
        };

        let vocab = self.config.vocab;
        let prefix_len = if let Some(prefix) = &self.config.continuing_subword_prefix {
            prefix.len()
        } else {
            0
        };
        let mut buffer: Vec<u8> = vec![0; max_len];
        let merge_map: MergeMap = self
            .config
            .merges
            .into_iter()
            .enumerate()
            .map(|(i, (a, b))| -> Result<(Pair, (u32, u32))> {
                let a_id = vocab
                    .get(&a)
                    .ok_or_else(|| Error::MergeTokenOutOfVocabulary(a.to_owned()))?;
                let b_id = vocab
                    .get(&b)
                    .ok_or_else(|| Error::MergeTokenOutOfVocabulary(b.to_owned()))?;
                buffer[0..a.len()].copy_from_slice(a.as_bytes());
                let b_len = b.len() - prefix_len;
                let merge_len = a.len() + b_len;
                buffer[a.len()..merge_len].copy_from_slice(&b.as_bytes()[prefix_len..]);
                // SAFETY: buffer contains a concatenation of two valid UTF-8 strings, so it is itself valid UTF-8, even considering prefix_len
                let new_token = unsafe { from_utf8_unchecked(&buffer[..merge_len]) };
                let new_id = vocab
                    .get(new_token)
                    .ok_or_else(|| Error::MergeTokenOutOfVocabulary(new_token.to_owned()))?;
                Ok(((*a_id, *b_id), (i as u32, *new_id)))
            })
            .collect::<Result<MergeMap>>()?;

        // merges.insert(pair, (rank as u32, *new_id));

        Ok(BPE {
            vocab,
            vocab_r,
            merges: merge_map,
            cache,
            dropout: self.config.dropout,
            unk_token: self.config.unk_token,
            continuing_subword_prefix: self.config.continuing_subword_prefix,
            end_of_word_suffix: self.config.end_of_word_suffix,
            fuse_unk: self.config.fuse_unk,
            byte_fallback: self.config.byte_fallback,
            ignore_merges: self.config.ignore_merges,
            byte_level_bypass: self.config.byte_level_bypass,
        })
    }
}

#[derive(PartialEq, Clone)]
pub(crate) struct ByteLevelBypass {
    /// Runtime toggle.
    /// Eligibility is intrinsic to the model and computed eagerly, but the fast path stays disabled
    /// until the `Tokenizer` confirms the surrounding pipeline (normalizer, pretokenizer, etc)
    /// is a no-op byte-level pipeline.
    pub(crate) active: bool,
    /// Maps each raw byte (0..=255) to its token id in the vocabulary.
    pub(crate) byte_to_token_id: Box<[u32; 256]>,
}

/// Single source of truth for byte-level fast-path eligibility.
/// Any path that rewrites `vocab`/prefix/suffix (builder, trainer) MUST re-run this so the bypass never goes stale.
pub(crate) fn compute_byte_level_bypass(
    vocab: &Vocab,
    continuing_subword_prefix: Option<&str>,
    end_of_word_suffix: Option<&str>,
) -> Option<ByteLevelBypass> {
    if continuing_subword_prefix.is_some_and(|p| !p.is_empty())
        || end_of_word_suffix.is_some_and(|s| !s.is_empty())
    {
        return None;
    }
    let mut table = [0u32; 256];
    for byte in 0..=255u8 {
        table[byte as usize] = *vocab.get(&BYTES_CHAR_LOOKUP[byte as usize].to_string())?;
    }
    Some(ByteLevelBypass {
        active: false,
        byte_to_token_id: Box::new(table),
    })
}

/// A [Byte Pair Encoding](https://www.aclweb.org/anthology/P16-1162/) model.
#[derive(PartialEq)]
pub struct BPE {
    /// The vocabulary assigns a number to each token.
    pub vocab: Vocab,
    /// Reversed vocabulary, to rebuild sentences.
    pub vocab_r: VocabR,
    /// Contains the mapping between Pairs and their (rank, new_id).
    pub merges: MergeMap,
    /// Contains the cache for optimizing the encoding step.
    cache: Option<BpeCache>,
    /// Dropout probability for merges. 0.0 = no dropout is the default. At 1.0, tokenization will
    /// perform no merges, so the result will just be characters.
    pub dropout: Option<f32>,
    /// The unknown token to be used when we encounter an unknown char
    pub unk_token: Option<String>,
    /// An optional prefix to use on any subword that exist only behind another one
    pub continuing_subword_prefix: Option<String>,
    /// An optional suffix to characterize and end-of-word subword
    pub end_of_word_suffix: Option<String>,
    /// Do multiple unk tokens get fused
    pub fuse_unk: bool,
    /// Byte fallback from sentence pieces, instead of UNK, uses `"<0x00>"`
    /// for each byte in the unk token
    pub byte_fallback: bool,
    /// Whether or not to direct output words if they are part of the vocab.
    pub ignore_merges: bool,

    /// If Some, the model is eligible to the "byte-level bypass" that skips byte-level projection
    /// Eligibility is computed in [`compute_byte_level_bypass`]
    pub(crate) byte_level_bypass: Option<ByteLevelBypass>,
}

impl std::fmt::Debug for BPE {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("BPE")
            .field("dropout", &self.dropout)
            .field("unk_token", &self.unk_token)
            .field("continuing_subword_prefix", &self.continuing_subword_prefix)
            .field("end_of_word_suffix", &self.end_of_word_suffix)
            .field("fuse_unk", &self.fuse_unk)
            .field("byte_fallback", &self.byte_fallback)
            .field("vocab", &self.vocab.len())
            .field("merges", &self.merges.len())
            .field("ignore_merges", &self.ignore_merges)
            .finish()
    }
}

impl Default for BPE {
    fn default() -> Self {
        Self::builder().build().unwrap()
    }
}

impl Clone for BPE {
    // `Clone` can't be derive because it's not implemented for `BpeCache`.
    // To keep things simple when we clone, the new BPE will start with a fresh cache.
    fn clone(&self) -> Self {
        let fresh_cache = self.cache.as_ref().map(|cache| cache.fresh());
        Self {
            vocab: self.vocab.clone(),
            vocab_r: self.vocab_r.clone(),
            merges: self.merges.clone(),
            byte_level_bypass: self.byte_level_bypass.clone(),
            cache: fresh_cache,
            dropout: self.dropout,
            unk_token: self.unk_token.clone(),
            continuing_subword_prefix: self.continuing_subword_prefix.clone(),
            end_of_word_suffix: self.end_of_word_suffix.clone(),
            fuse_unk: self.fuse_unk,
            byte_fallback: self.byte_fallback,
            ignore_merges: self.ignore_merges,
        }
    }
}

/// Converts the merges strings (for example from `merges.txt` file) with the format
/// "{pair_a} {pair_b}" into the format expected by the BPE struct
pub(crate) fn convert_merges_to_hashmap<I: Iterator<Item = String>>(
    iter: I,
    _vocab: &Vocab,
) -> Result<Merges> {
    let mut merges = vec![];

    let lines = iter.filter(|l| !l.starts_with("#version"));
    for (rank, line) in lines.enumerate() {
        let parts = line.split(' ').collect::<Vec<_>>();
        if parts.len() != 2 {
            return Err(Error::BadMerges(rank + 1).into());
        }

        merges.push((parts[0].to_string(), parts[1].to_string()));
    }

    Ok(merges)
}

impl BPE {
    /// Initialize a `BpeBuilder`.
    pub fn builder() -> BpeBuilder {
        BpeBuilder::new()
    }

    /// Create a new BPE model with the given vocab and merges.
    pub fn new(vocab: Vocab, merges: Merges) -> Self {
        Self::builder()
            .vocab_and_merges(vocab, merges)
            .build()
            .unwrap()
    }

    /// Initialize a BpeBuilder model from vocab and merges files
    pub fn from_file(vocab: &str, merges: &str) -> BpeBuilder {
        Self::builder().files(vocab.to_owned(), merges.to_owned())
    }

    /// Read the given files to extract the vocab and merges
    pub fn read_file(vocab: &str, merges: &str) -> Result<(Vocab, Merges)> {
        // Read vocab.json
        let vocab_file = File::open(vocab)?;
        let mut vocab_file = BufReader::new(vocab_file);

        let mut buffer = String::new();
        vocab_file.read_to_string(&mut buffer)?;
        let json: Value = serde_json::from_str(&buffer)?;
        let mut vocab = AHashMap::new();
        match json {
            Value::Object(m) => {
                for (token, id) in m {
                    if let Value::Number(id) = id {
                        let id = id.as_u64().ok_or(Error::BadVocabulary)? as u32;
                        vocab.insert(token, id);
                    }
                }
            }
            _ => return Err(Box::new(Error::BadVocabulary)),
        };

        // Read merges file
        let merge_file = File::open(merges)?;
        let merge_file = BufReader::new(merge_file);
        let merges = ResultShunt::process(merge_file.lines(), |iter| {
            convert_merges_to_hashmap(iter, &vocab)
        })??;

        Ok((vocab, merges))
    }

    /// Reset the cache.
    pub fn clear_cache(&self) {
        if let Some(ref cache) = self.cache {
            cache.clear()
        }
    }

    /// Resize the cache
    pub fn resize_cache(&mut self, capacity: usize) {
        if let Some(ref mut cache) = self.cache {
            cache.resize(capacity);
        }
    }

    pub fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.clone().into_iter().collect()
    }

    pub fn get_unk_token(&self) -> &Option<String> {
        &self.unk_token
    }

    pub fn get_continuing_subword_prefix(&self) -> &Option<String> {
        &self.continuing_subword_prefix
    }

    fn merge_word(&self, w: &str) -> Result<Word> {
        let mut indices = w.char_indices().map(|(idx, _)| idx).peekable();
        let mut word = Word::with_capacity(w.len());
        let mut unk: Option<(u32, usize)> = None;
        while let Some(i) = indices.next() {
            let end = indices.peek();
            let is_first = i == 0;
            let is_last = end.is_none();

            let mut s = if let Some(e) = end {
                Cow::Borrowed(&w[i..*e])
            } else {
                Cow::Borrowed(&w[i..])
            };
            let byte_len = s.len();

            // Add the `continuing_subword_prefix` if relevant
            if !is_first {
                if let Some(ref prefix) = self.continuing_subword_prefix {
                    s = format!("{prefix}{s}").into()
                }
            }
            // Add the `end_of_word_suffix` if relevant
            if is_last {
                if let Some(ref suffix) = self.end_of_word_suffix {
                    s = format!("{s}{suffix}").into()
                }
            }

            if let Some(id) = self.vocab.get(s.as_ref()) {
                if let Some((unk_id, unk_len)) = unk {
                    word.add(unk_id, unk_len);
                    unk = None;
                }
                word.add(*id, byte_len);
            } else {
                if self.byte_fallback {
                    let tokens: Option<Vec<_>> = s
                        .bytes()
                        .map(|b| -> Option<&u32> {
                            let code = format!("<{b:#04X}>");

                            self.vocab.get(&code)
                        })
                        .collect();
                    if let Some(tokens) = tokens {
                        for t in tokens {
                            word.add(*t, 1);
                        }
                        continue;
                    }
                }
                if let Some(unk_token) = &self.unk_token {
                    unk = match (unk, self.fuse_unk) {
                        (Some((unk_id, unk_len)), true) => {
                            // Fuse unk
                            Some((unk_id, unk_len + byte_len))
                        }
                        (Some((unk_id, unk_len)), false) => {
                            // Do not fuse unk, add the previous one
                            word.add(unk_id, unk_len);
                            Some((
                                *self.vocab.get(unk_token).ok_or_else(|| {
                                    Error::UnkTokenOutOfVocabulary(unk_token.to_owned())
                                })?,
                                byte_len,
                            ))
                        }
                        _ => Some((
                            *self.vocab.get(unk_token).ok_or_else(|| {
                                Error::UnkTokenOutOfVocabulary(unk_token.to_owned())
                            })?,
                            byte_len,
                        )),
                    };
                }
            }
        }
        if let Some((unk_id, unk_len)) = unk {
            word.add(unk_id, unk_len);
        }

        word.merge_all(&self.merges, self.dropout);

        Ok(word)
    }

    fn word_to_tokens<'a>(&'a self, word: &'a Word) -> impl Iterator<Item = Token> + 'a {
        word.get_chars_iter()
            .zip(word.get_offsets_iter())
            .map(move |(id, offsets)| Token::new(id, self.vocab_r[&id].clone(), offsets))
    }

    fn tokenize_with_cache(&self, sequence: &str) -> Result<Vec<Token>> {
        if self.ignore_merges {
            if let Some(id) = self.vocab.get(sequence) {
                return Ok(vec![Token::new(
                    *id,
                    sequence.to_string(),
                    (0, sequence.len()),
                )]);
            }
        }
        let Some(cache) = self.cache.as_ref() else {
            // Cache disabled (capacity 0): fall back to the uncached path.
            let word = self.merge_word(sequence)?;
            return Ok(self.word_to_tokens(&word).collect());
        };
        let cache_id = cache.id();
        BPE_LOCAL_CACHE.with(|cell| {
            let mut by_bpe = cell.borrow_mut();
            let local = by_bpe.entry(cache_id).or_default();
            if let Some(hit) = local.get(sequence) {
                return Ok(self.word_to_tokens(hit).collect());
            }
            let word = self.merge_word(sequence)?;
            let ret: Vec<Token> = self.word_to_tokens(&word).collect();
            if sequence.len() < MAX_LENGTH && local.len() < cache.capacity {
                local.insert(sequence.to_owned(), word);
            }
            Ok(ret)
        })
    }

    fn tokenize_bytes(&self, bytes: &[u8], byte_to_token_id: &[u32; 256]) -> Result<Vec<Token>> {
        if bytes.is_empty() {
            return Ok(vec![]);
        }
        if self.dropout.is_none() || self.dropout == Some(0.0) {
            self.tokenize_bytes_with_cache(bytes, byte_to_token_id)
        } else {
            let word = self.merge_word_from_bytes(bytes, byte_to_token_id);
            Ok(self.word_to_tokens(&word).collect())
        }
    }

    fn merge_word_from_bytes(&self, bytes: &[u8], byte_to_token_id: &[u32; 256]) -> Word {
        let mut word = Word::with_capacity(bytes.len());
        for byte in bytes.iter().copied() {
            word.add(byte_to_token_id[byte as usize], 1);
        }
        word.merge_all(&self.merges, self.dropout);
        word
    }

    fn tokenize_bytes_with_cache(
        &self,
        bytes: &[u8],
        byte_to_token_id: &[u32; 256],
    ) -> Result<Vec<Token>> {
        if self.ignore_merges {
            // One byte-level projection per pre-token: project the raw bytes back into
            // "vocabulary space" so we can reuse `self.vocab` unchanged. Storing the vocab in
            // byte space at load time would drop this alloc and the duplicate byte cache.
            let mapped_string: String = bytes
                .iter()
                .map(|byte| BYTES_CHAR_LOOKUP[*byte as usize])
                .collect();
            if let Some(id) = self.vocab.get(&mapped_string) {
                return Ok(vec![Token::new(*id, mapped_string, (0, bytes.len()))]);
            }
        }
        let Some(cache) = self.cache.as_ref() else {
            let word = self.merge_word_from_bytes(bytes, byte_to_token_id);
            return Ok(self.word_to_tokens(&word).collect());
        };
        let cache_id = cache.id();
        BPE_LOCAL_CACHE_BYTES.with(|cell| {
            let mut by_bpe = cell.borrow_mut();
            let local = by_bpe.entry(cache_id).or_default();
            if let Some(hit) = local.get(bytes) {
                return Ok(self.word_to_tokens(hit).collect());
            }
            let word = self.merge_word_from_bytes(bytes, byte_to_token_id);
            let ret: Vec<Token> = self.word_to_tokens(&word).collect();
            if bytes.len() < MAX_LENGTH && local.len() < cache.capacity {
                local.insert(bytes.to_owned(), word);
            }
            Ok(ret)
        })
    }
}

impl Model for BPE {
    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.clone().into_iter().collect()
    }

    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        if sequence.is_empty() {
            return Ok(vec![]);
        }
        if let Some(bypass) = &self.byte_level_bypass {
            if bypass.active {
                return self.tokenize_bytes(sequence.as_bytes(), &bypass.byte_to_token_id);
            }
        }

        if self.dropout.is_none() || self.dropout == Some(0.0) {
            self.tokenize_with_cache(sequence)
        } else {
            let word = self.merge_word(sequence)?;
            Ok(self.word_to_tokens(&word).collect())
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab_r.get(&id).cloned()
    }

    fn save(&self, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>> {
        let vocab_file_name = match name {
            Some(name) => format!("{name}-vocab.json"),
            None => "vocab.json".to_string(),
        };

        // Write vocab.json
        let vocab_path: PathBuf = [folder, Path::new(vocab_file_name.as_str())]
            .iter()
            .collect();
        let mut vocab_file = File::create(&vocab_path)?;
        let order_vocab_iter = OrderedVocabIter::new(&self.vocab_r);
        let serialized = serde_json::to_string(&order_vocab_iter)?;
        vocab_file.write_all(serialized.as_bytes())?;

        // Write merges.txt
        let merges_file_name = match name {
            Some(name) => format!("{name}-merges.txt"),
            None => "merges.txt".to_string(),
        };

        let merges_path: PathBuf = [folder, Path::new(merges_file_name.as_str())]
            .iter()
            .collect();
        let mut merges_file = File::create(&merges_path)?;
        let mut merges: Vec<(&Pair, &u32)> = self
            .merges
            .iter()
            .map(|(pair, (rank, _))| (pair, rank))
            .collect();
        merges.sort_unstable_by_key(|k| *k.1);
        merges_file.write_all(b"#version: 0.2\n")?;
        merges_file.write_all(
            &merges
                .into_iter()
                .flat_map(|(pair, _)| {
                    format!("{} {}\n", self.vocab_r[&pair.0], self.vocab_r[&pair.1]).into_bytes()
                })
                .collect::<Vec<_>>()[..],
        )?;

        Ok(vec![vocab_path, merges_path])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_cache_is_per_bpe_instance() {
        // Two BPE instances with different merges must tokenize the same
        // input differently even when they share a thread, i.e. the BPE
        // thread-local cache must not leak entries across instances.
        let vocab_a: Vocab = [
            ("h", 0u32),
            ("e", 1),
            ("l", 2),
            ("o", 3),
            ("he", 4),
            ("hel", 5),
            ("hell", 6),
            ("hello", 7),
        ]
        .iter()
        .map(|(s, i)| ((*s).into(), *i))
        .collect();
        let merges_a: Merges = vec![
            ("h".into(), "e".into()),
            ("he".into(), "l".into()),
            ("hel".into(), "l".into()),
            ("hell".into(), "o".into()),
        ];
        let bpe_a = BpeBuilder::default()
            .vocab_and_merges(vocab_a, merges_a)
            .build()
            .unwrap();

        let vocab_b: Vocab = [("h", 0u32), ("e", 1), ("l", 2), ("o", 3)]
            .iter()
            .map(|(s, i)| ((*s).into(), *i))
            .collect();
        let bpe_b = BpeBuilder::default()
            .vocab_and_merges(vocab_b, vec![])
            .build()
            .unwrap();

        // Interleave the two models so any cross-instance cache pollution
        // is visible on the second lookup.
        let ids_a: Vec<u32> = bpe_a
            .tokenize("hello")
            .unwrap()
            .iter()
            .map(|t| t.id)
            .collect();
        let ids_b: Vec<u32> = bpe_b
            .tokenize("hello")
            .unwrap()
            .iter()
            .map(|t| t.id)
            .collect();
        let ids_a2: Vec<u32> = bpe_a
            .tokenize("hello")
            .unwrap()
            .iter()
            .map(|t| t.id)
            .collect();
        let ids_b2: Vec<u32> = bpe_b
            .tokenize("hello")
            .unwrap()
            .iter()
            .map(|t| t.id)
            .collect();

        assert_eq!(ids_a, vec![7u32], "bpe_a must merge to [hello]");
        assert_eq!(ids_b, vec![0u32, 1, 2, 2, 3], "bpe_b has no merges");
        assert_eq!(ids_a2, ids_a, "bpe_a second call must match first");
        assert_eq!(ids_b2, ids_b, "bpe_b second call must match first");
    }

    #[test]
    fn test_ordered_vocab_iter() {
        let vocab_r: VocabR = [
            (0, "a".into()),
            (1, "b".into()),
            (2, "c".into()),
            (3, "ab".into()),
        ]
        .iter()
        .cloned()
        .collect();
        let order_vocab_iter = OrderedVocabIter::new(&vocab_r);
        let serialized = serde_json::to_string(&order_vocab_iter).unwrap();
        assert_eq!(serialized, "{\"a\":0,\"b\":1,\"c\":2,\"ab\":3}");
    }

    #[test]
    fn test_unk_not_fused() {
        let vocab: Vocab = [("<unk>".into(), 0), ("a".into(), 1), ("b".into(), 2)]
            .iter()
            .cloned()
            .collect();
        let bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, vec![])
            .unk_token("<unk>".to_string())
            .build()
            .unwrap();
        let tokens = bpe.tokenize("c").unwrap();
        assert_eq!(tokens, vec![Token::new(0u32, "<unk>".into(), (0, 1)),]);

        let tokens = bpe.tokenize("cc").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::new(0u32, "<unk>".into(), (0, 1)),
                Token::new(0u32, "<unk>".into(), (1, 2)),
            ]
        );

        let tokens = bpe.tokenize("accb").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::new(1u32, "a".into(), (0, 1)),
                Token::new(0u32, "<unk>".into(), (1, 2)),
                Token::new(0u32, "<unk>".into(), (2, 3)),
                Token::new(2u32, "b".into(), (3, 4)),
            ]
        );
    }
    #[test]
    fn test_unk_get_fused() {
        let vocab: Vocab = [("<unk>".into(), 0), ("a".into(), 1), ("b".into(), 2)]
            .iter()
            .cloned()
            .collect();
        let bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, vec![])
            .unk_token("<unk>".to_string())
            .fuse_unk(true)
            .build()
            .unwrap();
        let tokens = bpe.tokenize("c").unwrap();
        assert_eq!(tokens, vec![Token::new(0u32, "<unk>".into(), (0, 1)),]);

        let tokens = bpe.tokenize("cc").unwrap();
        assert_eq!(tokens, vec![Token::new(0u32, "<unk>".into(), (0, 2)),]);

        let tokens = bpe.tokenize("accb").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::new(1u32, "a".into(), (0, 1)),
                Token::new(0u32, "<unk>".into(), (1, 3)),
                Token::new(2u32, "b".into(), (3, 4)),
            ]
        );
    }

    #[test]
    // Test tokenization. With dropout set to 0 tokenization is deterministic,
    // so we know exactly what the result should be.
    //
    // To test this, we'll build a simple model to tokenize the word 'unrelated'.
    fn test_tokenize_with_and_without_dropout() {
        let vocab: Vocab = [
            ("u".into(), 0),
            ("n".into(), 1),
            ("r".into(), 2),
            ("e".into(), 3),
            ("l".into(), 4),
            ("a".into(), 5),
            ("t".into(), 6),
            ("d".into(), 7),
            ("re".into(), 8),
            ("at".into(), 9),
            ("ed".into(), 10),
            ("un".into(), 11),
            ("ated".into(), 12),
            ("rel".into(), 13),
            ("related".into(), 14),
            ("unrelated".into(), 15),
        ]
        .iter()
        .cloned()
        .collect();
        let merges: Merges = vec![
            ("r".to_string(), "e".to_string()),
            ("a".to_string(), "t".to_string()),
            ("e".to_string(), "d".to_string()),
            ("u".to_string(), "n".to_string()),
            ("at".to_string(), "ed".to_string()),
            ("re".to_string(), "l".to_string()),
            ("rel".to_string(), "ated".to_string()),
            ("un".to_string(), "related".to_string()),
        ];
        let mut bpe = BPE::new(vocab, merges);

        // With no dropout:
        let tokens = bpe.tokenize("unrelated").unwrap();
        assert_eq!(tokens, vec![Token::new(15u32, "unrelated".into(), (0, 9))]);

        // With dropout = 0.0 (equivalent to dropout == none)
        bpe.dropout = Some(0.0);
        let tokens = bpe.tokenize("unrelated").unwrap();
        assert_eq!(tokens, vec![Token::new(15u32, "unrelated".into(), (0, 9))]);

        // Now set dropout to 1.0. Result should be no merges performed.
        bpe.dropout = Some(1.0);
        let tokens = bpe.tokenize("unrelated").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::new(0u32, "u".into(), (0, 1)),
                Token::new(1u32, "n".into(), (1, 2)),
                Token::new(2u32, "r".into(), (2, 3)),
                Token::new(3u32, "e".into(), (3, 4)),
                Token::new(4u32, "l".into(), (4, 5)),
                Token::new(5u32, "a".into(), (5, 6)),
                Token::new(6u32, "t".into(), (6, 7)),
                Token::new(3u32, "e".into(), (7, 8)),
                Token::new(7u32, "d".into(), (8, 9)),
            ]
        );

        // Now try with dropout between 0 and 1.
        bpe.dropout = Some(0.5);
        let tokens = bpe.tokenize("unrelated").unwrap();
        assert!(!tokens.is_empty() && tokens.len() <= 9);
    }

    #[test]
    // Ensure `BPE::from_file` works as expected.
    fn test_bpe_from_file() {
        // Set up vocab file.
        let mut vocab_file = NamedTempFile::new().unwrap();
        vocab_file
            .write_all(b"{\"a\": 0, \"b\": 1, \"c\": 2, \"ab\": 3}")
            .unwrap();

        // Set up merges file.
        let mut merges_file = NamedTempFile::new().unwrap();
        merges_file.write_all(b"#version: 0.2\na b").unwrap();

        // Make sure we can instantiate a BPE model from the files.
        let builder = BPE::from_file(
            vocab_file.path().to_str().unwrap(),
            merges_file.path().to_str().unwrap(),
        );
        let bpe = builder.build().unwrap();

        // Check merges.
        assert_eq!(bpe.merges.get(&(0, 1)).unwrap(), &(0u32, 3u32));

        // Check vocab.
        assert_eq!(bpe.vocab.get("a").unwrap(), &0u32);
        assert_eq!(bpe.vocab.get("b").unwrap(), &1u32);
        assert_eq!(bpe.vocab.get("c").unwrap(), &2u32);
        assert_eq!(bpe.vocab.get("ab").unwrap(), &3u32);
    }

    #[test]
    // Ensure BPEBuilder with dropout = 0.0 doesn't error
    fn test_bpe_with_dropout_0() {
        let bpe = BPE::builder().dropout(0.0).build().unwrap();
        assert_eq!(bpe.dropout, Some(0.0));
    }

    #[test]
    // Ensure `BPE::from_file` works as expected.
    fn test_bpe_with_continuing_subword_prefix() {
        let vocab: Vocab = vec![
            ("a".to_string(), 0),
            ("##b".to_string(), 1),
            ("##c".to_string(), 2),
            ("ab".to_string(), 3),
            ("abc".to_string(), 4),
        ]
        .into_iter()
        .collect();

        let merges = vec![
            ("a".to_string(), "##b".to_string()),
            ("ab".to_string(), "##c".to_string()),
        ];

        let bpe = BPE::builder()
            .vocab_and_merges(vocab, merges)
            .unk_token("[UNK]".to_string())
            .continuing_subword_prefix("##".to_string())
            .build()
            .unwrap();

        let res = bpe.tokenize("ab");
        assert_eq!(
            res.unwrap(),
            vec![Token {
                id: 3,
                value: "ab".to_string(),
                offsets: (0, 2)
            }]
        );
        let res = bpe.tokenize("abc");
        assert_eq!(
            res.unwrap(),
            vec![Token {
                id: 4,
                value: "abc".to_string(),
                offsets: (0, 3)
            }]
        );
    }

    #[test]
    // Ensure `MergeTokenOutOfVocabulary` error is returned when it should be.
    fn test_bpe_from_file_merge_token_oov() {
        // Set up vocab file.
        let mut vocab_file = NamedTempFile::new().unwrap();
        vocab_file
            .write_all(b"{\"a\": 0, \"b\": 1, \"c\": 2, \"ab\": 3}")
            .unwrap();

        // Set up merges file.
        let mut merges_file = NamedTempFile::new().unwrap();
        merges_file.write_all(b"#version: 0.2\na b\na d").unwrap();

        // Ensure the result of BPE::from_file is a MergeTokenOutOfVocabulary error.
        match BPE::from_file(
            vocab_file.path().to_str().unwrap(),
            merges_file.path().to_str().unwrap(),
        )
        .build()
        {
            Ok(_) => unreachable!(),
            Err(err) => match err.downcast_ref::<Error>() {
                Some(Error::MergeTokenOutOfVocabulary(token)) => {
                    assert_eq!(*token, String::from("d"))
                }
                _ => unreachable!(),
            },
        }
    }

    #[test]
    // Ensure `BadMerges` error is returned when there is an invalid line in the
    // merges.txt file.
    fn test_bpe_from_file_bad_merges() {
        // Set up vocab file.
        let mut vocab_file = NamedTempFile::new().unwrap();
        vocab_file
            .write_all("{\"a\": 0, \"b\": 1, \"c\": 2, \"ab\": 3}".as_bytes())
            .unwrap();

        // Set up merges file with a bad line.
        let mut merges_file = NamedTempFile::new().unwrap();
        merges_file.write_all(b"#version: 0.2\na b\nc").unwrap();

        // Ensure the result of BPE::from_file is a BadMerges error.
        match BPE::from_file(
            vocab_file.path().to_str().unwrap(),
            merges_file.path().to_str().unwrap(),
        )
        .build()
        {
            Ok(_) => unreachable!(),
            Err(err) => match err.downcast_ref::<Error>() {
                Some(Error::BadMerges(line)) => assert_eq!(*line, 2),
                _ => unreachable!(),
            },
        }
    }

    #[test]
    fn test_bpe_byte_fallback() {
        // 0x61 == 'a' in bytes
        let vocab: Vocab = [("<unk>".into(), 0), ("<0x61>".into(), 1)]
            .iter()
            .cloned()
            .collect();
        let bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, vec![])
            .unk_token("<unk>".to_string())
            .byte_fallback(true)
            .build()
            .unwrap();
        let tokens = bpe.tokenize("c").unwrap();
        assert_eq!(tokens, vec![Token::new(0u32, "<unk>".into(), (0, 1)),]);

        let tokens = bpe.tokenize("a").unwrap();
        assert_eq!(tokens, vec![Token::new(1u32, "<0x61>".into(), (0, 1)),]);
    }

    #[test]
    fn test_bpe_byte_fallback_newline() {
        // 0x0A == '\n' in bytes
        let vocab: Vocab = [("<unk>".into(), 0), ("<0x0A>".into(), 1)]
            .iter()
            .cloned()
            .collect();
        let bpe = BpeBuilder::default()
            .vocab_and_merges(vocab, vec![])
            .unk_token("<unk>".to_string())
            .byte_fallback(true)
            .build()
            .unwrap();
        let tokens = bpe.tokenize("\n").unwrap();
        assert_eq!(tokens, vec![Token::new(1u32, "<0x0A>".into(), (0, 1)),]);
    }

    #[test]
    fn test_ignore_merges() {
        // 0x0A == '\n' in bytes
        let vocab: Vocab = [
            (".:.:".into(), 0),
            ("Ġbelirtilen".into(), 1),
            (".".into(), 2),
            (":".into(), 3),
            ("bel".into(), 4),
            ("irtilen".into(), 5),
            ("Ġ".into(), 6),
            (".:".into(), 7),
            ("belirtilen".into(), 8),
            (".:.".into(), 9),
            ("be".into(), 10),
            ("l".into(), 11),
            ("ir".into(), 12),
            ("ti".into(), 13),
            ("en".into(), 14),
            ("irtil".into(), 15),
            ("irti".into(), 16),
            ("i".into(), 17),
            ("r".into(), 18),
            ("t".into(), 19),
            ("b".into(), 20),
            ("e".into(), 21),
            ("n".into(), 22),
        ]
        .iter()
        .cloned()
        .collect();
        let mut bpe = BpeBuilder::default()
            .vocab_and_merges(
                vocab,
                vec![
                    (".".into(), ":".into()),
                    ("b".into(), "e".into()),
                    ("be".into(), "l".into()),
                    ("i".into(), "r".into()),
                    ("t".into(), "i".into()),
                    ("ir".into(), "ti".into()),
                    ("e".into(), "n".into()),
                    ("irti".into(), "l".into()),
                ],
            )
            .ignore_merges(true)
            .build()
            .unwrap();
        let tokens = bpe.tokenize(".:.:").unwrap();
        assert_eq!(tokens, vec![Token::new(0u32, ".:.:".into(), (0, 4))]);

        let tokens = bpe.tokenize("Ġbelirtilen").unwrap();
        assert_eq!(
            tokens,
            vec![Token::new(1u32, "Ġbelirtilen".into(), (0, 12))]
        );

        bpe.ignore_merges = false;

        let tokens = bpe.tokenize(".:.:").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token::new(7u32, ".:".into(), (0, 2)),
                Token::new(7u32, ".:".into(), (2, 4))
            ]
        );

        let tokens = bpe.tokenize("Ġbelirtilen").unwrap();
        assert_eq!(
            tokens,
            vec![
                Token {
                    id: 6,
                    value: "Ġ".into(),
                    offsets: (0, 2)
                },
                Token {
                    id: 4,
                    value: "bel".into(),
                    offsets: (2, 5)
                },
                Token {
                    id: 15,
                    value: "irtil".into(),
                    offsets: (5, 10)
                },
                Token {
                    id: 14,
                    value: "en".into(),
                    offsets: (10, 12)
                }
            ]
        )
    }

    mod byte_level_bypass {
        use super::*;

        mod builder {
            use super::*;
            use crate::utils::byte_level::BYTES_CHAR_LOOKUP;

            fn byte_level_vocab() -> Vocab {
                (0..=255u8)
                    .map(|byte| {
                        (
                            BYTES_CHAR_LOOKUP[byte as usize].to_string(),
                            (255 - byte) as u32,
                        )
                    })
                    .collect()
            }

            #[test]
            fn test_byte_level_bypass_built_when_vocab_has_all_bytes() {
                let bpe = BpeBuilder::default()
                    .vocab_and_merges(byte_level_vocab(), vec![])
                    .build()
                    .unwrap();
                let table = bpe
                    .byte_level_bypass
                    .as_ref()
                    .map(|bypass| &bypass.byte_to_token_id)
                    .expect("a complete byte-level vocab must produce a byte_to_token_id table");

                for byte in 0..=255u8 {
                    assert_eq!(
                        table[byte as usize],
                        (255 - byte) as u32,
                        "byte 0x{byte:02X} must map to the id of its byte-level token"
                    );
                }
            }

            #[test]
            fn test_byte_level_bypass_is_none_when_vocab_misses_bytes() {
                let mut vocab = byte_level_vocab();
                // Remove the space ' ' from the vocab
                vocab.remove(&BYTES_CHAR_LOOKUP[b' ' as usize].to_string());

                let bpe = BpeBuilder::default()
                    .vocab_and_merges(vocab, vec![])
                    .build()
                    .unwrap();

                assert!(
                    bpe.byte_level_bypass.is_none(),
                    "incomplete byte-level vocab disables the fast path"
                );
            }

            #[test]
            fn test_byte_level_bypass_is_none_with_continuing_subword_prefix() {
                // The fast path builds words straight from raw bytes and never applies
                // `continuing_subword_prefix`, so a vocab that needs it must NOT be eligible.
                let bpe = BpeBuilder::default()
                    .vocab_and_merges(byte_level_vocab(), vec![])
                    .continuing_subword_prefix("##".to_string())
                    .build()
                    .unwrap();

                assert!(
                    bpe.byte_level_bypass.is_none(),
                    "continuing_subword_prefix must disable the fast path"
                );
            }

            #[test]
            fn test_byte_level_bypass_is_none_with_end_of_word_suffix() {
                let bpe = BpeBuilder::default()
                    .vocab_and_merges(byte_level_vocab(), vec![])
                    .end_of_word_suffix("</w>".to_string())
                    .build()
                    .unwrap();

                assert!(
                    bpe.byte_level_bypass.is_none(),
                    "end_of_word_suffix must disable the fast path"
                );
            }

            #[test]
            fn test_byte_level_bypass_eligible_with_empty_prefix_and_suffix() {
                // Real models serialize these as "" (not null); empty is inert.
                let bpe = BpeBuilder::default()
                    .vocab_and_merges(byte_level_vocab(), vec![])
                    .continuing_subword_prefix(String::new())
                    .end_of_word_suffix(String::new())
                    .build()
                    .unwrap();

                assert!(
                    bpe.byte_level_bypass.is_some(),
                    "empty prefix/suffix must stay eligible"
                );
            }
        }

        // Exclude on windows (depends on test fixtures the CI does not download)
        #[cfg(not(target_os = "windows"))]
        mod tokenize_bytes {
            use super::*;
            use crate::utils::byte_level::BYTES_CHAR_LOOKUP;

            const MISSING_FIXTURES: &str =
                "test fixtures not found — run `make test` to download them";

            fn gpt2_bpe() -> BPE {
                BPE::from_file("../data/gpt2-vocab.json", "../data/gpt2-merges.txt")
                    .build()
                    .expect(MISSING_FIXTURES)
            }

            fn read_fixture(path: &str) -> String {
                std::fs::read_to_string(path).expect(MISSING_FIXTURES)
            }

            fn bytes_to_byte_level_string(raw: &[u8]) -> String {
                raw.iter()
                    .map(|byte| BYTES_CHAR_LOOKUP[*byte as usize])
                    .collect()
            }

            /// The step-2 invariant: tokenizing raw bytes yields the same token ids as
            /// tokenizing the ByteLevel-mapped string the slow path sees. Offsets differ by
            /// design (raw-byte space vs mapped-char space), so we compare ids only — offsets
            /// get their own invariant below.
            fn assert_same_ids(bpe: &BPE, raw: &[u8]) {
                let ids =
                    |tokens: Vec<Token>| -> Vec<u32> { tokens.into_iter().map(|t| t.id).collect() };
                let from_string = ids(bpe.tokenize(&bytes_to_byte_level_string(raw)).unwrap());
                let from_bytes = ids(bpe
                    .tokenize_bytes(
                        raw,
                        bpe.byte_level_bypass
                            .as_ref()
                            .map(|bypass| &bypass.byte_to_token_id)
                            .unwrap(),
                    )
                    .unwrap());
                assert_eq!(
                    from_string, from_bytes,
                    "token ids diverge for raw bytes {raw:?}"
                );
            }

            #[test]
            fn test_equivalent_on_big_txt() {
                let bpe = gpt2_bpe();
                for line in read_fixture("../data/big.txt").lines() {
                    assert_same_ids(&bpe, line.as_bytes());
                }
            }

            #[test]
            fn test_equivalent_on_non_latin() {
                let bpe = gpt2_bpe();
                for line in read_fixture("../data/unigram_wagahaiwa_nekodearu.txt").lines() {
                    assert_same_ids(&bpe, line.as_bytes());
                }
            }

            #[test]
            fn test_equivalent_on_empty_and_whitespace() {
                let bpe = gpt2_bpe();
                for raw in [
                    "", " ", "  ", "   ", " a", "a ", "\n", "\t", "a\nb", " \t \n ",
                ] {
                    assert_same_ids(&bpe, raw.as_bytes());
                }
            }

            #[test]
            fn test_equivalent_on_every_single_byte() {
                let bpe = gpt2_bpe();
                for byte in 0..=255u8 {
                    assert_same_ids(&bpe, &[byte]);
                }
            }

            #[test]
            fn test_equivalent_on_byte_pairs_and_triples() {
                let bpe = gpt2_bpe();
                assert_same_ids(&bpe, b"ab");
                assert_same_ids(&bpe, b" the");
                assert_same_ids(&bpe, &[0x20, b't', b'h', b'e']);
                assert_same_ids(&bpe, &[0x00, 0x01, 0x02]);
                assert_same_ids(&bpe, &[0xff, 0xfe]);
            }

            #[test]
            fn test_equivalent_on_multibyte_unicode() {
                let bpe = gpt2_bpe();
                // Accents, Greek, CJK, an emoji, a flag, a ZWJ family, and a mixed string —
                // all force the high bytes 0x80–0xFF through the byte path.
                for raw in [
                    "café",
                    "naïve",
                    "Ωμέγα",
                    "日本語",
                    "👍",
                    "🇫🇷",
                    "👨‍👩‍👧",
                    "a日b👍c",
                ] {
                    assert_same_ids(&bpe, raw.as_bytes());
                }
            }

            #[test]
            fn test_equivalent_on_deep_merges() {
                let bpe = gpt2_bpe();
                for raw in [
                    "a".repeat(64),
                    "the the the the the".to_string(),
                    format!(" {}", "ab".repeat(32)),
                ] {
                    assert_same_ids(&bpe, raw.as_bytes());
                }
            }

            #[test]
            fn test_equivalent_when_cache_is_warm() {
                let bpe = gpt2_bpe();
                // First call populates the cache, second hits it; both must agree.
                for _ in 0..2 {
                    assert_same_ids(&bpe, b"the quick brown fox");
                }
            }

            #[test]
            fn test_equivalent_when_ignore_merges_is_set() {
                // A complete byte-level base vocab (all 256 mapped chars, id = byte), so
                // `byte_level_bypass` is built.
                let mut vocab: Vocab = (0..=255u8)
                    .map(|byte| (BYTES_CHAR_LOOKUP[byte as usize].to_string(), byte as u32))
                    .collect();
                // '.' and ':' are printable, so they map to themselves. Add the pair ".:" and
                // the whole sequence ".:.:" as their own tokens.
                vocab.insert(".:".to_string(), 256);
                vocab.insert(".:.:".to_string(), 257);

                // Merge "." + ":" -> ".:", but nothing merges ".:" + ".:". So the merge-based
                // tokenization of ".:.:" is [".:", ".:"], while the `ignore_merges` shortcut
                // returns the whole token ".:.:".
                let bpe = BpeBuilder::default()
                    .vocab_and_merges(vocab, vec![(".".to_string(), ":".to_string())])
                    .ignore_merges(true)
                    .build()
                    .unwrap();

                assert_same_ids(&bpe, b".:.:");
            }

            #[test]
            fn test_ignore_merges_fast_path_preserves_byte_level_token_value() {
                // Complete byte-level base vocab (all 256 mapped chars, id = byte).
                let mut vocab: Vocab = (0..=255u8)
                    .map(|byte| (BYTES_CHAR_LOOKUP[byte as usize].to_string(), byte as u32))
                    .collect();
                // Add " the" as a single token. In byte-level space the leading space maps to
                // 'Ġ', so the token's mapped form is "Ġthe" — i.e. the vocab string differs
                // from the raw bytes.
                let mapped = bytes_to_byte_level_string(b" the");
                vocab.insert(mapped.clone(), 256);

                let bpe = BpeBuilder::default()
                    .vocab_and_merges(vocab, vec![])
                    .ignore_merges(true)
                    .build()
                    .unwrap();

                let values = |tokens: Vec<Token>| -> Vec<String> {
                    tokens.into_iter().map(|t| t.value).collect()
                };
                // Slow path sees the mapped string and hits the ignore_merges shortcut,
                // emitting value "Ġthe".
                let from_string = values(bpe.tokenize(&mapped).unwrap());
                // Fast path takes raw bytes; its token value must match the slow path's
                // byte-level-mapped value, not the raw decoded bytes.
                let from_bytes = values(
                    bpe.tokenize_bytes(
                        b" the",
                        bpe.byte_level_bypass
                            .as_ref()
                            .map(|bypass| &bypass.byte_to_token_id)
                            .unwrap(),
                    )
                    .unwrap(),
                );

                assert_eq!(
                    from_string, from_bytes,
                    "ignore_merges fast path must emit the byte-level-mapped token value"
                );
            }

            #[test]
            fn test_equivalent_when_unk_and_byte_fallback_are_set() {
                let mut vocab: Vocab = (0..=255u8)
                    .map(|byte| (BYTES_CHAR_LOOKUP[byte as usize].to_string(), byte as u32))
                    .collect();
                vocab.insert("<unk>".to_string(), 256);
                vocab.insert("th".to_string(), 257);
                vocab.insert("the".to_string(), 258);

                let bpe = BpeBuilder::default()
                    .vocab_and_merges(
                        vocab,
                        vec![
                            ("t".to_string(), "h".to_string()),
                            ("th".to_string(), "e".to_string()),
                        ],
                    )
                    .unk_token("<unk>".to_string())
                    .fuse_unk(true)
                    .byte_fallback(true)
                    .build()
                    .unwrap();

                assert!(
                    bpe.byte_level_bypass.is_some(),
                    "unk/fuse_unk/byte_fallback must not disable the bypass"
                );
                for raw in [
                    b"the".as_slice(),
                    b" the the the",
                    &[0xff, 0xfe, 0x00],
                    "café 日本 👍".as_bytes(),
                ] {
                    assert_same_ids(&bpe, raw);
                }
            }

            #[test]
            fn test_fast_path_offsets_tile_the_input() {
                let bpe = gpt2_bpe();
                let byte_to_token_id = bpe
                    .byte_level_bypass
                    .as_ref()
                    .map(|bypass| &bypass.byte_to_token_id)
                    .unwrap();
                for raw in ["", "hello", " a b c", "café 日本 👍", "the the the"] {
                    let tokens = bpe
                        .tokenize_bytes(raw.as_bytes(), byte_to_token_id)
                        .unwrap();
                    let mut cursor = 0;
                    for token in &tokens {
                        assert_eq!(token.offsets.0, cursor, "gap/overlap in {raw:?}");
                        cursor = token.offsets.1;
                    }
                    assert_eq!(cursor, raw.len(), "offsets must cover all of {raw:?}");
                }
            }

            #[test]
            fn test_equivalent_with_zero_dropout() {
                // dropout == 0.0 must take the cached path, identical to None.
                let bpe = BPE::from_file("../data/gpt2-vocab.json", "../data/gpt2-merges.txt")
                    .dropout(0.0)
                    .build()
                    .expect(MISSING_FIXTURES);
                for raw in ["the quick brown fox", "café 日本 👍", " a b c", ""] {
                    assert_same_ids(&bpe, raw.as_bytes());
                }
            }
        }
    }
}
