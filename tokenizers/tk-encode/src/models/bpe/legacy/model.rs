use super::word::Word;
use crate::models::OrderedVocabIter;
use crate::models::bpe::{Error, MergeMap, Merges, Pair, Vocab, VocabR};
use crate::tokenizer::{Model, Result, Token};
use crate::utils::cache::{DEFAULT_CACHE_CAPACITY, MAX_LENGTH};
use crate::utils::iter::ResultShunt;
use crate::vocab::bucket_vocab_store::BucketVocabStore;
use ahash::AHashMap;
use dary_heap::QuaternaryHeap;
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
}
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

    /// Returns a `BPE` model that uses the `BpeBuilder`'s configuration.
    pub fn build(mut self) -> Result<BPE> {
        // Validate dropout.
        if let Some(p) = self.config.dropout
            && !(0.0..=1.0).contains(&p)
        {
            return Err(Error::InvalidDropout.into());
        }

        // Read files if necessary
        if let Some((vocab, merges)) = self.config.files {
            let (v, m) = BPE::read_file(&vocab, &merges)?;
            self.config.vocab = v;
            self.config.merges = m;
        }

        let mut max_len = 0;
        for key in self.config.vocab.keys() {
            if max_len < key.len() {
                max_len = key.len();
            }
        }
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

        let vocab = if vocab.is_empty() {
            BucketVocabStore::new()
        } else {
            BucketVocabStore::build(
                vocab
                    .into_iter()
                    .map(|(k, v)| (k.into_bytes(), v))
                    .collect(),
            )
        };

        Ok(BPE {
            vocab,
            merges: merge_map,
            cache,
            dropout: self.config.dropout,
            unk_token: self.config.unk_token,
            continuing_subword_prefix: self.config.continuing_subword_prefix,
            end_of_word_suffix: self.config.end_of_word_suffix,
            fuse_unk: self.config.fuse_unk,
            byte_fallback: self.config.byte_fallback,
            ignore_merges: self.config.ignore_merges,
        })
    }
}

/// A [Byte Pair Encoding](https://www.aclweb.org/anthology/P16-1162/) model.
#[derive(PartialEq)]
pub struct BPE {
    /// The vocabulary, mapping tokens <-> ids both ways.
    pub vocab: BucketVocabStore,
    /// Contains the mapping between Pairs and their (rank, new_id).
    pub merges: MergeMap,
    /// Contains the cache for optimizing the encoding step.
    pub(crate) cache: Option<BpeCache>,
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
            merges: self.merges.clone(),
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
        self.vocab.get_vocab().into_iter().collect()
    }

    pub fn get_unk_token(&self) -> &Option<String> {
        &self.unk_token
    }

    pub fn get_continuing_subword_prefix(&self) -> &Option<String> {
        &self.continuing_subword_prefix
    }

    pub(super) fn merge_word(&self, w: &str) -> Result<Word> {
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
            if !is_first && let Some(ref prefix) = self.continuing_subword_prefix {
                s = format!("{prefix}{s}").into()
            }
            // Add the `end_of_word_suffix` if relevant
            if is_last && let Some(ref suffix) = self.end_of_word_suffix {
                s = format!("{s}{suffix}").into()
            }

            if let Some(id) = self.vocab.token_to_id(s.as_ref()) {
                if let Some((unk_id, unk_len)) = unk {
                    word.add(unk_id, unk_len);
                    unk = None;
                }
                word.add(id, byte_len);
            } else {
                if self.byte_fallback {
                    let tokens: Option<Vec<_>> = s
                        .bytes()
                        .map(|b| -> Option<u32> {
                            let code = format!("<{b:#04X}>");

                            self.vocab.token_to_id(&code)
                        })
                        .collect();
                    if let Some(tokens) = tokens {
                        for t in tokens {
                            word.add(t, 1);
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
                                self.vocab.token_to_id(unk_token).ok_or_else(|| {
                                    Error::UnkTokenOutOfVocabulary(unk_token.to_owned())
                                })?,
                                byte_len,
                            ))
                        }
                        _ => Some((
                            self.vocab.token_to_id(unk_token).ok_or_else(|| {
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

        let mut queue = QuaternaryHeap::with_capacity(word.len_symbols());
        let mut skip = Vec::with_capacity(queue.len());
        word.merge_all(&self.merges, self.dropout, &mut queue, &mut skip);

        Ok(word)
    }

    fn word_to_tokens<'a>(&'a self, word: &'a Word) -> impl Iterator<Item = Token> + 'a {
        word.get_chars_iter()
            .zip(word.get_offsets_iter())
            .map(move |(id, offsets)| {
                Token::new(id, self.vocab.id_to_token(id).unwrap_or_default(), offsets)
            })
    }

    fn tokenize_with_cache(&self, sequence: &str) -> Result<Vec<Token>> {
        if self.ignore_merges
            && let Some(id) = self.vocab.token_to_id(sequence)
        {
            return Ok(vec![Token::new(
                id,
                sequence.to_string(),
                (0, sequence.len()),
            )]);
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
}

impl Model for BPE {
    fn get_vocab(&self) -> HashMap<String, u32> {
        self.vocab.get_vocab().into_iter().collect()
    }

    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab.id_to_token(id)
    }

    fn save(&self, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>> {
        let vocab_r: VocabR = self
            .vocab
            .get_vocab()
            .into_iter()
            .map(|(s, id)| (id, s))
            .collect();
        let vocab_file_name = match name {
            Some(name) => format!("{name}-vocab.json"),
            None => "vocab.json".to_string(),
        };

        // Write vocab.json
        let vocab_path: PathBuf = [folder, Path::new(vocab_file_name.as_str())]
            .iter()
            .collect();
        let mut vocab_file = File::create(&vocab_path)?;
        let order_vocab_iter = OrderedVocabIter::new(&vocab_r);
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
                    format!("{} {}\n", vocab_r[&pair.0], vocab_r[&pair.1]).into_bytes()
                })
                .collect::<Vec<_>>()[..],
        )?;

        Ok(vec![vocab_path, merges_path])
    }
}
