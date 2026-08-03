use super::{super::OrderedVocabIter, Error, Pair, Word};
use crate::models::bpe::Merge;
use crate::models::bpe::tables::BpeTables;
use crate::models::bpe::word_cache::WordCache;
use crate::pipeline::{self, ModelScratch, PipelineToken};
use crate::tokenizer::{Model, Result, Token};
use crate::utils::byte_level::{self};
use crate::utils::cache::{DEFAULT_CACHE_CAPACITY, MAX_LENGTH};
use crate::utils::iter::ResultShunt;
use crate::vocab::bucket_vocab_store::BucketVocabStore;
use crate::vocab_store::VocabStore;
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
            VocabStore::new()
        } else {
            VocabStore::build(
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
    pub vocab: VocabStore,
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

    fn tokenize(&self, sequence: &str) -> Result<Vec<Token>> {
        if sequence.is_empty() {
            return Ok(vec![]);
        }

        if self.dropout.is_none() || self.dropout == Some(0.0) {
            self.tokenize_with_cache(sequence)
        } else {
            let word = self.merge_word(sequence)?;
            Ok(self.word_to_tokens(&word).collect())
        }
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

pub struct PipelineBPE {
    atoms: Atoms,
    tables: BpeTables,
    vocab: BucketVocabStore,
    merges: MergeMap,
    ignore_merges: bool,
    cache_capacity: Option<usize>,
}

enum Atoms {
    Bytes {
        byte_to_id: [u32; 256],
    },
    Chars {
        byte_fallback: Option<[u32; 256]>,
        unk_token: Option<u32>,
        fuse_unk: bool,
    },
}

impl PipelineBPE {
    pub fn from_bpe(model: BPE, with_byte_level: bool) -> Result<Self> {
        if matches!(&model.continuing_subword_prefix, Some(prefix) if !prefix.is_empty()) {
            return Err("BPE models with continuing_subword_prefix are not supported yet".into());
        }
        if matches!(&model.end_of_word_suffix, Some(suffix) if !suffix.is_empty()) {
            return Err("BPE models with end_of_word_suffix are not supported yet".into());
        }
        if matches!(&model.dropout, Some(dropout) if *dropout > 0.0) {
            return Err("BPE models with dropout not supported yet".into());
        }
        let BPE {
            vocab,
            merges,
            ignore_merges,
            byte_fallback,
            unk_token,
            fuse_unk,
            ..
        } = model;

        let tables = BpeTables::build(
            vocab.get_vocab().into_iter().collect(),
            merges.clone(),
            with_byte_level,
        );
        let (vocab, atoms) = if with_byte_level {
            let mut vocab = BucketVocabStore::build(vocab.byte_content());
            vocab = byte_level::transform_vocab(vocab);
            let mut byte_to_id = [0u32; 256];
            for b in 0u8..=255 {
                byte_to_id[b as usize] = vocab
                    .get_bytes(&[b])
                    .ok_or(Error::ByteAtomOutOfVocabulary(b))?;
            }
            (vocab, Atoms::Bytes { byte_to_id })
        } else {
            let vocab = BucketVocabStore::build(vocab.byte_content());
            let unk_token = if let Some(unk_str) = unk_token {
                let token_id = vocab
                    .token_to_id(&unk_str)
                    .ok_or_else(|| Error::UnkTokenOutOfVocabulary(unk_str.clone()))?;
                Some(token_id)
            } else {
                None
            };
            let fallback_lookup = if byte_fallback {
                let mut fallback_lookup = [0u32; 256];
                for b in 0u8..=255 {
                    let code = format!("<{b:#04X}>");
                    fallback_lookup[b as usize] = vocab
                        .token_to_id(&code)
                        .ok_or(Error::ByteFallbackOutOfVocabulary(b))?;
                }
                Some(fallback_lookup)
            } else {
                None
            };
            (
                vocab,
                Atoms::Chars {
                    fuse_unk,
                    unk_token,
                    byte_fallback: fallback_lookup,
                },
            )
        };
        Ok(Self {
            atoms,
            tables,
            ignore_merges,
            merges,
            vocab,
            cache_capacity: model.cache.map(|c| c.capacity).filter(|&c| c > 0),
        })
    }

    // We start by converting the sequence to the corresponding token id of each char/byte depending
    // on the settings. Tokenizers that use bytelevel pretokenizer work on bytes, others on chars.
    // TODO: this also means we are iterating twice on the string. Her and then on merge_all
    fn merge_word(
        &self,
        sequence: &str,
        merge_queue: &mut QuaternaryHeap<Merge>,
        skip: &mut Vec<Merge>,
        word: &mut Word,
    ) {
        let mut to_merge = Vec::new();
        // 1. we convert the codepoint to internal ID (rank)
        let mut global_min = 0u64;
        let mut past_rank = u32::MAX;
        for c in sequence.chars() {
            let rank = self
                .tables
                .fold
                .get(c as usize)
                .unwrap_or(&self.tables.non_bmp[&c]);
            // we compute the min rank as this will be the first merge we'll do
            let merge_rank = self.tables.get_value(&past_rank, &rank);
            global_min = std::cmp::min(global_min, merge_rank);
            past_rank = *rank;
            to_merge.push(*rank);
        }
        self.multipass_merge(&mut to_merge, global_min);
        // Finally, we use the unmap
    }

    /// `M` is false only for the first written symbol, which has no left neighbour and therefore no
    /// pair to rank.
    ///
    /// `&mut [u32]` rather than `&mut Vec<u32>` so the length is a local and the reads can have
    /// their bounds checks removed..
    #[inline(always)]
    fn advance_one<const M: bool>(
        &self,
        to_merge: &mut [u32],
        mut read_id: usize,
        global_min: u64,
        mut write_id: usize,
        mut running_min: u64,
    ) -> (u64, usize, usize) {
        let (ia, ib) = (to_merge[read_id], to_merge[read_id + 1]);
        let value = self.tables.get_value(&ia, &ib);
        // TODO: we are adding the `SAFE` flag on bit 31 this has to become `(value & ID_MASK) as u32`.
        let id = value as u32;
        // only merge pairs that have the min rank
        let written = if value == global_min {
            read_id += 1;
            id
        } else {
            ia
        };
        to_merge[write_id] = written;
        if M {
            let merge_rank = self.tables.get_value(&to_merge[write_id - 1], &written);
            running_min = std::cmp::min(running_min, merge_rank);
        }
        write_id += 1;
        read_id += 1;
        (running_min, read_id, write_id)
    }
    fn multipass_merge(&self, to_merge: &mut Vec<u32>, mut global_min: u64) {
        const FALLBACK_THRESHOLD: u8 = 8;
        let mut i = 0u8;
        // in multi-pass, we read and write in the same buffer
        let mut read_id = 0usize;
        let mut write_id = 0usize;
        let mut last_id = to_merge.len() - 1;
        loop {
            if i == FALLBACK_THRESHOLD {
                self.two_tier_queue_merge(to_merge, global_min);
            }
            let mut running_min = u64::MAX;
            (running_min, read_id, write_id) =
                self.advance_one::<false>(to_merge, read_id, global_min, write_id, running_min);
            while read_id + 1 < last_id {
                (running_min, read_id, write_id) =
                    self.advance_one::<true>(to_merge, read_id, global_min, write_id, running_min);
            }
            i += 1;
            last_id = write_id;
            if running_min == u64::MAX {
                break;
            }
            global_min = running_min;
        }
    }

    fn two_tier_queue_merge(&self, to_merge: &mut Vec<u32>, mut global_min: u64) {
        todo!()
    }
}

impl pipeline::Model for PipelineBPE {
    type Scratch = BpeScratch;

    fn tokenize_pipeline(
        &self,
        sequence: &str,
        scratch: &mut Self::Scratch,
        output: &mut Vec<PipelineToken>,
    ) -> Result<()> {
        if sequence.is_empty() {
            return Ok(());
        }

        if self.ignore_merges
            && let Some(id) = self.vocab.get_bytes(sequence.as_bytes())
        {
            output.push(PipelineToken { id });
            return Ok(());
        }

        let BpeScratch {
            merge_queue,
            skip,
            word,
            word_cache,
        } = scratch;

        if let Some(cache) = word_cache
            && let Some(hit) = cache.get(sequence.as_bytes())
        {
            output.extend(hit.iter().map(|&id| PipelineToken { id }));
            return Ok(());
        }

        // merges is close-adressing
        self.merge_word(sequence, merge_queue, skip, word);
        output.extend(word.get_chars_iter().map(|id| PipelineToken { id }));
        if let Some(cache) = word_cache {
            cache.insert(sequence.as_bytes(), word.get_chars_iter());
        }

        Ok(())
    }

    fn init_scratch(&self) -> Self::Scratch {
        Self::Scratch {
            merge_queue: QuaternaryHeap::with_capacity(64),
            word: Word::with_capacity(64),
            skip: Vec::new(),
            word_cache: self.cache_capacity.map(WordCache::new),
        }
    }
}

#[derive(Default)]
pub struct BpeScratch {
    pub(crate) merge_queue: QuaternaryHeap<Merge>,
    pub(crate) skip: Vec<Merge>,
    pub(crate) word: Word,
    pub(crate) word_cache: Option<WordCache>,
}

impl ModelScratch for BpeScratch {
    fn clear(&mut self) {
        let Self {
            merge_queue,
            skip,
            word,
            word_cache: _,
        } = self;
        merge_queue.clear();
        skip.clear();
        word.clear();
        // The word cache is intentionally kept across clears so it stays warm for future callers
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
        assert_eq!(bpe.vocab.token_to_id("a").unwrap(), 0u32);
        assert_eq!(bpe.vocab.token_to_id("b").unwrap(), 1u32);
        assert_eq!(bpe.vocab.token_to_id("c").unwrap(), 2u32);
        assert_eq!(bpe.vocab.token_to_id("ab").unwrap(), 3u32);
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

    mod pipeline_bpe {
        use super::*;
        use crate::{
            Model, pipeline::Model as PipelineModel, utils::byte_level::BYTES_CHAR_LOOKUP,
        };

        const HELLO_VOCAB: &[(&str, u32)] = &[
            ("h", 0),
            ("e", 1),
            ("l", 2),
            ("o", 3),
            ("he", 4),
            ("hel", 5),
            ("hell", 6),
            ("hello", 7),
        ];
        const HELLO_MERGES: &[(&str, &str)] =
            &[("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o")];

        fn v(pairs: &[(&str, u32)]) -> Vocab {
            pairs.iter().map(|&(s, i)| (s.into(), i)).collect()
        }

        fn m(pairs: &[(&str, &str)]) -> Merges {
            pairs.iter().map(|&(a, b)| (a.into(), b.into())).collect()
        }

        fn hello_builder() -> BpeBuilder {
            BpeBuilder::default().vocab_and_merges(v(HELLO_VOCAB), m(HELLO_MERGES))
        }

        fn pipeline_ids(model: &PipelineBPE, sequence: &str) -> Vec<u32> {
            let mut out = Vec::new();
            let mut scratch = model.init_scratch();
            pipeline::Model::tokenize_pipeline(model, sequence, &mut scratch, &mut out).unwrap();
            out.iter().map(|t| t.id).collect()
        }

        fn reference_ids(model: &BPE, sequence: &str) -> Vec<u32> {
            model
                .tokenize(sequence)
                .unwrap()
                .iter()
                .map(|t| t.id)
                .collect()
        }

        #[test]
        fn applies_merges() {
            let bpe = hello_builder().build().unwrap();
            let reference = bpe.clone();
            let pipeline = PipelineBPE::from_bpe(bpe, false).unwrap();
            for (input, want) in [
                ("hello", vec![7]),
                ("hell", vec![6]),
                ("helo", vec![5, 3]),
                ("oleh", vec![3, 2, 1, 0]),
            ] {
                assert_eq!(pipeline_ids(&pipeline, input), want, "{input:?}");
                assert_eq!(
                    pipeline_ids(&pipeline, input),
                    reference_ids(&reference, input),
                    "{input:?} vs reference"
                );
            }
        }

        #[test]
        fn empty_input_yields_no_tokens() {
            let pipeline = PipelineBPE::from_bpe(hello_builder().build().unwrap(), false).unwrap();
            assert!(pipeline_ids(&pipeline, "").is_empty());
        }

        // The scratch pool hands the SAME scratch to successive encodes. A bug leaking
        // state between calls (an undrained merge queue, a stale word buffer) would
        // corrupt every encode after the first. Drive several inputs — including
        // repeats and an empty string — through one reused scratch and check each still
        // matches the fresh-scratch reference. This is the invariant the pool relies on.
        #[test]
        fn reused_scratch_matches_fresh() {
            let bpe = hello_builder().build().unwrap();
            let reference = bpe.clone();
            let model = PipelineBPE::from_bpe(bpe, false).unwrap();
            let mut scratch = model.init_scratch();
            for input in ["hello", "hell", "helo", "oleh", "hello", "", "hxe"] {
                let mut out = Vec::new();
                pipeline::Model::tokenize_pipeline(&model, input, &mut scratch, &mut out).unwrap();
                let got: Vec<u32> = out.iter().map(|t| t.id).collect();
                assert_eq!(got, reference_ids(&reference, input), "{input:?}");
            }
        }

        #[test]
        fn unknown_char_without_unk_is_dropped() {
            let bpe = hello_builder().build().unwrap();
            let reference = bpe.clone();
            let pipeline = PipelineBPE::from_bpe(bpe, false).unwrap();
            // 'x' vanishes, making 'h' and 'e' adjacent, so the (h,e) merge
            // applies — mirrors the reference model.
            assert_eq!(pipeline_ids(&pipeline, "hxe"), vec![4]);
            assert_eq!(
                pipeline_ids(&pipeline, "hxe"),
                reference_ids(&reference, "hxe")
            );
        }

        #[test]
        fn unk_replaces_unknown_chars() {
            let mut vocab = v(HELLO_VOCAB);
            vocab.insert("<unk>".into(), 8);
            let bpe = BpeBuilder::default()
                .vocab_and_merges(vocab, m(HELLO_MERGES))
                .unk_token("<unk>".into())
                .build()
                .unwrap();
            let reference = bpe.clone();
            let pipeline = PipelineBPE::from_bpe(bpe, false).unwrap();
            for (input, want) in [
                ("hxe", vec![0, 8, 1]),
                ("xh", vec![8, 0]),
                ("hxxe", vec![0, 8, 8, 1]),
                ("xx", vec![8, 8]),
            ] {
                assert_eq!(pipeline_ids(&pipeline, input), want, "{input:?}");
                assert_eq!(
                    pipeline_ids(&pipeline, input),
                    reference_ids(&reference, input),
                    "{input:?} vs reference"
                );
            }
        }

        #[test]
        fn fused_unk_collapses_runs() {
            let mut vocab = v(HELLO_VOCAB);
            vocab.insert("<unk>".into(), 8);
            let bpe = BpeBuilder::default()
                .vocab_and_merges(vocab, m(HELLO_MERGES))
                .unk_token("<unk>".into())
                .fuse_unk(true)
                .build()
                .unwrap();
            let reference = bpe.clone();
            let pipeline = PipelineBPE::from_bpe(bpe, false).unwrap();
            for (input, want) in [
                ("hxxe", vec![0, 8, 1]),
                ("xxh", vec![8, 0]),
                ("xxxx", vec![8]),
                ("xhx", vec![8, 0, 8]),
            ] {
                assert_eq!(pipeline_ids(&pipeline, input), want, "{input:?}");
                assert_eq!(
                    pipeline_ids(&pipeline, input),
                    reference_ids(&reference, input),
                    "{input:?} vs reference"
                );
            }
        }

        fn byte_fallback_vocab() -> Vocab {
            let mut vocab = v(&[("h", 300), ("e", 301), ("<unk>", 400)]);
            vocab.extend((0..=255u8).map(|b| (format!("<0x{b:02X}>"), u32::from(b))));
            vocab
        }

        #[test]
        fn byte_fallback_encodes_missing_chars_as_byte_tokens() {
            let bpe = BpeBuilder::default()
                .vocab_and_merges(byte_fallback_vocab(), vec![])
                .byte_fallback(true)
                .build()
                .unwrap();
            let reference = bpe.clone();
            let pipeline = PipelineBPE::from_bpe(bpe, false).unwrap();
            // 'é' is not in the vocab: falls back to its UTF-8 bytes C3 A9
            assert_eq!(pipeline_ids(&pipeline, "hé"), vec![300, 0xC3, 0xA9]);
            assert_eq!(pipeline_ids(&pipeline, "🤗"), vec![0xF0, 0x9F, 0xA4, 0x97]);
            for input in ["hé", "🤗", "he"] {
                assert_eq!(
                    pipeline_ids(&pipeline, input),
                    reference_ids(&reference, input),
                    "{input:?} vs reference"
                );
            }
        }

        #[test]
        fn byte_fallback_wins_over_unk() {
            let bpe = BpeBuilder::default()
                .vocab_and_merges(byte_fallback_vocab(), vec![])
                .byte_fallback(true)
                .unk_token("<unk>".into())
                .build()
                .unwrap();
            let reference = bpe.clone();
            let pipeline = PipelineBPE::from_bpe(bpe, false).unwrap();
            assert_eq!(pipeline_ids(&pipeline, "é"), vec![0xC3, 0xA9]);
            assert_eq!(pipeline_ids(&pipeline, "é"), reference_ids(&reference, "é"));
        }

        #[test]
        fn ignore_merges_prefers_whole_word() {
            let bpe = hello_builder().ignore_merges(true).build().unwrap();
            let reference = bpe.clone();
            let pipeline = PipelineBPE::from_bpe(bpe, false).unwrap();
            // direct vocab hit bypasses the merge loop; a miss falls through to it
            assert_eq!(pipeline_ids(&pipeline, "hello"), vec![7]);
            assert_eq!(pipeline_ids(&pipeline, "helo"), vec![5, 3]);
            for input in ["hello", "helo"] {
                assert_eq!(
                    pipeline_ids(&pipeline, input),
                    reference_ids(&reference, input),
                    "{input:?} vs reference"
                );
            }
        }

        #[test]
        fn rejects_unsupported_configs() {
            // no merges: BpeBuilder::build underflows on merges whose right token
            // is shorter than continuing_subword_prefix (pre-existing, unrelated)
            let build = |f: fn(BpeBuilder) -> BpeBuilder| {
                f(BpeBuilder::default().vocab_and_merges(v(HELLO_VOCAB), vec![]))
                    .build()
                    .unwrap()
            };
            assert!(
                PipelineBPE::from_bpe(build(|b| b.continuing_subword_prefix("##".into())), false)
                    .is_err()
            );
            assert!(
                PipelineBPE::from_bpe(build(|b| b.end_of_word_suffix("</w>".into())), false)
                    .is_err()
            );
            assert!(PipelineBPE::from_bpe(build(|b| b.dropout(0.5)), false).is_err());
            // no-op values must not be rejected: gpt2's tokenizer.json serializes
            // prefix/suffix as "" and the reference treats dropout 0.0 as disabled
            assert!(
                PipelineBPE::from_bpe(
                    build(|b| {
                        b.continuing_subword_prefix(String::new())
                            .end_of_word_suffix(String::new())
                            .dropout(0.0)
                    }),
                    false
                )
                .is_ok()
            );
        }

        #[test]
        fn rejects_unk_token_missing_from_vocab() {
            let bpe = hello_builder().unk_token("<unk>".into()).build().unwrap();
            assert!(PipelineBPE::from_bpe(bpe, false).is_err());
        }

        #[test]
        fn byte_fallback_with_missing_codes_errors() {
            // Incomplete <0xNN> coverage must be a build error, not a panic.
            let bpe = hello_builder().byte_fallback(true).build().unwrap();
            assert!(PipelineBPE::from_bpe(bpe, false).is_err());
        }

        fn projected(s: &str) -> String {
            s.bytes().map(|b| BYTES_CHAR_LOOKUP[b as usize]).collect()
        }

        /// A gpt2-shaped miniature: the 256 projected single-byte tokens
        /// (id == byte value) plus `extra` tokens and merges, given in raw
        /// space and projected here — like a real byte-level tokenizer.json,
        /// whose vocab is stored in the projected alphabet.
        fn byte_level_bpe(
            extra: &[(&str, u32)],
            merges: &[(&str, &str)],
            ignore_merges: bool,
        ) -> BPE {
            let mut vocab: Vocab = (0..=255u8)
                .map(|b| (BYTES_CHAR_LOOKUP[b as usize].to_string(), u32::from(b)))
                .collect();
            vocab.extend(extra.iter().map(|&(s, i)| (projected(s), i)));
            let merges: Merges = merges
                .iter()
                .map(|&(a, b)| (projected(a), projected(b)))
                .collect();
            BpeBuilder::default()
                .vocab_and_merges(vocab, merges)
                .ignore_merges(ignore_merges)
                .build()
                .unwrap()
        }

        #[test]
        fn byte_level_merges_raw_bytes() {
            let bpe = byte_level_bpe(
                &[("he", 300), (" he", 301)],
                &[("h", "e"), (" ", "he")],
                false,
            );
            let reference = bpe.clone();
            let pipeline = PipelineBPE::from_bpe(bpe, true).unwrap();
            assert_eq!(pipeline_ids(&pipeline, " he"), vec![301]);
            // single bytes hit the un-projected single-byte tokens (id == byte value)
            assert_eq!(pipeline_ids(&pipeline, "é"), vec![0xC3, 0xA9]);
            // the end-to-end invariant: raw input through the pipeline must equal
            // projected input through the reference model
            for input in [" he", "é", "\x00\x7f", "hé llo"] {
                assert_eq!(
                    pipeline_ids(&pipeline, input),
                    reference_ids(&reference, &projected(input)),
                    "{input:?}"
                );
            }
        }

        #[test]
        fn byte_level_ignore_merges_whole_word() {
            let bpe = byte_level_bpe(&[(" hello", 300)], &[], true);
            let pipeline = PipelineBPE::from_bpe(bpe, true).unwrap();
            assert_eq!(pipeline_ids(&pipeline, " hello"), vec![300]);
            // not in vocab → falls through to single-byte atoms
            assert_eq!(
                pipeline_ids(&pipeline, "zz"),
                vec![u32::from(b'z'), u32::from(b'z')]
            );
        }

        #[test]
        fn byte_level_requires_full_byte_coverage() {
            // An ASCII-only vocab covers no control/high bytes: building the
            // byte-level pipeline must be a build error, not a panic.
            let bpe = hello_builder().build().unwrap();
            assert!(PipelineBPE::from_bpe(bpe, true).is_err());
        }
    }
}
