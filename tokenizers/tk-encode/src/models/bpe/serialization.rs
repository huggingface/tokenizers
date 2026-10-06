//! The different functions used to go from the serialized json vocab / merges to building the BPE.

use super::model::{Atoms, PipelineBPE, build_byte_to_gate};
use crate::models::bpe::convert::{AFFIX_BUF, Affixes};
use crate::models::bpe::tables::BpeTables;
use crate::models::bpe::{Error, MergeMap, Merges, Pair, Vocab};
use crate::tokenizer::Result;
use crate::utils::byte_level::{self};
use crate::utils::cache::DEFAULT_CACHE_CAPACITY;
use crate::vocab::bucket_vocab_store::BucketVocabStore;
use std::str::from_utf8_unchecked;

pub struct BpeConfig {
    /// `{"token": id}`, in no particular order.
    pub vocab: Vocab,
    /// The merge list in rank order.
    pub merges: Merges,
    /// The vocabulary is written in the byte-level alphabet (gpt2 and everything after it), so its
    /// entries are decoded to their raw bytes at load and every byte has to be an atom. Not a
    /// `model` field itself, but the writer needs it to spell the vocabulary.
    pub byte_level: bool,
    /// The token to emit for a character with no vocabulary entry. Must itself be in the vocab.
    pub unk_token: Option<String>,
    /// Prefix carried by every subword that is not the first of its word (WordPiece's `"##"`).
    pub continuing_subword_prefix: Option<String>,
    /// Suffix carried by the last subword of a word (`"</w>"`).
    pub end_of_word_suffix: Option<String>,
    /// Whether a run of unknown characters collapses into one unk token.
    pub fuse_unk: bool,
    /// SentencePiece's byte fallback: an unknown character becomes one `"<0xNN>"` token per byte,
    /// which requires all 256 of them to be in the vocabulary.
    pub byte_fallback: bool,
    /// Emit any pretoken that is itself a vocabulary entry as that entry, without merging. The flag
    /// only *widens* what folds: entries that provably reduce to themselves fold either way, which
    /// is what `prove_fold` settles at load.
    pub ignore_merges: bool,
    /// Merge dropout. Only `None` and `Some(0.0)` are runnable here: dropout makes tokenization
    /// non-deterministic, which the tables and the word cache are both built on the assumption of.
    /// A value outside `0.0..=1.0` is rejected as a bad config, a value above it as unsupported.
    pub dropout: Option<f32>,
    /// Slots in the per-scratch word cache; `0` turns caching off. Defaults to
    /// [`DEFAULT_CACHE_CAPACITY`](crate::models::bpe::DEFAULT_CACHE_CAPACITY).
    pub cache_capacity: usize,
}

impl Default for BpeConfig {
    fn default() -> Self {
        Self {
            vocab: Vocab::default(),
            merges: Merges::default(),
            byte_level: false,
            unk_token: None,
            continuing_subword_prefix: None,
            end_of_word_suffix: None,
            fuse_unk: false,
            byte_fallback: false,
            ignore_merges: false,
            dropout: None,
            cache_capacity: DEFAULT_CACHE_CAPACITY,
        }
    }
}

impl PipelineBPE {
    pub fn to_config(&self) -> Result<BpeConfig> {
        let byte_level = self.is_byte_level();
        // TODO: For now we keep serializing the old "printable" chars, but this costs nothing to
        // drop on serialize.
        let spell = |bytes: &[u8]| -> Result<String> {
            if byte_level {
                Ok(bytes
                    .iter()
                    .map(|&b| byte_level::BYTES_CHAR_LOOKUP[b as usize])
                    .collect())
            } else {
                String::from_utf8(bytes.to_vec())
                    .map_err(|_| "a BPE vocabulary entry is not valid UTF-8".into())
            }
        };

        let mut vocab = Vec::with_capacity(self.vocab.len());
        for (bytes, id) in self.vocab.byte_content() {
            vocab.push((spell(&bytes)?, id));
        }

        let token = |internal: u32| -> Result<String> {
            let external = self
                .tables
                .external(internal)
                .ok_or_else(|| -> crate::Error { "a merge names an unknown symbol".into() })?;
            let bytes = self
                .vocab
                .id_to_token_bytes(external)
                .ok_or_else(|| -> crate::Error {
                    "a merge names an id outside the vocabulary".into()
                })?;
            spell(bytes)
        };
        let list = self.tables.merge_list();
        let mut merges = Merges::with_capacity(list.len());
        for (_rank, left, right) in list {
            merges.push((token(left)?, token(right)?));
        }

        let (unk_token, fuse_unk, byte_fallback) = match &self.atoms {
            Atoms::Bytes => (None, false, false),
            Atoms::Chars {
                unk_token,
                fuse_unk,
                byte_fallback,
            } => {
                let unk = match *unk_token {
                    Some(internal) => Some(token(internal)?),
                    None => None,
                };
                (unk, *fuse_unk, byte_fallback.is_some())
            }
        };
        let affix = |s: &str| (!s.is_empty()).then(|| s.to_string());
        Ok(BpeConfig {
            vocab: vocab.into_iter().collect(),
            merges,
            byte_level,
            unk_token,
            continuing_subword_prefix: self.affixes.as_ref().and_then(|a| affix(&a.prefix)),
            end_of_word_suffix: self.affixes.as_ref().and_then(|a| affix(&a.suffix)),
            fuse_unk,
            byte_fallback,
            ignore_merges: self.vocab.all_foldable(),
            dropout: None,
            cache_capacity: self.cache_capacity.unwrap_or(0),
        })
    }

    /// Build a `PipelineBPE` from a vocabulary and a merge list, exactly as they are written in a
    /// `tokenizer.json`.
    ///
    /// Each merge's two tokens are looked up in `vocab`, their concatenation (minus the
    /// continuing-subword prefix on the right-hand token) is looked up as well, and the rank plus
    /// that product id become the merge's value.
    pub fn from_config(mut config: BpeConfig) -> Result<Self> {
        let vocab = std::mem::take(&mut config.vocab);
        let merges = std::mem::take(&mut config.merges);
        if let Some(p) = config.dropout
            && !(0.0..=1.0).contains(&p)
        {
            return Err(Error::InvalidDropout.into());
        }

        let mut max_len = 0;
        for key in vocab.keys() {
            if max_len < key.len() {
                max_len = key.len();
            }
        }
        let prefix_len = config
            .continuing_subword_prefix
            .as_ref()
            .map_or(0, |prefix| prefix.len());
        let mut buffer: Vec<u8> = vec![0; max_len];
        let merges: MergeMap = merges
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

        Self::from_merge_map(vocab, merges, config)
    }

    /// The same construction, entered from a vocabulary store and a merge map that have *already*
    /// been resolved against each other.
    fn from_merge_map(
        vocab: BucketVocabStore,
        merges: MergeMap,
        config: BpeConfig,
    ) -> Result<Self> {
        let BpeConfig {
            unk_token,
            continuing_subword_prefix,
            end_of_word_suffix,
            fuse_unk,
            byte_fallback,
            ignore_merges,
            dropout,
            cache_capacity,
            byte_level,
            ..
        } = config;
        if matches!(&dropout, Some(dropout) if *dropout > 0.0) {
            return Err("BPE models with dropout not supported yet".into());
        }
        // A capacity of zero means "no cache"; anything else sizes the per-scratch table.
        let cache_capacity = Some(cache_capacity).filter(|&c| c > 0);
        let prefix = continuing_subword_prefix.unwrap_or_default();
        let suffix = end_of_word_suffix.unwrap_or_default();
        if prefix.len() + 4 + suffix.len() > AFFIX_BUF {
            return Err("BPE affixes too long: raise AFFIX_BUF".into());
        }

        let (tables, external_to_internal) =
            BpeTables::build(vocab.get_vocab().into_iter().collect(), merges, byte_level);
        // the symbol stream is internal ids, mapped back through `unmap` at the very end
        let to_internal = |external: u32| -> Option<u32> {
            external_to_internal
                .get(external as usize)
                .copied()
                .filter(|&internal| internal != u32::MAX)
        };
        let (vocab, atoms) = if byte_level {
            let mut vocab = BucketVocabStore::build(vocab.byte_content());
            vocab = byte_level::transform_vocab(vocab);
            // every byte has to be an atom, or a word containing it could not be encoded at all
            for b in 0u8..=255 {
                vocab
                    .get_bytes(&[b])
                    .ok_or(Error::ByteAtomOutOfVocabulary(b))?;
            }
            (vocab, Atoms::Bytes)
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
            let unk_token = unk_token.map(|external| to_internal(external).unwrap_or(u32::MAX));
            let fallback_lookup = if byte_fallback {
                let mut fallback_lookup = [0u32; 256];
                for b in 0u8..=255 {
                    let code = format!("<{b:#04X}>");
                    let external = vocab
                        .token_to_id(&code)
                        .ok_or(Error::ByteFallbackOutOfVocabulary(b))?;
                    fallback_lookup[b as usize] =
                        to_internal(external).ok_or(Error::ByteFallbackOutOfVocabulary(b))?;
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
        let affixes = (!prefix.is_empty() || !suffix.is_empty()).then(|| Affixes {
            prefix,
            suffix,
            to_internal: external_to_internal.into_boxed_slice(),
        });
        let mut built = Self {
            atoms,
            tables,
            affixes,
            cache_capacity,
            vocab,
            byte_to_gate: build_byte_to_gate(),
        };
        // Every entry carries a foldable bit, so the encode path is one probe and one bit test
        // with no policy left in it. The policy is decided here, once: a config that declares
        // `ignore_merges` asks for every hit to fold, so every entry gets the bit; otherwise only
        // the entries that prove they reduce to themselves earn it.
        //
        // Two phases because the proof runs the merge engine, which borrows `built`: work out the
        // answers first, then set the bit on each entry that earned it.
        let proven = if ignore_merges {
            vec![true; built.vocab.id_space()]
        } else {
            built.prove_fold()
        };
        for (id, foldable) in proven.iter().enumerate() {
            if *foldable {
                built.vocab.set_foldable(id as u32);
            }
        }
        Ok(built)
    }
}
