//! The different functions used to go from the serialized json vocab / merges to building the BPE.

use super::model::{Atoms, BPE, build_byte_to_gate};
use crate::models::bpe::convert::{AFFIX_BUF, Affixes};
use crate::models::bpe::tables::{BpeTables, NO_SYMBOL};
use crate::models::bpe::{Error, MergeMap, Merges, Pair, Vocab};
use crate::tokenizer::Result;
use crate::utils::byte_level::{self};
use crate::utils::cache::DEFAULT_CACHE_CAPACITY;
use crate::vocab::bucket_vocab_store::BucketVocabStore;
use std::collections::HashSet;
use std::ops::RangeInclusive;

/// Every byte a `&str` can hold except 0xF1 | 0xF2, which are UTF-8 lead bytes that prefix unassigned codepoints
const VALID_UTF8_BYTES: [RangeInclusive<u8>; 3] = [0x00..=0xBF, 0xC2..=0xF0, 0xF3..=0xF4];

#[non_exhaustive]
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
    /// Emit tokens the merge list cannot build whole when a pretoken starts with them. Off by
    /// default: a malformed vocab keeps tokenizing the way it always did.
    pub prefix_atoms: bool,
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
            prefix_atoms: false,
        }
    }
}

impl BPE {
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
        let (prefix, suffix): (&[u8], &[u8]) = match &self.affixes {
            Some(affixes) => (&affixes.prefix, &affixes.suffix),
            None => (&[], &[]),
        };
        let affix = |bytes: &[u8]| (!bytes.is_empty()).then(|| spell(bytes)).transpose();
        Ok(BpeConfig {
            vocab: vocab.into_iter().collect(),
            merges,
            byte_level,
            unk_token,
            continuing_subword_prefix: affix(prefix)?,
            end_of_word_suffix: affix(suffix)?,
            fuse_unk,
            byte_fallback,
            ignore_merges: self.vocab.all_foldable(),
            dropout: None,
            cache_capacity: self.cache_capacity.unwrap_or(0),
            prefix_atoms: !self.roots.is_empty(),
        })
    }

    /// Build a `BPE` from a vocabulary and a merge list, exactly as they are written in a
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

        let prefix = config
            .continuing_subword_prefix
            .as_deref()
            .unwrap_or_default();
        let mut buffer = String::new();
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
                let Some(b_suffix) = b.strip_prefix(prefix) else {
                    return Err(Error::MergeTokenWithoutPrefix {
                        token: b,
                        prefix: prefix.to_owned(),
                    }
                    .into());
                };
                buffer.clear();
                buffer.push_str(&a);
                buffer.push_str(b_suffix);
                let new_id = vocab
                    .get(&buffer)
                    .ok_or_else(|| Error::MergeTokenOutOfVocabulary(buffer.clone()))?;
                Ok(((*a_id, *b_id), (i as u32, *new_id)))
            })
            .collect::<Result<MergeMap>>()?;

        let roots = if config.prefix_atoms {
            unreachable_roots(&vocab, &merges, config.byte_level)
        } else {
            Vec::new()
        };

        Self::from_merge_map(vocab, merges, roots, config)
    }

    /// The same construction, entered from a vocabulary and a merge map that have *already*
    /// been resolved against each other.
    fn from_merge_map(
        vocab: Vocab,
        merges: MergeMap,
        roots: Vec<(Box<[u8]>, u32)>,
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

        let (tables, external_to_internal) = BpeTables::build(&vocab, merges, byte_level);
        // the symbol stream is internal ids, mapped back through `unmap` at the very end
        let to_internal = |external: u32| -> Option<u32> {
            external_to_internal
                .get(external as usize)
                .copied()
                .filter(|&internal| internal != NO_SYMBOL)
        };
        let (vocab, atoms) = if byte_level {
            let missing: Vec<u8> = VALID_UTF8_BYTES
                .into_iter()
                .flatten()
                .filter(|&byte| tables.byte_internal[byte as usize] == NO_SYMBOL)
                .collect();
            if !missing.is_empty() {
                warn!(
                    "this byte-level vocabulary has no token for bytes {missing:02X?}: they will be \
                     ignored at encoding."
                );
            }
            (byte_level::transform_vocab(vocab)?, Atoms::Bytes)
        } else {
            let vocab = BucketVocabStore::build(
                vocab
                    .into_iter()
                    .map(|(k, v)| (k.into_bytes(), v))
                    .collect(),
            )?;
            let unk_token = if let Some(unk_str) = unk_token {
                let token_id = vocab
                    .token_to_id(&unk_str)
                    .ok_or_else(|| Error::UnkTokenOutOfVocabulary(unk_str.clone()))?;
                Some(token_id)
            } else {
                None
            };
            let unk_token = unk_token.map(|external| to_internal(external).unwrap_or(NO_SYMBOL));
            let fallback_lookup = if byte_fallback {
                let mut fallback_lookup = [NO_SYMBOL; 256];
                for b in 0u8..=255 {
                    let code = format!("<{b:#04X}>");
                    if let Some(internal) = vocab.token_to_id(&code).and_then(to_internal) {
                        fallback_lookup[b as usize] = internal;
                    }
                }

                // NVIDIA NeMo models such as Parakeet set `byte_fallback` with no `<0xNN>` token at all.
                // In this case, we turn off the byte fallback at load time.
                if fallback_lookup.iter().all(|&id| id == NO_SYMBOL) {
                    None
                } else {
                    // Some model don't store fallback tokens for single-bytes characters that
                    // are already in the vocab. For example, Gemma doesn't store a <0x09> fallback token (`\t`).
                    // It's a valid configuration because the byte fallback only triggers when
                    // a character is missing from the vocabulary.

                    let has_affixes = !prefix.is_empty() || !suffix.is_empty();
                    let mut missing = Vec::new();
                    for b in 0u8..=255 {
                        if fallback_lookup[b as usize] != NO_SYMBOL {
                            continue;
                        }
                        let literal = (b.is_ascii() && !has_affixes)
                            .then(|| vocab.token_to_id(&(b as char).to_string()))
                            .flatten()
                            .and_then(to_internal);
                        match literal {
                            Some(internal) => fallback_lookup[b as usize] = internal,
                            None => missing.push(b),
                        }
                    }
                    if !missing.is_empty() {
                        return Err(Error::ByteFallbackOutOfVocabulary(missing).into());
                    }
                    Some(fallback_lookup)
                }
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
        // A byte-level vocab is keyed by raw bytes, so its affixes are mapped the same way.
        let convert_byte_level = |affix: &str| -> Box<[u8]> {
            if byte_level {
                affix.chars().flat_map(byte_level::reverse_lookup).collect()
            } else {
                affix.as_bytes().into()
            }
        };
        let affixes = (!prefix.is_empty() || !suffix.is_empty()).then(|| Affixes {
            prefix: convert_byte_level(&prefix),
            suffix: convert_byte_level(&suffix),
            to_internal: external_to_internal.into_boxed_slice(),
        });
        let mut built = Self {
            atoms,
            tables,
            affixes,
            cache_capacity,
            vocab,
            roots,
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

/// Every vocabulary token no merge list can build, longest first.
fn unreachable_roots(vocab: &Vocab, merges: &MergeMap, byte_level: bool) -> Vec<(Box<[u8]>, u32)> {
    let pairs: HashSet<Pair> = merges.keys().copied().collect();
    let mut tokens: Vec<(&str, u32)> = vocab.iter().map(|(s, &id)| (s.as_str(), id)).collect();
    tokens.sort_by_key(|(token, _)| token.len());
    let mut seen: HashSet<u32> = HashSet::new();
    let mut roots = Vec::new();
    for (token, id) in tokens {
        let ok = token.chars().count() == 1
            || token.char_indices().skip(1).any(|(i, _)| {
                let (a, b) = token.split_at(i);
                matches!((vocab.get(a), vocab.get(b)), (Some(&x), Some(&y))
                    if seen.contains(&x) && seen.contains(&y) && pairs.contains(&(x, y)))
            });
        if ok {
            seen.insert(id);
        } else if token.chars().count() > 1 {
            let bytes: Box<[u8]> = if byte_level {
                token.chars().flat_map(byte_level::reverse_lookup).collect()
            } else {
                token.as_bytes().into()
            };
            roots.push((bytes, id));
        }
    }
    roots.sort_by_key(|(token, _)| std::cmp::Reverse(token.len()));
    roots
}
