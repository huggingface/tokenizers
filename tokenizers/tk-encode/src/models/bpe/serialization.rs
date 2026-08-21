//! The config-shaped door in and out of [`PipelineBPE`], and nothing else.
//!
//! The other three models put their `Serialize`/`Deserialize` impls here. `PipelineBPE` has none:
//! it is built from, and recovered into, plain owned data ([`BpeConfig`]) with no serde on it,
//! because one of its readers ([`tk-serialize`](https://docs.rs/tk-serialize)) links no serde at
//! all. Same role in the module, one layer lower down.

use super::model::{Atoms, PipelineBPE, PipelineBpeOptions, build_byte_to_gate};
use crate::models::bpe::convert::{AFFIX_BUF, Affixes};
use crate::models::bpe::tables::BpeTables;
use crate::models::bpe::{Error, MergeMap, Merges, Pair, Vocab};
use crate::tokenizer::Result;
use crate::utils::byte_level::{self};
use crate::vocab::bucket_vocab_store::BucketVocabStore;
use std::str::from_utf8_unchecked;

/// A [`PipelineBPE`] in the shape a `tokenizer.json` spells it, as returned by
/// [`PipelineBPE::to_config`]. Plain owned data with no serde on it: the crate that writes the
/// JSON is the one that knows the format.
#[derive(Debug, Clone)]
pub struct BpeConfig {
    /// `{"token": id}`, in no particular order.
    pub vocab: Vec<(String, u32)>,
    /// The merge list in rank order.
    pub merges: Merges,
    /// Whether the model applies the byte-level map, i.e. whether the config had a `ByteLevel`
    /// pre-tokenizer. Not a `model` field itself, but the writer needs it to spell the vocabulary.
    pub byte_level: bool,
    pub unk_token: Option<String>,
    pub continuing_subword_prefix: Option<String>,
    pub end_of_word_suffix: Option<String>,
    pub fuse_unk: bool,
    pub byte_fallback: bool,
    pub ignore_merges: bool,
}

impl PipelineBPE {
    /// The model as a `tokenizer.json` spells it, recovered from the runtime tables.
    ///
    /// ## Why this is a recovery and not a getter
    ///
    /// Nothing here is stored in config shape. The merge list was consumed into a perfect-hash map
    /// plus a dense grid at load; the vocabulary became a [`BucketVocabStore`] whose entries are the
    /// *decoded* bytes when the model is byte-level; and every option was folded into
    /// [`Atoms`], [`Affixes`] or a per-entry bit. So a writer cannot ask this type what it was built
    /// from -- it has to run the construction backwards, which is what this does.
    ///
    /// It is exact for everything that can change an id, and deliberately not exact for the rest:
    ///
    /// - **Merges** come back in rank order, with the pairs the build dropped still dropped and a
    ///   pair repeated in the source collapsed to one. See [`BpeTables::merge_list`].
    /// - **A byte-level vocabulary** is re-encoded into byte-level characters, which is the
    ///   canonical spelling and not necessarily the file's: a token written with a character
    ///   outside the byte-level alphabet decoded to that character's UTF-8 bytes on the way in, and
    ///   comes back as the byte-level spelling of those same bytes. Both decode identically, so the
    ///   rebuilt model is the same model.
    /// - **`unk_token`, `fuse_unk` and `byte_fallback` on a byte-level model** are gone, because
    ///   the load path never keeps them: every byte is an atom there, so none of the three can
    ///   affect anything. They come back as `None`/`false`.
    /// - **`dropout`** is gone for the same reason, having been rejected at load unless it was
    ///   absent or zero.
    /// - **`ignore_merges`** is answered by [`BucketVocabStore::all_foldable`], which is equivalent:
    ///   see that method.
    pub fn to_config(&self) -> Result<BpeConfig> {
        let byte_level = self.is_byte_level();
        // A byte-level store holds decoded bytes, so spelling one back out is the byte-level
        // encoding of those bytes. Everything else holds the token as written.
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
        // `Affixes` is absent when both halves were empty, and each half is empty rather than
        // absent when only the other was given -- the same collapse the builder does, so an empty
        // affix and a missing one are the same model either way.
        let affix = |s: &str| (!s.is_empty()).then(|| s.to_string());
        Ok(BpeConfig {
            vocab,
            merges,
            byte_level,
            unk_token,
            continuing_subword_prefix: self.affixes.as_ref().and_then(|a| affix(&a.prefix)),
            end_of_word_suffix: self.affixes.as_ref().and_then(|a| affix(&a.suffix)),
            fuse_unk,
            byte_fallback,
            ignore_merges: self.vocab.all_foldable(),
        })
    }

    /// Build a `PipelineBPE` from a vocabulary and a merge list, exactly as they are written in a
    /// `tokenizer.json`.
    ///
    /// This is the crate boundary made concrete. `tk-serialize` reads a canonical config and links
    /// no serde at all, so it cannot go through the config-shaped
    /// `BPE` in `tk-convert` — that type is what makes an *old* serialized BPE still loadable, and
    /// depending on it here would invert the split (`tk-serialize` -> `tk-convert`) that the whole
    /// arrangement exists to prevent. So the runtime owns one serde-free door that takes the raw
    /// parts, and both readers walk through it: `tk-serialize` with what it parsed, `tk-convert`
    /// with what its builder resolved (see [`Self::from_merge_map`]).
    ///
    /// The work below is the merge-table derivation that used to live in `BpeBuilder::build`, moved
    /// verbatim: each merge's two tokens are looked up in `vocab`, their concatenation (minus the
    /// continuing-subword prefix on the right-hand token) is looked up as well, and the rank plus
    /// that product id become the merge's value. It has to happen on this side of the line because
    /// everything it feeds — the tables, the byte-level fold, the caches — is runtime state.
    pub fn from_vocab_and_merges(
        vocab: Vocab,
        merges: Merges,
        options: PipelineBpeOptions,
    ) -> Result<Self> {
        // The range check the builder used to do. It is separate from the "dropout is not
        // supported" rejection below: 0.5 is a *valid* config that this engine cannot run, while
        // 1.5 was never a valid config at all, and the two have to keep reporting differently.
        if let Some(p) = options.dropout
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
        let prefix_len = options
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

        Self::from_merge_map(vocab, merges, options)
    }

    /// The same construction, entered from a vocabulary store and a merge map that have *already*
    /// been resolved against each other.
    ///
    /// This is the door for the config-shaped `BPE` in `tk-convert`: that type keeps exactly these
    /// two tables as its own fields, because its legacy encode path needs them, so re-deriving them
    /// from re-inverted merge *strings* would be both slower and a second source of truth for what
    /// a merge means. [`Self::from_vocab_and_merges`] is this function plus that derivation.
    pub fn from_merge_map(
        vocab: BucketVocabStore,
        merges: MergeMap,
        options: PipelineBpeOptions,
    ) -> Result<Self> {
        let PipelineBpeOptions {
            unk_token,
            continuing_subword_prefix,
            end_of_word_suffix,
            fuse_unk,
            byte_fallback,
            ignore_merges,
            dropout,
            cache_capacity,
            with_byte_level,
        } = options;
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

        let (tables, external_to_internal) = BpeTables::build(
            vocab.get_vocab().into_iter().collect(),
            merges,
            with_byte_level,
        );
        // the symbol stream is internal ids, mapped back through `unmap` at the very end
        let to_internal = |external: u32| -> Option<u32> {
            external_to_internal
                .get(external as usize)
                .copied()
                .filter(|&internal| internal != u32::MAX)
        };
        let (vocab, atoms) = if with_byte_level {
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
