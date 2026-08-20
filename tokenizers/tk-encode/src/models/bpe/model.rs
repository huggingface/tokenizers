//! The pipeline BPE model: its tables, how it is built from a vocabulary and a merge list, and how
//! a pretokenized sequence is turned into tokens. Conversion to symbols lives in `convert`; the
//! merge engines are `merge_multipass` and `merge_hot_cold_queue`.
use crate::models::bpe::At;
use crate::models::bpe::convert::{AFFIX_BUF, Affixes};
use crate::models::bpe::merge_hot_cold_queue::{QueueScratch, merge_with_queue};
use crate::models::bpe::merge_multipass::merge_multipass;
use crate::models::bpe::tables::BpeTables;
use crate::models::bpe::{Error, MergeMap, Merges, Pair, Vocab};
use crate::pipeline::{self, PipelineToken};
use crate::tokenizer::Result;
use crate::utils::byte_level::{self};
use crate::utils::cache::DEFAULT_CACHE_CAPACITY;
use crate::utils::word_cache::{Lookup, WordCache};
use crate::vocab::bucket_vocab_store::BucketVocabStore;
use std::str::from_utf8_unchecked;

const GATE_MULTI: u16 = 8;
const GATE_ASCII: u16 = 24;

/// The gate, indexed by a word's first content byte: words no longer than their gate go to
/// multipass, longer ones to the hot/cold queue.
fn build_byte_to_gate() -> [u16; 256] {
    let mut b2g = [GATE_MULTI; 256];
    b2g[..0x80].fill(GATE_ASCII);
    // Kept for a word that is *only* a delimiter (a run of spaces), where there is no content to
    // classify. Words with content are indexed past their delimiter -- see [`content_start`].
    for ws in *b" \t\n\r" {
        b2g[ws as usize] = GATE_MULTI;
    }
    b2g
}

/// ByteLevel produces `" word"` or `"Ġword"`, Metaspace produces `"▁word"`. Indexing byte 0
/// classifies the delimiter instead of the content.
#[inline]
fn content_start(bytes: &[u8]) -> usize {
    match bytes {
        // Metaspace `▁` (U+2581).
        [0xE2, 0x96, 0x81, rest @ ..] if !rest.is_empty() => 3,
        // ByteLevel `Ġ` (U+0120) -- the byte-level spelling of a leading space.
        [0xC4, 0xA0, rest @ ..] if !rest.is_empty() => 2,
        // A literal leading space, which a ByteLevel pre-tokenizer also hands over.
        [ws, rest @ ..] if ws.is_ascii_whitespace() && !rest.is_empty() => 1,
        _ => 0,
    }
}

// The fused cache probe stores ids straight at a `*mut u32` pointing into the `Vec<PipelineToken>`
// the caller is filling. That is only sound while a token is layout-identical to its id.
const _: () = assert!(size_of::<PipelineToken>() == size_of::<u32>());
const _: () = assert!(align_of::<PipelineToken>() == align_of::<u32>());

/// Everything a [`PipelineBPE`] needs besides its vocabulary and its merge list.
///
/// A `Default`-able struct rather than a nine-argument constructor, so a caller spells only the
/// fields its config actually declares — which for most of the Hub is `with_byte_level` and nothing
/// else. It carries no serde: naming these options in a config file is the readers' business, and
/// one of those readers ([`tk-serialize`](https://docs.rs/tk-serialize)) links no serde at all.
pub struct PipelineBpeOptions {
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
    /// Merge dropout. Only `None` and `Some(0.0)` are runnable here: dropout makes tokenization
    /// non-deterministic, which the tables and the word cache are both built on the assumption of.
    /// A value outside `0.0..=1.0` is rejected as a bad config, a value above it as unsupported.
    pub dropout: Option<f32>,
    /// Emit any pretoken that is itself a vocabulary entry as that entry, without merging. The flag
    /// only *widens* what folds: entries that provably reduce to themselves fold either way, which
    /// is what `prove_fold` settles at load.
    pub ignore_merges: bool,
    /// Slots in the per-scratch word cache; `0` turns caching off. Defaults to
    /// [`DEFAULT_CACHE_CAPACITY`](crate::models::bpe::DEFAULT_CACHE_CAPACITY).
    pub cache_capacity: usize,
    /// The vocabulary is written in the byte-level alphabet (gpt2 and everything after it), so its
    /// entries are decoded to their raw bytes at load and every byte has to be an atom.
    pub with_byte_level: bool,
}

impl Default for PipelineBpeOptions {
    fn default() -> Self {
        Self {
            unk_token: None,
            continuing_subword_prefix: None,
            end_of_word_suffix: None,
            fuse_unk: false,
            byte_fallback: false,
            dropout: None,
            ignore_merges: false,
            cache_capacity: DEFAULT_CACHE_CAPACITY,
            with_byte_level: false,
        }
    }
}

pub struct PipelineBPE {
    pub(super) atoms: Atoms,
    pub(super) tables: BpeTables,
    pub(super) affixes: Option<Affixes>,
    pub(super) vocab: BucketVocabStore,
    byte_to_gate: [u16; 256],
    /// Slots for the per-scratch word cache, from the config. `None` disables it.
    cache_capacity: Option<usize>,
}

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

// A `PipelineBPE` holds exactly one `Atoms`, so `Chars`' 1 KB byte-fallback table costs nothing.
#[allow(clippy::large_enum_variant)]
pub(super) enum Atoms {
    /// The atoms are the 256 bytes; the symbol for each lives in `BpeTables::byte_internal`.
    Bytes,
    Chars {
        byte_fallback: Option<[u32; 256]>,
        unk_token: Option<u32>,
        fuse_unk: bool,
    },
}

impl PipelineBPE {
    /// True when this model was built with `with_byte_level`, which means
    /// [`byte_level::transform_vocab`] already turned every vocabulary entry into its
    /// **decoded raw bytes** at load time. Decoding is then a concatenation, and running a
    /// `ByteLevel` decoder over these entries would decode a second time.
    pub fn is_byte_level(&self) -> bool {
        matches!(self.atoms, Atoms::Bytes)
    }

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

    /// A token's bytes, borrowed from the vocab store's slab. For a byte-level model these are
    /// the decoded bytes (see [`Self::is_byte_level`]) and a single entry is not necessarily
    /// valid UTF-8 on its own -- only the concatenation of a whole id sequence usually is.
    pub(crate) fn id_to_token_bytes(&self, id: u32) -> Option<&[u8]> {
        self.vocab.id_to_token_bytes(id)
    }

    /// A token as a `String`, for the decoder-chain route. Only meaningful when the entries are
    /// the token strings as written, i.e. when [`Self::is_byte_level`] is false; a byte-level
    /// model decodes through [`Self::id_to_token_bytes`] instead.
    pub(crate) fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab.id_to_token(id)
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

    /// One bit per vocabulary id: can a pretoken equal to this entry be emitted as this entry,
    /// without running the merge loop?
    ///
    /// We replace the old "ignore_merges" with something that actually ignores whether or not the flag was set.
    fn prove_fold(&self) -> Vec<bool> {
        // The id space, not the entry count: ids may be sparse, and bounding the walk by
        // `vocab.len()` would leave every entry above it unproven.
        let len = self.vocab.id_space();
        let mut proven = vec![false; len];
        let mut symbols = Vec::with_capacity(64);
        let mut scratch = QueueScratch::default();
        for id in 0..len as u32 {
            let Some(bytes) = self.vocab.id_to_token_bytes(id) else {
                continue;
            };
            // An entry that is not valid UTF-8 can never equal a pretoken, which is always a
            // `&str` slice, so it can never be folded and needs no proof.
            let Ok(text) = std::str::from_utf8(bytes) else {
                continue;
            };
            let foldable = if text.chars().count() <= 1 {
                // A single atom has no pair to merge and is trivially its own encoding.
                true
            } else {
                self.merge_word(text, &mut symbols, &mut scratch);
                symbols.len() == 1 && self.tables.unmap.at(symbols[0] as usize) == id
            };
            proven[id as usize] = foldable;
        }
        proven
    }

    /// The id to emit for `sequence` without merging, when the whole pretoken is a vocabulary
    /// entry that may be folded. `None` sends the word to the merge engines.
    #[inline(always)]
    fn fold_id(&self, sequence: &str) -> Option<u32> {
        // One probe; the foldable bit is part of the id that probe already returned. Which entries
        // carry it was settled at load -- see `from_merge_map`.
        let (id, foldable) = self.vocab.get_bytes_foldable(sequence.as_bytes())?;
        foldable.then_some(id)
    }

    /// Converts a word to symbols and merges it. The gate, indexed by the word's first *content*
    /// byte (past any delimiter the pre-tokenizer prepended -- see [`content_start`]), says
    /// which engine gets it: short words go to multipass, longer ones to the hot/cold queue.
    /// `symbols` is the caller's reusable symbol buffer -- it lives in the scratch so that a word
    /// costs no allocation. On return it holds the merged word as internal ids, which the caller
    /// maps to external ids through `unmap`.
    pub(super) fn merge_word(
        &self,
        sequence: &str,
        symbols: &mut Vec<u32>,
        queue_scratch: &mut QueueScratch,
    ) {
        let bytes = sequence.as_bytes();
        // Classify on the first content byte, not on the delimiter the pre-tokenizer prepended.
        let gate: u16 = self.byte_to_gate[bytes[content_start(bytes)] as usize];

        if sequence.len() > gate as usize {
            // conversion writes the entries and cold keys directly: no intermediate symbol array
            self.convert_queue(sequence, symbols, queue_scratch);
            merge_with_queue(&self.tables, symbols, queue_scratch);
        } else {
            let first_merge = self.convert_multipass(sequence, symbols);
            merge_multipass(&self.tables, symbols, first_merge);
        }
    }
}

/// Per-thread scratch for BPE. Every buffer here is cleared, never reallocated, so tokenizing a
/// sequence does not allocate.
pub struct BpeScratch {
    /// Symbols of the word being merged.
    pub(crate) symbols: Vec<u32>,
    /// Entry arena and the two queue tiers.
    pub(crate) queue: QueueScratch,
    /// Words already seen, so a repeat costs a probe instead of a merge. It lives in the scratch
    /// so it outlives the encode call that fills it -- otherwise it would never see a word twice.
    pub(crate) word_cache: Option<WordCache>,
}

impl pipeline::ModelScratch for BpeScratch {}

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

        if let Some(id) = self.fold_id(sequence) {
            output.push(PipelineToken::from(id));
            return Ok(());
        }

        let BpeScratch {
            symbols,
            queue,
            word_cache,
        } = scratch;

        // A word seen before costs a probe instead of a merge.
        let insert_at = if let Some(cache) = word_cache.as_mut() {
            match cache.lookup(sequence.as_bytes()) {
                Lookup::Hit(ids) => {
                    output.extend(ids.iter().copied().map(PipelineToken::from));
                    return Ok(());
                }
                Lookup::Miss(at) => Some(at),
            }
        } else {
            None
        };

        let start = output.len();
        self.merge_word(sequence, symbols, queue);
        // the merge engines work in internal ids; `unmap` takes them back to the vocab's own ids
        output.extend(
            symbols
                .iter()
                .map(|&symbol| PipelineToken::from(self.tables.unmap.at(symbol as usize))),
        );
        if let Some(cache) = word_cache.as_mut()
            && let Some(at) = insert_at
        {
            cache.insert(at, output[start..].iter().map(|token| token.id()));
        }

        Ok(())
    }

    fn init_scratch(&self) -> Self::Scratch {
        Self::Scratch {
            symbols: Vec::with_capacity(64),
            queue: QueueScratch::default(),
            word_cache: self.cache_capacity.map(WordCache::new),
        }
    }
}
