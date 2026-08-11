//! The pipeline BPE model: its tables, how it is built from a [`BPE`], and how a pretokenized
//! sequence is turned into tokens. Conversion to symbols lives in `convert`; the merge engines
//! are `merge_multipass` and `merge_hot_cold_queue`.
use crate::models::bpe::At;
use crate::models::bpe::Error;
use crate::models::bpe::convert::{AFFIX_BUF, Affixes};
use crate::models::bpe::legacy::model::BPE;
use crate::models::bpe::merge_hot_cold_queue::{QueueScratch, merge_with_queue};
use crate::models::bpe::merge_multipass::merge_multipass;
use crate::models::bpe::tables::BpeTables;
use crate::pipeline::{self, PipelineToken};
use crate::tokenizer::Result;
use crate::utils::byte_level::{self};
use crate::utils::word_cache::{Lookup, WordCache};
use crate::vocab::bucket_vocab_store::BucketVocabStore;

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

pub struct PipelineBPE {
    pub(super) atoms: Atoms,
    pub(super) tables: BpeTables,
    pub(super) affixes: Option<Affixes>,
    pub(super) vocab: BucketVocabStore,
    byte_to_gate: [u16; 256],
    /// Slots for the per-scratch word cache, from the config. `None` disables it.
    cache_capacity: Option<usize>,
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
    pub(crate) fn is_byte_level(&self) -> bool {
        matches!(self.atoms, Atoms::Bytes)
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

    pub fn from_bpe(model: BPE, with_byte_level: bool) -> Result<Self> {
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
            continuing_subword_prefix,
            end_of_word_suffix,
            cache,
            ..
        } = model;
        // A capacity of zero means "no cache"; anything else sizes the per-scratch table.
        let cache_capacity = cache.map(|cache| cache.capacity).filter(|&c| c > 0);
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
        // carry it was settled at load -- see `from_bpe`.
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

// todo: tests
