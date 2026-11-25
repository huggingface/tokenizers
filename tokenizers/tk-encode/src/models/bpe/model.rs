//! The pipeline BPE model: its tables, and how a pretokenized sequence is turned into tokens.
//! Building it from a vocabulary and a merge list, and recovering the config back out of it, live
//! in `serialization`. Conversion to symbols lives in `convert`; the merge engines are
//! `merge_multipass` and `merge_hot_cold_queue`.
use std::collections::HashMap;

use crate::models::bpe::At;
use crate::models::bpe::convert::Affixes;
use crate::models::bpe::merge_hot_cold_queue::{QueueScratch, merge_with_queue};
use crate::models::bpe::merge_multipass::merge_multipass;
use crate::models::bpe::tables::BpeTables;
use crate::pipeline::{self, Token, Word};
use crate::tokenizer::Result;
use crate::utils::byte_level::{BYTES_CHAR_LOOKUP, char_to_byte};
use crate::vocab::bucket_vocab_store::BucketVocabStore;

const GATE_MULTI: u16 = 8;
const GATE_ASCII: u16 = 24;

/// The gate, indexed by a word's first content byte: words no longer than their gate go to
/// multipass, longer ones to the hot/cold queue.
pub(super) fn build_byte_to_gate() -> [u16; 256] {
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

pub struct BPE {
    pub(super) atoms: Atoms,
    pub(super) tables: BpeTables,
    pub(super) affixes: Option<Affixes>,
    pub(super) vocab: BucketVocabStore,
    pub(super) byte_to_gate: [u16; 256],
    /// Slots for the per-scratch word cache, from the config. `None` disables it.
    pub(super) cache_capacity: Option<usize>,
}

// A `BPE` holds exactly one `Atoms`, so `Chars`' 1 KB byte-fallback table costs nothing.
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

impl BPE {
    /// True when this model was built with `byte_level`, which means
    /// [`byte_level::transform_vocab`] already turned every vocabulary entry into its
    /// **decoded raw bytes** at load time. Decoding is then a concatenation, and running a
    /// `ByteLevel` decoder over these entries would decode a second time.
    pub fn is_byte_level(&self) -> bool {
        matches!(self.atoms, Atoms::Bytes)
    }

    /// A token's bytes, borrowed from the vocab store's slab. For a byte-level model these are
    /// the decoded bytes (see [`Self::is_byte_level`]) and a single entry is not necessarily
    /// valid UTF-8 on its own -- only the concatenation of a whole id sequence usually is.
    pub(crate) fn id_to_token_bytes(&self, id: u32) -> Option<&[u8]> {
        self.vocab.id_to_token_bytes(id)
    }

    /// The string representation of a token, with printable markers of unprintable bytes
    pub(crate) fn id_to_token(&self, id: u32) -> Option<String> {
        if !self.is_byte_level() {
            return self.vocab.id_to_token(id);
        }
        let bytes = self.id_to_token_bytes(id)?;
        Some(
            bytes
                .iter()
                .map(|&b| BYTES_CHAR_LOOKUP[b as usize])
                .collect(),
        )
    }

    /// The id of a token as [`Self::id_to_token`] spells it.
    pub(crate) fn token_to_id(&self, token: &str) -> Option<u32> {
        if !self.is_byte_level() {
            return self.vocab.token_to_id(token);
        }
        let bytes: Vec<u8> = token.chars().map(char_to_byte).collect::<Option<_>>()?;
        self.vocab.get_bytes(&bytes)
    }

    /// `{token: id}`, with each token as [`Self::id_to_token`] spells it.
    pub(crate) fn get_vocab(&self) -> HashMap<String, u32> {
        let ids = 0..self.vocab.id_space() as u32;
        ids.filter_map(|id| Some((self.id_to_token(id)?, id)))
            .collect()
    }

    pub(crate) fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// One bit per vocabulary id: can a pretoken equal to this entry be emitted as this entry,
    /// without running the merge loop?
    ///
    /// We replace the old "ignore_merges" with something that actually ignores whether or not the flag was set.
    pub(super) fn prove_fold(&self) -> Vec<bool> {
        // The id space, not the entry count: ids may be sparse, and bounding the walk by
        // `vocab.len()` would leave every entry above it unproven.
        let len = self.vocab.id_space();
        let mut proven = vec![false; len];
        let mut symbols = Vec::with_capacity(64);
        let mut scratch = QueueScratch::default();
        // With an `end_of_word_suffix`, a one-character word is looked up as `char + suffix`, so
        // a bare single-character entry is not its own encoding.
        let has_suffix = self.affixes.as_ref().is_some_and(|a| !a.suffix.is_empty());
        for id in 0..len as u32 {
            let Some(bytes) = self.vocab.id_to_token_bytes(id) else {
                continue;
            };
            // An entry that is not valid UTF-8 can never equal a pretoken, which is always a
            // `&str` slice, so it can never be folded and needs no proof.
            let Ok(text) = std::str::from_utf8(bytes) else {
                continue;
            };
            let chars = text.chars().count();
            let foldable = if chars == 0 || (chars == 1 && !has_suffix) {
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

    #[inline(always)]
    fn fold_id_keyed(&self, key: u64, hash: u64) -> Option<u32> {
        // One probe; the foldable bit is part of the id that probe already returned. Which entries
        // carry it was settled at load -- see `from_merge_map`.
        let (id, foldable) = self.vocab.get_keyed_foldable(key, hash)?;
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

/// The buffers BPE merges a word in. They are cleared, never reallocated, so merging a word does
/// not allocate.
pub struct BpeScratch {
    /// Symbols of the word being merged.
    pub(crate) symbols: Vec<u32>,
    /// Entry arena and the two queue tiers.
    pub(crate) queue: QueueScratch,
}

impl pipeline::Tokenize for BPE {
    type Scratch = BpeScratch;

    fn new_scratch(&self) -> BpeScratch {
        BpeScratch {
            symbols: Vec::with_capacity(64),
            queue: QueueScratch::default(),
        }
    }

    fn word_cache_capacity(&self) -> Option<usize> {
        self.cache_capacity
    }

    /// Appends one id when the whole word is a vocabulary entry the model may use as is (see
    /// [`BPE::prove_fold`]). Otherwise merges the word and appends the ids of its symbols.
    fn tokenize_word(
        &self,
        word: Word<'_>,
        scratch: &mut BpeScratch,
        output: &mut Vec<Token>,
    ) -> Result<()> {
        if let Some(id) = self.fold_id_keyed(word.lookup_key, word.lookup_hash) {
            output.push(Token::from(id));
            return Ok(());
        }
        self.merge_word(word.text, &mut scratch.symbols, &mut scratch.queue);
        // The merge engines work in internal ids. `unmap` turns them back into vocabulary ids.
        output.extend(
            scratch
                .symbols
                .iter()
                .map(|&symbol| Token::from(self.tables.unmap.at(symbol as usize))),
        );
        Ok(())
    }
}
