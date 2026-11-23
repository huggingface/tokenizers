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
use crate::pipeline::{self, PreTokens, Token};
use crate::tokenizer::Result;
use crate::utils::byte_level::{BYTES_CHAR_LOOKUP, char_to_byte};
use crate::utils::signature::Signature;
use crate::utils::word_cache::{Lookup, MAX_RUN_IDS, WordCache};
use crate::vocab::bucket_vocab_store::BucketVocabStore;

/// How many words [`BPE::tokenize_spans`] reads the signatures of before it looks any of them up.
// Measured at 8, 16, 32 and 64 on an EPYC 7R13: each step up ran fewer instructions per word, and
// 32 and 64 were the fastest, on text the cache had seen and on text it had not.
const BATCH: usize = 64;

/// Reads the signature of every word of `batch` into `signatures`, and starts loading each
/// word's home slot in the cache.
///
/// A signature costs the same whatever the word's length, with no branch. So every slot is on its
/// way to the CPU cache before the first lookup needs one.
// Out of line: inside the encode loop, the compiler reloaded the loop's addresses per word.
#[inline(never)]
fn read_signatures(batch: PreTokens<'_>, signatures: &mut [Signature], cache: Option<&WordCache>) {
    let chunk_len = batch.chunk().len();
    let words = signatures.iter_mut().zip(batch.iter());
    // One loop per case, so that neither tests for a cache on every word.
    match cache {
        Some(cache) => {
            for (signature, (span, sequence)) in words {
                let readable = chunk_len - span.start as usize;
                *signature = Signature::new_readable(sequence.as_bytes(), readable);
                cache.prefetch(*signature);
            }
        }
        None => {
            for (signature, (span, sequence)) in words {
                let readable = chunk_len - span.start as usize;
                *signature = Signature::new_readable(sequence.as_bytes(), readable);
            }
        }
    }
}

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

// The fused cache probe stores ids straight at a `*mut u32` pointing into the `Vec<Token>`
// the caller is filling. That is only sound while a token is layout-identical to its id.
const _: () = assert!(size_of::<Token>() == size_of::<u32>());
const _: () = assert!(align_of::<Token>() == align_of::<u32>());

pub struct BPE {
    pub(super) atoms: Atoms,
    pub(super) tables: BpeTables,
    pub(super) affixes: Option<Affixes>,
    pub(super) vocab: BucketVocabStore,
    pub(super) byte_to_gate: [u16; 256],
    /// The [`Signature::hash`] of every token longer than a [`Signature`] holds whole that is not
    /// what merging its own text gives, sorted. Only a word with one of these hashes needs the
    /// vocabulary: every other long word skips the fold. Empty for most models.
    pub(super) long_folds: Box<[u64]>,
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

    /// The sorted [`Signature::hash`] of every token longer than a [`Signature`] holds whole,
    /// that a word can equal, and that is not what merging its own text gives. `merged` is
    /// [`Self::prove_fold`]'s answer.
    pub(super) fn long_folds(&self, merged: &[bool]) -> Box<[u64]> {
        let mut hashes: Vec<u64> = Vec::new();
        for id in 0..self.vocab.id_space() as u32 {
            let Some(bytes) = self.vocab.id_to_token_bytes(id) else {
                continue;
            };
            if bytes.len() > Signature::BYTES
                && std::str::from_utf8(bytes).is_ok()
                && !merged[id as usize]
            {
                hashes.push(Signature::new(bytes).hash);
            }
        }
        hashes.sort_unstable();
        hashes.into_boxed_slice()
    }

    #[inline(always)]
    fn fold_id(&self, word: &[u8], signature: Signature) -> Option<u32> {
        // 99% of tokens fit a signature, and a word that does not is rarely a token. Merging it
        // gives the same ids, without hashing all of its bytes for a probe that misses.
        if signature.len > Signature::BYTES
            && self.long_folds.binary_search(&signature.hash).is_err()
        {
            return None;
        }
        // One probe; the foldable bit is part of the id that probe already returned. Which entries
        // carry it was settled at load -- see `from_merge_map`.
        let (id, foldable) = self.vocab.get_foldable(word, signature)?;
        foldable.then_some(id)
    }

    /// Tokenizes one word whose signature is already read: from the cache, else as its own
    /// vocabulary entry, else through the merge engine.
    ///
    /// The cache goes first. It is one load, where the fold is a perfect-hash probe into the
    /// whole vocabulary. The two agree on ids: `prove_fold` only sets the bit for an entry that
    /// merging its own text reproduces. A folded word is inserted too, so its later occurrences
    /// come off the cache.
    // Out of line: it holds the lookup, the fold and the merge call, inlined, and has two callers.
    #[inline(never)]
    fn tokenize_with_signature(
        &self,
        sequence: &str,
        signature: Signature,
        scratch: &mut BpeScratch,
        output: &mut Vec<Token>,
    ) {
        if sequence.is_empty() {
            return;
        }
        let bytes = sequence.as_bytes();
        let BpeScratch {
            symbols,
            queue,
            word_cache,
        } = scratch;

        let mut insert_at = None;
        if let Some(cache) = word_cache.as_ref() {
            match cache.lookup_with_signature(bytes, signature) {
                Lookup::Hit(ids) => {
                    output.reserve(ids.room());
                    // SAFETY: the spare capacity holds `ids.room()` ids, and a `Token` is laid
                    // out as its `u32` id. The hit wrote `written` of them.
                    unsafe {
                        let dst = output.as_mut_ptr().add(output.len()).cast::<u32>();
                        let written = ids.write(dst);
                        output.set_len(output.len() + written);
                    }
                    return;
                }
                Lookup::Miss(at) => insert_at = Some(at),
            }
        }

        let start = output.len();
        match self.fold_id(bytes, signature) {
            Some(id) => output.push(Token::from(id)),
            None => {
                self.merge_word(sequence, symbols, queue);
                // The merge engines work in internal ids; `unmap` takes them back to the vocab's own ids.
                output.extend(
                    symbols
                        .iter()
                        .map(|&symbol| Token::from(self.tables.unmap.at(symbol as usize))),
                );
            }
        }
        if let (Some(cache), Some(at)) = (word_cache.as_mut(), insert_at) {
            cache.insert(at, bytes, output[start..].iter().map(|token| token.id()));
        }
    }

    /// Tokenizes a batch of words whose signatures are already read.
    ///
    /// [`WordCache::emit_home_hits`] answers runs of words straight from the cache, and
    /// [`Self::tokenize_with_signature`] takes each word that a run stops on.
    // Out of line, with the run loop inlined into it: on text the cache has not seen a run is
    // 2 to 5 words long, and a call per run cost more than the run.
    #[inline(never)]
    fn tokenize_batch(
        &self,
        batch: PreTokens<'_>,
        signatures: &[Signature],
        scratch: &mut BpeScratch,
        output: &mut Vec<Token>,
    ) {
        let mut next = 0;
        while next < signatures.len() {
            if let Some(cache) = scratch.word_cache.as_ref() {
                let run = &signatures[next..];
                output.reserve(MAX_RUN_IDS * run.len());
                // SAFETY: the spare capacity holds `MAX_RUN_IDS` ids per word of the run, and a
                // `Token` is laid out as its `u32` id. The run wrote `written` of them.
                unsafe {
                    let dst = output.as_mut_ptr().add(output.len()).cast::<u32>();
                    let word = |i| batch.text(next + i).as_bytes();
                    let (answered, written) = cache.emit_home_hits(run, word, dst);
                    output.set_len(output.len() + written);
                    next += answered;
                }
                if next == signatures.len() {
                    break;
                }
            }
            self.tokenize_with_signature(batch.text(next), signatures[next], scratch, output);
            next += 1;
        }
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

impl pipeline::Tokenize for BPE {
    type Scratch = BpeScratch;

    fn tokenize(
        &self,
        sequence: &str,
        scratch: &mut Self::Scratch,
        output: &mut Vec<Token>,
    ) -> Result<()> {
        let signature = Signature::new(sequence.as_bytes());
        self.tokenize_with_signature(sequence, signature, scratch, output);
        Ok(())
    }

    /// Every pre-token of a chunk in one call, [`BATCH`] words at a time.
    ///
    /// For each batch, [`read_signatures`] reads every word's signature first, then
    /// [`BPE::tokenize_batch`] looks the words up.
    // Inlining measure faster on x86
    #[inline(always)]
    fn tokenize_spans(
        &self,
        pre_tokens: PreTokens<'_>,
        scratch: &mut Self::Scratch,
        output: &mut Vec<Token>,
    ) -> Result<()> {
        let mut signatures = [Signature::default(); BATCH];
        for batch in pre_tokens.batches(BATCH) {
            let signatures = &mut signatures[..batch.spans().len()];
            read_signatures(batch, signatures, scratch.word_cache.as_ref());
            self.tokenize_batch(batch, signatures, scratch, output);
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
