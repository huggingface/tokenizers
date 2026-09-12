//! The pipeline BPE model: its tables, and how a pretokenized sequence is turned into tokens.
//! Building it from a vocabulary and a merge list, and recovering the config back out of it, live
//! in `serialization`. Conversion to symbols lives in `convert`; the merge engines are
//! `merge_multipass` and `merge_hot_cold_queue`.
use crate::models::bpe::At;
use crate::models::bpe::convert::Affixes;
use crate::models::bpe::merge_hot_cold_queue::{QueueScratch, merge_with_queue};
use crate::models::bpe::merge_multipass::merge_multipass;
use crate::models::bpe::tables::BpeTables;
use crate::pipeline::{self, PipelineToken, Span};
use crate::tokenizer::Result;
use crate::utils::word_cache::{Lookup, MAX_INLINE_IDS, ProbeEmit, WordCache};
use crate::vocab::bucket_vocab_store::{BucketVocabStore, key_and_hash, key_and_hash_readable};

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

// The fused cache probe stores ids straight at a `*mut u32` pointing into the `Vec<PipelineToken>`
// the caller is filling. That is only sound while a token is layout-identical to its id.
const _: () = assert!(size_of::<PipelineToken>() == size_of::<u32>());
const _: () = assert!(align_of::<PipelineToken>() == align_of::<u32>());

pub struct PipelineBPE {
    pub(super) atoms: Atoms,
    pub(super) tables: BpeTables,
    pub(super) affixes: Option<Affixes>,
    pub(super) vocab: BucketVocabStore,
    pub(super) byte_to_gate: [u16; 256],
    /// Slots for the per-scratch word cache, from the config. `None` disables it.
    pub(super) cache_capacity: Option<usize>,
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
    /// The decode-path lookup: one adjacent-pair load instead of the `id_to_slot` -> `spans` ->
    /// `bytes` chase. Empty slice for an id the vocabulary does not hold, which is what a decoder
    /// appends for it anyway. See `BucketVocabStore::id_to_token_bytes_for_decode`.
    #[inline]
    pub(crate) fn id_to_token_bytes_for_decode(&self, id: u32) -> &[u8] {
        self.vocab.id_to_token_bytes_for_decode(id)
    }

    /// A token as a `String`, for the decoder-chain route. Only meaningful when the entries are
    /// the token strings as written, i.e. when [`Self::is_byte_level`] is false; a byte-level
    /// model decodes through [`Self::id_to_token_bytes_for_decode`] instead.
    pub(crate) fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab.id_to_token(id)
    }

    /// Every `(token, id)` the vocabulary holds, for a load-time decode transform.
    pub(crate) fn content(&self) -> Vec<(String, u32)> {
        self.vocab.content()
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

        let bytes = sequence.as_bytes();
        let (key, hash) = key_and_hash(bytes);

        let BpeScratch {
            symbols,
            queue,
            word_cache,
        } = scratch;

        // Cache before fold, for the reason given in `tokenize_spans`: the cache is one load and
        // the fold is an MPHF probe, so the fold must not run ahead of it.
        let insert_at = if let Some(cache) = word_cache.as_mut() {
            match cache.lookup_keyed(key, hash) {
                Lookup::Hit(ids) => {
                    output.extend(ids.iter().copied().map(PipelineToken::from));
                    return Ok(());
                }
                Lookup::Miss(at) => Some(at),
            }
        } else {
            None
        };

        if let Some(id) = self.fold_id_keyed(key, hash) {
            output.push(PipelineToken::from(id));
            if let Some(cache) = word_cache.as_mut()
                && let Some(at) = insert_at
            {
                cache.insert(at, std::iter::once(id));
            }
            return Ok(());
        }

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

    /// Every pre-token of a chunk in one call.
    ///
    /// Same work per word as [`Self::tokenize_pipeline`]; what changes is what is *not* repeated.
    /// The scratch is destructured once instead of once per word, the output is grown once for the
    /// whole batch instead of being capacity-checked on every push, and the virtual call, the
    /// slice and the `Result` happen once per chunk rather than once per pre-token.
    fn tokenize_spans(
        &self,
        chunk: &str,
        spans: &[Span],
        scratch: &mut Self::Scratch,
        output: &mut Vec<PipelineToken>,
    ) -> Result<()> {
        let BpeScratch {
            symbols,
            queue,
            word_cache,
        } = scratch;

        output.reserve(spans.len() + MAX_INLINE_IDS);
        let mut capacity = output.capacity();
        let mut cursor = output.len();
        // A raw write cursor held across the whole chunk. `output.as_mut_ptr()` inside the loop had
        // to be *reloaded from memory every span*: the loop also calls `set_len`, `reserve` and
        // `extend` on `output`, so the optimiser cannot assume the buffer stayed where it was. That
        // reload, the capacity test and the cursor arithmetic were a quarter of encode time in
        // `tokenize_spans` itself, with nothing of the model in it. Now the fast path -- a cache hit
        // writing inline ids -- touches only these two registers, and `dst` is refreshed solely on
        // the paths that can actually move the buffer. This is the other half of gigatoken's
        // `probe_emit_chunk`: loop-invariant cursors, refreshed only in the slow path.
        // SAFETY: `cursor <= output.len() <= capacity`, so this is inside the allocation.
        let mut dst = unsafe { output.as_mut_ptr().add(cursor) };

        // Unwrapped once for the whole chunk. `word_cache` is an `Option<WordCache>` living in the
        // scratch, so every `as_mut()` re-read its discriminant out of memory -- three times per
        // span, on a table that is either there for the entire call or not at all. Holding
        // `Option<&mut WordCache>` in a local keeps that test in a register, which is the cheap half
        // of what gigatoken's `ProbeView` does by carrying the table base and mask across a chunk.
        let mut cache_slot = word_cache.as_mut();

        for span in spans {
            // SAFETY: the pre-tokenizer cuts on char boundaries, so a span is always a valid slice
            // of this chunk. Bounds- and UTF-8-checking it again per word measured worth removing.
            let sequence = unsafe { chunk.get_unchecked(span.range()) };
            if sequence.is_empty() {
                continue;
            }

            if cursor + MAX_INLINE_IDS > capacity {
                // SAFETY: `cursor` counts what has been written so far.
                unsafe { output.set_len(cursor) };
                output.reserve(spans.len() + MAX_INLINE_IDS);
                capacity = output.capacity();
                // `reserve` may have moved the buffer.
                // SAFETY: `cursor` is what was written so far, so it is within the new allocation.
                dst = unsafe { output.as_mut_ptr().add(cursor) };
            }

            // The cache goes first. It is one direct-mapped load; the fold is an MPHF probe, which
            // is a pilot load plus a dependent entry load into the whole vocabulary. Running the
            // fold ahead of the cache paid that on every pre-token including the ones the cache
            // was about to answer, and on a warm cache that is nearly all of them.
            //
            // The two still agree on ids: `prove_fold` only sets the bit for an entry that merging
            // its own text reproduces, so a folded word and a merged word give the same answer.
            // What changes is that a foldable word now gets *inserted*, so its second and later
            // occurrences come off the cache instead of re-probing the vocabulary.
            let (key, hash) =
                key_and_hash_readable(sequence.as_bytes(), chunk.len() - span.start as usize);

            let mut placement = None;
            if let Some(cache) = cache_slot.as_deref_mut() {
                // SAFETY: the capacity check above leaves `MAX_INLINE_IDS` slots past `cursor`, and
                let found = unsafe { cache.probe_emit_keyed(key, hash, dst.cast::<u32>()) };
                match found {
                    ProbeEmit::Wrote(n) => {
                        cursor += n;
                        // SAFETY: the probe wrote `n <= MAX_INLINE_IDS` ids, which the capacity
                        // check above reserved room for.
                        dst = unsafe { dst.add(n) };
                        continue;
                    }
                    ProbeEmit::Hit(ids) => {
                        // SAFETY: `cursor` counts what has been written so far.
                        unsafe { output.set_len(cursor) };
                        output.extend(ids.iter().map(|&id| PipelineToken::from(id)));
                        cursor = output.len();
                        capacity = output.capacity();
                        // SAFETY: `cursor == output.len()`, inside the (possibly moved) allocation.
                        dst = unsafe { output.as_mut_ptr().add(cursor) };
                        continue;
                    }
                    ProbeEmit::Miss(at) => placement = Some(at),
                }
            }

            // Cache miss. The fold still answers a word that is its own vocabulary entry in one
            // probe, which beats running the merge engine for it.
            if let Some(id) = self.fold_id_keyed(key, hash) {
                // SAFETY: the check above leaves at least `MAX_INLINE_IDS >= 1` slots past `cursor`.
                unsafe { dst.write(PipelineToken::from(id)) };
                cursor += 1;
                // SAFETY: one id written, and the capacity check reserved MAX_INLINE_IDS >= 1.
                dst = unsafe { dst.add(1) };
                if let Some(cache) = cache_slot.as_deref_mut()
                    && let Some(at) = placement
                {
                    cache.insert(at, std::iter::once(id));
                }
                continue;
            }

            // SAFETY: `cursor` counts what the fast paths wrote; the merge below uses `output`
            unsafe { output.set_len(cursor) };
            let start = output.len();
            self.merge_word(sequence, symbols, queue);
            // the merge engines work in internal ids; `unmap` takes them back to the vocab's own ids
            output.extend(
                symbols
                    .iter()
                    .map(|&symbol| PipelineToken::from(self.tables.unmap.at(symbol as usize))),
            );
            if let Some(cache) = cache_slot.as_deref_mut()
                && let Some(at) = placement
            {
                cache.insert(at, output[start..].iter().map(|token| token.id()));
            }
            cursor = output.len();
            capacity = output.capacity();
            // SAFETY: `cursor == output.len()`; `extend` above may have moved the buffer.
            dst = unsafe { output.as_mut_ptr().add(cursor) };
        }
        // SAFETY: `cursor` counts every token written, by the fast paths and the slow one alike.
        unsafe { output.set_len(cursor) };
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
