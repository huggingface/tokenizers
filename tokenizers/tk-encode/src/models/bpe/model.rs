//! The pipeline BPE model: its tables, how it is built from a [`BPE`], and how a pretokenized
//! sequence is turned into tokens. Conversion to symbols lives in `convert`; the merge engines
//! are `merge_multipass` and `merge_hot_cold_queue`.
use crate::models::bpe::At;
use crate::models::bpe::Error;
use crate::models::bpe::convert::{AFFIX_BUF, Affixes};
use crate::models::bpe::legacy::model::BPE;
use crate::models::bpe::merge_hot_cold_queue::{QueueScratch, merge_hot_cold_queue};
use crate::models::bpe::merge_multipass::merge_multipass;
use crate::models::bpe::tables::BpeTables;
use crate::pipeline::{self, PipelineToken, Span};
use crate::tokenizer::Result;
use crate::utils::byte_level::{self};
use crate::utils::word_cache::{Lookup, MAX_INLINE_IDS, ProbeEmit, WordCache};
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

// `tokenize_spans` hands the word cache a `*mut u32` pointing into the `Vec<PipelineToken>` it is
// filling, so the probe can store ids straight at the cursor. That is only sound while a token is
// layout-identical to its id.
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
            vocab,
            byte_to_gate: build_byte_to_gate(),
            cache_capacity,
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
            self.convert_queue(
                sequence,
                symbols,
                &mut queue_scratch.entries,
                &mut queue_scratch.cold,
            );
            merge_hot_cold_queue(&self.tables, symbols, queue_scratch);
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

        // The fold goes first: it answers a word that is itself a vocabulary entry in one probe,
        // which is cheaper than a cache probe. Folded words therefore never enter the cache --
        // they are already as cheap as a hit.
        if let Some(id) = self.fold_id(sequence) {
            output.push(PipelineToken { id });
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
                    output.extend(ids.iter().map(|&id| PipelineToken { id }));
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
        output.extend(symbols.iter().map(|&symbol| PipelineToken {
            id: self.tables.unmap.at(symbol as usize),
        }));
        if let Some(cache) = word_cache.as_mut()
            && let Some(at) = insert_at
        {
            cache.insert(at, output[start..].iter().map(|token| token.id));
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
        // 92% of English pre-tokens are one id and 98% are at most two, so reserve for two apiece
        // plus the probe's headroom. `spans.len()` alone is a *lower* bound, which would make the
        // buffer grow -- and memcpy what it already holds -- partway through most chunks.
        output.reserve(2 * spans.len() + MAX_INLINE_IDS);
        let mut capacity = output.capacity();
        let mut cursor = output.len();

        for span in spans {
            // SAFETY: the pre-tokenizer cuts on char boundaries, so a span is always a valid slice
            // of this chunk. Bounds- and UTF-8-checking it again per word measured worth removing.
            let sequence = unsafe { chunk.get_unchecked(span.range()) };
            if sequence.is_empty() {
                continue;
            }
            // One capacity check per word, covering both the fold's single write and the probe's
            // `MAX_INLINE_IDS` lanes. After it, writing that many past `cursor` is in bounds.
            if cursor + MAX_INLINE_IDS > capacity {
                // SAFETY: `cursor` counts what has been written so far.
                unsafe { output.set_len(cursor) };
                output.reserve(spans.len() + MAX_INLINE_IDS);
                capacity = output.capacity();
            }
            // The fold goes first -- see `tokenize_pipeline` for why.
            if let Some(id) = self.fold_id(sequence) {
                // SAFETY: the check above leaves at least `MAX_INLINE_IDS >= 1` slots past `cursor`.
                unsafe { output.as_mut_ptr().add(cursor).write(PipelineToken { id }) };
                cursor += 1;
                continue;
            }
            let mut placement = None;
            if let Some(cache) = word_cache.as_mut() {
                // The probe writes the ids at the cursor itself, so a hit is one load of the slot
                // and one unconditional store of its lanes -- the ids never become a slice and the
                // line is never read twice.
                // SAFETY: the check above leaves `MAX_INLINE_IDS` slots past `cursor`, and
                // `PipelineToken` is a single `u32` (asserted below), so the cast is sound.
                let found = unsafe {
                    cache.probe_emit(
                        sequence.as_bytes(),
                        output.as_mut_ptr().add(cursor).cast::<u32>(),
                    )
                };
                match found {
                    ProbeEmit::Wrote(n) => {
                        cursor += n;
                        continue;
                    }
                    // A hit the fast path could not serve: the probe already found the ids, so
                    // copy those rather than probing a second time.
                    ProbeEmit::Hit(ids) => {
                        // SAFETY: `cursor` counts what has been written so far.
                        unsafe { output.set_len(cursor) };
                        output.extend(ids.iter().map(|&id| PipelineToken { id }));
                        cursor = output.len();
                        capacity = output.capacity();
                        continue;
                    }
                    ProbeEmit::Miss(at) => placement = Some(at),
                }
            }
            // SAFETY: `cursor` counts what the fast paths wrote; the merge below uses `output`
            // through its normal API, so its length has to be true again first.
            unsafe { output.set_len(cursor) };
            let start = output.len();
            self.merge_word(sequence, symbols, queue);
            output.extend(symbols.iter().map(|&symbol| PipelineToken {
                id: self.tables.unmap.at(symbol as usize),
            }));
            if let Some(cache) = word_cache.as_mut()
                && let Some(at) = placement
            {
                cache.insert(at, output[start..].iter().map(|token| token.id));
            }
            cursor = output.len();
            capacity = output.capacity();
        }
        // SAFETY: `cursor` counts every token written above.
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

#[cfg(test)]
mod fold_tests {
    use crate::pipeline::PipelineTokenizer;
    use crate::Tokenizer;

    /// The proven fold emits a vocabulary entry without merging, so it is only valid if the merge
    /// loop would have produced that same entry. gpt2 does not declare `ignore_merges`, so here
    /// the fold is on purely because the proof enabled it -- which makes it the config where a
    /// wrong proof would show up.
    ///
    /// These strings mix words that are a single vocabulary entry (folded) with words that are
    /// not (merged), and include the special token whose entry does NOT fold: `<|endoftext|>`
    /// decomposes to seven tokens, and folding it would emit one.
    #[test]
    fn the_proven_fold_never_changes_the_ids() {
        let reference = Tokenizer::from_file("../data/gpt2.json").unwrap();
        let pipe = PipelineTokenizer::try_from(&reference).unwrap();

        for text in [
            " the quick brown fox jumps over the lazy dog",
            "unprefixed words and internationalisation",
            "def foo(bar):\n    return bar + 1\n",
            "<|endoftext|> literal in the middle <|endoftext|>",
            " 语言模型 mixed with ASCII and ελληνικά",
            "aaaaaaaaaaaaaaaaaaaaaaaa",
            "   ",
            "",
        ] {
            let want: Vec<u32> = reference
                .encode_fast(text, false)
                .unwrap()
                .get_ids()
                .to_vec();
            let got: Vec<u32> = pipe
                .encode(text, false)
                .wait()
                .unwrap()
                .remove(0)
                .iter()
                .map(|t| t.id)
                .collect();
            assert_eq!(want, got, "the fold changed the ids for {text:?}");
        }
    }
}
