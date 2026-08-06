//! The pipeline BPE model: its tables, how it is built from a [`BPE`], and how a pretokenized
//! sequence is turned into tokens. The merge engines themselves live in `bpe_pretoken_to_rank`, `merge_multipass`
//! and `merge_hot_cold_queue`.
use crate::models::bpe::bpe_build_tables::BpeTables;
use crate::models::bpe::bpe_scratch::BpeScratch;
use crate::models::bpe::legacy_model::BPE;
use crate::models::bpe::merge_hot_cold_queue::{
    MergeScratch, build_byte_to_gate, two_tier_queue_merge,
};
use crate::models::bpe::merge_multipass::merge_multipass;
use crate::models::bpe::{Error, bpe_build_tables::At};
use crate::pipeline::{self, PipelineToken, Span};
use crate::tokenizer::Result;
use crate::utils::byte_level::{self};
use crate::utils::word_cache::{Lookup, MAX_INLINE_IDS, ProbeEmit, WordCache};
use crate::vocab::bucket_vocab_store::BucketVocabStore;

/// Set only for the few models that decorate their atoms: `end_of_word_suffix` (CLIP, openai-gpt,
/// XLM) and `continuing_subword_prefix`. A character's atom then depends on its position in the
/// word, so those models take a slow path that looks each decorated character up in the vocab.
pub(super) struct Affixes {
    pub(super) prefix: String,
    pub(super) suffix: String,
    /// Dense `external vocab id -> internal symbol id`, `u32::MAX` where there is none. Dense
    /// beats a hash here because external ids are `0..vocab_size`: 4 bytes a slot and one load,
    /// against 8-16 for any map. It is the array `BpeTables::build` makes anyway.
    pub(super) to_internal: Box<[u32]>,
}

/// Longest `prefix + one character + suffix` the stack buffer holds.
pub(super) const AFFIX_BUF: usize = 64;

pub struct PipelineBPE {
    pub(super) atoms: Atoms,
    pub(super) tables: BpeTables,
    pub(super) affixes: Option<Affixes>,
    pub(super) vocab: BucketVocabStore,
    ignore_merges: bool,
    byte_to_mode: [u16; 256],
    /// `None` disables the per-thread word cache; the builder's `0` means the same thing.
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
        Ok(Self {
            atoms,
            tables,
            affixes,
            ignore_merges,
            vocab,
            byte_to_mode: build_byte_to_gate(),
            cache_capacity,
        })
    }

    /// Converts a word to symbols and merges it. The gate, indexed by the word's first byte, says
    /// which engine gets it: short words go to multipass, longer ones to the two-tier queue.
    /// `to_merge` is the caller's reusable symbol buffer -- it lives in the scratch so that a word
    /// costs no allocation. On return it holds the merged word as internal ids, which the caller
    /// maps to external ids through `unmap`.
    pub(super) fn merge_word(
        &self,
        sequence: &str,
        symbols: &mut Vec<u32>,
        merge_scratch: &mut MergeScratch,
    ) {
        // Index the gate by the first *content* byte, not the first byte of the
        // word. A ByteLevel pre-tokenizer writes a leading space as `Ġ`
        // (C4 A0) and Metaspace writes `▁` (E2 96 81). Both are >= 0x80, so
        // indexing the raw first byte handed every space-prefixed word the
        // multibyte gate -- the one meant for CJK, at 8 bytes -- and sent any
        // English word longer than that to the two-tier queue, which is the
        // wrong engine for a short word.
        //
        // `build_byte_to_gate` already makes this argument for the ASCII space
        // and maps " \t\n\r" to the multibyte gate, precisely because a
        // leading delimiter says nothing about the script of the rest. The
        // conclusion was right and the remedy was backwards: what follows the
        // delimiter is what should choose, so look past it.
        let bytes = sequence.as_bytes();
        let first = match bytes {
            [0xC4, 0xA0, rest @ ..] if !rest.is_empty() => rest[0],
            [0xE2, 0x96, 0x81, rest @ ..] if !rest.is_empty() => rest[0],
            [ws, rest @ ..] if ws.is_ascii_whitespace() && !rest.is_empty() => rest[0],
            _ => bytes[0],
        };
        let gate: u16 = self.byte_to_mode[first as usize];

        if sequence.len() > gate as usize {
            // conversion writes the entries and cold keys directly: no intermediate rank array
            self.convert_queue(
                sequence,
                symbols,
                &mut merge_scratch.entries,
                &mut merge_scratch.cold,
            );
            two_tier_queue_merge(&self.tables, symbols, merge_scratch);
        } else {
            let first_merge = self.convert_multipass(sequence, symbols);
            merge_multipass(&self.tables, symbols, first_merge);
        }
    }
}

impl pipeline::Model for PipelineBPE {
    type Scratch = BpeScratch;

    /// Every pre-token of a chunk in one call.
    ///
    /// Same work per word as [`Self::tokenize_pipeline`]; what changes is what is *not* repeated.
    /// The scratch is destructured once instead of once per word, the output is grown once for
    /// the whole batch instead of being capacity-checked on every push, and the virtual call,
    /// the slice and the `Result` happen once per chunk rather than once per word.
    fn tokenize_spans(
        &self,
        chunk: &str,
        spans: &[Span],
        scratch: &mut Self::Scratch,
        output: &mut Vec<PipelineToken>,
    ) -> Result<()> {
        let BpeScratch {
            symbols,
            merge: merge_scratch,
            word_cache,
        } = scratch;
        // 92% of English pre-tokens are one id and 98% are at most two, so reserve for two apiece
        // and emit a cache hit by writing straight at a running cursor: `extend` would re-check
        // capacity and re-read the length for every word.
        output.reserve(2 * spans.len() + MAX_INLINE_IDS);
        let mut capacity = output.capacity();
        let mut cursor = output.len();
        for span in spans {
            // SAFETY: the spans come from the pre-tokenizer, which only ever cuts on char
            // boundaries -- so the boundary check `&chunk[..]` would run is already satisfied, and
            // it is not free when it runs once per word.
            let sequence = unsafe { chunk.get_unchecked(span.range()) };
            if sequence.is_empty() {
                continue;
            }
            let mut placement = None;
            if let Some(cache) = word_cache.as_mut() {
                // The probe needs somewhere to put the ids before it knows how many there are, so
                // make the room first: after this, `MAX_INLINE_IDS` writes past the cursor are
                // always inside the allocation.
                if cursor + MAX_INLINE_IDS > capacity {
                    // SAFETY: `cursor` counts what has been written so far.
                    unsafe { output.set_len(cursor) };
                    output.reserve(spans.len() + MAX_INLINE_IDS);
                    capacity = output.capacity();
                }
                // The probe writes the ids at the cursor itself, so a hit is one load of the slot
                // and one unconditional store of its lanes -- the ids never become a slice and the
                // line is never read twice.
                // SAFETY: the check above leaves `MAX_INLINE_IDS` slots past `cursor`.
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
                    // An arena-backed hit: it does not fit in the slot's lanes, but the probe
                    // already found the ids, so copy those rather than probing a second time.
                    ProbeEmit::Hit(ids) => {
                        // SAFETY: `cursor` counts what has been written so far.
                        unsafe { output.set_len(cursor) };
                        output.extend(ids.iter().map(|&id| PipelineToken { id }));
                        cursor = output.len();
                        capacity = output.capacity();
                        continue;
                    }
                    ProbeEmit::Miss(at) => placement = at,
                }
            }
            // SAFETY: `cursor` counts what the fast path wrote; the slow paths below use `output`
            // through its normal API, so its length has to be true again first.
            unsafe { output.set_len(cursor) };
            let start = output.len();
            if self.ignore_merges
                && let Some(id) = self.vocab.get_bytes(sequence.as_bytes())
            {
                output.push(PipelineToken { id });
            } else {
                self.merge_word(sequence, symbols, merge_scratch);
                // the merge engines work in internal ids; `unmap` takes them back to the vocab's
                // own ids
                output.extend(symbols.iter().map(|&symbol| PipelineToken {
                    id: self.tables.unmap.at(symbol as usize),
                }));
            }
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

    fn tokenize_pipeline(
        &self,
        sequence: &str,
        scratch: &mut Self::Scratch,
        output: &mut Vec<PipelineToken>,
    ) -> Result<()> {
        if sequence.is_empty() {
            return Ok(());
        }

        let BpeScratch {
            symbols,
            merge: merge_scratch,
            word_cache,
        } = scratch;

        let mut placement = None;
        if let Some(cache) = word_cache.as_mut() {
            match cache.lookup(sequence.as_bytes()) {
                Lookup::Hit(ids) => {
                    output.extend(ids.iter().map(|&id| PipelineToken { id }));
                    return Ok(());
                }
                Lookup::Miss(at) => placement = at,
            }
        }
        let start = output.len();
        if self.ignore_merges
            && let Some(id) = self.vocab.get_bytes(sequence.as_bytes())
        {
            output.push(PipelineToken { id });
        } else {
            self.merge_word(sequence, symbols, merge_scratch);
            // the merge engines work in internal ids; `unmap` takes them back to the vocab's own ids
            output.extend(symbols.iter().map(|&symbol| PipelineToken {
                id: self.tables.unmap.at(symbol as usize),
            }));
        }
        // The ids come back out of `output` because that is the only place both
        // branches above leave them: `ignore_merges` never touches `symbols`.
        if let Some(cache) = word_cache.as_mut()
            && let Some(at) = placement
        {
            cache.insert(at, output[start..].iter().map(|token| token.id));
        }

        Ok(())
    }

    fn init_scratch(&self) -> Self::Scratch {
        Self::Scratch {
            symbols: Vec::with_capacity(64),
            merge: MergeScratch::default(),
            word_cache: self.cache_capacity.map(WordCache::new),
        }
    }
}

#[cfg(test)]
mod gate_tests {
    use crate::pipeline::PipelineTokenizer;
    use crate::Tokenizer;

    /// The gate picks which merge engine handles a word; both must produce the
    /// same ids, so a change to the gate is only ever a performance change.
    ///
    /// This is the case that made the gate wrong: with a ByteLevel
    /// pre-tokenizer the leading space becomes `Ġ` (C4 A0), so the raw first
    /// byte is >= 0x80 and every space-prefixed word took the multibyte gate.
    /// The words below straddle that 8-byte boundary in both directions, so
    /// they exercise both engines, and any future change to the gate that also
    /// changes an id fails here rather than in a benchmark.
    #[test]
    fn the_gate_never_changes_the_ids() {
        let reference = Tokenizer::from_file("../data/gpt2.json").unwrap();
        let pipe = PipelineTokenizer::try_from(&reference).unwrap();

        // Short and long, space-prefixed and not, plus scripts whose own first
        // byte is genuinely multibyte.
        let cases = [
            " a", " the", " short", " straddle", " understanding",
            " internationalisation", "unprefixed", " ", "  ", "\tindented",
            " 语言模型", " ελληνικά", " naïve café", " 🎉 emoji",
            "mixed ascii and 中文 in one word",
        ];
        for text in cases {
            let want: Vec<u32> = reference
                .encode_fast(text, false)
                .unwrap()
                .get_ids()
                .to_vec();
            let got: Vec<u32> = pipe
                .encode_one(text, false)
                .unwrap()
                .iter()
                .map(|t| t.id)
                .collect();
            assert_eq!(want, got, "gate changed the ids for {text:?}");
        }
    }
}
