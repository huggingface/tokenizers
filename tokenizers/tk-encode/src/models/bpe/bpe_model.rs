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
use crate::pipeline::{self, PipelineToken};
use crate::tokenizer::Result;
use crate::utils::byte_level::{self};
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
    /// Whether the model seeds on the 256 bytes rather than on characters.
    pub fn is_byte_level(&self) -> bool {
        matches!(self.atoms, Atoms::Bytes)
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
            ..
        } = model;
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
        let gate: u16 = self.byte_to_mode[sequence.as_bytes()[0] as usize];

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

    fn tokenize_pipeline(
        &self,
        sequence: &str,
        scratch: &mut Self::Scratch,
        output: &mut Vec<PipelineToken>,
    ) -> Result<()> {
        if sequence.is_empty() {
            return Ok(());
        }

        if self.ignore_merges
            && let Some(id) = self.vocab.get_bytes(sequence.as_bytes())
        {
            output.push(PipelineToken { id });
            return Ok(());
        }

        let BpeScratch {
            symbols,
            merge: merge_scratch,
            ..
        } = scratch;

        self.merge_word(sequence, symbols, merge_scratch);
        // the merge engines work in internal ids; `unmap` takes them back to the vocab's own ids
        output.extend(symbols.iter().map(|&symbol| PipelineToken {
            id: self.tables.unmap.at(symbol as usize),
        }));

        Ok(())
    }

    fn init_scratch(&self) -> Self::Scratch {
        Self::Scratch {
            symbols: Vec::with_capacity(64),
            merge: MergeScratch::default(),
        }
    }
}
