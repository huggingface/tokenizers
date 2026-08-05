//! v0 -> v1: a legacy `tokenizer.json` becomes a `.tok`.
//!
//! Conversion runs once, offline, on a machine that already has the JSON stack. This is the only
//! crate that names `Tokenizer` or a wrapper enum, and the only one that links serde — which is
//! the whole point: `tk-encode` reads `.tok` and can do neither.
//!
//! The pipeline is the validation. Building one is what the loader will have to do, so whatever it
//! accepts but the format cannot carry is reported by name rather than silently dropped: a
//! conversion either round-trips exactly or fails.

use tk_encode::pre_tokenizers::split::SplitPattern;
use tk_encode::tokenizer::pipeline::{PipelineModel, PipelinePreTokenizer, PipelineTokenizer};
use tk_encode::tokenizer::{ModelWrapper, NormalizerWrapper, Result, SplitDelimiterBehavior, Tokenizer};
use tk_serialization::{AddedEntry, Config, Entry, Writer, added_flag, behavior, flag, kind, pretok, strings};

/// Read a `tokenizer.json` and return the equivalent `.tok` v1 image.
pub fn convert_file(path: impl AsRef<std::path::Path>) -> Result<Vec<u8>> {
    let path = path.as_ref();
    let tokenizer = Tokenizer::from_file(path)?;
    to_tok(&tokenizer)
}


/// Serialise `tokenizer` as a `.tok` v1 image.
///
/// Whatever the pipeline accepts but the format does not carry is reported by name rather than
/// silently dropped, so a conversion either round-trips exactly or fails.
pub fn to_tok(tokenizer: &Tokenizer) -> Result<Vec<u8>> {

    // Building the pipeline is the validation: it is the thing that will have to load this
    // file, and it also reduces the post-processor to the two id lists the file stores.
    let pipeline = PipelineTokenizer::try_from(tokenizer)?;
    let normalizer = normalizer_strings(&pipeline, tokenizer)?;
    let ModelWrapper::BPE(bpe) = tokenizer.get_model() else {
        return Err(".tok v1 only carries BPE".into());
    };
    let (pretok_id, pretok_param, pretok_pattern) = pretokenizer_id(&pipeline)?;

    let mut flags = 0;
    if bpe.ignore_merges {
        flags |= flag::IGNORE_MERGES;
    }
    if bpe.byte_fallback {
        flags |= flag::BYTE_FALLBACK;
    }
    if bpe.fuse_unk {
        flags |= flag::FUSE_UNK;
    }
    if matches!(pipeline.get_model(), PipelineModel::BPE(m) if m.is_byte_level()) {
        flags |= flag::BYTE_LEVEL;
    }
    if tokenizer.get_added_vocabulary().get_encode_special_tokens() {
        flags |= flag::ENCODE_SPECIAL_TOKENS;
    }
    if let PipelinePreTokenizer::Split(split) = pipeline.get_pre_tokenizer()
        && split.invert
    {
        flags |= flag::PRETOK_INVERT;
    }

    // ── vocabulary ────────────────────────────────────────────────────────────────────────
    let mut vocab = bpe.vocab.get_vocab();
    // Sorted by id so the file is deterministic: the same tokenizer always converts to the
    // same bytes, which is what makes a checksum meaningful.
    vocab.sort_unstable_by_key(|(_, id)| *id);
    let mut slab = Vec::new();
    let mut entries = Vec::with_capacity(vocab.len());
    for (token, id) in &vocab {
        entries.push(Entry {
            start: slab.len() as u32,
            len: token.len() as u32,
            id: *id,
        });
        slab.extend_from_slice(token.as_bytes());
    }

    // ── merges, written in rank order so the rank is the index ────────────────────────────
    let mut ranked: Vec<(u32, (u32, u32))> = bpe
        .merges
        .iter()
        .map(|(&(left, right), &(rank, _))| (rank, (left, right)))
        .collect();
    ranked.sort_unstable();
    let mut pairs = Vec::with_capacity(ranked.len() * 2);
    for (_, (left, right)) in ranked {
        pairs.push(left);
        pairs.push(right);
    }

    // ── added tokens ──────────────────────────────────────────────────────────────────────
    let mut added: Vec<_> = tokenizer
        .get_added_vocabulary()
        .get_added_tokens_decoder()
        .into_iter()
        .collect();
    added.sort_unstable_by_key(|(id, _)| *id);
    let mut added_first = [0u64; 4];
    let mut added_slab = Vec::new();
    let mut added_entries = Vec::with_capacity(added.len());
    for (id, token) in &added {
        let bytes = token.content.as_bytes();
        let Some(&first) = bytes.first() else {
            return Err(".tok v1 has no empty added token".into());
        };
        added_first[(first >> 6) as usize] |= 1u64 << (first & 63);
        let mut token_flags = 0;
        if token.lstrip {
            token_flags |= added_flag::LSTRIP;
        }
        if token.rstrip {
            token_flags |= added_flag::RSTRIP;
        }
        if token.special {
            token_flags |= added_flag::SPECIAL;
        }
        if token.single_word {
            token_flags |= added_flag::SINGLE_WORD;
        }
        if token.normalized {
            token_flags |= added_flag::NORMALIZED;
        }
        added_entries.push(AddedEntry {
            start: added_slab.len() as u32,
            len: bytes.len() as u32,
            id: **id,
            flags: token_flags,
        });
        added_slab.extend_from_slice(bytes);
    }

    let mut model_strings = Vec::new();
    for value in [
        &bpe.unk_token,
        &bpe.continuing_subword_prefix,
        &bpe.end_of_word_suffix,
    ] {
        strings::push(&mut model_strings, value.as_deref().unwrap_or(""));
    }
    let mut normalizer_bytes = Vec::new();
    for part in &normalizer {
        strings::push(&mut normalizer_bytes, part);
    }
    let mut pretok_bytes = Vec::new();
    if let Some(pattern) = &pretok_pattern {
        strings::push(&mut pretok_bytes, pattern);
    }

    let config = Config {
        pretok: pretok_id,
        pretok_param,
        flags,
        _pad0: 0,
        added_first,
    };

    let mut w = Writer::new();
    w.push_one(kind::CONFIG, &config);
    w.push(kind::VOCAB_SLAB, &slab);
    w.push(kind::VOCAB_ENTRY, &entries);
    w.push(kind::MERGE_PAIRS, &pairs);
    w.push(kind::ADDED_SLAB, &added_slab);
    w.push(kind::ADDED_ENTRY, &added_entries);
    w.push(kind::POST_PREFIX, pipeline.get_post_processor().prefix_ids());
    w.push(kind::POST_SUFFIX, pipeline.get_post_processor().suffix_ids());
    w.push(kind::MODEL_STRINGS, &model_strings);
    w.push(kind::NORMALIZER, &normalizer_bytes);
    w.push(kind::PRETOK_STRINGS, &pretok_bytes);
    Ok(w.finish())
}

/// The normalizer as a string list, or empty when there is none. v1 carries a literal
/// `Replace` and nothing else — that covers the SentencePiece-derived configs, and every
/// other normalizer would drag a regex engine or a Unicode table into the read path.
fn normalizer_strings(
    pipeline: &PipelineTokenizer,
    tokenizer: &Tokenizer,
) -> Result<Vec<String>> {
    use tk_encode::normalizers::replace::ReplacePattern;

    if !pipeline.has_normalizer() {
        return Ok(Vec::new());
    }
    match tokenizer.get_normalizer() {
        Some(NormalizerWrapper::Replace(replace)) => match replace.pattern() {
            ReplacePattern::String(pattern) => Ok(vec![
                "replace".to_owned(),
                pattern.clone(),
                replace.content.clone(),
            ]),
            ReplacePattern::Regex(_) => {
                Err(".tok v1 has no regex `Replace` normalizer, only a literal one".into())
            }
        },
        other => Err(format!(".tok v1 has no normalizer for {other:?}").into()),
    }
}

/// Name the pre-tokenizer as a `(family, param)` pair. Recognising a regex is work the loader
/// should not have to redo, and storing the source would let a `.tok` demand a regex engine.
fn pretokenizer_id(pipeline: &PipelineTokenizer) -> Result<(u32, u32, Option<String>)> {
    use tk_encode::utils::GptFsm;

    // A byte-level tokenizer ships as `Sequence([Split(regex), ByteLevel])`, and the pipeline
    // converts that trailing byte-map member to `None` because it splits nothing. Look through
    // it so the sequence reduces to the one member that does.
    let pre_tokenizer = match pipeline.get_pre_tokenizer() {
        PipelinePreTokenizer::Sequence(seq) if !seq.is_deepseek() => {
            let mut splitting = seq
                .members()
                .iter()
                .filter(|m| !matches!(m, PipelinePreTokenizer::None));
            match (splitting.next(), splitting.next()) {
                (Some(only), None) => only,
                _ => pipeline.get_pre_tokenizer(),
            }
        }
        other => other,
    };

    match pre_tokenizer {
        PipelinePreTokenizer::None => Ok((pretok::NONE, 0, None)),
        PipelinePreTokenizer::Sequence(seq) if seq.is_deepseek() => {
            Ok((pretok::DEEPSEEK, 0, None))
        }
        PipelinePreTokenizer::Split(split) => match split.gpt_fsm() {
            Some(GptFsm::Gpt2) => Ok((pretok::BYTE_LEVEL, 0, None)),
            Some(GptFsm::O200k) => Ok((pretok::O200K, 0, None)),
            Some(GptFsm::Tekken) => Ok((pretok::TEKKEN, 0, None)),
            Some(GptFsm::Cl100k { digit_cap }) => Ok((
                pretok::CL100K,
                if digit_cap == usize::MAX {
                    u32::MAX
                } else {
                    digit_cap as u32
                },
                None,
            )),
            // A literal pattern is searched for directly, so it needs no engine either.
            None => match &split.pattern {
                SplitPattern::String(pattern) => Ok((
                    pretok::LITERAL,
                    write_behavior(split.behavior),
                    Some(pattern.clone()),
                )),
                SplitPattern::Regex(_) => Err(format!(
                    ".tok v1 has no pre-tokenizer for the pattern {:?}",
                    split.pattern
                )
                .into()),
            },
        },
        other => Err(format!(".tok v1 has no pre-tokenizer for {other:?}").into()),
    }
}

fn write_behavior(value: SplitDelimiterBehavior) -> u32 {
match value {
    SplitDelimiterBehavior::Removed => behavior::REMOVED,
    SplitDelimiterBehavior::Isolated => behavior::ISOLATED,
    SplitDelimiterBehavior::MergedWithPrevious => behavior::MERGED_WITH_PREVIOUS,
    SplitDelimiterBehavior::MergedWithNext => behavior::MERGED_WITH_NEXT,
    SplitDelimiterBehavior::Contiguous => behavior::CONTIGUOUS,
}
}

