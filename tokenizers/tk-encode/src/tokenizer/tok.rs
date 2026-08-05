//! Reading and writing the `.tok` v1 container — see the [`tk_serialization`] crate for the layout.
//!
//! The read half is what an inference build links, and it is deliberately dull: pull each section
//! out as a slice, hand the pieces to the same builders `Tokenizer::from_file` would have. Nothing
//! here can reach `serde_json`, which is the whole reason the format exists — a binary that cannot
//! parse JSON does not carry a JSON parser, worth 583 KB gzipped on this workspace.
//!
//! The write half is behind `tok-write` and belongs to `tk-convert`.

use ahash::AHashMap;

use tk_serialization::{
    AddedEntry, Config, Entry, Reader, added_flag, behavior, flag, kind, pretok, strings,
};

use crate::models::bpe::BPE;
use crate::pre_tokenizers::byte_level::ByteLevel;
use crate::pre_tokenizers::sequence::Sequence;
use crate::normalizers::replace::{Replace, ReplacePattern};
use crate::pre_tokenizers::split::{Split, SplitPattern};
use crate::tokenizer::pipeline::{PipelineModel, PipelinePostProcessor, PipelineTokenizer};
use crate::tokenizer::{
    AddedToken, ModelWrapper, NormalizerWrapper, PreTokenizerWrapper, Result,
    SplitDelimiterBehavior, Tokenizer,
};
use crate::utils::cl100k_pattern;

// ── read ───────────────────────────────────────────────────────────────────────────────────────

impl PipelineTokenizer {
    /// Build a pipeline from a `.tok` v1 image.
    ///
    /// `bytes` must be 8-byte aligned, which `tk_serialization::TokFile` and any `mmap` give you.
    pub fn from_tok(bytes: &[u8]) -> Result<Self> {
        let reader = Reader::new(bytes).map_err(|e| e.to_string())?;
        let config = reader.config;

        let model = read_model(&reader, config)?;
        let pre_tokenizer = read_pre_tokenizer(&reader, config)?;

        // Route through `Tokenizer` so added tokens get the same id assignment and the same
        // "already in the model vocabulary" reuse they get on the JSON path. This costs one
        // wrapper and keeps a second implementation of that rule from existing.
        let mut tokenizer = Tokenizer::new(ModelWrapper::BPE(model));
        tokenizer.with_pre_tokenizer(pre_tokenizer);
        tokenizer.with_normalizer(read_normalizer(&reader)?)?;
        let added = read_added_tokens(&reader)?;
        if !added.is_empty() {
            tokenizer.add_tokens(added)?;
        }

        let mut pipeline = Self::try_from(&tokenizer)?;
        pipeline
            .added_vocabulary
            .set_encode_special_tokens(config.flags & flag::ENCODE_SPECIAL_TOKENS != 0);
        // The post-processor reduces to two id lists, so that is what the file carries; there is
        // no `PostProcessorWrapper` to rebuild.
        pipeline.post_processor = PipelinePostProcessor::from_ids(
            reader.section::<u32>(kind::POST_PREFIX).map_err(|e| e.to_string())?,
            reader.section::<u32>(kind::POST_SUFFIX).map_err(|e| e.to_string())?,
        );
        Ok(pipeline)
    }
}

fn read_model(reader: &Reader<'_>, config: &Config) -> Result<BPE> {
    let slab: &[u8] = reader.require(kind::VOCAB_SLAB).map_err(|e| e.to_string())?;
    let entries: &[Entry] = reader.require(kind::VOCAB_ENTRY).map_err(|e| e.to_string())?;
    let pairs: &[u32] = reader.section(kind::MERGE_PAIRS).map_err(|e| e.to_string())?;
    if pairs.len() % 2 != 0 {
        return Err("corrupt .tok: MERGE_PAIRS holds an odd number of ids".into());
    }
    let [unk, prefix, suffix] = read_model_strings(reader)?;

    let token = |e: &Entry| -> Result<String> {
        let end = e.start as usize + e.len as usize;
        let bytes = slab
            .get(e.start as usize..end)
            .ok_or("corrupt .tok: vocabulary entry points outside the slab")?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| "corrupt .tok: vocabulary token is not valid UTF-8".into())
    };

    // `id_to_token` is only needed to name the merge operands, which the builder wants as strings.
    let mut vocab: AHashMap<String, u32> = AHashMap::with_capacity(entries.len());
    let mut by_id: Vec<&Entry> = Vec::new();
    for entry in entries {
        let text = token(entry)?;
        if entry.id as usize >= by_id.len() {
            by_id.resize(entry.id as usize + 1, entry);
        }
        by_id[entry.id as usize] = entry;
        vocab.insert(text, entry.id);
    }
    let name = |id: u32| -> Result<String> {
        let entry = by_id
            .get(id as usize)
            .ok_or("corrupt .tok: a merge names an id outside the vocabulary")?;
        if entry.id != id {
            return Err("corrupt .tok: a merge names an id with no vocabulary entry".into());
        }
        token(entry)
    };

    // Merges are stored in rank order, so a pair's rank is its index — nothing to sort.
    let mut merges = Vec::with_capacity(pairs.len() / 2);
    for pair in pairs.chunks_exact(2) {
        merges.push((name(pair[0])?, name(pair[1])?));
    }

    let mut builder = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .fuse_unk(config.flags & flag::FUSE_UNK != 0)
        .byte_fallback(config.flags & flag::BYTE_FALLBACK != 0)
        .ignore_merges(config.flags & flag::IGNORE_MERGES != 0);
    if let Some(unk) = unk {
        builder = builder.unk_token(unk);
    }
    if let Some(prefix) = prefix {
        builder = builder.continuing_subword_prefix(prefix);
    }
    if let Some(suffix) = suffix {
        builder = builder.end_of_word_suffix(suffix);
    }
    builder.build()
}

/// `MODEL_STRINGS` is three length-prefixed strings: unk, continuing prefix, end-of-word suffix.
/// Empty means absent, which is also what the JSON path treats an empty string as.
fn read_model_strings(reader: &Reader<'_>) -> Result<[Option<String>; 3]> {
    let raw: &[u8] = reader.section(kind::MODEL_STRINGS).map_err(|e| e.to_string())?;
    let mut out = [const { None }; 3];
    let mut at = 0usize;
    for slot in &mut out {
        if at == raw.len() {
            break;
        }
        let len_bytes = raw
            .get(at..at + 4)
            .ok_or("corrupt .tok: truncated MODEL_STRINGS length")?;
        let len = u32::from_le_bytes(len_bytes.try_into().unwrap()) as usize;
        at += 4;
        let bytes = raw
            .get(at..at + len)
            .ok_or("corrupt .tok: truncated MODEL_STRINGS value")?;
        at += len;
        if len > 0 {
            *slot = Some(
                String::from_utf8(bytes.to_vec())
                    .map_err(|_| "corrupt .tok: MODEL_STRINGS value is not valid UTF-8")?,
            );
        }
    }
    Ok(out)
}

/// The normalizer, if the file carries one. v1 knows a single literal `Replace`, which is what
/// SentencePiece-derived configs (the gemma family) use for their ` ` -> `U+2581` rewrite.
fn read_normalizer(reader: &Reader<'_>) -> Result<Option<NormalizerWrapper>> {
    let raw: &[u8] = reader.section(kind::NORMALIZER).map_err(|e| e.to_string())?;
    if raw.is_empty() {
        return Ok(None);
    }
    let parts = strings::parse(raw).ok_or("corrupt .tok: malformed NORMALIZER section")?;
    match parts.as_slice() {
        ["replace", pattern, content] => Ok(Some(NormalizerWrapper::Replace(Replace::new(
            ReplacePattern::String((*pattern).to_owned()),
            *content,
        )?))),
        [other, ..] => Err(format!("corrupt .tok: unknown normalizer kind `{other}`").into()),
        [] => Ok(None),
    }
}

fn read_added_tokens(reader: &Reader<'_>) -> Result<Vec<AddedToken>> {
    let slab: &[u8] = reader.section(kind::ADDED_SLAB).map_err(|e| e.to_string())?;
    let entries: &[AddedEntry] = reader.section(kind::ADDED_ENTRY).map_err(|e| e.to_string())?;
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        let end = entry.start as usize + entry.len as usize;
        let bytes = slab
            .get(entry.start as usize..end)
            .ok_or("corrupt .tok: added token points outside the slab")?;
        let content = std::str::from_utf8(bytes)
            .map_err(|_| "corrupt .tok: added token is not valid UTF-8")?;
        out.push(
            AddedToken::from(content, entry.flags & added_flag::SPECIAL != 0)
                .single_word(entry.flags & added_flag::SINGLE_WORD != 0)
                .lstrip(entry.flags & added_flag::LSTRIP != 0)
                .rstrip(entry.flags & added_flag::RSTRIP != 0)
                .normalized(entry.flags & added_flag::NORMALIZED != 0),
        );
    }
    Ok(out)
}

/// Spell the pre-tokenizer back out from its family id. The file names the FSM rather than
/// carrying a regex, so a `.tok` never needs a regex engine to load: every pattern produced here
/// is one `gpt_fsm` recognises, and `Split::new` falls back to the native FSM without a backend.
fn read_pre_tokenizer(reader: &Reader<'_>, config: &Config) -> Result<Option<PreTokenizerWrapper>> {
    let split = |pattern: String| -> Result<PreTokenizerWrapper> {
        Ok(PreTokenizerWrapper::Split(Split::new(
            SplitPattern::Regex(pattern),
            SplitDelimiterBehavior::Isolated,
            false,
        )?))
    };
    // A trailing `ByteLevel` with `use_regex: false` is the byte-map half: it does no splitting,
    // it tells the pipeline the model seeds on bytes.
    let byte_map = PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, true, false));
    let with_byte_map = |mut parts: Vec<PreTokenizerWrapper>| -> Option<PreTokenizerWrapper> {
        if config.flags & flag::BYTE_LEVEL != 0 {
            parts.push(byte_map);
        }
        match parts.len() {
            0 => None,
            1 => parts.pop(),
            _ => Some(PreTokenizerWrapper::Sequence(Sequence::new(parts))),
        }
    };

    Ok(match config.pretok {
        // GPT-2 ships as a single `ByteLevel` that both splits and byte-maps.
        pretok::BYTE_LEVEL => Some(PreTokenizerWrapper::ByteLevel(ByteLevel::new(
            false, true, true,
        ))),
        pretok::CL100K => with_byte_map(vec![split(cl100k_pattern(match config.pretok_param {
            u32::MAX => usize::MAX,
            cap => cap as usize,
        }))?]),
        pretok::O200K => with_byte_map(vec![split(atomsplit::regexes::O200K.to_owned())?]),
        pretok::TEKKEN => with_byte_map(vec![split(atomsplit::regexes::TEKKEN.to_owned())?]),
        pretok::DEEPSEEK => with_byte_map(
            atomsplit::regexes::DEEPSEEK
                .iter()
                .map(|r| split((*r).to_owned()))
                .collect::<Result<Vec<_>>>()?,
        ),
        pretok::LITERAL => {
            let raw: &[u8] = reader.section(kind::PRETOK_STRINGS).map_err(|e| e.to_string())?;
            let [pattern] = strings::parse(raw)
                .ok_or("corrupt .tok: malformed PRETOK_STRINGS section")?[..]
            else {
                return Err("corrupt .tok: a literal split needs exactly one pattern".into());
            };
            with_byte_map(vec![PreTokenizerWrapper::Split(Split::new(
                SplitPattern::String(pattern.to_owned()),
                read_behavior(config.pretok_param)?,
                config.flags & flag::PRETOK_INVERT != 0,
            )?)])
        }
        pretok::NONE => with_byte_map(Vec::new()),
        other => return Err(format!("corrupt .tok: unknown pre-tokenizer id {other}").into()),
    })
}

fn read_behavior(value: u32) -> Result<SplitDelimiterBehavior> {
    Ok(match value {
        behavior::REMOVED => SplitDelimiterBehavior::Removed,
        behavior::ISOLATED => SplitDelimiterBehavior::Isolated,
        behavior::MERGED_WITH_PREVIOUS => SplitDelimiterBehavior::MergedWithPrevious,
        behavior::MERGED_WITH_NEXT => SplitDelimiterBehavior::MergedWithNext,
        behavior::CONTIGUOUS => SplitDelimiterBehavior::Contiguous,
        other => return Err(format!("corrupt .tok: unknown split behaviour {other}").into()),
    })
}

// ── write ──────────────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "tok-write")]
mod write {
    use super::*;
    use tk_serialization::Writer;

    /// Serialise `tokenizer` as a `.tok` v1 image.
    ///
    /// Whatever the pipeline accepts but the format does not carry is reported by name rather than
    /// silently dropped, so a conversion either round-trips exactly or fails.
    pub fn to_tok(tokenizer: &Tokenizer) -> Result<Vec<u8>> {
        use crate::tokenizer::pipeline::PipelinePreTokenizer;

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
        w.push(kind::POST_PREFIX, pipeline.post_processor.prefix_ids());
        w.push(kind::POST_SUFFIX, pipeline.post_processor.suffix_ids());
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
        use crate::normalizers::replace::ReplacePattern;

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
        use crate::tokenizer::pipeline::PipelinePreTokenizer;
        use crate::utils::GptFsm;

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
}

#[cfg(feature = "tok-write")]
fn write_behavior(value: SplitDelimiterBehavior) -> u32 {
    match value {
        SplitDelimiterBehavior::Removed => behavior::REMOVED,
        SplitDelimiterBehavior::Isolated => behavior::ISOLATED,
        SplitDelimiterBehavior::MergedWithPrevious => behavior::MERGED_WITH_PREVIOUS,
        SplitDelimiterBehavior::MergedWithNext => behavior::MERGED_WITH_NEXT,
        SplitDelimiterBehavior::Contiguous => behavior::CONTIGUOUS,
    }
}

#[cfg(feature = "tok-write")]
pub use write::to_tok;
