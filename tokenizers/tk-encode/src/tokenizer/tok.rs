//! Reading the `.tok` v1 container — see the [`tk_serialization`] crate for the layout.
//!
//! This is how a v1 build constructs a pipeline, and it is deliberately dull: pull each section
//! out as a slice and hand the pieces to the builders. Nothing here can reach a JSON parser, a
//! wrapper enum, or serde, which is the whole point — the v0 `tokenizer.json` reader lives in
//! `tk-convert` behind the `config` feature, along with the writer that produced this file.

use ahash::AHashMap;

use tk_serialization::{
    AddedEntry, Config, Entry, Reader, added_flag, behavior, flag, kind, model, pretok, strings,
};

use crate::models::bpe::{BPE, PipelineBPE};
use crate::models::unigram::Unigram;
use crate::models::wordlevel::WordLevel;
use crate::models::wordpiece::{PipelineWordPiece, WordPiece};
use crate::normalizers::replace::{Replace, ReplacePattern};
use crate::pre_tokenizers::sequence::PipelineSequence;
use crate::pre_tokenizers::split::{Split, SplitPattern};
use crate::tokenizer::pipeline::{
    PipelineModel, PipelineNormalizer, PipelinePostProcessor, PipelinePreTokenizer,
    PipelineTokenizer,
};
use crate::tokenizer::{Result, SplitDelimiterBehavior};
use crate::utils::{DEEPSEEK_PATTERNS, cl100k_pattern};
use crate::vocab::bucket_added_vocabulary::{AddedToken, AddedVocabulary as BucketAddedVocabulary};

// ── read ───────────────────────────────────────────────────────────────────────────────────────

impl PipelineTokenizer {
    /// Build a pipeline from a `.tok` v1 image.
    ///
    /// `bytes` must be 8-byte aligned, which `tk_serialization::TokFile` and any `mmap` give you.
    pub fn from_tok(bytes: &[u8]) -> Result<Self> {
        let reader = Reader::new(bytes).map_err(|e| e.to_string())?;
        let config = reader.config;

        let vocab = Vocabulary::read(&reader)?;
        let normalizer = read_normalizer(&reader)?;

        // Added tokens are written in id order, and `add_tokens` reuses a model id when the token
        // is already in the vocabulary, so replaying them in order reproduces the JSON path's
        // assignment. The model is passed as a concrete `BPE` and the normalizer as a concrete
        // `Replace`: routing either through its wrapper enum would make every other variant
        // reachable, which is most of what this format exists to avoid.
        let added = read_added_tokens(&reader)?;
        let mut added_vocabulary = BucketAddedVocabulary::new();
        // `add_tokens` needs the model only to ask whether a token is already in the vocabulary,
        // so it gets the pre-pipeline form and a concrete normalizer — never a wrapper.
        let model = match read_model(&reader, config, vocab)? {
            Built::Bpe(bpe) => {
                added_vocabulary.add_tokens(added, &bpe, normalizer.as_ref())?;
                PipelineModel::BPE(PipelineBPE::from_bpe(
                    bpe,
                    config.flags & flag::BYTE_LEVEL != 0,
                )?)
            }
            Built::Unigram(unigram) => {
                added_vocabulary.add_tokens(added, &unigram, normalizer.as_ref())?;
                PipelineModel::Unigram(unigram)
            }
            Built::WordPiece(wordpiece) => {
                added_vocabulary.add_tokens(added, &wordpiece, normalizer.as_ref())?;
                PipelineModel::WordPiece(wordpiece.try_into()?)
            }
            Built::WordLevel(wordlevel) => {
                added_vocabulary.add_tokens(added, &wordlevel, normalizer.as_ref())?;
                PipelineModel::WordLevel(wordlevel)
            }
        };
        added_vocabulary
            .set_encode_special_tokens(config.flags & flag::ENCODE_SPECIAL_TOKENS != 0);

        Ok(Self {
            added_vocabulary,
            normalizers: normalizer.map(PipelineNormalizer::Replace).into_iter().collect(),
            pre_tokenizer: read_pre_tokenizer(&reader, config)?,
            model,
            post_processor: PipelinePostProcessor::from_ids(
                reader.section::<u32>(kind::POST_PREFIX).map_err(|e| e.to_string())?,
                reader.section::<u32>(kind::POST_SUFFIX).map_err(|e| e.to_string())?,
            ),
        })
    }
}

/// The vocabulary as the file stores it: a byte slab plus one entry per token, and — for Unigram —
/// one score each. Decoded once and shared by all four model builders.
struct Vocabulary {
    /// `(token, id)` in the file's order, which is id order.
    tokens: Vec<(String, u32)>,
    /// Parallel to `tokens`; empty unless the model is Unigram.
    scores: Vec<f64>,
}

impl Vocabulary {
    fn read(reader: &Reader<'_>) -> Result<Self> {
        let slab: &[u8] = reader.require(kind::VOCAB_SLAB).map_err(|e| e.to_string())?;
        let entries: &[Entry] = reader.require(kind::VOCAB_ENTRY).map_err(|e| e.to_string())?;
        let scores: &[f64] = reader.section(kind::VOCAB_SCORES).map_err(|e| e.to_string())?;
        if !scores.is_empty() && scores.len() != entries.len() {
            return Err("corrupt .tok: VOCAB_SCORES and VOCAB_ENTRY disagree in length".into());
        }

        let mut tokens = Vec::with_capacity(entries.len());
        for entry in entries {
            let end = entry.start as usize + entry.len as usize;
            let bytes = slab
                .get(entry.start as usize..end)
                .ok_or("corrupt .tok: vocabulary entry points outside the slab")?;
            let text = std::str::from_utf8(bytes)
                .map_err(|_| "corrupt .tok: vocabulary token is not valid UTF-8")?;
            tokens.push((text.to_owned(), entry.id));
        }
        Ok(Self {
            tokens,
            scores: scores.to_vec(),
        })
    }

    fn map(&self) -> AHashMap<String, u32> {
        self.tokens.iter().cloned().collect()
    }

    /// `id -> token`, for naming merge operands. Sparse ids leave `None` holes.
    fn by_id(&self) -> Vec<Option<&str>> {
        let max = self.tokens.iter().map(|(_, id)| *id).max().unwrap_or(0);
        let mut out = vec![None; max as usize + 1];
        for (text, id) in &self.tokens {
            out[*id as usize] = Some(text.as_str());
        }
        out
    }
}

/// A model in its pre-pipeline form. `add_tokens` wants one of these to ask whether an added token
/// is already in the vocabulary, so the dispatch happens before the pipeline conversion.
enum Built {
    Bpe(BPE),
    Unigram(Unigram),
    WordPiece(WordPiece),
    WordLevel(WordLevel),
}

fn read_model(reader: &Reader<'_>, config: &Config, vocab: Vocabulary) -> Result<Built> {
    let [unk, prefix, suffix] = read_model_strings(reader)?;
    match config.model {
        model::BPE => {
            let pairs: &[u32] = reader.section(kind::MERGE_PAIRS).map_err(|e| e.to_string())?;
            if pairs.len() % 2 != 0 {
                return Err("corrupt .tok: MERGE_PAIRS holds an odd number of ids".into());
            }
            let by_id = vocab.by_id();
            let name = |id: u32| -> Result<String> {
                by_id
                    .get(id as usize)
                    .copied()
                    .flatten()
                    .map(str::to_owned)
                    .ok_or_else(|| "corrupt .tok: a merge names an id with no vocabulary entry".into())
            };
            // Merges are stored in rank order, so a pair's rank is its index — nothing to sort.
            let mut merges = Vec::with_capacity(pairs.len() / 2);
            for pair in pairs.chunks_exact(2) {
                merges.push((name(pair[0])?, name(pair[1])?));
            }

            let mut builder = BPE::builder()
                .vocab_and_merges(vocab.map(), merges)
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
            Ok(Built::Bpe(builder.build()?))
        }
        model::UNIGRAM => {
            if vocab.scores.len() != vocab.tokens.len() {
                return Err("corrupt .tok: a Unigram model needs one score per token".into());
            }
            // Unigram's vocabulary is positional: a piece's index *is* its id, which is why the
            // writer refuses a sparse one.
            let pieces: Vec<(String, f64)> = vocab
                .tokens
                .iter()
                .map(|(text, _)| text.clone())
                .zip(vocab.scores.iter().copied())
                .collect();
            let unk_id = match config.model_param {
                u32::MAX => None,
                id => Some(id as usize),
            };
            Ok(Built::Unigram(Unigram::from(
                pieces,
                unk_id,
                config.flags & flag::BYTE_FALLBACK != 0,
            )?))
        }
        model::WORDPIECE => {
            let mut builder = WordPiece::builder()
                .vocab(vocab.map())
                .max_input_chars_per_word(config.model_param as usize);
            if let Some(unk) = unk {
                builder = builder.unk_token(unk);
            }
            if let Some(prefix) = prefix {
                builder = builder.continuing_subword_prefix(prefix);
            }
            Ok(Built::WordPiece(builder.build()?))
        }
        model::WORDLEVEL => {
            let mut builder = WordLevel::builder().vocab(vocab.map());
            if let Some(unk) = unk {
                builder = builder.unk_token(unk);
            }
            Ok(Built::WordLevel(builder.build()?))
        }
        other => Err(format!("corrupt .tok: unknown model id {other}").into()),
    }
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
fn read_normalizer(reader: &Reader<'_>) -> Result<Option<Replace>> {
    let raw: &[u8] = reader.section(kind::NORMALIZER).map_err(|e| e.to_string())?;
    if raw.is_empty() {
        return Ok(None);
    }
    let parts = strings::parse(raw).ok_or("corrupt .tok: malformed NORMALIZER section")?;
    match parts.as_slice() {
        ["replace", pattern, content] => Ok(Some(Replace::new(
            ReplacePattern::String((*pattern).to_owned()),
            *content,
        )?)),
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

/// Spell the pre-tokenizer back out from its family id.
///
/// The file names the FSM family rather than carrying a regex, so loading a `.tok` never needs a
/// regex engine: every pattern produced here is one `gpt_fsm` recognises and drives natively, and
/// a literal pattern is searched for directly. This builds `PipelinePreTokenizer` rather than the
/// config-level `PreTokenizerWrapper` — the wrapper holds every pre-tokenizer variant, so touching
/// it would link all of them.
fn read_pre_tokenizer(reader: &Reader<'_>, config: &Config) -> Result<PipelinePreTokenizer> {
    let regex = |pattern: &str| -> Result<PipelinePreTokenizer> {
        Ok(PipelinePreTokenizer::Split(Split::native(
            SplitPattern::Regex(pattern.to_owned()),
            SplitDelimiterBehavior::Isolated,
            false,
        )?))
    };

    Ok(match config.pretok {
        // The byte-map half of a byte-level pre-tokenizer splits nothing, so it does not appear
        // here at all — `Config::flags` carries it as `BYTE_LEVEL` and the model reads it there.
        pretok::BYTE_LEVEL => regex(atomsplit::regexes::GPT2)?,
        pretok::CL100K => regex(&cl100k_pattern(match config.pretok_param {
            u32::MAX => usize::MAX,
            cap => cap as usize,
        }))?,
        pretok::O200K => regex(atomsplit::regexes::O200K)?,
        pretok::TEKKEN => regex(atomsplit::regexes::TEKKEN)?,
        // The three deepseek regexes as a sequence, which the pipeline recognises and runs as one
        // native pass.
        pretok::DEEPSEEK => PipelinePreTokenizer::Sequence(PipelineSequence::new(
            DEEPSEEK_PATTERNS
                .iter()
                .map(|r| regex(r))
                .collect::<Result<Vec<_>>>()?,
        )),
        pretok::LITERAL => {
            let raw: &[u8] = reader.section(kind::PRETOK_STRINGS).map_err(|e| e.to_string())?;
            let [pattern] = strings::parse(raw)
                .ok_or("corrupt .tok: malformed PRETOK_STRINGS section")?[..]
            else {
                return Err("corrupt .tok: a literal split needs exactly one pattern".into());
            };
            PipelinePreTokenizer::Split(Split::native(
                SplitPattern::String(pattern.to_owned()),
                read_behavior(config.pretok_param)?,
                config.flags & flag::PRETOK_INVERT != 0,
            )?)
        }
        pretok::NONE => PipelinePreTokenizer::None,
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
