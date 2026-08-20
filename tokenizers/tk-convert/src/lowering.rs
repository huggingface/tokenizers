//! Lowering a config into `tk-encode`'s runtime pipeline.
//!
//! One direction only, and one entry point: `PipelineTokenizer::try_from(&Tokenizer)`. Each stage
//! has its own `TryFrom` next to it, so the rejections stay where the reader will look for them.
//!
//! The lowering is deliberately *not* a mirror of `tk-encode`'s slim JSON reader. They meet at
//! [`PipelineTokenizer::from_parts`], which is the only constructor of the pipeline's inner state;
//! everything above that door is per-reader, everything below it is shared. Two copies of the door
//! is how the two readers drift, and `json_oracle` is what catches it when they do.
//!
//! ## Where the wrapper enums are *not* named
//!
//! `PipelineNormalizer` has one variant per concrete normalizer rather than a single
//! `NormalizerWrapper` one, because a match arm on a wrapper makes every variant of it reachable —
//! and that reachability is most of what the split exists to remove. So a declared normalizer is
//! flattened here, member by member, exactly as the slim reader flattens a config `Sequence`.

use std::convert::{TryFrom, TryInto};

use tk_encode::normalizers::metaspace::MetaspaceNormalizer;
use tk_encode::pipeline::{
    NormalizerChain, PipelineModel, PipelineNormalizer, PipelinePostProcessor,
    PipelinePreTokenizer, PipelineToken, PipelineTokenizer, Seq, Slice, Template, build_slices,
    compose,
};
use tk_encode::pre_tokenizers::metaspace::normalizer_and_split;
use tk_encode::pre_tokenizers::split::{Split, SplitPattern};
use tk_encode::processors::bert::BertProcessing;
use tk_encode::processors::roberta::RobertaProcessing;
use tk_encode::utils::byte_level::GPT2_REGEX_STR;
use tk_encode::vocab::bucket_added_vocabulary::{
    AddedToken as BucketAddedToken, AddedVocabulary as BucketAddedVocabulary,
};
use tk_encode::{DecoderRuntime, Result, SplitDelimiterBehavior};

use crate::models::ModelWrapper;
use crate::normalizers::NormalizerWrapper;
use crate::pre_tokenizers::PreTokenizerWrapper;
use crate::processors::PostProcessorWrapper;
use crate::tokenizer::Tokenizer;

/// The same decomposition of a `Metaspace` pre-tokenizer the slim reader performs, driven off the
/// wrapper: a `Metaspace` on its own, or a `WhitespaceSplit` followed by one (t5 and albert ship
/// that shape). `None` for anything else, and for a `Metaspace` whose settings
/// [`normalizer_and_split`] cannot reproduce — the caller then converts the pre-tokenizer the usual
/// way and rejects a residual `Metaspace`, leaving the model out of the pipeline instead of quietly
/// encoding it differently.
pub fn to_normalizer_and_split(
    pre_tokenizer: Option<&PreTokenizerWrapper>,
) -> Option<(MetaspaceNormalizer, Split)> {
    match pre_tokenizer {
        Some(PreTokenizerWrapper::Metaspace(metaspace)) => normalizer_and_split(metaspace, false),
        Some(PreTokenizerWrapper::Sequence(sequence)) => match sequence.as_ref() {
            [
                PreTokenizerWrapper::WhitespaceSplit(_),
                PreTokenizerWrapper::Metaspace(metaspace),
            ] => normalizer_and_split(metaspace, true),
            _ => None,
        },
        _ => None,
    }
}

/// Flatten a declared normalizer into the pipeline's per-normalizer variants, appending to `out`.
///
/// A config `Sequence` is flattened rather than kept, which is what makes a `NormalizerChain` over
/// the result equivalent to the wrapper's own `Normalizer::normalize` — and that equivalence is what
/// keeps a `normalized: true` added token's id where the reference path put it. Nested `Sequence`s
/// flatten too, since running a nested sequence in order is the same as running its members in
/// order.
fn lower_normalizer(declared: &NormalizerWrapper, out: &mut Vec<PipelineNormalizer>) {
    match declared {
        NormalizerWrapper::Sequence(seq) => {
            for member in seq.as_ref() {
                lower_normalizer(member, out);
            }
        }
        NormalizerWrapper::StripNormalizer(n) => out.push(PipelineNormalizer::Strip(*n)),
        NormalizerWrapper::Lowercase(n) => out.push(PipelineNormalizer::Lowercase(*n)),
        NormalizerWrapper::Replace(n) => out.push(PipelineNormalizer::Replace(n.clone())),
        NormalizerWrapper::Prepend(n) => out.push(PipelineNormalizer::Prepend(n.clone())),
        NormalizerWrapper::ByteLevel(n) => out.push(PipelineNormalizer::ByteLevel(n.clone())),
        NormalizerWrapper::BertNormalizer(n) => out.push(PipelineNormalizer::Bert(*n)),
        NormalizerWrapper::StripAccents(n) => out.push(PipelineNormalizer::StripAccents(*n)),
        NormalizerWrapper::NFC(n) => out.push(PipelineNormalizer::NFC(*n)),
        NormalizerWrapper::NFD(n) => out.push(PipelineNormalizer::NFD(*n)),
        NormalizerWrapper::NFKC(n) => out.push(PipelineNormalizer::NFKC(*n)),
        NormalizerWrapper::NFKD(n) => out.push(PipelineNormalizer::NFKD(*n)),
        NormalizerWrapper::Nmt(n) => out.push(PipelineNormalizer::Nmt(*n)),
        NormalizerWrapper::Precompiled(n) => out.push(PipelineNormalizer::Precompiled(n.clone())),
    }
}

impl TryFrom<PreTokenizerWrapper> for PipelinePreTokenizer {
    type Error = tk_encode::Error;

    fn try_from(value: PreTokenizerWrapper) -> Result<Self> {
        match value {
            PreTokenizerWrapper::BertPreTokenizer(p) => Ok(PipelinePreTokenizer::Bert(p)),
            PreTokenizerWrapper::Delimiter(p) => Ok(PipelinePreTokenizer::Delimiter(p)),
            PreTokenizerWrapper::Digits(p) => Ok(PipelinePreTokenizer::Digits(p)),
            PreTokenizerWrapper::FixedLength(p) => Ok(PipelinePreTokenizer::FixedLength(p)),
            PreTokenizerWrapper::Punctuation(p) => Ok(PipelinePreTokenizer::Punctuation(p)),
            PreTokenizerWrapper::Split(p) => {
                Ok(PipelinePreTokenizer::Split(p.canonicalized_for_pipeline()?))
            }
            PreTokenizerWrapper::UnicodeScripts(p) => Ok(PipelinePreTokenizer::UnicodeScripts(p)),
            PreTokenizerWrapper::Whitespace(p) => Ok(PipelinePreTokenizer::Whitespace(p)),
            PreTokenizerWrapper::WhitespaceSplit(p) => Ok(PipelinePreTokenizer::WhitespaceSplit(p)),
            PreTokenizerWrapper::ByteLevel(byte_level) => {
                if byte_level.add_prefix_space {
                    return Err(
                        "ByteLevel add_prefix_space=true is not supported by the pipeline yet"
                            .into(),
                    );
                }
                if byte_level.use_regex {
                    Ok(PipelinePreTokenizer::Split(Split::new(
                        SplitPattern::Regex(GPT2_REGEX_STR.to_owned()),
                        SplitDelimiterBehavior::Isolated,
                        false,
                    )?))
                } else {
                    Ok(PipelinePreTokenizer::None)
                }
            }
            PreTokenizerWrapper::Sequence(p) => Ok(PipelinePreTokenizer::Sequence(p.try_into()?)),
            other => {
                Err(format!("PipelineTokenizer does not support PreTokenizer: {other:?}").into())
            }
        }
    }
}

impl TryFrom<&PostProcessorWrapper> for PipelinePostProcessor {
    type Error = tk_encode::Error;

    fn try_from(value: &PostProcessorWrapper) -> Result<Self> {
        fn one(id: u32, tid: u8) -> Slice {
            Slice::Specials {
                tokens: Box::new([PipelineToken::from(id)]),
                type_id: tid,
            }
        }
        fn multi(ids: &[u32], tid: u8) -> Slice {
            Slice::Specials {
                tokens: ids.iter().map(|&id| PipelineToken::from(id)).collect(),
                type_id: tid,
            }
        }
        use Seq::{A, B};
        let sq = |seq, type_id| Slice::Sequence { seq, type_id };

        match value {
            PostProcessorWrapper::Bert(BertProcessing {
                cls: (_, cls_id),
                sep: (_, sep_id),
            }) => Ok(Self::new(
                Template::new(vec![one(*cls_id, 0), sq(A, 0), one(*sep_id, 0)]),
                Template::new(vec![
                    one(*cls_id, 0),
                    sq(A, 0),
                    one(*sep_id, 0),
                    sq(B, 1),
                    one(*sep_id, 1),
                ]),
            )),
            PostProcessorWrapper::Roberta(RobertaProcessing {
                cls: (_, cls_id),
                sep: (_, sep_id),
                ..
            }) => Ok(Self::new(
                Template::new(vec![one(*cls_id, 0), sq(A, 0), one(*sep_id, 0)]),
                Template::new(vec![
                    one(*cls_id, 0),
                    sq(A, 0),
                    multi(&[*sep_id, *sep_id], 0),
                    sq(B, 0),
                    one(*sep_id, 0),
                ]),
            )),
            PostProcessorWrapper::Template(pp) => Ok(Self::new(
                Template::new(build_slices(
                    pp.single.as_slice(),
                    pp.get_special_tokens(),
                    false,
                )?),
                Template::new(build_slices(
                    pp.get_pair().as_slice(),
                    pp.get_special_tokens(),
                    true,
                )?),
            )),
            PostProcessorWrapper::ByteLevel(_) => Ok(Self::new(
                Template::new(vec![sq(A, 0)]),
                Template::new(vec![sq(A, 0), sq(B, 1)]),
            )),
            PostProcessorWrapper::Sequence(sequence) => {
                let members = sequence
                    .as_ref()
                    .iter()
                    .map(Self::try_from)
                    .collect::<Result<Vec<_>>>()?;
                Ok(Self::new(
                    compose(members.iter().map(|m| m.templates().0))?,
                    compose(members.iter().map(|m| m.templates().1))?,
                ))
            }
        }
    }
}

impl TryFrom<&Tokenizer> for PipelineTokenizer {
    type Error = tk_encode::Error;

    /// Build a pipeline from an existing [`Tokenizer`], cloning its components.
    ///
    /// The base [`Tokenizer`] carries the legacy [`crate::AddedVocabulary`]; the pipeline uses the
    /// fast bucket [`BucketAddedVocabulary`], so we rebuild it from the tokenizer's added tokens.
    /// Adding them in id order preserves ids (tokens present in the model reuse their model id, the
    /// rest keep their dense order), so the pipeline emits the same ids as the reference tokenizer.
    fn try_from(tok: &Tokenizer) -> Result<Self> {
        let mut normalizers = Vec::new();
        // An empty `Sequence` is how a config spells "no normalization" (deepseek ships one), so drop
        // it here instead of calling into a no-op for every segment. `lower_normalizer` would
        // produce nothing for it anyway; the `filter` keeps that explicit.
        let declared = tok.get_normalizer().filter(|declared| {
            !matches!(declared, NormalizerWrapper::Sequence(seq) if seq.as_ref().is_empty())
        });
        if let Some(declared) = declared {
            lower_normalizer(declared, &mut normalizers);
        }

        let mut metaspace_normalizer_to_append: Option<MetaspaceNormalizer> = None;
        // A `Metaspace` pre-tokenizer does two jobs at once: it writes `▁` delimiters into the text,
        // then cuts on them. The pipeline keeps rewriting and cutting apart, so we rebuild it as a
        // normalizer plus a `Split`. That normalizer runs after the declared one, matching the order
        // the config asks for: the whole normalizer first, then the pre-tokenizer.
        let pre_tokenizer = match to_normalizer_and_split(tok.get_pre_tokenizer()) {
            // One shift this brings: added tokens flagged `normalized` are matched against text
            // that already carries the delimiters, so such a token containing a space would stop
            // matching. The t5 and albert configs we test have no normalized added token at all.
            //
            // The normalizer itself is appended after the added-token replay below, purely so the
            // replay's borrow of `normalizers` ends first; it is a no-op in the legacy trait, so the
            // order makes no difference to the ids.
            Some((metaspace_normalizer, split)) => {
                metaspace_normalizer_to_append = Some(metaspace_normalizer);
                PipelinePreTokenizer::Split(split)
            }
            // Every other pre-tokenizer converts on its own.
            None => tok
                .get_pre_tokenizer()
                .cloned()
                .map(TryInto::try_into)
                .transpose()?
                .unwrap_or(PipelinePreTokenizer::None),
        };

        let legacy_av = tok.get_added_vocabulary();
        let mut added_tokens: Vec<_> = legacy_av.get_added_tokens_decoder().iter().collect();
        added_tokens.sort_by_key(|(id, _)| **id);
        let mut added_vocabulary = BucketAddedVocabulary::new();
        // The whole lowered chain, not `tok.get_normalizer()`: a config `Sequence` was flattened
        // into `normalizers`, so the chain is what the wrapper's own `normalize` would have done —
        // and a `normalized: true` added token has to see all of it or its id moves. The `Metaspace`
        // member appended below is a deliberate no-op in the legacy trait, so replaying before or
        // after it gives the same ids; the slim reader replays through the same chain.
        let chain = NormalizerChain(&normalizers);
        added_vocabulary.add_tokens(
            added_tokens.into_iter().map(|(_, t)| BucketAddedToken {
                content: t.content.clone(),
                single_word: t.single_word,
                lstrip: t.lstrip,
                rstrip: t.rstrip,
                normalized: t.normalized,
                special: t.special,
            }),
            tok.get_model(),
            Some(&chain),
        )?;
        added_vocabulary.set_encode_special_tokens(legacy_av.get_encode_special_tokens());
        if let Some(metaspace_normalizer) = metaspace_normalizer_to_append {
            normalizers.push(PipelineNormalizer::Metaspace(metaspace_normalizer));
        }

        let with_byte_level = {
            if let Some(pt) = tok.get_pre_tokenizer() {
                if let PreTokenizerWrapper::ByteLevel(_) = pt {
                    true
                } else if let PreTokenizerWrapper::Sequence(seq) = pt {
                    if seq
                        .as_ref()
                        .iter()
                        .any(|pt| matches!(pt, PreTokenizerWrapper::Sequence(_)))
                    {
                        return Err("Nesting Sequence pre tokenizers is not supported".into());
                    }
                    if let Some(pos) = seq
                        .as_ref()
                        .iter()
                        .position(|p| matches!(p, PreTokenizerWrapper::ByteLevel(_)))
                    {
                        if pos != seq.as_ref().len() - 1 {
                            return Err("ByteLevel pre tokenizer must be the last pre tokenizer in the Sequence".into());
                        }
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            }
        };

        let model = tok.get_model();
        if with_byte_level && !matches!(&model, ModelWrapper::BPE(_)) {
            let model_name = match model {
                ModelWrapper::BPE(_) => "BPE",
                ModelWrapper::Unigram(_) => "Unigram",
                ModelWrapper::WordLevel(_) => "WordLevel",
                ModelWrapper::WordPiece(_) => "WordPiece",
            };
            return Err(format!(
                "ByteLevel pre tokenizer is not supported with model {model_name}"
            )
            .into());
        }

        let model = match model.clone() {
            ModelWrapper::BPE(model) => {
                PipelineModel::BPE(crate::models::bpe::into_pipeline(model, with_byte_level)?)
            }
            ModelWrapper::Unigram(model) => PipelineModel::Unigram(model),
            ModelWrapper::WordLevel(model) => PipelineModel::WordLevel(model),
            ModelWrapper::WordPiece(model) => PipelineModel::WordPiece(model.try_into()?),
        };

        Ok(PipelineTokenizer::from_parts(
            added_vocabulary,
            normalizers,
            pre_tokenizer,
            model,
            tok.get_post_processor()
                .map(PipelinePostProcessor::try_from)
                .transpose()?
                .unwrap_or_default(),
            tok.get_decoder().map(DecoderRuntime::from),
        ))
    }
}
