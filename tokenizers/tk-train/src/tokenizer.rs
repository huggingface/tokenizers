use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{Trainer, TrainerWrapper};
use tk_encode::{
    Result,
    pipeline::{
        NormalizerChain, PipelineModel, PipelineNormalizer, PipelineTokenizer, PreTokenizer,
        PreTokenizerScratch, normalize_all,
    },
    vocab::bucket_added_vocabulary::AddedVocabulary,
};

/// Train a fresh model using an existing tokenizer's normalization and pre-tokenization.
///
/// The returned tokenizer owns a new model, added vocabulary and scratch pool. The template
/// (including its clones) is never modified. Its decoder, post-processor, token roles, padding
/// and truncation are retained; token ids in the post-processor and padding are reassigned by
/// token content, not copied. References resolve through the new added-token matcher before
/// the model vocabulary, matching encode precedence. Referenced tokens and roles must exist
/// in the new vocabulary.
///
/// This is fresh training, not incremental training: the old model vocabulary and old added
/// tokens are not used to split the corpus or copied into the result. Only the trainer's added
/// tokens are registered. Include any special tokens needed by retained settings in the trainer.
/// Normalization runs on each complete input sequence, followed by pre-tokenization, with no
/// truncation, padding, decoding or post-processing during training. Empty sequences and
/// normalizer-erased content contribute no training words; training that yields no vocabulary
/// entries is rejected with an error.
///
/// The trainer must match the template's model family. A mismatch is rejected before consuming
/// the iterator or modifying the trainer. Otherwise `feed` replaces the trainer's word counts,
/// even if subsequent training or validation fails. Model options not determined by the trainer
/// are preserved, including BPE `ignore_merges`, preserved across training even when the
/// template vocabulary folds either way. A sequence longer than `u32::MAX` bytes after
/// normalization is rejected before pre-tokenization. Byte-level BPE requires the caller to
/// supply a sufficient initial alphabet; Unigram byte fallback is currently rejected because
/// its trainer does not preserve that option.
/// A post-processor or padding reference to a normalized, added-only token is also rejected:
/// the template does not retain that token's original spelling.
///
/// ```
/// # #[cfg(feature = "bpe")] {
/// use tk_encode::{models::bpe::{BpeConfig, PipelineBPE}, pipeline::*};
/// use tk_train::{train_tokenizer, trainers::BpeTrainer};
/// let template = PipelineTokenizer::from_parts(
///     Default::default(), vec![], PipelinePreTokenizer::None,
///     PipelineModel::BPE(PipelineBPE::from_config(BpeConfig {
///         vocab: [("a".into(), 0)].into(),
///         ..Default::default()
///     })?),
///     Default::default(), None, Default::default(), None, None,
/// );
/// let mut trainer = BpeTrainer::builder().show_progress(false).vocab_size(32).build().into();
/// let trained = train_tokenizer(&template, &mut trainer, ["hello", "hello world"].into_iter())?;
/// assert!(!trained.encode("hello", &EncodeOptions::no_specials()).wait()?[0].is_empty());
/// # }
/// # Ok::<(), tk_encode::Error>(())
/// ```
pub fn train_tokenizer<I, S>(
    template: &PipelineTokenizer,
    trainer: &mut TrainerWrapper,
    sequences: I,
) -> Result<PipelineTokenizer>
where
    I: Iterator<Item = S> + Send,
    S: AsRef<str> + Send,
{
    // Validate before `feed`: neither an incompatible trainer nor its iterator should be touched.
    #[allow(unreachable_patterns)]
    match (&*trainer, template.get_model()) {
        #[cfg(feature = "bpe")]
        (TrainerWrapper::BpeTrainer(_), PipelineModel::BPE(_)) => (),
        #[cfg(feature = "wordpiece")]
        (TrainerWrapper::WordPieceTrainer(_), PipelineModel::WordPiece(_)) => (),
        #[cfg(feature = "wordlevel")]
        (TrainerWrapper::WordLevelTrainer(_), PipelineModel::WordLevel(_)) => (),
        #[cfg(feature = "unigram")]
        (TrainerWrapper::UnigramTrainer(_), PipelineModel::Unigram(model)) => {
            if model.byte_fallback() {
                return Err("train_tokenizer does not yet support Unigram byte fallback".into());
            }
        }
        _ => {
            return Err("train_tokenizer: trainer and template model families do not match".into());
        }
    }

    // Words actually handed to the trainer. Zero means the corpus was empty or every
    // sequence was empty/erased: the Unigram trainer builds a suffix array over its input
    // and panics on zero words (algorithm internals, out of scope), so its arm rejects
    // that before training. The other arms train to an empty vocabulary, rejected below.
    let produced = AtomicUsize::new(0);
    trainer.feed(sequences, |sequence| {
        let normalized = normalize_all(template.get_normalizers(), sequence, 0)?;
        // A `Span` holds `u32` offsets, matching the encode path's guard: normalization can
        // grow the text (a widening normalizer on a near-`u32::MAX` input), so check the
        // normalized length before pre-tokenization truncates it to `u32`.
        ensure_pretokenizable_len(normalized.len())?;
        let mut spans = Vec::new();
        template.get_pre_tokenizer().pre_tokenize(
            &normalized,
            &mut PreTokenizerScratch::default(),
            &mut spans,
        )?;
        // The canonical byte-level model consumes raw UTF-8 bytes, whereas the BPE trainer
        // learns characters. Spell those bytes in the model's serialized alphabet exactly once,
        // after normalization and splitting (a ByteLevel normalizer is a separate pipeline step).
        let byte_level =
            matches!(template.get_model(), PipelineModel::BPE(bpe) if bpe.is_byte_level());
        spans
            .into_iter()
            // Empty spans (an empty sequence through `None`, or content fully erased by
            // normalization) carry no text: feeding them would train an "" entry that displaces
            // real words under a small vocabulary.
            .filter(|span| span.start < span.end)
            .map(|span| {
                let word = &normalized[span.range()];
                Ok(if byte_level {
                    word.bytes()
                        .map(|b| tk_encode::utils::byte_level::BYTES_CHAR_LOOKUP[b as usize])
                        .collect()
                } else {
                    word.to_owned()
                })
            })
            .collect::<Result<Vec<String>>>()
            .inspect(|words| {
                produced.fetch_add(words.len(), Ordering::Relaxed);
            })
    })?;

    #[allow(unreachable_patterns)]
    let (model, special_tokens, vocab): (_, _, HashMap<String, u32>) =
        match (&*trainer, template.get_model()) {
            #[cfg(feature = "bpe")]
            (TrainerWrapper::BpeTrainer(trainer), PipelineModel::BPE(model)) => {
                let mut config = model.to_config()?;
                // `to_config` derives `ignore_merges` from foldability for the writer, which is
                // ambiguous when the template vocabulary folds either way. Retraining keeps the
                // configured policy so a fully-foldable `false` template does not broaden.
                config.ignore_merges = model.ignore_merges();
                let (vocab, merges, special_tokens) = trainer.train_vocab()?;
                if vocab.is_empty() {
                    return Err(empty_vocabulary_error());
                }
                config.vocab = vocab;
                config.merges = merges;
                config.continuing_subword_prefix = trainer.continuing_subword_prefix.clone();
                config.end_of_word_suffix = trainer.end_of_word_suffix.clone();
                let vocab = config
                    .vocab
                    .iter()
                    .map(|(t, id)| (t.clone(), *id))
                    .collect();
                (
                    PipelineModel::BPE(tk_encode::models::bpe::PipelineBPE::from_config(config)?),
                    special_tokens,
                    vocab,
                )
            }
            #[cfg(feature = "wordpiece")]
            (TrainerWrapper::WordPieceTrainer(trainer), PipelineModel::WordPiece(model)) => {
                let mut model = tk_encode::models::wordpiece::WordPiece {
                    vocab: Default::default(),
                    vocab_r: Default::default(),
                    unk_token: model
                        .unk_token()
                        .ok_or("train_tokenizer: template WordPiece has no unknown token")?
                        .to_owned(),
                    // The affix is trainer-determined: seed from the trainer (defaulting to none)
                    // rather than the template, or unprefixed learned pieces would pair with a
                    // prefixed runtime and compounds would become unknown tokens.
                    continuing_subword_prefix: trainer
                        .continuing_subword_prefix()
                        .clone()
                        .unwrap_or_default(),
                    max_input_chars_per_word: model.max_input_chars_per_word(),
                };
                let special_tokens = trainer.train(&mut model)?;
                if model.get_vocab_size() == 0 {
                    return Err(empty_vocabulary_error());
                }
                if model.token_to_id(&model.unk_token).is_none() {
                    return Err(
                    "train_tokenizer: trained WordPiece vocabulary is missing its unknown token"
                        .into(),
                );
                }
                let vocab = model.get_vocab();
                (
                    PipelineModel::WordPiece(model.try_into()?),
                    special_tokens,
                    vocab,
                )
            }
            #[cfg(feature = "wordlevel")]
            (TrainerWrapper::WordLevelTrainer(trainer), PipelineModel::WordLevel(model)) => {
                let mut model = model.clone();
                let special_tokens = trainer.train(&mut model)?;
                if model.get_vocab_size() == 0 {
                    return Err(empty_vocabulary_error());
                }
                if model.token_to_id(&model.unk_token).is_none() {
                    return Err(
                    "train_tokenizer: trained WordLevel vocabulary is missing its unknown token"
                        .into(),
                );
                }
                let vocab = model.get_vocab();
                (PipelineModel::WordLevel(model), special_tokens, vocab)
            }
            #[cfg(feature = "unigram")]
            (TrainerWrapper::UnigramTrainer(trainer), PipelineModel::Unigram(template_model)) => {
                // The Unigram trainer builds a suffix array over its input and panics on zero
                // words (algorithm internals, out of scope): reject that before training.
                if produced.load(Ordering::Relaxed) == 0 {
                    return Err(empty_vocabulary_error());
                }
                let mut model = tk_encode::models::unigram::Unigram::default();
                let special_tokens = trainer.train(&mut model)?;
                if model.get_vocab_size() == 0 {
                    return Err(empty_vocabulary_error());
                }
                model.alpha = template_model.alpha;
                model.nbest_size = template_model.nbest_size;
                let vocab = model.get_vocab();
                (PipelineModel::Unigram(model), special_tokens, vocab)
            }
            _ => unreachable!("model family was validated before feeding the trainer"),
        };

    let mut added = AddedVocabulary::new();
    // Match the reader: a trailing Metaspace is a folded pre-tokenizer, not part of the
    // normalization of added-token spellings.
    let normalizers = template.get_normalizers();
    let declared = match normalizers.last() {
        Some(PipelineNormalizer::Metaspace(_)) => &normalizers[..normalizers.len() - 1],
        _ => normalizers,
    };
    added.add_tokens(
        special_tokens,
        vocab.len(),
        |t| vocab.get(t).copied(),
        Some(&NormalizerChain(declared)),
    )?;
    // Encode resolves added tokens before the model, so retained references must follow the
    // same precedence: the new added matcher first (raw, then normalized form), then the
    // model spelling. The two maps are deliberately not merged: `get_vocab` iteration order
    // is nondeterministic, which would let a normalized alias collision resolve either way.
    let lookup = |token: &str| -> Option<u32> {
        added
            .token_to_id(token)
            .or_else(|| vocab.get(token).copied())
    };

    let old_added = template.get_added_vocabulary().get_added_tokens_decoder();
    let remap = |id: u32| -> Result<u32> {
        let token = if let Some(token) = old_added.get(&id).filter(|token| !token.normalized) {
            // Raw added tokens retain their original content, even when they reuse a model id.
            token.content.clone()
        } else if let Some(token) = template.get_model().id_to_token(id) {
            token
        } else if let Some(token) = old_added.get(&id) {
            if token.normalized {
                return Err(format!("train_tokenizer: cannot recover the original spelling of normalized added token at id {id}").into());
            }
            token.content.clone()
        } else {
            return Err(format!(
                "train_tokenizer: retained setting references unknown template token id {id}"
            )
            .into());
        };
        lookup(&token).ok_or_else(|| format!("train_tokenizer: retained setting references token {token:?}, absent from the trained vocabulary").into())
    };
    let mut post_processor = template.get_post_processor().clone();
    for template in [&mut post_processor.single, &mut post_processor.pair] {
        for run in [
            &mut template.prefix,
            &mut template.infix,
            &mut template.suffix,
        ] {
            for (id, _) in run.iter_mut() {
                *id = remap(id.id())?.into();
            }
        }
    }
    let mut padding = template.get_padding().cloned();
    if let Some(padding) = &mut padding {
        padding.pad_id = remap(padding.pad_id)?;
    }
    for (role, token) in template.get_role_to_token() {
        if lookup(token).is_none() {
            return Err(format!("train_tokenizer: role {role:?} references token {token:?}, absent from the trained vocabulary").into());
        }
    }
    Ok(PipelineTokenizer::from_parts(
        added,
        normalizers.to_vec(),
        template.get_pre_tokenizer().clone(),
        model,
        post_processor,
        template.get_decoder().cloned(),
        template.get_role_to_token().clone(),
        padding,
        template.get_truncation().cloned(),
    ))
}

/// Training that yields no vocabulary entries (an empty corpus, or every sequence empty or
/// erased by normalization) cannot build a model: reject it descriptively before a model
/// constructor meets the empty input.
fn empty_vocabulary_error() -> tk_encode::Error {
    "train_tokenizer: training produced an empty vocabulary (empty corpus or input erased by normalization)".into()
}

/// The pre-tokenizer contract needs `u32` offsets into the normalized text, so a normalized
/// sequence longer than `u32::MAX` bytes cannot be split. Checked here rather than on the raw
/// input because normalization can grow the text.
fn ensure_pretokenizable_len(len: usize) -> Result<()> {
    if len > u32::MAX as usize {
        return Err(format!(
            "sequence too long to pre-tokenize: {len} bytes after normalization, the limit is {}",
            u32::MAX
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::ensure_pretokenizable_len;

    #[test]
    fn normalized_length_guard_matches_the_encode_path() {
        assert!(ensure_pretokenizable_len(0).is_ok());
        assert!(ensure_pretokenizable_len(u32::MAX as usize).is_ok());
        // `u32::MAX as usize + 1` overflows where `usize` is 32 bits; probe the boundary
        // without assuming a 64-bit address space.
        match (u32::MAX as usize).checked_add(1) {
            Some(overflow) => {
                let error = ensure_pretokenizable_len(overflow).unwrap_err();
                assert!(error.to_string().contains("pre-tokenize"), "{error}");
            }
            None => {
                // 32-bit: every addressable length is pre-tokenizable; nothing to reject.
                assert_eq!(u32::MAX as usize, usize::MAX);
            }
        }
    }
}
