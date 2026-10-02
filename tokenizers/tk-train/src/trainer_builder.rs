use std::collections::BTreeMap;

use log::warn;
use tk_encode::decoders::byte_level::ByteLevelDecoder;
use tk_encode::pipeline::{PipelineModel, PipelineTokenizer, Template};
use tk_encode::{DecoderRuntime, TruncationParams};
use tk_encode::{
    pipeline::{PipelineNormalizer, PipelinePreTokenizer},
    vocab::bucket_added_vocabulary::AddedToken,
};

use crate::ProgressFormat;
use crate::error::{Result, TrainingError};
use crate::trainer::{
    ModelTrainer, PaddingSpec, PostProcessorSpec, TemplateSpec, TokenizerTrainer, TrainingParams,
    TrainingPipeline,
};
use crate::trainers::TrainerWrapper;

#[derive(Debug)]
pub struct TokenizerTrainerBuilder<T> {
    trainer: T,

    // ---- Training Params ----
    vocab_size: Option<usize>,
    special_tokens: Option<Vec<AddedToken>>,
    unk_token: Option<String>,
    progress: Option<ProgressFormat>,
    byte_level: bool,

    // ---- Training Pipeline ----
    normalizers: Option<Vec<PipelineNormalizer>>,
    pre_tokenizer: Option<PipelinePreTokenizer>,
    post_processor: Option<PostProcessorSpec>,
    decoder: Option<DecoderRuntime>,
    padding: Option<PaddingSpec>,
    truncation: Option<TruncationParams>,
    role_to_token: Option<BTreeMap<String, String>>,
}

impl<T: ModelTrainer> TokenizerTrainerBuilder<T> {
    /// Checks the settings against each other, so that a bad configuration fails before any
    /// text is read.
    pub fn build(self) -> Result<TokenizerTrainer<T>> {
        let vocab_size = self.vocab_size.ok_or(TrainingError::MissingVocabSize)?;
        let special_tokens = self.special_tokens.unwrap_or_default();
        let post_processor = self.post_processor.unwrap_or_default();
        let is_special = |content: &str| special_tokens.iter().any(|tk| tk.content == content);

        if let Some(unk_token) = &self.unk_token
            && !is_special(unk_token)
        {
            return Err(TrainingError::UnkTokenNotSpecial(unk_token.clone()));
        }
        if let Some(token) = template_tokens(&post_processor).find(|token| !is_special(token)) {
            return Err(TrainingError::TemplateTokenNotSpecial(token.to_owned()));
        }
        if let Some(padding) = &self.padding
            && !is_special(&padding.pad_token)
        {
            return Err(TrainingError::PadTokenNotSpecial(padding.pad_token.clone()));
        }
        let role_to_token = self.role_to_token.unwrap_or_default();
        if let Some((role, token)) = role_to_token.iter().find(|(_, token)| !is_special(token)) {
            return Err(TrainingError::RoleTokenNotSpecial {
                role: role.clone(),
                token: token.clone(),
            });
        }
        if vocab_size <= special_tokens.len() {
            return Err(TrainingError::VocabTooSmall {
                vocab_size,
                reserved: special_tokens.len(),
            });
        }

        let normalizers = self.normalizers.unwrap_or_default();
        let mut decoder = self.decoder;
        if self.byte_level {
            decoder.get_or_insert(DecoderRuntime::ByteLevel(ByteLevelDecoder::new()));
        }
        let byte_level_normalizer = normalizers.iter().any(is_byte_level_normalizer);
        if self.byte_level && byte_level_normalizer {
            return Err(TrainingError::ByteLevelTwice);
        }
        let byte_level_text = self.byte_level || byte_level_normalizer;
        // No decoder is fine: a tokenizer that never decodes cannot decode wrongly.
        match &decoder {
            Some(decoder) if byte_level_text && !has_byte_level_decoder(decoder) => {
                return Err(TrainingError::ByteLevelDecoderMissing);
            }
            Some(decoder) if !byte_level_text && has_byte_level_decoder(decoder) => {
                return Err(TrainingError::ByteLevelDecoderWithoutByteLevel);
            }
            _ => {}
        }

        let params = TrainingParams {
            vocab_size,
            byte_level: self.byte_level,
            progress: self.progress.unwrap_or_default(),
            special_tokens,
            unk_token: self.unk_token,
        };
        self.trainer.check(&params)?;

        Ok(TokenizerTrainer {
            trainer: self.trainer,
            pipeline: TrainingPipeline {
                normalizers,
                pre_tokenizer: self.pre_tokenizer.unwrap_or(PipelinePreTokenizer::None),
                post_processor,
                decoder,
                padding: self.padding,
                truncation: self.truncation,
                role_to_token,
            },
            params,
        })
    }
}

impl<T> TokenizerTrainerBuilder<T> {
    pub fn new(trainer: T) -> Self {
        Self {
            trainer,
            vocab_size: None,
            special_tokens: None,
            unk_token: None,
            progress: None,
            byte_level: false,
            normalizers: None,
            pre_tokenizer: None,
            post_processor: None,
            decoder: None,
            padding: None,
            truncation: None,
            role_to_token: None,
        }
    }

    /// Replaces the trainer and keeps every other setting, for example to retrain a tokenizer from
    /// [`TokenizerTrainerBuilder::from_tokenizer`] with other trainer options.
    #[must_use]
    pub fn trainer<U>(self, trainer: U) -> TokenizerTrainerBuilder<U> {
        TokenizerTrainerBuilder {
            trainer,
            vocab_size: self.vocab_size,
            special_tokens: self.special_tokens,
            unk_token: self.unk_token,
            progress: self.progress,
            byte_level: self.byte_level,
            normalizers: self.normalizers,
            pre_tokenizer: self.pre_tokenizer,
            post_processor: self.post_processor,
            decoder: self.decoder,
            padding: self.padding,
            truncation: self.truncation,
            role_to_token: self.role_to_token,
        }
    }

    /// The number of tokens in the trained vocabulary, special tokens included.
    #[must_use]
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.vocab_size = Some(size);
        self
    }

    /// Tokens that are never split or normalized, and take the first ids of the vocabulary.
    /// A token given twice keeps its first position.
    #[must_use]
    pub fn special_tokens<I, S>(mut self, tokens: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut special_tokens: Vec<AddedToken> = vec![];
        for token in tokens {
            let token = token.into();
            if special_tokens
                .iter()
                .any(|special| special.content == token)
            {
                warn!("special token {token:?} is present more than once, dropping the duplicate");
                continue;
            }
            special_tokens.push(AddedToken::from(token, true));
        }
        self.special_tokens = Some(special_tokens);
        self
    }

    /// The token for text the vocabulary cannot spell. It has to be one of the special tokens.
    #[must_use]
    pub fn unk_token(mut self, unk_token: impl Into<String>) -> Self {
        self.unk_token = Some(unk_token.into());
        self
    }

    #[must_use]
    pub fn progress(mut self, progress: ProgressFormat) -> Self {
        self.progress = Some(progress);
        self
    }

    /// The model reads text as bytes, each spelled with one of 256 visible characters (GPT-2 and
    /// most models after it). Adds the ByteLevel decoder when no decoder is set.
    #[must_use]
    pub fn byte_level(mut self) -> Self {
        self.byte_level = true;
        self
    }

    #[must_use]
    pub fn normalizers(mut self, normalizers: Vec<PipelineNormalizer>) -> Self {
        self.normalizers = Some(normalizers);
        self
    }

    #[must_use]
    pub fn pre_tokenizer(mut self, pre_tokenizer: PipelinePreTokenizer) -> Self {
        self.pre_tokenizer = Some(pre_tokenizer);
        self
    }

    /// Every token its templates use has to be one of the special tokens.
    #[must_use]
    pub fn post_processor(mut self, post_processor: PostProcessorSpec) -> Self {
        self.post_processor = Some(post_processor);
        self
    }

    #[must_use]
    pub fn decoder(mut self, decoder: DecoderRuntime) -> Self {
        self.decoder = Some(decoder);
        self
    }

    /// Its pad token has to be one of the special tokens.
    #[must_use]
    pub fn padding(mut self, padding: PaddingSpec) -> Self {
        self.padding = Some(padding);
        self
    }

    #[must_use]
    pub fn truncation(mut self, truncation: TruncationParams) -> Self {
        self.truncation = Some(truncation);
        self
    }

    /// Which special token plays which role (`"eos_token"` to `"</s>"`). Every token has to be one
    /// of the special tokens.
    #[must_use]
    pub fn role_to_token(mut self, role_to_token: BTreeMap<String, String>) -> Self {
        self.role_to_token = Some(role_to_token);
        self
    }
}

impl TokenizerTrainerBuilder<TrainerWrapper> {
    /// A builder with every setting of `tokenizer`: its normalizers, pre-tokenizer, post-processor,
    /// decoder, padding, truncation, roles and special tokens, and a trainer for its model's kind with the model's subword
    /// affixes, vocabulary size and unknown token.
    ///
    /// Not carried over: added tokens that are not special, and byte fallback.
    pub fn from_tokenizer(tokenizer: &PipelineTokenizer) -> Result<Self> {
        let model = ModelSettings::read(tokenizer.get_model())?;

        let added_tokens = tokenizer.get_added_vocabulary().get_added_tokens_decoder();
        let mut special_tokens: Vec<_> = added_tokens
            .iter()
            .filter(|(_, token)| token.special)
            .collect();
        special_tokens.sort_by_key(|(id, _)| **id);
        let special_tokens = special_tokens
            .into_iter()
            .map(|(_, token)| token.clone())
            .collect();

        let token = |id: u32| {
            added_tokens
                .get(&id)
                .map(|token| token.content.clone())
                .or_else(|| tokenizer.get_model().id_to_token(id))
                .ok_or(TrainingError::UnknownTemplateId(id))
        };
        let tokens = |run: &[(tk_encode::pipeline::PipelineToken, u8)]| {
            run.iter()
                .map(|&(id, type_id)| Ok((token(id.id())?, type_id)))
                .collect::<Result<Vec<_>>>()
        };
        let template = |template: &Template| -> Result<TemplateSpec> {
            Ok(TemplateSpec {
                prefix: tokens(&template.prefix)?,
                infix: tokens(&template.infix)?,
                suffix: tokens(&template.suffix)?,
                a_type_id: template.a_type_id,
                b_type_id: template.b_type_id,
            })
        };
        let post_processor = tokenizer.get_post_processor();

        Ok(Self {
            trainer: model.trainer,
            vocab_size: Some(model.vocab_size),
            special_tokens: Some(special_tokens),
            unk_token: model.unk_token,
            progress: None,
            byte_level: model.byte_level,
            normalizers: Some(tokenizer.get_normalizers().to_vec()),
            pre_tokenizer: Some(tokenizer.get_pre_tokenizer().clone()),
            post_processor: Some(PostProcessorSpec {
                single: template(&post_processor.single)?,
                pair: template(&post_processor.pair)?,
            }),
            decoder: tokenizer.get_decoder().cloned(),
            padding: tokenizer.get_padding().map(|padding| PaddingSpec {
                strategy: padding.strategy.clone(),
                direction: padding.direction,
                pad_to_multiple_of: padding.pad_to_multiple_of,
                pad_type_id: padding.pad_type_id,
                pad_token: padding.pad_token.clone(),
            }),
            truncation: tokenizer.get_truncation().cloned(),
            role_to_token: Some(tokenizer.get_role_to_token().clone()),
        })
    }
}

/// What a retrained tokenizer keeps from its model.
struct ModelSettings {
    trainer: TrainerWrapper,
    vocab_size: usize,
    unk_token: Option<String>,
    byte_level: bool,
}

impl ModelSettings {
    // tk-encode can have a model compiled in whose trainer tk-train was built without.
    #[allow(unreachable_patterns)]
    fn read(model: &PipelineModel) -> Result<Self> {
        match model {
            #[cfg(feature = "bpe")]
            PipelineModel::BPE(bpe) => {
                let config = bpe.to_config()?;
                let mut trainer = crate::trainers::BpeTrainer::builder();
                if let Some(prefix) = config.continuing_subword_prefix {
                    trainer = trainer.continuing_subword_prefix(prefix);
                }
                if let Some(suffix) = config.end_of_word_suffix {
                    trainer = trainer.end_of_word_suffix(suffix);
                }
                trainer = trainer
                    .fuse_unk(config.fuse_unk)
                    .ignore_merges(config.ignore_merges);
                Ok(Self {
                    trainer: trainer.build().into(),
                    vocab_size: config.vocab.len(),
                    unk_token: config.unk_token,
                    byte_level: config.byte_level,
                })
            }
            #[cfg(feature = "wordpiece")]
            PipelineModel::WordPiece(wordpiece) => Ok(Self {
                trainer: crate::trainers::WordPieceTrainer::builder()
                    .continuing_subword_prefix(wordpiece.continuing_subword_prefix().to_owned())
                    .build()
                    .into(),
                vocab_size: wordpiece.vocab().len(),
                unk_token: wordpiece.unk_token().map(str::to_owned),
                byte_level: false,
            }),
            #[cfg(feature = "wordlevel")]
            PipelineModel::WordLevel(wordlevel) => Ok(Self {
                trainer: crate::trainers::WordLevelTrainer::default().into(),
                vocab_size: wordlevel.get_vocab_size(),
                // A WordLevel always names an unknown token, but only uses it when it is in the
                // vocabulary.
                unk_token: wordlevel
                    .token_to_id(&wordlevel.unk_token)
                    .map(|_| wordlevel.unk_token.clone()),
                byte_level: false,
            }),
            #[cfg(feature = "unigram")]
            PipelineModel::Unigram(unigram) => Ok(Self {
                trainer: crate::trainers::UnigramTrainer::default().into(),
                vocab_size: unigram.get_vocab_size(),
                unk_token: unigram.unk_id().map(|id| unigram.vocab()[id].0.clone()),
                byte_level: false,
            }),
            PipelineModel::BPE(_) => Err(TrainingError::TrainerNotCompiled("BPE")),
            _ => Err(TrainingError::TrainerNotCompiled("matching")),
        }
    }
}

fn is_byte_level_normalizer(normalizer: &PipelineNormalizer) -> bool {
    matches!(normalizer, PipelineNormalizer::ByteLevel(_))
}

fn has_byte_level_decoder(decoder: &DecoderRuntime) -> bool {
    match decoder {
        DecoderRuntime::ByteLevel(_) => true,
        DecoderRuntime::Sequence(decoders) => decoders.iter().any(has_byte_level_decoder),
        _ => false,
    }
}

fn template_tokens(post_processor: &PostProcessorSpec) -> impl Iterator<Item = &str> {
    [&post_processor.single, &post_processor.pair]
        .into_iter()
        .flat_map(|template| [&template.prefix, &template.infix, &template.suffix])
        .flatten()
        .map(|(token, _type_id)| token.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainer::TemplateSpec;
    use tk_encode::normalizers::ByteLevel;

    struct NoopTrainer;

    impl ModelTrainer for NoopTrainer {
        type Model = tk_encode::pipeline::PipelineModel;

        fn train_model(&self, _params: &TrainingParams) -> Result<Self::Model> {
            unimplemented!()
        }

        // Stands in for any model, byte-level ones included.
        fn check(&self, _params: &TrainingParams) -> Result<()> {
            Ok(())
        }

        fn feed<I, S, F>(&mut self, _iterator: I, _process: F) -> Result<()>
        where
            I: Iterator<Item = S> + Send,
            S: AsRef<str> + Send,
            F: Fn(&str) -> Result<Vec<String>> + Sync,
        {
            unimplemented!()
        }
    }

    fn builder() -> TokenizerTrainerBuilder<NoopTrainer> {
        TokenizerTrainerBuilder::new(NoopTrainer)
    }

    fn template(prefix: &str) -> PostProcessorSpec {
        PostProcessorSpec {
            single: TemplateSpec {
                prefix: vec![(prefix.into(), 0)],
                ..TemplateSpec::default()
            },
            pair: TemplateSpec::default(),
        }
    }

    #[test]
    fn builds_a_valid_configuration() {
        let built = builder()
            .vocab_size(100)
            .special_tokens(["[UNK]", "[CLS]", "[PAD]"])
            .unk_token("[UNK]")
            .post_processor(template("[CLS]"))
            .padding(padding("[PAD]"))
            .role_to_token(BTreeMap::from([(
                "pad_token".to_owned(),
                "[PAD]".to_owned(),
            )]))
            .build();

        assert!(built.is_ok());
    }

    fn padding(pad_token: &str) -> PaddingSpec {
        PaddingSpec {
            strategy: tk_encode::PaddingStrategy::BatchLongest,
            direction: tk_encode::PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_type_id: 0,
            pad_token: pad_token.to_owned(),
        }
    }

    #[test]
    fn refuses_a_pad_token_that_is_not_special() {
        let built = builder()
            .vocab_size(100)
            .special_tokens(["[UNK]"])
            .padding(padding("[PAD]"))
            .build();

        assert!(matches!(built, Err(TrainingError::PadTokenNotSpecial(t)) if t == "[PAD]"));
    }

    #[test]
    fn refuses_a_role_whose_token_is_not_special() {
        let built = builder()
            .vocab_size(100)
            .special_tokens(["[UNK]"])
            .role_to_token(BTreeMap::from([(
                "eos_token".to_owned(),
                "</s>".to_owned(),
            )]))
            .build();

        assert!(matches!(
            built,
            Err(TrainingError::RoleTokenNotSpecial { role, token })
                if role == "eos_token" && token == "</s>"
        ));
    }

    #[test]
    fn drops_duplicate_special_tokens_keeping_the_first() {
        let builder = builder().special_tokens(["[CLS]", "[SEP]", "[CLS]"]);

        let contents: Vec<_> = builder
            .special_tokens
            .unwrap()
            .into_iter()
            .map(|token| token.content)
            .collect();
        assert_eq!(contents, ["[CLS]", "[SEP]"]);
    }

    #[test]
    fn refuses_a_missing_vocab_size() {
        assert!(matches!(
            builder().build(),
            Err(TrainingError::MissingVocabSize)
        ));
    }

    #[test]
    fn checks_unk_token_whatever_the_setter_order() {
        let built = builder()
            .vocab_size(100)
            .unk_token("[UNK]")
            .special_tokens(["[UNK]"])
            .build();

        assert!(built.is_ok());
    }

    #[test]
    fn refuses_an_unk_token_that_is_not_special() {
        let built = builder()
            .vocab_size(100)
            .special_tokens(["[CLS]"])
            .unk_token("[UNK]")
            .build();

        assert!(matches!(built, Err(TrainingError::UnkTokenNotSpecial(t)) if t == "[UNK]"));
    }

    #[test]
    fn refuses_a_template_token_that_is_not_special() {
        let built = builder()
            .vocab_size(100)
            .post_processor(template("[CLS]"))
            .build();

        assert!(matches!(built, Err(TrainingError::TemplateTokenNotSpecial(t)) if t == "[CLS]"));
    }

    #[test]
    fn refuses_a_vocab_with_no_room_for_a_learned_token() {
        let built = builder()
            .vocab_size(2)
            .special_tokens(["[UNK]", "[CLS]"])
            .build();

        assert!(matches!(
            built,
            Err(TrainingError::VocabTooSmall {
                vocab_size: 2,
                reserved: 2
            })
        ));
    }

    fn wordpiece_decoder() -> DecoderRuntime {
        DecoderRuntime::WordPiece(tk_encode::decoders::wordpiece::WordPiece::new(
            "##".into(),
            true,
        ))
    }

    fn byte_level_normalizer() -> PipelineNormalizer {
        PipelineNormalizer::ByteLevel(ByteLevel::new())
    }

    #[test]
    fn byte_level_makes_a_byte_level_model_and_adds_the_decoder() {
        let built = builder()
            .vocab_size(100)
            .byte_level()
            .normalizers(vec![PipelineNormalizer::Lowercase(
                tk_encode::normalizers::Lowercase,
            )])
            .build()
            .unwrap();

        assert!(matches!(
            built.pipeline.normalizers[..],
            [PipelineNormalizer::Lowercase(_)]
        ));
        assert!(matches!(
            built.pipeline.decoder,
            Some(DecoderRuntime::ByteLevel(_))
        ));
        assert!(built.params.byte_level);
    }

    #[test]
    fn refuses_byte_level_with_the_byte_level_normalizer() {
        let built = builder()
            .vocab_size(100)
            .normalizers(vec![byte_level_normalizer()])
            .byte_level()
            .build();

        assert!(matches!(built, Err(TrainingError::ByteLevelTwice)));
    }

    #[test]
    fn byte_level_keeps_a_decoder_that_contains_byte_level() {
        let decoder = DecoderRuntime::Sequence(vec![
            DecoderRuntime::ByteLevel(ByteLevelDecoder::new()),
            wordpiece_decoder(),
        ]);
        let built = builder()
            .vocab_size(100)
            .byte_level()
            .decoder(decoder)
            .build()
            .unwrap();

        assert!(matches!(
            built.pipeline.decoder,
            Some(DecoderRuntime::Sequence(_))
        ));
    }

    #[test]
    fn byte_level_refuses_a_decoder_without_byte_level() {
        let built = builder()
            .vocab_size(100)
            .byte_level()
            .decoder(wordpiece_decoder())
            .build();

        assert!(matches!(built, Err(TrainingError::ByteLevelDecoderMissing)));
    }

    #[test]
    fn byte_level_normalizer_alone_refuses_a_decoder_without_byte_level() {
        let built = builder()
            .vocab_size(100)
            .normalizers(vec![byte_level_normalizer()])
            .decoder(wordpiece_decoder())
            .build();

        assert!(matches!(built, Err(TrainingError::ByteLevelDecoderMissing)));
    }

    #[test]
    fn byte_level_normalizer_alone_needs_no_byte_level_model_nor_decoder() {
        let built = builder()
            .vocab_size(100)
            .normalizers(vec![byte_level_normalizer()])
            .build()
            .unwrap();

        assert!(!built.params.byte_level);
        assert!(built.pipeline.decoder.is_none());
    }

    #[test]
    fn byte_level_normalizer_accepts_the_byte_level_decoder() {
        let built = builder()
            .vocab_size(100)
            .normalizers(vec![byte_level_normalizer()])
            .decoder(DecoderRuntime::ByteLevel(ByteLevelDecoder::new()))
            .build();

        assert!(built.is_ok());
    }

    #[test]
    fn refuses_a_byte_level_decoder_without_byte_level_text() {
        let built = builder()
            .vocab_size(100)
            .decoder(DecoderRuntime::ByteLevel(ByteLevelDecoder::new()))
            .build();

        assert!(matches!(
            built,
            Err(TrainingError::ByteLevelDecoderWithoutByteLevel)
        ));
    }

    #[test]
    fn is_not_byte_level_by_default() {
        let built = builder().vocab_size(100).build().unwrap();

        assert!(!built.params.byte_level);
        assert!(built.pipeline.normalizers.is_empty());
        assert!(built.pipeline.decoder.is_none());
    }

    fn tokenizer(json: &str) -> PipelineTokenizer {
        tk_serialize::from_json(&tk_convert::canonicalize_str(json).unwrap()).unwrap()
    }

    #[cfg(any(feature = "bpe", feature = "unigram"))]
    fn fixture(name: &str) -> PipelineTokenizer {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/").to_owned() + name;
        tokenizer(&std::fs::read_to_string(path).unwrap())
    }

    fn special_contents(built: &TokenizerTrainer<TrainerWrapper>) -> Vec<&str> {
        let tokens = built.params.special_tokens.iter();
        tokens.map(|token| token.content.as_str()).collect()
    }

    #[cfg(feature = "wordpiece")]
    #[test]
    fn from_tokenizer_keeps_a_wordpiece_tokenizer() {
        let built = TokenizerTrainerBuilder::from_tokenizer(&fixture("bert-base-uncased.json"))
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(built.params.vocab_size, 30522);
        assert_eq!(
            special_contents(&built),
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        );
        assert_eq!(built.params.unk_token.as_deref(), Some("[UNK]"));
        assert!(!built.params.byte_level);
        assert!(matches!(
            built.pipeline.normalizers[..],
            [PipelineNormalizer::Bert(_)]
        ));
        assert!(matches!(
            built.pipeline.decoder,
            Some(DecoderRuntime::WordPiece(_))
        ));
        assert_eq!(
            built.pipeline.post_processor.single.prefix,
            [("[CLS]".to_owned(), 0)]
        );
        assert_eq!(
            built.pipeline.post_processor.single.suffix,
            [("[SEP]".to_owned(), 0)]
        );
        let TrainerWrapper::WordPieceTrainer(trainer) = &built.trainer else {
            panic!("expected a WordPiece trainer, got {:?}", built.trainer);
        };
        assert_eq!(trainer.continuing_subword_prefix(), Some("##"));
    }

    #[cfg(feature = "bpe")]
    #[test]
    fn from_tokenizer_keeps_a_byte_level_bpe_tokenizer() {
        let built = TokenizerTrainerBuilder::from_tokenizer(&fixture("gpt2.json"))
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(built.params.vocab_size, 50257);
        assert_eq!(special_contents(&built), ["<|endoftext|>"]);
        assert_eq!(built.params.unk_token, None);
        assert!(built.params.byte_level);
        assert!(built.pipeline.normalizers.is_empty());
        assert!(matches!(
            built.pipeline.decoder,
            Some(DecoderRuntime::ByteLevel(_))
        ));
        assert!(matches!(built.trainer, TrainerWrapper::BpeTrainer(_)));
    }

    #[cfg(feature = "bpe")]
    #[test]
    fn from_tokenizer_keeps_fuse_unk_and_ignore_merges() {
        let source = tokenizer(
            r#"{
                "version": "1.0",
                "truncation": null,
                "padding": null,
                "added_tokens": [],
                "normalizer": null,
                "pre_tokenizer": null,
                "post_processor": null,
                "decoder": null,
                "model": {
                    "type": "BPE",
                    "unk_token": "<unk>",
                    "fuse_unk": true,
                    "ignore_merges": true,
                    "vocab": {"<unk>": 0, "a": 1, "b": 2},
                    "merges": []
                }
            }"#,
        );

        let builder = TokenizerTrainerBuilder::from_tokenizer(&source).unwrap();

        let TrainerWrapper::BpeTrainer(trainer) = &builder.trainer else {
            panic!("expected a BPE trainer, got {:?}", builder.trainer);
        };
        assert!(trainer.fuse_unk());
        assert!(trainer.ignore_merges());
    }

    #[cfg(feature = "unigram")]
    #[test]
    fn from_tokenizer_keeps_a_unigram_tokenizer() {
        let built =
            TokenizerTrainerBuilder::from_tokenizer(&fixture("albert-base-v1-tokenizer.json"))
                .unwrap()
                .build()
                .unwrap();

        assert_eq!(built.params.vocab_size, 30000);
        assert_eq!(
            special_contents(&built),
            ["<pad>", "<unk>", "[CLS]", "[SEP]", "[MASK]"]
        );
        assert_eq!(built.params.unk_token.as_deref(), Some("<unk>"));
        assert!(matches!(built.trainer, TrainerWrapper::UnigramTrainer(_)));
    }

    #[cfg(feature = "wordlevel")]
    fn wordlevel(unk_token: &str) -> PipelineTokenizer {
        wordlevel_with(unk_token, "null", "null", "null")
    }

    #[cfg(feature = "wordlevel")]
    fn wordlevel_with(
        unk_token: &str,
        padding: &str,
        truncation: &str,
        role_to_token: &str,
    ) -> PipelineTokenizer {
        tokenizer(&format!(
            r#"{{
                "version": "1.0",
                "truncation": {truncation},
                "padding": {padding},
                "role_to_token": {role_to_token},
                "added_tokens": [{{
                    "id": 0, "content": "[UNK]", "single_word": false, "lstrip": false,
                    "rstrip": false, "normalized": false, "special": true
                }}],
                "normalizer": null,
                "pre_tokenizer": {{"type": "Whitespace"}},
                "post_processor": null,
                "decoder": null,
                "model": {{
                    "type": "WordLevel",
                    "vocab": {{"[UNK]": 0, "hello": 1, "world": 2}},
                    "unk_token": "{unk_token}"
                }}
            }}"#
        ))
    }

    #[cfg(feature = "wordlevel")]
    #[test]
    fn from_tokenizer_keeps_a_wordlevel_tokenizer() {
        let built = TokenizerTrainerBuilder::from_tokenizer(&wordlevel("[UNK]"))
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(built.params.vocab_size, 3);
        assert_eq!(special_contents(&built), ["[UNK]"]);
        assert_eq!(built.params.unk_token.as_deref(), Some("[UNK]"));
        assert!(matches!(built.trainer, TrainerWrapper::WordLevelTrainer(_)));
    }

    #[cfg(feature = "wordlevel")]
    #[test]
    fn from_tokenizer_drops_a_wordlevel_unk_token_missing_from_the_vocab() {
        let built = TokenizerTrainerBuilder::from_tokenizer(&wordlevel("<unk>"))
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(built.params.unk_token, None);
    }

    #[cfg(feature = "wordlevel")]
    #[test]
    fn from_tokenizer_keeps_padding_truncation_and_roles() {
        let tokenizer = wordlevel_with(
            "[UNK]",
            r#"{"strategy": {"Fixed": 16}, "direction": "Left", "pad_to_multiple_of": null,
                "pad_id": 0, "pad_type_id": 1, "pad_token": "[UNK]"}"#,
            r#"{"direction": "Right", "max_length": 64, "strategy": "OnlyFirst", "stride": 0}"#,
            r#"{"unk_token": "[UNK]"}"#,
        );

        let built = TokenizerTrainerBuilder::from_tokenizer(&tokenizer)
            .unwrap()
            .build()
            .unwrap();

        let padding = built.pipeline.padding.as_ref().unwrap();
        assert_eq!(padding.pad_token, "[UNK]");
        assert_eq!(padding.strategy, tk_encode::PaddingStrategy::Fixed(16));
        assert_eq!(padding.direction, tk_encode::PaddingDirection::Left);
        assert_eq!(padding.pad_type_id, 1);
        assert_eq!(
            built.pipeline.truncation,
            tokenizer.get_truncation().cloned()
        );
        assert_eq!(&built.pipeline.role_to_token, tokenizer.get_role_to_token());
    }

    #[cfg(feature = "wordpiece")]
    #[test]
    fn trainer_replaces_the_trainer_and_keeps_the_other_settings() {
        let trainer = crate::WordPieceTrainer::builder().min_frequency(3).build();

        let built = TokenizerTrainerBuilder::from_tokenizer(&fixture("bert-base-uncased.json"))
            .unwrap()
            .trainer(TrainerWrapper::from(trainer))
            .build()
            .unwrap();

        let TrainerWrapper::WordPieceTrainer(trainer) = &built.trainer else {
            panic!("expected a WordPiece trainer, got {:?}", built.trainer);
        };
        assert_eq!(trainer.min_frequency(), 3);
        assert_eq!(built.params.vocab_size, 30522);
        assert_eq!(built.params.unk_token.as_deref(), Some("[UNK]"));
    }

    #[cfg(feature = "wordlevel")]
    #[test]
    fn refuses_byte_level_for_a_model_that_reads_characters() {
        let built = TokenizerTrainerBuilder::new(crate::WordLevelTrainer::default())
            .vocab_size(100)
            .byte_level()
            .build();

        assert!(matches!(built, Err(TrainingError::ByteLevelUnsupported)));
    }

    #[cfg(feature = "bpe")]
    #[test]
    fn refuses_a_byte_level_bpe_vocab_without_room_for_the_byte_tokens() {
        let built = TokenizerTrainerBuilder::new(crate::BpeTrainer::default())
            .vocab_size(257)
            .special_tokens(["<|endoftext|>"])
            .byte_level()
            .build();

        assert!(matches!(
            built,
            Err(TrainingError::VocabTooSmall {
                vocab_size: 257,
                reserved: 257
            })
        ));
    }

    #[cfg(feature = "bpe")]
    #[test]
    fn refuses_a_byte_level_bpe_alphabet_limit_below_the_byte_tokens() {
        let trainer = crate::BpeTrainer::builder().limit_alphabet(255).build();

        let built = TokenizerTrainerBuilder::new(trainer)
            .vocab_size(1000)
            .byte_level()
            .build();

        assert!(matches!(
            built,
            Err(TrainingError::AlphabetTooSmall {
                limit_alphabet: 255
            })
        ));
    }

    #[cfg(feature = "bpe")]
    #[test]
    fn accepts_a_byte_level_bpe_alphabet_limit_that_keeps_the_byte_tokens() {
        let trainer = crate::BpeTrainer::builder().limit_alphabet(256).build();

        let built = TokenizerTrainerBuilder::new(trainer)
            .vocab_size(1000)
            .byte_level()
            .build();

        assert!(built.is_ok());
    }
}
