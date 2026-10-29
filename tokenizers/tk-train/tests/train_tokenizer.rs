#[cfg(feature = "wordlevel")]
mod wordlevel {
    use tk_encode::{
        models::wordlevel::WordLevel,
        normalizers::utils::Lowercase,
        pipeline::{
            EncodeOptions, PipelineModel, PipelineNormalizer, PipelinePostProcessor,
            PipelinePreTokenizer, PipelineTokenizer,
        },
        pre_tokenizers::whitespace::WhitespaceSplit,
        vocab::bucket_added_vocabulary::{AddedToken, AddedVocabulary},
    };
    use tk_train::{TrainerWrapper, train_tokenizer, trainers::WordLevelTrainer};

    fn template() -> PipelineTokenizer {
        let model = WordLevel::builder()
            .vocab([("<unk>".into(), 0), ("old".into(), 1)].into())
            .unk_token("<unk>".into())
            .build()
            .unwrap();
        PipelineTokenizer::from_parts(
            AddedVocabulary::new(),
            vec![PipelineNormalizer::Lowercase(Lowercase)],
            PipelinePreTokenizer::WhitespaceSplit(WhitespaceSplit),
            PipelineModel::WordLevel(model),
            PipelinePostProcessor::default(),
            None,
            Default::default(),
            None,
            None,
        )
    }

    fn trainer() -> TrainerWrapper {
        WordLevelTrainer::builder()
            .min_frequency(1)
            .show_progress(false)
            .special_tokens(vec![AddedToken::from("<unk>", true)])
            .build()
            .unwrap()
            .into()
    }

    #[test]
    fn trains_through_the_pipeline_without_changing_the_template() {
        let original = template();
        let clone = original.clone();
        let before = original
            .encode("old", &EncodeOptions::no_specials())
            .wait()
            .unwrap();
        let trained = train_tokenizer(
            &original,
            &mut trainer(),
            ["HELLO hello", "World"].into_iter(),
        )
        .unwrap();
        let PipelineModel::WordLevel(model) = trained.get_model() else {
            unreachable!()
        };
        assert!(
            model.token_to_id("hello").is_some(),
            "the new model must learn normalized pre-tokens"
        );
        assert!(model.token_to_id("HELLO").is_none());
        assert!(model.token_to_id("hello hello").is_none());
        assert!(model.token_to_id("old").is_none());
        let encoded = trained
            .encode("HELLO World", &EncodeOptions::no_specials())
            .wait()
            .unwrap();
        assert_eq!(
            trained
                .decode(
                    &encoded[0]
                        .ids()
                        .iter()
                        .map(|id| id.id())
                        .collect::<Vec<_>>(),
                    true
                )
                .unwrap(),
            "hello world"
        );
        assert_eq!(
            original
                .encode("old", &EncodeOptions::no_specials())
                .wait()
                .unwrap(),
            before
        );
        assert_eq!(
            clone
                .encode("old", &EncodeOptions::no_specials())
                .wait()
                .unwrap(),
            before
        );
    }

    fn configured(
        post_processor: PipelinePostProcessor,
        padding: Option<tk_encode::PaddingParams>,
        roles: std::collections::BTreeMap<String, String>,
    ) -> PipelineTokenizer {
        let model = WordLevel::builder()
            .vocab(
                [
                    ("<unk>".into(), 0),
                    ("<bos>".into(), 1),
                    ("<sep>".into(), 2),
                    ("<pad>".into(), 3),
                    ("old".into(), 4),
                ]
                .into(),
            )
            .unk_token("<unk>".into())
            .build()
            .unwrap();
        let mut added = AddedVocabulary::new();
        added
            .add_tokens::<tk_encode::pipeline::NormalizerChain<'_>>(
                ["<unk>", "<bos>", "<sep>", "<pad>"].map(|s| AddedToken::from(s, true)),
                model.get_vocab_size(),
                |t| model.token_to_id(t),
                None,
            )
            .unwrap();
        PipelineTokenizer::from_parts(
            added,
            vec![],
            PipelinePreTokenizer::WhitespaceSplit(WhitespaceSplit),
            PipelineModel::WordLevel(model),
            post_processor,
            None,
            roles,
            padding,
            Some(tk_encode::TruncationParams {
                max_length: 12,
                ..Default::default()
            }),
        )
    }

    fn special_trainer() -> TrainerWrapper {
        WordLevelTrainer::builder()
            .min_frequency(1)
            .show_progress(false)
            .special_tokens(
                ["<sep>", "<pad>", "<unk>", "<bos>"]
                    .map(|s| AddedToken::from(s, true))
                    .into(),
            )
            .build()
            .unwrap()
            .into()
    }

    #[test]
    fn remaps_all_templates_and_padding_by_content_not_existing_id() {
        use tk_encode::pipeline::{Override, Template};
        let run = |ids: &[u32]| ids.iter().map(|&id| (id.into(), 7)).collect();
        let original = configured(
            PipelinePostProcessor {
                single: Template {
                    prefix: run(&[1]),
                    suffix: run(&[2]),
                    ..Default::default()
                },
                pair: Template {
                    prefix: run(&[1]),
                    infix: run(&[2, 1]),
                    suffix: run(&[2]),
                    a_type_id: 4,
                    b_type_id: Some(5),
                },
            },
            Some(tk_encode::PaddingParams {
                pad_id: 3,
                pad_token: "<pad>".into(),
                strategy: tk_encode::PaddingStrategy::Fixed(10),
                ..Default::default()
            }),
            [("bos_token".into(), "<bos>".into())].into(),
        );
        let clone = original.clone();
        let before = original
            .encode(("old", "old"), &EncodeOptions::default())
            .wait()
            .unwrap();
        let trained = train_tokenizer(
            &original,
            &mut special_trainer(),
            ["hello hello world"].into_iter(),
        )
        .unwrap();
        // Old ids 1, 2 and 3 are all still valid but now name DIFFERENT special tokens.
        let post = trained.get_post_processor();
        assert_eq!(post.single.prefix[0].0.id(), 3);
        assert_eq!(post.single.suffix[0].0.id(), 0);
        assert_eq!(post.pair.prefix[0].0.id(), 3);
        assert_eq!(
            post.pair.infix.iter().map(|x| x.0.id()).collect::<Vec<_>>(),
            [0, 3]
        );
        assert_eq!(post.pair.suffix[0].0.id(), 0);
        assert_eq!(trained.get_padding().unwrap().pad_id, 1);
        assert_eq!(trained.get_truncation(), original.get_truncation());
        assert_eq!(trained.get_role_to_token(), original.get_role_to_token());
        let enc = trained
            .encode(("hello", "world"), &EncodeOptions::default())
            .wait()
            .unwrap();
        assert_eq!(
            enc[0].ids().iter().map(|x| x.id()).collect::<Vec<_>>(),
            [3, 4, 0, 3, 5, 0, 1, 1, 1, 1]
        );
        assert_eq!(enc[0].type_ids().unwrap()[..6], [7, 4, 7, 7, 5, 7]);
        let no_padding = EncodeOptions {
            padding: Override::Off,
            ..EncodeOptions::default()
        };
        let special = trained.encode("<bos>", &no_padding).wait().unwrap();
        assert_eq!(special[0].ids()[1].id(), 3);
        assert_eq!(trained.decode(&[3, 4, 0], true).unwrap(), "hello");
        assert_eq!(
            original
                .encode(("old", "old"), &EncodeOptions::default())
                .wait()
                .unwrap(),
            before
        );
        assert_eq!(
            clone
                .encode(("old", "old"), &EncodeOptions::default())
                .wait()
                .unwrap(),
            before
        );
    }

    #[test]
    fn rejects_missing_retained_tokens_without_changing_shared_template() {
        use tk_encode::pipeline::Template;
        let original = configured(
            PipelinePostProcessor {
                single: Template {
                    prefix: vec![(1.into(), 0)].into(),
                    ..Default::default()
                },
                ..Default::default()
            },
            None,
            Default::default(),
        );
        let clone = original.clone();
        let before = original
            .encode("old", &EncodeOptions::default())
            .wait()
            .unwrap();
        let error = train_tokenizer(&original, &mut trainer(), ["new new"].into_iter())
            .err()
            .unwrap();
        assert!(error.to_string().contains("<bos>"), "{error}");
        assert_eq!(
            original
                .encode("old", &EncodeOptions::default())
                .wait()
                .unwrap(),
            before
        );
        assert_eq!(
            clone
                .encode("old", &EncodeOptions::default())
                .wait()
                .unwrap(),
            before
        );
        let missing_pad = configured(
            Default::default(),
            Some(tk_encode::PaddingParams {
                pad_id: 3,
                pad_token: "<pad>".into(),
                ..Default::default()
            }),
            Default::default(),
        );
        assert!(
            train_tokenizer(&missing_pad, &mut trainer(), ["new"].into_iter())
                .err()
                .unwrap()
                .to_string()
                .contains("<pad>")
        );
        let missing_role = configured(
            Default::default(),
            None,
            [("bos_token".into(), "<bos>".into())].into(),
        );
        assert!(
            train_tokenizer(&missing_role, &mut trainer(), ["new"].into_iter())
                .err()
                .unwrap()
                .to_string()
                .contains("bos_token")
        );
    }

    #[test]
    fn fresh_training_does_not_replay_or_split_on_old_normalized_added_tokens() {
        use tk_encode::normalizers::prepend::Prepend;
        use tk_encode::pipeline::{NormalizerChain, Template};
        let normalizers = vec![PipelineNormalizer::Prepend(Prepend::new("x".into()))];
        let mut old_added = AddedVocabulary::new();
        old_added
            .add_tokens(
                [AddedToken::from("mark", false)],
                2,
                |_| None,
                Some(&NormalizerChain(&normalizers)),
            )
            .unwrap();
        let make = |post| {
            PipelineTokenizer::from_parts(
                old_added.clone(),
                normalizers.clone(),
                PipelinePreTokenizer::None,
                PipelineModel::WordLevel(
                    WordLevel::builder()
                        .vocab([("<unk>".into(), 0), ("old".into(), 1)].into())
                        .unk_token("<unk>".into())
                        .build()
                        .unwrap(),
                ),
                post,
                None,
                Default::default(),
                None,
                None,
            )
        };
        let original = make(Default::default());
        let trained =
            train_tokenizer(&original, &mut trainer(), ["mark mark"].into_iter()).unwrap();
        let PipelineModel::WordLevel(model) = trained.get_model() else {
            unreachable!()
        };
        assert!(
            model.token_to_id("xmark mark").is_some(),
            "old added token must not carve up the corpus"
        );
        assert!(
            trained
                .get_added_vocabulary()
                .token_to_id("xxmark")
                .is_none()
        );
        assert!(
            trained
                .get_added_vocabulary()
                .token_to_id("xmark")
                .is_none()
        );
        assert_eq!(
            original.get_added_vocabulary().token_to_id("xmark"),
            Some(2)
        );
        let referenced = make(PipelinePostProcessor {
            single: Template {
                prefix: vec![(2.into(), 0)].into(),
                ..Default::default()
            },
            ..Default::default()
        });
        assert!(
            train_tokenizer(&referenced, &mut trainer(), ["mark"].into_iter())
                .err()
                .unwrap()
                .to_string()
                .contains("original spelling")
        );
    }

    #[test]
    fn rejects_a_trained_vocabulary_without_the_configured_unknown_token() {
        let mut trainer = WordLevelTrainer::builder()
            .show_progress(false)
            .min_frequency(1)
            .build()
            .unwrap()
            .into();
        let error = train_tokenizer(&template(), &mut trainer, ["hello"].into_iter())
            .err()
            .unwrap();
        assert!(error.to_string().contains("unknown token"), "{error}");
    }

    #[test]
    fn remaps_raw_added_only_references_and_normalizes_new_tokens_once() {
        use tk_encode::normalizers::prepend::Prepend;
        use tk_encode::pipeline::{NormalizerChain, Template};
        let normalizers = vec![PipelineNormalizer::Prepend(Prepend::new("x".into()))];
        let model = WordLevel::builder()
            .vocab([("<unk>".into(), 0), ("old".into(), 1)].into())
            .unk_token("<unk>".into())
            .build()
            .unwrap();
        let mut added = AddedVocabulary::new();
        added
            .add_tokens(
                [AddedToken::from("<bos>", true)],
                2,
                |t| model.token_to_id(t),
                Some(&NormalizerChain(&normalizers)),
            )
            .unwrap();
        let original = PipelineTokenizer::from_parts(
            added,
            normalizers,
            PipelinePreTokenizer::None,
            PipelineModel::WordLevel(model),
            PipelinePostProcessor {
                single: Template {
                    prefix: vec![(2.into(), 0)].into(),
                    ..Default::default()
                },
                ..Default::default()
            },
            None,
            Default::default(),
            None,
            None,
        );
        let mut trainer = WordLevelTrainer::builder()
            .min_frequency(1)
            .show_progress(false)
            .special_tokens(vec![
                AddedToken::from("<bos>", true),
                AddedToken::from("<unk>", true),
                AddedToken::from("mark", false),
            ])
            .build()
            .unwrap()
            .into();
        let trained = train_tokenizer(&original, &mut trainer, ["hello"].into_iter()).unwrap();
        assert_eq!(trained.get_post_processor().single.prefix[0].0.id(), 0);
        assert_eq!(trained.get_added_vocabulary().token_to_id("xmark"), Some(2));
        assert_eq!(trained.get_added_vocabulary().token_to_id("xxmark"), None);
        let encoded = trained
            .encode("mark", &EncodeOptions::no_specials())
            .wait()
            .unwrap();
        assert_eq!(encoded[0].ids(), [2]);
    }

    #[test]
    fn metaspace_first_uses_sequence_start_and_is_not_applied_to_added_tokens() {
        use tk_encode::normalizers::metaspace::{MetaspaceNormalizer, PrependBehavior};
        let model = WordLevel::builder()
            .vocab([("<unk>".into(), 0)].into())
            .unk_token("<unk>".into())
            .build()
            .unwrap();
        let original = PipelineTokenizer::from_parts(
            Default::default(),
            vec![PipelineNormalizer::Metaspace(MetaspaceNormalizer::new(
                '▁',
                PrependBehavior::First,
                false,
            ))],
            PipelinePreTokenizer::None,
            PipelineModel::WordLevel(model),
            Default::default(),
            None,
            Default::default(),
            None,
            None,
        );
        let mut trainer = WordLevelTrainer::builder()
            .min_frequency(1)
            .show_progress(false)
            .special_tokens(vec![
                AddedToken::from("<unk>", true),
                AddedToken::from("<s>", true).normalized(true),
            ])
            .build()
            .unwrap()
            .into();
        let trained =
            train_tokenizer(&original, &mut trainer, ["<s>hello", "world"].into_iter()).unwrap();
        let PipelineModel::WordLevel(model) = trained.get_model() else {
            unreachable!()
        };
        assert!(model.token_to_id("▁<s>hello").is_some());
        assert!(model.token_to_id("▁world").is_some());
        assert_eq!(trained.get_added_vocabulary().token_to_id("<s>"), Some(1));
        assert_eq!(trained.get_added_vocabulary().token_to_id("▁<s>"), None);
    }

    #[cfg(feature = "bpe")]
    #[test]
    fn mismatched_trainer_does_not_consume_input_or_replace_its_counts() {
        use tk_train::{Trainer, trainers::BpeTrainer};
        let mut trainer: TrainerWrapper = BpeTrainer::builder().show_progress(false).build().into();
        trainer
            .feed(["already fed"].into_iter(), |s| Ok(vec![s.to_owned()]))
            .unwrap();
        let before = serde_json::to_value(&trainer).unwrap();
        let sequences =
            std::iter::from_fn(|| -> Option<&str> { panic!("mismatch must not consume input") });
        let error = train_tokenizer(&template(), &mut trainer, sequences)
            .err()
            .unwrap();
        assert!(error.to_string().contains("families"));
        assert_eq!(serde_json::to_value(&trainer).unwrap(), before);
    }

    #[test]
    fn empty_sequences_do_not_enter_the_vocabulary() {
        // `None` maps every sequence, including "", to one span: without a guard the empty
        // sequence feeds "" as a training word, which then displaces real words under a small
        // `vocab_size` ("" sorts before every word).
        let model = WordLevel::builder()
            .vocab([("<unk>".into(), 0)].into())
            .unk_token("<unk>".into())
            .build()
            .unwrap();
        let original = PipelineTokenizer::from_parts(
            AddedVocabulary::new(),
            vec![],
            PipelinePreTokenizer::None,
            PipelineModel::WordLevel(model),
            Default::default(),
            None,
            Default::default(),
            None,
            None,
        );
        let clone = original.clone();
        let before = original
            .encode("hello", &EncodeOptions::no_specials())
            .wait()
            .unwrap();
        let mut trainer: TrainerWrapper = WordLevelTrainer::builder()
            .min_frequency(1)
            .show_progress(false)
            .vocab_size(3)
            .special_tokens(vec![AddedToken::from("<unk>", true)])
            .build()
            .unwrap()
            .into();
        let trained =
            train_tokenizer(&original, &mut trainer, ["hello", "", "world"].into_iter()).unwrap();
        let PipelineModel::WordLevel(model) = trained.get_model() else {
            unreachable!()
        };
        let vocab = model.get_vocab();
        assert!(vocab.contains_key("hello"), "{vocab:?}");
        assert!(vocab.contains_key("world"), "{vocab:?}");
        assert!(!vocab.contains_key(""), "{vocab:?}");
        assert_eq!(
            original
                .encode("hello", &EncodeOptions::no_specials())
                .wait()
                .unwrap(),
            before
        );
        assert_eq!(
            clone
                .encode("hello", &EncodeOptions::no_specials())
                .wait()
                .unwrap(),
            before
        );
    }

    #[test]
    fn normalized_added_alias_resolves_through_the_added_matcher() {
        use tk_encode::pipeline::{PipelinePostProcessor, Template};
        // Lowercase template: the trainer registers "The" (normalized), the model learns "the".
        // A retained reference to "the" must follow the added matcher, like encode does.
        let model = WordLevel::builder()
            .vocab([("<unk>".into(), 0), ("the".into(), 1)].into())
            .unk_token("<unk>".into())
            .build()
            .unwrap();
        let mut added = AddedVocabulary::new();
        added
            .add_tokens::<tk_encode::pipeline::NormalizerChain<'_>>(
                [AddedToken::from("<unk>", true)],
                model.get_vocab_size(),
                |t| model.token_to_id(t),
                None,
            )
            .unwrap();
        let original = PipelineTokenizer::from_parts(
            added,
            vec![PipelineNormalizer::Lowercase(Lowercase)],
            PipelinePreTokenizer::WhitespaceSplit(WhitespaceSplit),
            PipelineModel::WordLevel(model),
            PipelinePostProcessor {
                single: Template {
                    prefix: vec![(1.into(), 0)].into(),
                    ..Default::default()
                },
                ..Default::default()
            },
            None,
            [("bos_token".into(), "the".into())].into(),
            Some(tk_encode::PaddingParams {
                pad_id: 1,
                pad_token: "the".into(),
                ..Default::default()
            }),
            None,
        );
        let mut trainer: TrainerWrapper = WordLevelTrainer::builder()
            .min_frequency(1)
            .show_progress(false)
            .special_tokens(vec![
                AddedToken::from("<unk>", true),
                AddedToken::from("The", false),
            ])
            .build()
            .unwrap()
            .into();
        let trained = train_tokenizer(&original, &mut trainer, ["the the"].into_iter()).unwrap();
        let PipelineModel::WordLevel(trained_model) = trained.get_model() else {
            unreachable!()
        };
        // "The" normalizes to "the" and reuses the model id, so the added matcher owns "the".
        let added_id = trained.get_added_vocabulary().token_to_id("the").unwrap();
        assert_eq!(added_id, trained_model.token_to_id("The").unwrap());
        assert_eq!(
            trained.get_post_processor().single.prefix[0].0.id(),
            added_id
        );
        assert_eq!(trained.get_padding().unwrap().pad_id, added_id);
        assert_eq!(trained.get_role_to_token().get("bos_token").unwrap(), "the");
        let encoded = trained
            .encode("the", &EncodeOptions::no_specials())
            .wait()
            .unwrap();
        assert_eq!(encoded[0].ids(), [added_id]);
    }

    #[test]
    fn empty_training_corpus_returns_an_error() {
        let mut trainer: TrainerWrapper = WordLevelTrainer::builder()
            .show_progress(false)
            .build()
            .unwrap()
            .into();
        let error = train_tokenizer(&template(), &mut trainer, std::iter::empty::<&str>())
            .err()
            .unwrap();
        assert!(error.to_string().contains("empty"), "{error}");
    }

    #[test]
    fn normalizer_erased_corpus_returns_an_empty_vocabulary_error() {
        use tk_encode::normalizers::replace::Replace;
        let model = WordLevel::builder()
            .vocab([("<unk>".into(), 0)].into())
            .unk_token("<unk>".into())
            .build()
            .unwrap();
        let original = PipelineTokenizer::from_parts(
            AddedVocabulary::new(),
            vec![PipelineNormalizer::Replace(
                Replace::new("hello", "").unwrap(),
            )],
            PipelinePreTokenizer::None,
            PipelineModel::WordLevel(model),
            Default::default(),
            None,
            Default::default(),
            None,
            None,
        );
        let mut trainer: TrainerWrapper = WordLevelTrainer::builder()
            .min_frequency(1)
            .show_progress(false)
            .build()
            .unwrap()
            .into();
        let error = train_tokenizer(&original, &mut trainer, ["hello"].into_iter())
            .err()
            .unwrap();
        assert!(error.to_string().contains("empty"), "{error}");
    }
}

#[cfg(feature = "bpe")]
// With only `bpe` compiled in, `PipelineModel` has a single variant, so the
// `let ... else` assertions below are irrefutable. They stay `let-else` so the
// all-features build still rejects a wrong model family.
#[allow(irrefutable_let_patterns)]
mod bpe {
    use tk_encode::{
        models::bpe::{BpeConfig, PipelineBPE},
        normalizers::byte_level::ByteLevel,
        pipeline::{EncodeOptions, PipelineModel, PipelinePreTokenizer, PipelineTokenizer},
        vocab::bucket_added_vocabulary::AddedToken,
    };
    use tk_train::{train_tokenizer, trainers::BpeTrainer};

    fn pipeline(config: BpeConfig) -> PipelineTokenizer {
        PipelineTokenizer::from_parts(
            Default::default(),
            vec![],
            PipelinePreTokenizer::None,
            PipelineModel::BPE(PipelineBPE::from_config(config).unwrap()),
            Default::default(),
            None,
            Default::default(),
            None,
            None,
        )
    }

    fn ids(tokenizer: &PipelineTokenizer, input: &str) -> Vec<u32> {
        tokenizer
            .encode(input, &EncodeOptions::no_specials())
            .wait()
            .unwrap()[0]
            .ids()
            .iter()
            .map(|x| x.id())
            .collect()
    }

    #[test]
    fn byte_level_projects_utf8_after_preprocessing_and_keeps_held_out_bytes() {
        let alphabet = ByteLevel::alphabet();
        let original = pipeline(BpeConfig {
            byte_level: true,
            vocab: alphabet
                .iter()
                .enumerate()
                .map(|(id, c)| (c.to_string(), id as u32))
                .collect(),
            cache_capacity: 23,
            ..Default::default()
        });
        let clone = original.clone();
        let text = " café 東京 🙂";
        let before = ids(&original, text);
        let mut trainer = BpeTrainer::builder()
            .show_progress(false)
            .vocab_size(300)
            .initial_alphabet(alphabet.into_iter().collect())
            .build()
            .into();
        let trained = train_tokenizer(
            &original,
            &mut trainer,
            ["café café", " 東京 東京"].into_iter(),
        )
        .unwrap();
        let PipelineModel::BPE(model) = trained.get_model() else {
            unreachable!()
        };
        assert!(model.is_byte_level());
        assert_eq!(model.to_config().unwrap().cache_capacity, 23);
        for text in [text, "a 🦀 Z", " Ġ ", ""] {
            assert_eq!(trained.decode(&ids(&trained, text), false).unwrap(), text);
        }
        assert_eq!(ids(&original, text), before);
        assert_eq!(ids(&clone, text), before);
        // New merges must be learned in the printable byte alphabet, not from Unicode characters.
        assert!(ids(&trained, "café café").len() < "café café".len());
    }

    #[test]
    fn preserves_bpe_runtime_options_and_uses_trainer_affixes() {
        let original = pipeline(BpeConfig {
            vocab: [("<unk>".into(), 0), ("a".into(), 1)].into(),
            unk_token: Some("<unk>".into()),
            fuse_unk: true,
            cache_capacity: 19,
            continuing_subword_prefix: Some("##".into()),
            end_of_word_suffix: Some("</w>".into()),
            ..Default::default()
        });
        let mut trainer = BpeTrainer::builder()
            .show_progress(false)
            .vocab_size(24)
            .continuing_subword_prefix("##".into())
            .end_of_word_suffix("</w>".into())
            .special_tokens(vec![AddedToken::from("<unk>", true)])
            .build()
            .into();
        let trained =
            train_tokenizer(&original, &mut trainer, ["hello hello"].into_iter()).unwrap();
        let PipelineModel::BPE(model) = trained.get_model() else {
            unreachable!()
        };
        let config = model.to_config().unwrap();
        assert_eq!(config.unk_token.as_deref(), Some("<unk>"));
        assert!(config.fuse_unk);
        assert_eq!(config.cache_capacity, 19);
        assert_eq!(config.continuing_subword_prefix.as_deref(), Some("##"));
        assert_eq!(config.end_of_word_suffix.as_deref(), Some("</w>"));
        // The existing affixed encoder emits unknowns per character, even with fuse_unk.
        // Training must preserve its behavior rather than changing that inference path.
        assert_eq!(ids(&trained, "?¿"), ids(&original, "?¿"));
    }

    #[test]
    fn preserves_fused_unknowns_without_affixes() {
        let original = pipeline(BpeConfig {
            vocab: [("<unk>".into(), 0)].into(),
            unk_token: Some("<unk>".into()),
            fuse_unk: true,
            ..Default::default()
        });
        assert_eq!(ids(&original, "?¿"), [0]);
        let mut trainer = BpeTrainer::builder()
            .show_progress(false)
            .special_tokens(vec![AddedToken::from("<unk>", true)])
            .build()
            .into();
        let trained = train_tokenizer(&original, &mut trainer, ["hello"].into_iter()).unwrap();
        assert_eq!(ids(&trained, "?¿"), [0]);
    }

    #[test]
    fn empty_training_corpus_returns_an_error_without_panicking() {
        let original = pipeline(BpeConfig {
            vocab: [("a".into(), 0)].into(),
            ..Default::default()
        });
        let mut trainer = BpeTrainer::builder().show_progress(false).build().into();
        let error = train_tokenizer(&original, &mut trainer, std::iter::empty::<&str>())
            .err()
            .unwrap();
        assert!(error.to_string().contains("empty"), "{error}");
    }

    #[test]
    fn preserves_ignore_merges_even_when_the_template_vocab_folds_either_way() {
        // A chars-only template folds under either policy, so the writer's derived flag cannot
        // tell `false` from `true`. The trained model must keep the template's configured policy:
        // `false` must not silently broaden to whole-word folding of fresh specials.
        for ignore_merges in [false, true] {
            let original = pipeline(BpeConfig {
                vocab: [("a".into(), 0), ("b".into(), 1)].into(),
                ignore_merges,
                ..Default::default()
            });
            let PipelineModel::BPE(template_model) = original.get_model() else {
                unreachable!()
            };
            assert_eq!(
                template_model.ignore_merges(),
                ignore_merges,
                "the template must keep its configured policy"
            );
            let mut trainer = BpeTrainer::builder()
                .show_progress(false)
                .vocab_size(10)
                .special_tokens(vec![AddedToken::from("<unk>", true)])
                .build()
                .into();
            let trained =
                train_tokenizer(&original, &mut trainer, ["a a b b"].into_iter()).unwrap();
            let PipelineModel::BPE(model) = trained.get_model() else {
                unreachable!()
            };
            assert_eq!(
                model.ignore_merges(),
                ignore_merges,
                "fresh training must preserve ignore_merges={ignore_merges}"
            );
        }
    }

    #[test]
    fn insufficient_byte_alphabet_is_an_error_not_a_fallback_to_character_bpe() {
        let original = pipeline(BpeConfig {
            byte_level: true,
            vocab: ByteLevel::alphabet()
                .iter()
                .enumerate()
                .map(|(id, c)| (c.to_string(), id as u32))
                .collect(),
            ..Default::default()
        });
        let clone = original.clone();
        let before = ids(&original, "café");
        let mut trainer = BpeTrainer::builder().show_progress(false).build().into();
        let error = train_tokenizer(&original, &mut trainer, ["hello"].into_iter())
            .err()
            .unwrap();
        assert!(error.to_string().contains("vocabulary"), "{error}");
        assert_eq!(ids(&clone, "café"), before);
    }

    #[test]
    fn bpe_unknown_token_must_be_in_new_model_vocabulary() {
        let original = pipeline(BpeConfig {
            vocab: [("<unk>".into(), 0)].into(),
            unk_token: Some("<unk>".into()),
            ..Default::default()
        });
        let mut trainer = BpeTrainer::builder().show_progress(false).build().into();
        let error = train_tokenizer(&original, &mut trainer, ["hello"].into_iter())
            .err()
            .unwrap();
        assert!(error.to_string().contains("<unk>"), "{error}");
    }

    #[test]
    fn remaps_byte_level_model_spellings_and_added_tokens_reusing_model_ids() {
        use tk_encode::{
            pipeline::{PipelinePostProcessor, Template},
            vocab::bucket_added_vocabulary::AddedVocabulary,
        };
        let mut alphabet: Vec<_> = ByteLevel::alphabet().into_iter().collect();
        alphabet.sort_unstable();
        let space = alphabet.iter().position(|&c| c == 'Ġ').unwrap();
        alphabet.swap(0, space);
        let mut vocab: ahash::AHashMap<_, _> = alphabet
            .iter()
            .enumerate()
            .map(|(id, c)| (c.to_string(), id as u32))
            .collect();
        vocab.insert("<bos>".into(), 256);
        let model = PipelineBPE::from_config(BpeConfig {
            vocab,
            byte_level: true,
            ..Default::default()
        })
        .unwrap();
        let config = model.to_config().unwrap();
        let mut added = AddedVocabulary::new();
        added
            .add_tokens::<tk_encode::pipeline::NormalizerChain<'_>>(
                [AddedToken::from("<bos>", true)],
                config.vocab.len(),
                |t| config.vocab.get(t).copied(),
                None,
            )
            .unwrap();
        let original = PipelineTokenizer::from_parts(
            added,
            vec![],
            PipelinePreTokenizer::None,
            PipelineModel::BPE(model),
            PipelinePostProcessor {
                single: Template {
                    prefix: vec![(0.into(), 0), (256.into(), 0)].into(),
                    ..Default::default()
                },
                ..Default::default()
            },
            None,
            Default::default(),
            None,
            None,
        );
        let mut trainer = BpeTrainer::builder()
            .show_progress(false)
            .vocab_size(270)
            .initial_alphabet(alphabet.into_iter().collect())
            .special_tokens(vec![AddedToken::from("<bos>", true)])
            .build()
            .into();
        let trained = train_tokenizer(&original, &mut trainer, ["hello"].into_iter()).unwrap();
        let PipelineModel::BPE(model) = trained.get_model() else {
            unreachable!()
        };
        let config = model.to_config().unwrap();
        let prefix = &trained.get_post_processor().single.prefix;
        assert_eq!(prefix[0].0.id(), config.vocab["Ġ"]);
        assert_ne!(prefix[0].0.id(), 0);
        assert_eq!(prefix[1].0.id(), 0);
        assert_eq!(trained.get_added_vocabulary().token_to_id("<bos>"), Some(0));
    }
}

#[cfg(feature = "wordpiece")]
mod wordpiece {
    use tk_encode::{
        models::wordpiece::WordPiece,
        pipeline::{EncodeOptions, PipelineModel, PipelinePreTokenizer, PipelineTokenizer},
        vocab::bucket_added_vocabulary::AddedToken,
    };
    use tk_train::{train_tokenizer, trainers::WordPieceTrainer};

    #[test]
    fn trains_wordpiece_preserving_unknown_limit_and_decoder() {
        let model = WordPiece::builder()
            .vocab(
                [("<unk>".into(), 0)]
                    .into_iter()
                    .collect::<ahash::AHashMap<_, _>>(),
            )
            .unk_token("<unk>".into())
            .max_input_chars_per_word(4)
            .build()
            .unwrap();
        let decoder = tk_encode::DecoderRuntime::WordPiece(
            tk_encode::decoders::wordpiece::WordPiece::new("##".into(), false),
        );
        let original = PipelineTokenizer::from_parts(
            Default::default(),
            vec![],
            PipelinePreTokenizer::None,
            PipelineModel::WordPiece(model.try_into().unwrap()),
            Default::default(),
            Some(decoder),
            Default::default(),
            None,
            None,
        );
        let mut trainer = WordPieceTrainer::builder()
            .show_progress(false)
            .vocab_size(24)
            .special_tokens(vec![AddedToken::from("<unk>", true)])
            .build()
            .into();
        let trained = train_tokenizer(
            &original,
            &mut trainer,
            ["hello", "help", "hell"].into_iter(),
        )
        .unwrap();
        let PipelineModel::WordPiece(model) = trained.get_model() else {
            unreachable!()
        };
        assert_eq!(model.unk_token(), Some("<unk>"));
        assert_eq!(model.max_input_chars_per_word(), 4);
        assert_eq!(model.continuing_subword_prefix(), "##");
        let enc = trained
            .encode("hello", &EncodeOptions::no_specials())
            .wait()
            .unwrap();
        assert_eq!(enc[0].ids(), [0]);
        let enc = trained
            .encode("hell", &EncodeOptions::no_specials())
            .wait()
            .unwrap();
        let ids = enc[0].ids().iter().map(|x| x.id()).collect::<Vec<_>>();
        assert_eq!(trained.decode(&ids, false).unwrap(), "hell");
        assert!(trained.get_decoder().is_some());
    }

    #[test]
    fn default_trainer_prefix_seeds_the_new_wordpiece_config() {
        use tk_encode::pre_tokenizers::whitespace::WhitespaceSplit;
        let model = WordPiece::builder()
            .vocab(
                [("<unk>".into(), 0)]
                    .into_iter()
                    .collect::<ahash::AHashMap<_, _>>(),
            )
            .unk_token("<unk>".into())
            .build()
            .unwrap();
        assert_eq!(model.continuing_subword_prefix, "##");
        let original = PipelineTokenizer::from_parts(
            Default::default(),
            vec![],
            PipelinePreTokenizer::WhitespaceSplit(WhitespaceSplit),
            PipelineModel::WordPiece(model.try_into().unwrap()),
            Default::default(),
            None,
            Default::default(),
            None,
            None,
        );
        // `WordPieceTrainer::default` carries no prefix, unlike the builder's "##".
        let mut trainer = WordPieceTrainer::default();
        trainer.set_special_tokens(vec![AddedToken::from("<unk>", true)]);
        trainer.set_vocab_size(8);
        let mut trainer: tk_train::TrainerWrapper = trainer.into();
        let trained = train_tokenizer(
            &original,
            &mut trainer,
            ["hellohello hellohello"].into_iter(),
        )
        .unwrap();
        let PipelineModel::WordPiece(model) = trained.get_model() else {
            unreachable!()
        };
        // Unprefixed learned pieces need an unprefixed runtime: the held-out compound
        // must encode without the unknown token.
        assert_eq!(model.continuing_subword_prefix(), "");
        let unk = model
            .vocab()
            .into_iter()
            .find(|(token, _)| token == "<unk>")
            .map(|(_, id)| id)
            .unwrap();
        let encoded = trained
            .encode("hellohello", &EncodeOptions::no_specials())
            .wait()
            .unwrap();
        assert!(!encoded[0].is_empty());
        assert!(
            encoded[0].ids().iter().all(|id| id.id() != unk),
            "{:?}",
            encoded[0].ids()
        );
    }

    #[test]
    fn empty_training_corpus_returns_an_error() {
        let model = WordPiece::builder()
            .vocab(
                [("<unk>".into(), 0)]
                    .into_iter()
                    .collect::<ahash::AHashMap<_, _>>(),
            )
            .unk_token("<unk>".into())
            .build()
            .unwrap();
        let original = PipelineTokenizer::from_parts(
            Default::default(),
            vec![],
            PipelinePreTokenizer::None,
            PipelineModel::WordPiece(model.try_into().unwrap()),
            Default::default(),
            None,
            Default::default(),
            None,
            None,
        );
        let mut trainer: tk_train::TrainerWrapper = WordPieceTrainer::builder()
            .show_progress(false)
            .build()
            .into();
        let error = train_tokenizer(&original, &mut trainer, std::iter::empty::<&str>())
            .err()
            .unwrap();
        assert!(error.to_string().contains("empty"), "{error}");
    }
}

#[cfg(feature = "unigram")]
mod unigram {
    use tk_encode::{
        models::unigram::Unigram,
        pipeline::{EncodeOptions, PipelineModel, PipelinePreTokenizer, PipelineTokenizer},
        vocab::bucket_added_vocabulary::AddedToken,
    };
    use tk_train::{TrainerWrapper, train_tokenizer, trainers::UnigramTrainer};

    fn pipeline(byte_fallback: bool) -> PipelineTokenizer {
        PipelineTokenizer::from_parts(
            Default::default(),
            vec![],
            PipelinePreTokenizer::None,
            PipelineModel::Unigram(
                Unigram::from(vec![("<unk>".into(), 0.0)], Some(0), byte_fallback).unwrap(),
            ),
            Default::default(),
            None,
            Default::default(),
            None,
            None,
        )
    }

    fn trainer() -> TrainerWrapper {
        UnigramTrainer::builder()
            .show_progress(false)
            .vocab_size(16)
            .unk_token(Some("<unk>".into()))
            .special_tokens(vec![AddedToken::from("<s>", true)])
            .build()
            .unwrap()
            .into()
    }

    #[test]
    fn trains_unigram_and_registers_specials_with_the_new_ids() {
        let original = pipeline(false);
        let trained = train_tokenizer(
            &original,
            &mut trainer(),
            ["hello hello", "help", "world"].into_iter(),
        )
        .unwrap();
        let PipelineModel::Unigram(model) = trained.get_model() else {
            unreachable!()
        };
        assert_eq!(
            trained.get_added_vocabulary().token_to_id("<s>"),
            model.token_to_id("<s>")
        );
        let encoded = trained
            .encode("hello", &EncodeOptions::no_specials())
            .wait()
            .unwrap();
        assert!(!encoded[0].is_empty());
        assert!(
            encoded[0]
                .ids()
                .iter()
                .all(|id| id.id() as usize != model.unk_id().unwrap())
        );
        let special = trained
            .encode("<s>", &EncodeOptions::no_specials())
            .wait()
            .unwrap();
        assert_eq!(special[0].ids(), [model.token_to_id("<s>").unwrap()]);
        assert_eq!(
            trained
                .decode(&[model.token_to_id("<s>").unwrap()], true)
                .unwrap(),
            ""
        );
    }

    #[test]
    fn byte_fallback_is_explicitly_rejected_before_feeding() {
        let mut trainer = trainer();
        let before = serde_json::to_value(&trainer).unwrap();
        let sequences = std::iter::from_fn(|| -> Option<&str> {
            panic!("unsupported options must not consume input")
        });
        let error = train_tokenizer(&pipeline(true), &mut trainer, sequences)
            .err()
            .unwrap();
        assert!(error.to_string().contains("Unigram byte fallback"));
        assert_eq!(serde_json::to_value(&trainer).unwrap(), before);
    }

    #[test]
    fn empty_training_corpus_returns_an_error() {
        let mut trainer: TrainerWrapper = UnigramTrainer::builder()
            .show_progress(false)
            .vocab_size(16)
            .build()
            .unwrap()
            .into();
        let error = train_tokenizer(&pipeline(false), &mut trainer, std::iter::empty::<&str>())
            .err()
            .unwrap();
        assert!(error.to_string().contains("empty"), "{error}");
    }
}
