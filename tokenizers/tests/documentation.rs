use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::normalizers::{Sequence, Strip, NFC};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, TokenizerBuilder};
use tokenizers::{DecoderWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper};
use tokenizers::{Tokenizer, TokenizerImpl};

#[test]
fn train_tokenizer() {
    let vocab_size: usize = 100;
    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_normalizer(Some(Sequence::new(vec![
            Strip::new(true, true).into(),
            NFC.into(),
        ])))
        .with_pre_tokenizer(Some(ByteLevel::default()))
        .with_post_processor(Some(ByteLevel::default()))
        .with_decoder(Some(ByteLevel::default()))
        .build()
        .unwrap();

    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(false)
        .vocab_size(vocab_size)
        .min_frequency(0)
        .special_tokens(vec![
            AddedToken::from(String::from("<s>"), true),
            AddedToken::from(String::from("<pad>"), true),
            AddedToken::from(String::from("</s>"), true),
            AddedToken::from(String::from("<unk>"), true),
            AddedToken::from(String::from("<mask>"), true),
        ])
        .build();

    let pretty = true;
    tokenizer
        .train_from_files(&mut trainer, vec!["data/small.txt".to_string()])
        .unwrap()
        .save("data/tokenizer.json", pretty)
        .unwrap();
}

#[test]
fn load_tokenizer() {
    let tokenizer = Tokenizer::from_file("data/roberta.json").unwrap();

    let example = "This is an example";
    let ids = vec![713, 16, 41, 1246];
    let tokens = vec!["This", "Ġis", "Ġan", "Ġexample"];

    let encodings = tokenizer.encode(example, false).unwrap();

    assert_eq!(encodings.get_ids(), ids);
    assert_eq!(encodings.get_tokens(), tokens);

    let decoded = tokenizer.decode(ids, false).unwrap();
    assert_eq!(decoded, example);
}

#[test]
#[ignore]
fn quicktour_slow_train() -> tokenizers::Result<()> {
    // START quicktour_init_tokenizer
    use tokenizers::models::bpe::BPE;

    let mut tokenizer: TokenizerImpl<
        BPE,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    > = TokenizerImpl::new(
        BPE::builder()
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap(),
    );
    // END quicktour_init_tokenizer
    // START quicktour_init_trainer
    use tokenizers::models::bpe::BpeTrainer;

    let mut trainer = BpeTrainer::builder()
        .special_tokens(vec![
            AddedToken::from("[UNK]", true),
            AddedToken::from("[CLS]", true),
            AddedToken::from("[SEP]", true),
            AddedToken::from("[PAD]", true),
            AddedToken::from("[MASK]", true),
        ])
        .build();
    // END quicktour_init_trainer
    // START quicktour_init_pretok
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    tokenizer.with_pre_tokenizer(Whitespace::default());
    // END quicktour_init_pretok

    // START quicktour_train
    let files = vec![
        "data/wikitext-103-raw/wiki.train.raw".into(),
        "data/wikitext-103-raw/wiki.test.raw".into(),
        "data/wikitext-103-raw/wiki.valid.raw".into(),
    ];
    tokenizer.train_from_files(&mut trainer, files)?;
    // END quicktour_train
    // START quicktour_save
    tokenizer.save("data/tokenizer-wiki.json", false)?;
    // END quicktour_save

    Ok(())
}

#[test]
fn quicktour() -> tokenizers::Result<()> {
    // START quicktour_reload_tokenizer
    let mut tokenizer = Tokenizer::from_file("data/tokenizer-wiki.json")?;
    // END quicktour_reload_tokenizer
    // START quicktour_encode
    let output = tokenizer.encode("Hello, y'all! How are you 😁 ?", true)?;
    // END quicktour_encode
    // START quicktour_print_tokens
    println!("{:?}", output.get_tokens());
    // ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?",]
    // END quicktour_print_tokens
    assert_eq!(
        output.get_tokens(),
        ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?",]
    );
    // START quicktour_print_ids
    println!("{:?}", output.get_ids());
    // [27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35]
    // END quicktour_print_ids
    assert_eq!(
        output.get_ids(),
        [27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35]
    );
    // START quicktour_print_offsets
    println!("{:?}", output.get_offsets()[9]);
    // (26, 30)
    // END quicktour_print_offsets
    assert_eq!(output.get_offsets()[9], (26, 30));
    // START quicktour_use_offsets
    let sentence = "Hello, y'all! How are you 😁 ?";
    println!("{}", &sentence[26..30]);
    // "😁"
    // END quicktour_use_offsets
    // START quicktour_check_sep
    println!("{}", tokenizer.token_to_id("[SEP]").unwrap());
    // 2
    // END quicktour_check_sep
    assert_eq!(tokenizer.token_to_id("[SEP]"), Some(2));
    // START quicktour_init_template_processing
    use tokenizers::processors::template::TemplateProcessing;

    let special_tokens = vec![
        ("[CLS]", tokenizer.token_to_id("[CLS]").unwrap()),
        ("[SEP]", tokenizer.token_to_id("[SEP]").unwrap()),
    ];
    tokenizer.with_post_processor(
        TemplateProcessing::builder()
            .try_single("[CLS] $A [SEP]")
            .unwrap()
            .try_pair("[CLS] $A [SEP] $B:1 [SEP]:1")
            .unwrap()
            .special_tokens(special_tokens)
            .build()?,
    );
    // END quicktour_init_template_processing
    // START quicktour_print_special_tokens
    let output = tokenizer.encode("Hello, y'all! How are you 😁 ?", true)?;
    println!("{:?}", output.get_tokens());
    // ["[CLS]", "Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?", "[SEP]"]
    // END quicktour_print_special_tokens
    assert_eq!(
        output.get_tokens(),
        ["[CLS]", "Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?", "[SEP]"]
    );
    // START quicktour_print_special_tokens_pair
    let output = tokenizer.encode(("Hello, y'all!", "How are you 😁 ?"), true)?;
    println!("{:?}", output.get_tokens());
    // ["[CLS]", "Hello", ",", "y", "'", "all", "!", "[SEP]", "How", "are", "you", "[UNK]", "?", "[SEP]"]
    // END quicktour_print_special_tokens_pair
    assert_eq!(
        output.get_tokens(),
        [
            "[CLS]", "Hello", ",", "y", "'", "all", "!", "[SEP]", "How", "are", "you", "[UNK]",
            "?", "[SEP]"
        ]
    );
    // START quicktour_print_type_ids
    println!("{:?}", output.get_type_ids());
    // [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    // END quicktour_print_type_ids
    assert_eq!(
        output.get_type_ids(),
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    );
    // START quicktour_encode_batch
    let output = tokenizer.encode_batch(vec!["Hello, y'all!", "How are you 😁 ?"], true)?;
    // END quicktour_encode_batch
    println!("{:?}", output);
    // START quicktour_encode_batch_pair
    let output = tokenizer.encode_batch(
        vec![
            ("Hello, y'all!", "How are you 😁 ?"),
            ("Hello to you too!", "I'm fine, thank you!"),
        ],
        true,
    )?;
    // END quicktour_encode_batch_pair
    println!("{:?}", output);
    // START quicktour_enable_padding
    use tokenizers::PaddingParams;

    tokenizer.with_padding(Some(PaddingParams {
        pad_id: 3,
        pad_token: "[PAD]".to_string(),
        ..PaddingParams::default()
    }));
    // END quicktour_enable_padding
    // START quicktour_print_batch_tokens
    let output = tokenizer.encode_batch(vec!["Hello, y'all!", "How are you 😁 ?"], true)?;
    println!("{:?}", output[1].get_tokens());
    // ["[CLS]", "How", "are", "you", "[UNK]", "?", "[SEP]", "[PAD]"]
    // END quicktour_print_batch_tokens
    assert_eq!(
        output[1].get_tokens(),
        ["[CLS]", "How", "are", "you", "[UNK]", "?", "[SEP]", "[PAD]"]
    );
    // START quicktour_print_attention_mask
    println!("{:?}", output[1].get_attention_mask());
    // [1, 1, 1, 1, 1, 1, 1, 0]
    // END quicktour_print_attention_mask
    assert_eq!(output[1].get_attention_mask(), [1, 1, 1, 1, 1, 1, 1, 0]);
    Ok(())
}

#[test]
fn pipeline() -> tokenizers::Result<()> {
    // START pipeline_reload_tokenizer
    use tokenizers::Tokenizer;

    let mut tokenizer = Tokenizer::from_file("data/tokenizer-wiki.json")?;
    // END pipeline_reload_tokenizer
    // START pipeline_setup_normalizer
    use tokenizers::normalizers::{
        strip::StripAccents, unicode::NFD, utils::Sequence as NormalizerSequence,
    };

    let normalizer = NormalizerSequence::new(vec![NFD.into(), StripAccents.into()]);
    // END pipeline_setup_normalizer
    // START pipeline_test_normalizer
    use tokenizers::{NormalizedString, Normalizer};

    let mut normalized = NormalizedString::from("Héllò hôw are ü?");
    normalizer.normalize(&mut normalized)?;

    println!("{}", normalized.get());
    // "Hello how are u?"
    // END pipeline_test_normalizer
    assert_eq!(normalized.get(), "Hello how are u?");
    // START pipeline_replace_normalizer
    tokenizer.with_normalizer(normalizer);
    // END pipeline_replace_normalizer
    // START pipeline_setup_pre_tokenizer
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer};

    let pre_tokenizer = Whitespace::default();
    let mut pre_tokenized = PreTokenizedString::from("Hello! How are you? I'm fine, thank you.");

    pre_tokenizer.pre_tokenize(&mut pre_tokenized)?;

    println!(
        "{:?}",
        pre_tokenized.get_splits(OffsetReferential::Original, OffsetType::Byte)
    );
    // [("Hello", (0, 5), None), ("!", (5, 6), None), ("How", (7, 10), None),
    //  ("are", (11, 14), None), ("you", (15, 18), None), ("?", (18, 19), None),
    //  ("I", (20, 21), None), ("\'", (21, 22), None), ("m", (22, 23), None),
    //  ("fine", (24, 28), None), (",", (28, 29), None), ("thank", (30, 35), None),
    //  ("you", (36, 39), None), (".", (39, 40), None)]
    // END pipeline_setup_pre_tokenizer
    assert_eq!(
        pre_tokenized.get_splits(OffsetReferential::Original, OffsetType::Byte),
        vec![
            ("Hello", (0, 5), &None),
            ("!", (5, 6), &None),
            ("How", (7, 10), &None),
            ("are", (11, 14), &None),
            ("you", (15, 18), &None),
            ("?", (18, 19), &None),
            ("I", (20, 21), &None),
            ("\'", (21, 22), &None),
            ("m", (22, 23), &None),
            ("fine", (24, 28), &None),
            (",", (28, 29), &None),
            ("thank", (30, 35), &None),
            ("you", (36, 39), &None),
            (".", (39, 40), &None)
        ]
    );
    // START pipeline_combine_pre_tokenizer
    use tokenizers::pre_tokenizers::{digits::Digits, sequence::Sequence};

    let pre_tokenizer = Sequence::new(vec![Whitespace::default().into(), Digits::new(true).into()]);
    let mut pre_tokenized = PreTokenizedString::from("Call 911!");

    pre_tokenizer.pre_tokenize(&mut pre_tokenized)?;

    println!(
        "{:?}",
        pre_tokenized.get_splits(OffsetReferential::Original, OffsetType::Byte)
    );
    // END pipeline_combine_pre_tokenizer
    assert_eq!(
        pre_tokenized.get_splits(OffsetReferential::Original, OffsetType::Byte),
        vec![
            ("Call", (0, 4), &None),
            ("9", (5, 6), &None),
            ("1", (6, 7), &None),
            ("1", (7, 8), &None),
            ("!", (8, 9), &None)
        ]
    );
    // START pipeline_replace_pre_tokenizer
    tokenizer.with_pre_tokenizer(pre_tokenizer);
    // END pipeline_replace_pre_tokenizer
    // START pipeline_setup_processor
    use tokenizers::processors::template::TemplateProcessing;

    tokenizer.with_post_processor(
        TemplateProcessing::builder()
            .try_single("[CLS] $A [SEP]")
            .unwrap()
            .try_pair("[CLS] $A [SEP] $B:1 [SEP]:1")
            .unwrap()
            .special_tokens(vec![("[CLS]", 1), ("[SEP]", 2)])
            .build()
            .unwrap(),
    );
    // END pipeline_setup_processor
    // START pipeline_test_decoding
    let output = tokenizer.encode("Hello, y'all! How are you 😁 ?", true)?;
    println!("{:?}", output.get_ids());
    // [1, 27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35, 2]

    let decoded = tokenizer.decode(
        vec![1, 27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35, 2],
        true,
    )?;
    println!("{}", decoded);
    // "Hello , y ' all ! How are you ?"
    // END pipeline_test_decoding

    Ok(())
}

#[test]
#[ignore]
fn train_pipeline_bert() -> tokenizers::Result<()> {
    // START bert_setup_tokenizer
    use tokenizers::models::wordpiece::WordPiece;
    use tokenizers::Tokenizer;

    let mut bert_tokenizer = Tokenizer::new(
        WordPiece::builder()
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap(),
    );
    // END bert_setup_tokenizer
    // START bert_setup_normalizer
    use tokenizers::normalizers::utils::Sequence as NormalizerSequence;
    use tokenizers::normalizers::{strip::StripAccents, unicode::NFD, utils::Lowercase};

    bert_tokenizer.with_normalizer(NormalizerSequence::new(vec![
        NFD.into(),
        Lowercase.into(),
        StripAccents.into(),
    ]));
    // END bert_setup_normalizer
    // START bert_setup_pre_tokenizer
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    bert_tokenizer.with_pre_tokenizer(Whitespace::default());
    // END bert_setup_pre_tokenizer
    // START bert_setup_processor
    use tokenizers::processors::template::TemplateProcessing;

    bert_tokenizer.with_post_processor(
        TemplateProcessing::builder()
            .try_single("[CLS] $A [SEP]")
            .unwrap()
            .try_pair("[CLS] $A [SEP] $B:1 [SEP]:1")
            .unwrap()
            .special_tokens(vec![("[CLS]", 1), ("[SEP]", 2)])
            .build()
            .unwrap(),
    );
    // END bert_setup_processor
    // START bert_train_tokenizer
    use tokenizers::models::{wordpiece::WordPieceTrainer, TrainerWrapper};

    let mut trainer: TrainerWrapper = WordPieceTrainer::builder()
        .vocab_size(30_522)
        .special_tokens(vec![
            AddedToken::from("[UNK]", true),
            AddedToken::from("[CLS]", true),
            AddedToken::from("[SEP]", true),
            AddedToken::from("[PAD]", true),
            AddedToken::from("[MASK]", true),
        ])
        .build()
        .into();
    let files = vec![
        "data/wikitext-103-raw/wiki.train.raw".into(),
        "data/wikitext-103-raw/wiki.test.raw".into(),
        "data/wikitext-103-raw/wiki.valid.raw".into(),
    ];
    bert_tokenizer.train_from_files(&mut trainer, files)?;

    bert_tokenizer.save("data/bert-wiki.json", false)?;
    // END bert_train_tokenizer
    Ok(())
}

#[test]
fn pipeline_bert() -> tokenizers::Result<()> {
    let mut bert_tokenizer = Tokenizer::from_file("data/bert-wiki.json")?;

    // START bert_test_decoding
    let output = bert_tokenizer.encode("Welcome to the 🤗 Tokenizers library.", true)?;
    println!("{:?}", output.get_tokens());
    // ["[CLS]", "welcome", "to", "the", "[UNK]", "tok", "##eni", "##zer", "##s", "library", ".", "[SEP]"]

    let decoded = bert_tokenizer.decode(output.get_ids().to_vec(), true)?;
    println!("{}", decoded);
    // "welcome to the tok ##eni ##zer ##s library ."
    // END bert_test_decoding
    assert_eq!(
        output.get_tokens(),
        &[
            "[CLS]", "welcome", "to", "the", "[UNK]", "tok", "##eni", "##zer", "##s", "library",
            ".", "[SEP]"
        ]
    );
    assert_eq!(decoded, "welcome to the tok ##eni ##zer ##s library .");
    // START bert_proper_decoding
    use tokenizers::decoders::wordpiece::WordPiece as WordPieceDecoder;

    bert_tokenizer.with_decoder(WordPieceDecoder::default());
    let decoded = bert_tokenizer.decode(output.get_ids().to_vec(), true)?;
    // "welcome to the tokenizers library."
    // END bert_proper_decoding
    assert_eq!(decoded, "welcome to the tokenizers library.");

    Ok(())
}
