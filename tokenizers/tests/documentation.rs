use tokenizers::models::bpe::{BpeTrainer, BpeTrainerBuilder, BPE};
use tokenizers::normalizers::{Sequence, Strip, NFC};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, TokenizerBuilder};
use tokenizers::{DecoderWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper};
use tokenizers::{Tokenizer, TokenizerImpl};

#[test]
fn train_tokenizer() {
    let vocab_size: usize = 100;
    let tokenizer = TokenizerBuilder::new()
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

    let trainer = BpeTrainerBuilder::new()
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
        .train(&trainer, vec!["data/small.txt".to_string()])
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
    let (mut tokenizer, trainer) = quicktour_get_tokenizer_trainer()?;

    // START train
    let files = ["test", "train", "valid"]
        .iter()
        .map(|split| format!("data/wikitext-103-raw/wiki.{}.raw", split))
        .collect::<Vec<_>>();
    tokenizer.train_and_replace(&trainer, files)?;
    // END train
    // START reload_model
    use std::path::Path;
    use tokenizers::Model;

    let saved_files = tokenizer
        .get_model()
        .save(&Path::new("data"), Some("wiki"))?;
    tokenizer.with_model(
        BPE::from_file(
            saved_files[0].to_str().unwrap(),
            &saved_files[1].to_str().unwrap(),
        )
        .unk_token("[UNK]".to_string())
        .build()?,
    );
    // END reload_model
    // START save
    tokenizer.save("data/tokenizer-wiki.json", false)?;
    // END save

    Ok(())
}

#[allow(unused_imports)]
fn quicktour_get_tokenizer_trainer() -> tokenizers::Result<(
    TokenizerImpl<
        BPE,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >,
    BpeTrainer,
)> {
    // START init_tokenizer
    use tokenizers::models::bpe::BPE;
    use tokenizers::TokenizerBuilder;

    let mut tokenizer: TokenizerImpl<
        BPE,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    > = TokenizerImpl::new(BPE::default());
    // END init_tokenizer
    // START init_trainer
    use tokenizers::models::bpe::BpeTrainer;

    let trainer = BpeTrainer::builder()
        .special_tokens(vec![
            AddedToken::from("[UNK]", true),
            AddedToken::from("[CLS]", true),
            AddedToken::from("[SEP]", true),
            AddedToken::from("[PAD]", true),
            AddedToken::from("[MASK]", true),
        ])
        .build();
    // END init_trainer
    // START init_pretok
    use tokenizers::pre_tokenizers::whitespace::Whitespace;

    tokenizer.with_pre_tokenizer(Whitespace::default());
    // END init_pretok

    Ok((tokenizer, trainer))
}

#[test]
fn quicktour() -> tokenizers::Result<()> {
    // START reload_tokenizer
    let mut tokenizer = Tokenizer::from_file("data/tokenizer-wiki.json")?;
    // END reload_tokenizer
    // START encode
    let output = tokenizer.encode("Hello, y'all! How are you 😁 ?", true)?;
    // END encode
    // START print_tokens
    println!("{:?}", output.get_tokens());
    // ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?",]
    // END print_tokens
    assert_eq!(
        output.get_tokens(),
        ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?",]
    );
    // START print_ids
    println!("{:?}", output.get_ids());
    // [27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35]
    // END print_ids
    assert_eq!(
        output.get_ids(),
        [27253, 16, 93, 11, 5097, 5, 7961, 5112, 6218, 0, 35]
    );
    // START print_offsets
    println!("{:?}", output.get_offsets()[9]);
    // (26, 30)
    // END print_offsets
    assert_eq!(output.get_offsets()[9], (26, 30));
    // START use_offsets
    let sentence = "Hello, y'all! How are you 😁 ?";
    println!("{}", &sentence[26..30]);
    // "😁"
    // END use_offsets
    // START check_sep
    println!("{}", tokenizer.token_to_id("[SEP]").unwrap());
    // 2
    // END check_sep
    assert_eq!(tokenizer.token_to_id("[SEP]"), Some(2));
    // START init_template_processing
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
    // END init_template_processing
    // START print_special_tokens
    let output = tokenizer.encode("Hello, y'all! How are you 😁 ?", true)?;
    println!("{:?}", output.get_tokens());
    // ["[CLS]", "Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?", "[SEP]"]
    // END print_special_tokens
    assert_eq!(
        output.get_tokens(),
        ["[CLS]", "Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?", "[SEP]"]
    );
    // START print_special_tokens_pair
    let output = tokenizer.encode(("Hello, y'all!", "How are you 😁 ?"), true)?;
    println!("{:?}", output.get_tokens());
    // ["[CLS]", "Hello", ",", "y", "'", "all", "!", "[SEP]", "How", "are", "you", "[UNK]", "?", "[SEP]"]
    // END print_special_tokens_pair
    assert_eq!(
        output.get_tokens(),
        [
            "[CLS]", "Hello", ",", "y", "'", "all", "!", "[SEP]", "How", "are", "you", "[UNK]",
            "?", "[SEP]"
        ]
    );
    // START print_type_ids
    println!("{:?}", output.get_type_ids());
    // [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    // END print_type_ids
    assert_eq!(
        output.get_type_ids(),
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    );
    // START encode_batch
    let output = tokenizer.encode_batch(vec!["Hello, y'all!", "How are you 😁 ?"], true)?;
    // END encode_batch
    println!("{:?}", output);
    // START encode_batch_pair
    let output = tokenizer.encode_batch(
        vec![
            ("Hello, y'all!", "How are you 😁 ?"),
            ("Hello to you too!", "I'm fine, thank you!"),
        ],
        true,
    )?;
    // END encode_batch_pair
    println!("{:?}", output);
    // START enable_padding
    use tokenizers::PaddingParams;

    tokenizer.with_padding(Some(PaddingParams {
        pad_id: 3,
        pad_token: "[PAD]".to_string(),
        ..PaddingParams::default()
    }));
    // END enable_padding
    // START print_batch_tokens
    let output = tokenizer.encode_batch(vec!["Hello, y'all!", "How are you 😁 ?"], true)?;
    println!("{:?}", output[1].get_tokens());
    // ["[CLS]", "How", "are", "you", "[UNK]", "?", "[SEP]", "[PAD]"]
    // END print_batch_tokens
    assert_eq!(
        output[1].get_tokens(),
        ["[CLS]", "How", "are", "you", "[UNK]", "?", "[SEP]", "[PAD]"]
    );
    // START print_attention_mask
    println!("{:?}", output[1].get_attention_mask());
    // [1, 1, 1, 1, 1, 1, 1, 0]
    // END print_attention_mask
    assert_eq!(output[1].get_attention_mask(), [1, 1, 1, 1, 1, 1, 1, 0]);
    Ok(())
}
