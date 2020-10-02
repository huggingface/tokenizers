use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::normalizers::{Sequence, Strip, NFC};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::Tokenizer;
use tokenizers::{AddedToken, TokenizerBuilder};

#[test]
fn train_tokenizer() {
    // START train_tokenizer
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
        .save("data/trained-tokenizer-tests.json", pretty)
        .unwrap();
    // END train_tokenizer
}

#[test]
#[ignore]
fn load_tokenizer() {
    // START load_tokenizer
    let tokenizer = Tokenizer::from_file("data/roberta.json").unwrap();
    // END load_tokenizer

    let example = "This is an example";
    let ids = vec![713, 16, 41, 1246];
    let tokens = vec!["This", "Ġis", "Ġan", "Ġexample"];

    let encodings = tokenizer.encode(example, false).unwrap();

    assert_eq!(encodings.get_ids(), ids);
    assert_eq!(encodings.get_tokens(), tokens);

    let decoded = tokenizer.decode(ids, false).unwrap();
    assert_eq!(decoded, example);
}
