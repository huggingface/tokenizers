#[cfg(not(debug_assertions))]
use assert_approx_eq::assert_approx_eq;
use tokenizers::models::bne::{BneTrainer, BNE};
use tokenizers::{
    normalizers, AddedToken, NormalizerWrapper, PreTokenizerWrapper, SplitDelimiterBehavior,
};
use tokenizers::{pre_tokenizers, DecoderWrapper, Model, Tokenizer};

#[test]
fn test_load_bne_from_file() {
    let model = BNE::from_file("data/bne-vocab.json", "data/bne-merges.txt")
        .build()
        .unwrap();
}
/*
#[test]
fn test_decoding_with_added_bpe() {
    let mut tokenizer = Tokenizer::from_file("data/bne.json").unwrap();
    tokenizer.with_pre_tokenizer(Some(PreTokenizerWrapper::from(
        pre_tokenizers::byte_level::ByteLevel::new(true, true, true),
    )));
    tokenizer.with_decoder(Some(DecoderWrapper::from(
        pre_tokenizers::byte_level::ByteLevel::new(true, true, true),
    )));
    let encoded = tokenizer.encode("understanding", false).unwrap();
    println!("{}", encoded.get_ids()[0]);
    let encoded = tokenizer
        .encode("Hey! how is this token: 嗎", false)
        .unwrap();
    assert_eq!(
        encoded.get_ids(),
        [11754, 5, 1853, 325, 558, 29106, 30, 11098, 198, 189]
    );
    assert_eq!(
        encoded.get_tokens(),
        ["ĠHey", "!", "Ġhow", "Ġis", "Ġthis", "Ġtoken", ":", "Ġå", "Ĺ", "İ"]
    );

    let decoded = tokenizer.decode(encoded.get_ids(), false);
    assert_eq!(decoded.unwrap(), "Hey! how is this token: 嗎");

    tokenizer.add_tokens(&[AddedToken::from("д", false).normalized(true)]);
    let encoded = tokenizer
        .encode("Hey! how is this token: д", false)
        .unwrap();
    assert_eq!(
        encoded.get_ids(),
        [19182, 0, 1268, 602, 82, 62428, 82, 4037, 25, 220, 128257]
    );
    assert_eq!(
        encoded.get_tokens(),
        ["Hey", "!", "Ġhow", "Ġi", "s", "Ġthi", "s", "Ġtoken", ":", "Ġ", "Ð´"]
    );
    let decoded = tokenizer.decode(encoded.get_ids(), false);
    assert_eq!(decoded.unwrap(), "Hey! how is this token: д")
}*/
/*
#[test]
fn test_tokenize_bne_from_file() {
    let model = BNE::from_file("data/bne-vocab.json", "data/bne-merges.txt").build().unwrap();
    let string = "This is a teststring. Hopefully this works well!";
    let string2 = "吾輩《わがはい》は猫である。名前はまだ無い。";
    assert_eq!(
        model
            .tokenize(string)
            .unwrap()
            .iter()
            .map(|tok| tok.value.clone())
            .collect::<Vec<_>>(),
        vec![
            "吾輩",
            "《",
            "わが",
            "はい",
            "》",
            "は",
            "猫",
            "である",
            "。",
            "名前",
            "はまだ",
            "無い",
            "。"
        ]
    );
}

#[test]
fn test_train_unigram_from_file() {
    let content = read_to_string("data/small.txt").unwrap();
    let mut word_counts = HashMap::new();
    content.split_whitespace().for_each(|word| {
        // This is important for the test of char vs u8
        let word = format!("▁{word}");
        *word_counts.entry(word).or_insert(0) += 1;
    });

    // println!("Words counts {:?}", word_counts);

    let trainer = BneTrainer::builder()
        .show_progress(false)
        //.unk_token(Some("<UNK>".into()))
        .build();
    let mut model = BNE::default();

    let sentences: Vec<_> = word_counts
        .iter()
        .map(|(s, i)| (s.to_owned(), *i))
        .collect();
    trainer.do_train(sentences, &mut model).unwrap();
    assert_eq!(model.get_vocab_size(), 719);
}
*/
