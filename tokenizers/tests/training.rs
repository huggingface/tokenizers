use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::{DecoderWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper};
use tokenizers::{Model, Tokenizer, TokenizerBuilder};

#[test]
fn bpe_values_after_training() {
    let mut tokenizer = TokenizerBuilder::<
        BPE,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >::default()
    .with_model(
        BPE::builder()
            .unk_token("[UNK]".to_string())
            .dropout(0.1)
            .build()
            .unwrap(),
    )
    .build()
    .unwrap();
    let mut trainer = tokenizer.get_model().get_trainer();
    tokenizer
        .train_from_files(&mut trainer, vec!["./data/small.txt".to_string()])
        .unwrap();
    assert_eq!(tokenizer.get_model().dropout, Some(0.1));
    assert_eq!(tokenizer.get_model().unk_token, Some("[UNK]".to_string()));
}

#[test]
fn bpe_continuing_subword_prefix_error() {
    let mut tokenizer = TokenizerBuilder::<
        BPE,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >::default()
    .with_model(
        BPE::builder()
            .unk_token("[UNK]".to_string())
            .continuing_subword_prefix("##".to_string())
            .build()
            .unwrap(),
    )
    .with_pre_tokenizer(Some(PreTokenizerWrapper::Whitespace(Whitespace::default())))
    .build()
    .unwrap();
    let mut trainer = tokenizer.get_model().get_trainer();
    tokenizer
        .train_from_files(&mut trainer, vec!["./data/small.txt".to_string()])
        .unwrap();
    tokenizer.save("tokenizer.json", true).unwrap();
    let tokenizer = Tokenizer::from_file("tokenizer.json").unwrap();
    assert_eq!(tokenizer.get_vocab_size(false), 1526);

    std::fs::remove_file("tokenizer.json").unwrap();
}
