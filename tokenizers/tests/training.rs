use tokenizers::models::bpe::BPE;
use tokenizers::{DecoderWrapper, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper};
use tokenizers::{Model, TokenizerBuilder};

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
