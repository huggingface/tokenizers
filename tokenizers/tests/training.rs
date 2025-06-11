use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::pre_tokenizers::random_chunk::RandomChunkSplit;
use tokenizers::pre_tokenizers::random_whitespace::RandomWhitespaceSplit;
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
    .with_pre_tokenizer(Some(PreTokenizerWrapper::Whitespace(Whitespace {})))
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

#[test]
fn bpe_training_with_random_chunk_split() {
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
            .build()
            .unwrap(),
    )
    .with_pre_tokenizer(Some(PreTokenizerWrapper::RandomChunkSplit(
        RandomChunkSplit::new(2, 5)
    )))
    .build()
    .unwrap();
    
    let mut trainer = tokenizer.get_model().get_trainer();
    tokenizer
        .train_from_files(&mut trainer, vec!["./data/small.txt".to_string()])
        .unwrap();
    
    // Save and reload the tokenizer to test serialization
    tokenizer.save("random_chunk_tokenizer.json", true).unwrap();
    let tokenizer = Tokenizer::from_file("random_chunk_tokenizer.json").unwrap();
    
    // Verify the model works by encoding a text
    let encoding = tokenizer.encode("Hello world", false).unwrap();
    assert!(encoding.get_tokens().len() > 0);
    
    std::fs::remove_file("random_chunk_tokenizer.json").unwrap();
}

#[test]
fn bpe_training_with_random_whitespace_split() {
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
            .build()
            .unwrap(),
    )
    .with_pre_tokenizer(Some(PreTokenizerWrapper::RandomWhitespaceSplit(
        RandomWhitespaceSplit::new(0.3)
    )))
    .build()
    .unwrap();
    
    let mut trainer = tokenizer.get_model().get_trainer();
    tokenizer
        .train_from_files(&mut trainer, vec!["./data/small.txt".to_string()])
        .unwrap();
    
    // Save and reload the tokenizer to test serialization
    tokenizer.save("random_whitespace_tokenizer.json", true).unwrap();
    let tokenizer = Tokenizer::from_file("random_whitespace_tokenizer.json").unwrap();
    
    // Verify the model works by encoding a text
    let encoding = tokenizer.encode("Hello world", false).unwrap();
    assert!(encoding.get_tokens().len() > 0);
    
    std::fs::remove_file("random_whitespace_tokenizer.json").unwrap();
}

#[test]
fn bpe_training_with_deterministic_random_pretokenizers() {
    // Test with RandomChunkSplit in deterministic mode
    let mut tokenizer1 = TokenizerBuilder::<
        BPE,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >::default()
    .with_model(
        BPE::builder()
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap(),
    )
    .with_pre_tokenizer(Some(PreTokenizerWrapper::RandomChunkSplit(
        RandomChunkSplit::new(2, 4).with_deterministic(true)
    )))
    .build()
    .unwrap();
    
    let mut trainer1 = tokenizer1.get_model().get_trainer();
    tokenizer1
        .train_from_files(&mut trainer1, vec!["./data/small.txt".to_string()])
        .unwrap();
    
    // Test with RandomWhitespaceSplit in deterministic mode
    let mut tokenizer2 = TokenizerBuilder::<
        BPE,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >::default()
    .with_model(
        BPE::builder()
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap(),
    )
    .with_pre_tokenizer(Some(PreTokenizerWrapper::RandomWhitespaceSplit(
        RandomWhitespaceSplit::new(0.3).with_deterministic(true)
    )))
    .build()
    .unwrap();
    
    let mut trainer2 = tokenizer2.get_model().get_trainer();
    tokenizer2
        .train_from_files(&mut trainer2, vec!["./data/small.txt".to_string()])
        .unwrap();
    
    // Encode the same text with both tokenizers to verify they work
    let sample_text = "Hello world, this is a test for multi-word tokenization";
    let encoding1 = tokenizer1.encode(sample_text, false).unwrap();
    let encoding2 = tokenizer2.encode(sample_text, false).unwrap();
    
    assert!(encoding1.get_tokens().len() > 0);
    assert!(encoding2.get_tokens().len() > 0);
}
