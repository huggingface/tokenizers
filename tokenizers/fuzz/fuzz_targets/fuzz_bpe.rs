#![no_main]
use arbitrary::Arbitrary;
use std::collections::HashSet;
use tokenizers::{
    models::bpe::{trainer::BpeTrainerBuilder, BpeBuilder, BpeTrainer, Merges, Vocab, BPE},
    normalizers::NormalizerWrapper,
    pre_tokenizers::byte_level::ByteLevel,
    AddedToken, Result, TokenizerBuilder,
};

use libfuzzer_sys::fuzz_target;

macro_rules! unwrap_or_return {
    ( $e:expr ) => {
        match $e {
            Ok(x) => x,
            Err(_) => return,
        }
    };
}

#[derive(Debug, Arbitrary)]
struct BpeBuilderParams {
    vocab: Vocab,
    merges: Merges,
    #[arbitrary(with = |u: &mut arbitrary::Unstructured| u.int_in_range(0..=1024))]
    cache_capacity: usize,
    dropout: f32,
    unk_token: String,
    continuing_sub_word_prefix: String,
    end_of_word_suffix: String,
    fuse_unk: bool,
    byte_fallback: bool,
}

fn fuzzed_bpe(p: BpeBuilderParams) -> Result<BPE> {
    BpeBuilder::new()
        .vocab_and_merges(p.vocab, p.merges)
        //.cache_capacity(p.cache_capacity)
        .dropout(p.dropout)
        .unk_token(p.unk_token)
        .continuing_subword_prefix(p.continuing_sub_word_prefix)
        .end_of_word_suffix(p.end_of_word_suffix)
        .fuse_unk(p.fuse_unk)
        .byte_fallback(p.byte_fallback)
        .build()
}

#[derive(Debug, Arbitrary)]
struct BpeTrainerBuilderParams {
    #[arbitrary(with = |u: &mut arbitrary::Unstructured| u.int_in_range(0..=1024))]
    min_frequency: u32,
    #[arbitrary(with = |u: &mut arbitrary::Unstructured| u.int_in_range(0..=1024))]
    vocab_size: usize,
    special_tokens: Vec<AddedToken>,
    limit_alphabet: usize,
    initial_alphabet: HashSet<char>,
    continuing_subword_prefix: String,
    end_of_word_suffix: String,
    max_token_length: Option<usize>,
}

fn fuzzed_bpe_trainer(t: BpeTrainerBuilderParams) -> BpeTrainer {
    BpeTrainerBuilder::new()
        .min_frequency(t.min_frequency)
        .vocab_size(t.vocab_size)
        .show_progress(false)
        .special_tokens(t.special_tokens)
        .limit_alphabet(t.limit_alphabet)
        .initial_alphabet(t.initial_alphabet)
        .continuing_subword_prefix(t.continuing_subword_prefix)
        .end_of_word_suffix(t.end_of_word_suffix)
        .max_token_length(t.max_token_length)
        .build()
}

#[derive(Arbitrary, Debug)]
struct TokenizerBuilderParams {
    normalizer: Option<NormalizerWrapper>,
}

#[derive(Arbitrary, Debug)]
struct Ctx {
    bpe_builder_params: BpeBuilderParams,
    bpe_trainer_builder_params: BpeTrainerBuilderParams,
    tokenizer_builder_params: TokenizerBuilderParams,
    training_set: Vec<String>,
}

fuzz_target!(|ctx: Ctx| {
    let bpe = unwrap_or_return!(fuzzed_bpe(ctx.bpe_builder_params));
    let mut trainer = fuzzed_bpe_trainer(ctx.bpe_trainer_builder_params);
    let mut tokenizer = unwrap_or_return!(TokenizerBuilder::new()
        .with_model(bpe)
        .with_normalizer(ctx.tokenizer_builder_params.normalizer)
        .with_pre_tokenizer(Some(ByteLevel::default()))
        .with_post_processor(Some(ByteLevel::default()))
        .with_decoder(Some(ByteLevel::default()))
        .build());
    let _ = std::hint::black_box(tokenizer.train(&mut trainer, ctx.training_set.iter()));
});
