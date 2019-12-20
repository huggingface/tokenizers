#[macro_use]
extern crate criterion;

use std::collections::HashMap;

use criterion::black_box;
use criterion::Criterion;
use tokenizers::models::bpe::{BpeTrainer, BpeTrainerConfig};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::tokenizer::{EncodeInput, Tokenizer, Trainer};

fn bpe_benchmark(c: &mut Criterion) {
    let word_counts: HashMap<String, u32> = [
        (String::from("The"), 1),
        (String::from("quick"), 1),
        (String::from("brown"), 1),
        (String::from("fox"), 1),
        (String::from("jumps"), 1),
        (String::from("over"), 1),
        (String::from("the"), 1),
        (String::from("lazy"), 1),
        (String::from("dog"), 1),
    ].iter().cloned().collect();
    let trainer = BpeTrainer::new(BpeTrainerConfig::new(0, 100));

    c.bench_function("BPE train", |b| {
        b.iter(|| trainer.train(black_box(word_counts.clone())))
    });

    let bpe = trainer.train(word_counts);
    let mut tokenizer = Tokenizer::new(Box::new(bpe).unwrap());
    tokenizer.with_pre_tokenizer(Box::new(ByteLevel::new(true)));
    tokenizer.with_decoder(Box::new(ByteLevel::new(false)));
    let sentence = "The quick brown fox jumps over the lazy dog";

    c.bench_function("BPE tokenize", |b| {
        b.iter(|| tokenizer.encode(EncodeInput::Single(String::from(sentence))))
    });
}

criterion_group!(benches, bpe_benchmark);
criterion_main!(benches);
