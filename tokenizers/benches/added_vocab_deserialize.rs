#[macro_use]
extern crate criterion;

use criterion::{black_box, Criterion};
use tokenizers::tokenizer::{AddedToken, Tokenizer};
use tokenizers::models::wordlevel::WordLevel;
use std::collections::HashMap;

fn serialized_tokenizer(size: usize) -> String {
    // Create a very small model
    let mut vocab = HashMap::new();
    vocab.insert("a".to_string(), 0);
    let model = WordLevel::builder().vocab(vocab).unk_token("[UNK]".into()).build().unwrap();
    let mut tokenizer = Tokenizer::new(model);
    // Add many tokens to the added vocabulary
    let tokens: Vec<_> = (0..size)
        .map(|i| AddedToken::from(format!("tok{i}"), false))
        .collect();
    tokenizer.add_tokens(&tokens);
    serde_json::to_string(&tokenizer).unwrap()
}

fn bench_deserialize(c: &mut Criterion) {
    for &size in &[10_000usize, 100_000, 400_000] {
        let json = serialized_tokenizer(size);
        let label = format!("deserialize_added_vocab_{size}");
        c.bench_function(&label, |b| {
            b.iter(|| {
                let tok: Tokenizer = black_box(serde_json::from_str(&json).unwrap());
                black_box(tok);
            })
        });
    }
}

criterion_group!(benches, bench_deserialize);
criterion_main!(benches);
